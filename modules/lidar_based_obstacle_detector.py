"""
LiDAR-Based Obstacle Detector Module
=====================================
Phase 1: Standard LiDAR (sensor.lidar.ray_cast) integration.
Attaches to an existing ego vehicle, streams Nx4 point clouds,
provides thread-safe latest-frame access, debug stats, and watchdog.

No camera fusion, no ROS, no deep learning.

Usage (integrated):
    from modules.lidar_based_obstacle_detector import LidarManager
    lidar_mgr = LidarManager(world, vehicle)  # attach & start
    ...
    data = lidar_mgr.get_latest()              # {'points': Nx4, 'frame': int, ...}
    ...
    lidar_mgr.shutdown()                       # clean destroy

Phase 2 ready:
    - get_latest() returns a dict easy to serialize to ROS2 PointCloud2
    - All LiDAR config is centralized in LIDAR_CONFIG dict
    - Module is self-contained — no global state
"""

import carla
import numpy as np
import time
import os
import threading
import weakref
from typing import Optional, Dict, Any


# ─── Default LiDAR Configuration ───────────────────────────────────────────────

LIDAR_CONFIG = {
    # Blueprint attributes (sensor.lidar.ray_cast)
    'channels': 32,
    'range': 100.0,                 # meters
    'points_per_second': 600000,
    'rotation_frequency': 20.0,     # Hz
    'upper_fov': 10.0,              # degrees
    'lower_fov': -30.0,             # degrees
    'sensor_tick': 0.0,             # 0 = every sim tick

    # Mounting transform (relative to vehicle origin)
    'location': {'x': 0.0, 'y': 0.0, 'z': 2.4},
    'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0},

    # Debug / monitoring
    'watchdog_timeout': 5.0,        # seconds — alert if no data
    'stats_interval': 10.0,         # seconds — print throughput
    'save_interval': 0,             # 0 = disabled; N = save every Nth frame
    'output_dir': 'output/lidar',   # directory for saved .npy files
    'verbose': True,                # per-frame debug prints
}


class LidarManager:
    """
    Manages a Standard LiDAR sensor attached to an ego vehicle in CARLA.

    Responsibilities:
        - Create sensor.lidar.ray_cast blueprint with configurable attributes
        - Attach to supplied ego vehicle via rigid transform
        - Stream data via listen() callback
        - Decode raw_data to numpy float32 Nx4 = [x, y, z, intensity]
        - Thread-safe latest frame buffer
        - Periodic throughput stats (fps, pts/s)
        - Watchdog timer for stalled stream detection
        - Clean shutdown (stop + destroy)

    Thread safety:
        The callback runs on CARLA's sensor thread. All shared state is
        protected by self._lock. get_latest() returns a *copy* of the
        latest point cloud so callers never hold the lock while processing.
    """

    def __init__(
        self,
        world: carla.World,
        vehicle: carla.Vehicle,
        config: Optional[Dict[str, Any]] = None,
        auto_start: bool = True,
    ):
        """
        Create and optionally start the LiDAR sensor.

        Args:
            world:      carla.World instance
            vehicle:    carla.Vehicle to attach the LiDAR to
            config:     Optional config dict (merged over LIDAR_CONFIG defaults)
            auto_start: If True, attach + start streaming immediately
        """
        self.world = world
        self.vehicle = vehicle
        self.config = {**LIDAR_CONFIG, **(config or {})}
        self.lidar_sensor: Optional[carla.Actor] = None

        # Thread-safe latest data
        self._lock = threading.Lock()
        self._latest_cloud: Optional[np.ndarray] = None  # Nx4 float32
        self._latest_frame: int = -1
        self._latest_timestamp: float = 0.0
        self._latest_transform: Optional[carla.Transform] = None

        # Statistics
        self._total_frames: int = 0
        self._total_points: int = 0
        self._stats_start_time: float = time.time()
        self._last_receive_time: Optional[float] = None
        self._last_stats_print: float = time.time()

        # Watchdog
        self._shutdown_event = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None

        # Output dir
        out_dir = self.config.get('output_dir', 'output/lidar')
        if self.config.get('save_interval', 0) > 0:
            os.makedirs(out_dir, exist_ok=True)

        if auto_start:
            self.attach_and_start()

    # ── Public API ─────────────────────────────────────────────────────────

    def attach_and_start(self):
        """Attach the LiDAR sensor and begin streaming."""
        self._create_and_attach()
        self._start_listening()
        self._start_watchdog()

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Thread-safe access to the most recent point cloud.

        Returns:
            dict with:
                'points'    : np.ndarray float32 (N, 4) — [x, y, z, intensity]
                'frame'     : int — CARLA simulation frame
                'timestamp' : float — CARLA simulation time (seconds)
                'transform' : carla.Transform — sensor world pose at capture
            or None if no data received yet.
        """
        with self._lock:
            if self._latest_cloud is None:
                return None
            return {
                'points': self._latest_cloud.copy(),
                'frame': self._latest_frame,
                'timestamp': self._latest_timestamp,
                'transform': self._latest_transform,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Return current throughput statistics."""
        with self._lock:
            elapsed = time.time() - self._stats_start_time
            frames = self._total_frames
            points = self._total_points
        fps = frames / elapsed if elapsed > 0 else 0
        pps = points / elapsed if elapsed > 0 else 0
        avg = points / frames if frames > 0 else 0
        return {
            'elapsed_s': elapsed,
            'total_frames': frames,
            'total_points': points,
            'fps': fps,
            'points_per_sec': pps,
            'avg_points_per_frame': avg,
        }

    @property
    def is_alive(self) -> bool:
        """True if LiDAR sensor actor exists and is listening."""
        return self.lidar_sensor is not None and self.lidar_sensor.is_alive

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._total_frames

    def print_stats(self):
        """Print throughput stats to console."""
        s = self.get_stats()
        print(
            f"[LiDAR] stats: {s['elapsed_s']:.1f}s | "
            f"{s['total_frames']} frames ({s['fps']:.1f} fps) | "
            f"{s['total_points']:,} pts ({s['points_per_sec']:,.0f} pts/s) | "
            f"avg {s['avg_points_per_frame']:,.0f} pts/frame"
        )

    def maybe_print_stats(self):
        """Print stats if enough time has passed since last print."""
        now = time.time()
        if now - self._last_stats_print >= self.config['stats_interval']:
            self.print_stats()
            self._last_stats_print = now

    def shutdown(self):
        """Stop streaming, destroy sensor actor, stop watchdog."""
        print("[LiDAR] Shutting down LiDAR manager...")

        # Stop watchdog
        self._shutdown_event.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)

        # Stop and destroy sensor
        if self.lidar_sensor is not None:
            try:
                self.lidar_sensor.stop()
                print("[LiDAR]   ✓ Sensor stopped")
            except Exception as e:
                print(f"[LiDAR]   ⚠ Sensor stop error: {e}")

            time.sleep(0.2)  # let pending callbacks drain

            try:
                self.lidar_sensor.destroy()
                print("[LiDAR]   ✓ Sensor destroyed")
            except Exception as e:
                print(f"[LiDAR]   ⚠ Sensor destroy error: {e}")
            self.lidar_sensor = None

        self.print_stats()
        print("[LiDAR] ✓ LiDAR manager shutdown complete")

    # ── Internal ───────────────────────────────────────────────────────────

    def _create_and_attach(self):
        """Create LiDAR blueprint, set attributes, attach to vehicle."""
        bp_lib = self.world.get_blueprint_library()
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')

        # Set blueprint attributes from config
        attr_keys = [
            'channels', 'range', 'points_per_second',
            'rotation_frequency', 'upper_fov', 'lower_fov', 'sensor_tick',
        ]
        for key in attr_keys:
            if key in self.config:
                lidar_bp.set_attribute(key, str(self.config[key]))

        # Build transform
        loc = self.config.get('location', {})
        rot = self.config.get('rotation', {})
        transform = carla.Transform(
            carla.Location(
                x=loc.get('x', 0.0),
                y=loc.get('y', 0.0),
                z=loc.get('z', 2.4),
            ),
            carla.Rotation(
                pitch=rot.get('pitch', 0.0),
                yaw=rot.get('yaw', 0.0),
                roll=rot.get('roll', 0.0),
            ),
        )

        # Spawn and attach
        self.lidar_sensor = self.world.spawn_actor(
            lidar_bp, transform, attach_to=self.vehicle
        )

        # Log configuration
        print(f"[LiDAR] ✓ sensor.lidar.ray_cast attached (actor id={self.lidar_sensor.id})")
        print(f"[LiDAR]   channels={self.config['channels']}, "
              f"range={self.config['range']}m, "
              f"pps={self.config['points_per_second']}, "
              f"freq={self.config['rotation_frequency']}Hz")
        print(f"[LiDAR]   fov=[{self.config['lower_fov']}°, {self.config['upper_fov']}°], "
              f"mount=({loc.get('x',0)},{loc.get('y',0)},{loc.get('z',2.4)})")

    def _start_listening(self):
        """Register listen() callback using weak reference pattern."""
        self._stats_start_time = time.time()
        self._last_receive_time = time.time()
        self._last_stats_print = time.time()

        weak_self = weakref.ref(self)
        self.lidar_sensor.listen(
            lambda measurement: LidarManager._on_lidar_data(weak_self, measurement)
        )
        print("[LiDAR] ✓ listen() callback registered — streaming")

    def _start_watchdog(self):
        """Start background watchdog thread."""
        self._shutdown_event.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="LiDAR-Watchdog"
        )
        self._watchdog_thread.start()

    @staticmethod
    def _on_lidar_data(weak_self, measurement):
        """
        Static callback for LiDAR data.

        Receives carla.LidarMeasurement and decodes raw_data to numpy Nx4.
        Protected by try/except for robustness — a crash here would kill
        the CARLA sensor thread silently.
        """
        self = weak_self()
        if self is None:
            return

        try:
            frame = measurement.frame
            timestamp = measurement.timestamp
            transform = measurement.transform
            num_points = len(measurement)

            # ── Decode raw_data → numpy float32 Nx4 ──
            raw = np.frombuffer(measurement.raw_data, dtype=np.float32)

            if raw.size == 0:
                if self.config.get('verbose'):
                    print(f"[LiDAR] ⚠ Frame {frame}: empty point cloud")
                return

            if raw.size % 4 != 0:
                print(f"[LiDAR] ⚠ Frame {frame}: raw size {raw.size} not divisible by 4")
                return

            points = raw.reshape((-1, 4))  # (N, 4) = [x, y, z, intensity]

            # ── Store (thread-safe) ──
            with self._lock:
                self._latest_cloud = points
                self._latest_frame = frame
                self._latest_timestamp = timestamp
                self._latest_transform = transform
                self._total_frames += 1
                self._total_points += num_points
                self._last_receive_time = time.time()

            # ── Optional verbose logging ──
            verbose = self.config.get('verbose', False)
            total = self._total_frames
            if verbose and (total <= 3 or total % 100 == 0):
                loc = transform.location
                print(
                    f"[LiDAR] frame={frame:>6d} | pts={num_points:>6d} | "
                    f"shape={points.shape} | t={timestamp:.3f}s | "
                    f"pos=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f})"
                )

            # ── Optional save ──
            save_interval = self.config.get('save_interval', 0)
            if save_interval > 0 and total % save_interval == 0:
                out_dir = self.config.get('output_dir', 'output/lidar')
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, f"lidar_{frame:06d}.npy")
                np.save(path, points)
                print(f"[LiDAR] 💾 Saved {path} ({points.shape[0]} pts)")

        except Exception as e:
            print(f"[LiDAR] ❌ Callback error: {type(e).__name__}: {e}")

    def _watchdog_loop(self):
        """Background loop: alert if LiDAR stream stalls."""
        timeout = self.config.get('watchdog_timeout', 5.0)
        while not self._shutdown_event.is_set():
            self._shutdown_event.wait(timeout=1.0)
            if self._shutdown_event.is_set():
                break

            with self._lock:
                last = self._last_receive_time
                total = self._total_frames

            if last is None:
                continue

            gap = time.time() - last
            if gap > timeout:
                print(
                    f"[LiDAR] ⚠ WATCHDOG: No data for {gap:.1f}s "
                    f"(timeout={timeout}s, total_frames={total})"
                )
