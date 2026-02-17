#!/usr/bin/env python3
"""
lidar_basic.py — Standalone Standard LiDAR demo for CARLA
=========================================================
Phase 1: Spawn vehicle, attach sensor.lidar.ray_cast, stream point clouds.
No camera fusion, no ROS, no deep learning.

Usage:
    python lidar_basic.py

Requirements:
    - CARLA server running on localhost:2000
    - carla Python package installed
    - numpy installed

Controls:
    Ctrl+C to exit (clean shutdown)
"""

import carla
import numpy as np
import time
import os
import sys
import signal
import threading
import weakref


# ─── Configuration ─────────────────────────────────────────────────────────────

CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0
CARLA_MAP = 'Town04'

# LiDAR sensor attributes (sensor.lidar.ray_cast)
LIDAR_CHANNELS = 32                # Number of vertical laser channels
LIDAR_RANGE = 100.0                # Maximum range in meters
LIDAR_POINTS_PER_SECOND = 600000   # Total points emitted per second
LIDAR_ROTATION_FREQUENCY = 20.0    # Rotations per second (Hz)
LIDAR_UPPER_FOV = 10.0             # Upper vertical FOV in degrees
LIDAR_LOWER_FOV = -30.0            # Lower vertical FOV in degrees
LIDAR_SENSOR_TICK = 0.0            # 0.0 = every simulation tick

# LiDAR mounting transform (relative to vehicle center)
LIDAR_X = 0.0       # Forward offset (m)
LIDAR_Y = 0.0       # Lateral offset (m)
LIDAR_Z = 2.4       # Height above vehicle origin (m)
LIDAR_PITCH = 0.0   # Pitch rotation (degrees)
LIDAR_YAW = 0.0     # Yaw rotation (degrees)
LIDAR_ROLL = 0.0    # Roll rotation (degrees)

# Debug / output settings.
SAVE_INTERVAL = 100            # Save a sample .npy file every N frames
STATS_INTERVAL = 5.0           # Print throughput stats every N seconds
WATCHDOG_TIMEOUT = 5.0         # Alert if no LiDAR frame received for N seconds
OUTPUT_DIR = 'output/lidar'    # Directory for saved point clouds

# Driving settings
AUTOPILOT = True               # Enable autopilot for ego vehicle
DURATION = 120                 # Run duration in seconds


# ─── LiDAR Streaming Core ──────────────────────────────────────────────────────

class LidarStreamer:
    """
    Manages a Standard LiDAR (sensor.lidar.ray_cast) attached to an ego vehicle.
    Streams point cloud data via listen() callback with debugging and watchdog.
    """

    def __init__(self):
        # CARLA handles
        self.client = None
        self.world = None
        self.vehicle = None
        self.lidar_sensor = None

        # Latest point cloud (thread-safe via lock)
        self._lock = threading.Lock()
        self._latest_cloud = None       # numpy Nx4 float32
        self._latest_frame = -1
        self._latest_timestamp = 0.0
        self._latest_transform = None

        # Statistics
        self._total_frames = 0
        self._total_points = 0
        self._stats_start_time = None
        self._last_receive_time = None

        # Watchdog
        self._watchdog_thread = None
        self._shutdown_event = threading.Event()

        # Output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Setup ──────────────────────────────────────────────────────────────

    def connect(self):
        """Step 1: Connect to CARLA server."""
        print(f"[LiDAR] Connecting to CARLA at {CARLA_HOST}:{CARLA_PORT} ...")
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(CARLA_TIMEOUT)
        self.world = self.client.load_world(CARLA_MAP)
        print(f"[LiDAR] ✓ Connected — Map: {CARLA_MAP}")

    def spawn_vehicle(self):
        """Step 2: Spawn ego vehicle (Tesla Model 3)."""
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available on this map!")
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        print(f"[LiDAR] ✓ Ego vehicle spawned: {self.vehicle.type_id} (id={self.vehicle.id})")
        if AUTOPILOT:
            self.vehicle.set_autopilot(True)
            print(f"[LiDAR]   Autopilot: ON")

    def attach_lidar(self):
        """
        Step 3: Create and attach Standard LiDAR (sensor.lidar.ray_cast).

        Blueprint attributes set:
            channels, range, points_per_second, rotation_frequency,
            upper_fov, lower_fov, sensor_tick
        """
        bp_lib = self.world.get_blueprint_library()
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')

        # Set LiDAR attributes
        lidar_bp.set_attribute('channels', str(LIDAR_CHANNELS))
        lidar_bp.set_attribute('range', str(LIDAR_RANGE))
        lidar_bp.set_attribute('points_per_second', str(LIDAR_POINTS_PER_SECOND))
        lidar_bp.set_attribute('rotation_frequency', str(LIDAR_ROTATION_FREQUENCY))
        lidar_bp.set_attribute('upper_fov', str(LIDAR_UPPER_FOV))
        lidar_bp.set_attribute('lower_fov', str(LIDAR_LOWER_FOV))
        lidar_bp.set_attribute('sensor_tick', str(LIDAR_SENSOR_TICK))

        # Print configured attributes for verification
        print(f"[LiDAR] Blueprint: sensor.lidar.ray_cast")
        print(f"[LiDAR]   channels            = {LIDAR_CHANNELS}")
        print(f"[LiDAR]   range                = {LIDAR_RANGE} m")
        print(f"[LiDAR]   points_per_second    = {LIDAR_POINTS_PER_SECOND}")
        print(f"[LiDAR]   rotation_frequency   = {LIDAR_ROTATION_FREQUENCY} Hz")
        print(f"[LiDAR]   upper_fov            = {LIDAR_UPPER_FOV}°")
        print(f"[LiDAR]   lower_fov            = {LIDAR_LOWER_FOV}°")
        print(f"[LiDAR]   sensor_tick          = {LIDAR_SENSOR_TICK}")

        # Mounting transform
        lidar_transform = carla.Transform(
            carla.Location(x=LIDAR_X, y=LIDAR_Y, z=LIDAR_Z),
            carla.Rotation(pitch=LIDAR_PITCH, yaw=LIDAR_YAW, roll=LIDAR_ROLL)
        )
        print(f"[LiDAR]   mount position       = ({LIDAR_X}, {LIDAR_Y}, {LIDAR_Z})")
        print(f"[LiDAR]   mount rotation       = ({LIDAR_PITCH}, {LIDAR_YAW}, {LIDAR_ROLL})")

        # Attach to ego vehicle
        self.lidar_sensor = self.world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=self.vehicle
        )
        print(f"[LiDAR] ✓ LiDAR attached (actor id={self.lidar_sensor.id})")

    def start_streaming(self):
        """
        Step 4: Start streaming LiDAR data via listen() callback.
        Also starts watchdog timer.
        """
        self._stats_start_time = time.time()
        self._last_receive_time = time.time()

        # Use weak reference to prevent preventing garbage collection
        weak_self = weakref.ref(self)
        self.lidar_sensor.listen(lambda data: LidarStreamer._lidar_callback(weak_self, data))
        print(f"[LiDAR] ✓ Streaming started — listen() callback registered")

        # Start watchdog
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="LiDAR-Watchdog"
        )
        self._watchdog_thread.start()
        print(f"[LiDAR] ✓ Watchdog started (timeout={WATCHDOG_TIMEOUT}s)")

    # ── Callback ───────────────────────────────────────────────────────────

    @staticmethod
    def _lidar_callback(weak_self, measurement):
        """
        Callback invoked by CARLA for each LiDAR sweep.

        Receives: carla.LidarMeasurement
            - measurement.frame         : simulation frame number
            - measurement.timestamp     : simulation time (seconds)
            - measurement.transform     : sensor world transform at capture time
            - measurement.raw_data      : buffer of float32 [x,y,z,intensity, ...]
            - measurement.channels      : number of laser channels
            - len(measurement)          : total number of 3D points

        Decodes raw_data into numpy float32 array of shape (N, 4):
            Each row = [x, y, z, intensity]
        """
        self = weak_self()
        if self is None:
            return

        try:
            frame = measurement.frame
            timestamp = measurement.timestamp
            transform = measurement.transform
            num_points = len(measurement)

            # ── Decode raw_data to numpy Nx4 ──
            # raw_data is a buffer of float32 values: x,y,z,intensity repeating
            raw = np.frombuffer(measurement.raw_data, dtype=np.float32)

            if raw.size == 0:
                print(f"[LiDAR] ⚠ Frame {frame}: Empty point cloud (0 points)")
                return

            if raw.size % 4 != 0:
                print(f"[LiDAR] ⚠ Frame {frame}: raw_data size {raw.size} not divisible by 4!")
                return

            points = raw.reshape((-1, 4))  # (N, 4) = [x, y, z, intensity]

            # Validate shape
            assert points.shape[1] == 4, f"Expected Nx4, got {points.shape}"
            assert points.dtype == np.float32, f"Expected float32, got {points.dtype}"

            # ── Store latest data (thread-safe) ──
            with self._lock:
                self._latest_cloud = points
                self._latest_frame = frame
                self._latest_timestamp = timestamp
                self._latest_transform = transform
                self._total_frames += 1
                self._total_points += num_points
                self._last_receive_time = time.time()

            # ── Per-frame debug print ──
            if self._total_frames <= 5 or self._total_frames % 50 == 0:
                loc = transform.location
                print(
                    f"[LiDAR] Frame {frame:>6d} | "
                    f"pts={num_points:>6d} | "
                    f"shape={points.shape} | "
                    f"t={timestamp:.3f}s | "
                    f"pos=({loc.x:.1f},{loc.y:.1f},{loc.z:.1f})"
                )

            # ── Save sample to .npy every N frames ──
            if SAVE_INTERVAL > 0 and self._total_frames % SAVE_INTERVAL == 0:
                filename = os.path.join(OUTPUT_DIR, f"lidar_frame_{frame:06d}.npy")
                np.save(filename, points)
                print(f"[LiDAR] 💾 Saved {filename} ({points.shape[0]} points)")

        except Exception as e:
            print(f"[LiDAR] ❌ Callback error: {type(e).__name__}: {e}")

    # ── Watchdog ───────────────────────────────────────────────────────────

    def _watchdog_loop(self):
        """Background thread that checks for stalled LiDAR stream."""
        while not self._shutdown_event.is_set():
            self._shutdown_event.wait(timeout=1.0)
            if self._shutdown_event.is_set():
                break

            with self._lock:
                last_time = self._last_receive_time
                total = self._total_frames

            if last_time is None:
                continue

            gap = time.time() - last_time
            if gap > WATCHDOG_TIMEOUT:
                print(
                    f"[LiDAR] ⚠ WATCHDOG: No data for {gap:.1f}s! "
                    f"(last frame {self._latest_frame}, total={total})"
                )

    # ── Throughput Stats ───────────────────────────────────────────────────

    def print_stats(self):
        """Print periodic throughput statistics."""
        with self._lock:
            elapsed = time.time() - self._stats_start_time
            frames = self._total_frames
            points = self._total_points

        if elapsed <= 0 or frames == 0:
            return

        fps = frames / elapsed
        pps = points / elapsed
        avg_pts = points / frames if frames > 0 else 0

        print(
            f"[LiDAR] ── Stats ──  "
            f"elapsed={elapsed:.1f}s | "
            f"frames={frames} ({fps:.1f} fps) | "
            f"total_pts={points:,} ({pps:,.0f} pts/s) | "
            f"avg_pts/frame={avg_pts:,.0f}"
        )

    # ── Data Access (thread-safe) ──────────────────────────────────────────

    def get_latest(self):
        """
        Get the latest point cloud data (thread-safe).

        Returns:
            dict with keys: 'points' (Nx4 float32), 'frame', 'timestamp', 'transform'
            or None if no data received yet.
        """
        with self._lock:
            if self._latest_cloud is None:
                return None
            return {
                'points': self._latest_cloud.copy(),
                'frame': self._latest_frame,
                'timestamp': self._latest_timestamp,
                'transform': self._latest_transform
            }

    # ── Cleanup ────────────────────────────────────────────────────────────

    def shutdown(self):
        """Clean shutdown: stop sensor, destroy actors."""
        print("\n[LiDAR] Shutting down...")

        # Signal watchdog to stop
        self._shutdown_event.set()

        # Stop LiDAR sensor listening
        if self.lidar_sensor is not None:
            try:
                self.lidar_sensor.stop()
                print("[LiDAR]   ✓ Sensor stopped")
            except Exception as e:
                print(f"[LiDAR]   ⚠ Sensor stop error: {e}")
            time.sleep(0.2)  # Allow pending callbacks to complete
            try:
                self.lidar_sensor.destroy()
                print("[LiDAR]   ✓ Sensor destroyed")
            except Exception as e:
                print(f"[LiDAR]   ⚠ Sensor destroy error: {e}")
            self.lidar_sensor = None

        # Destroy vehicle
        if self.vehicle is not None:
            try:
                self.vehicle.set_autopilot(False)
            except:
                pass
            try:
                self.vehicle.destroy()
                print("[LiDAR]   ✓ Vehicle destroyed")
            except Exception as e:
                print(f"[LiDAR]   ⚠ Vehicle destroy error: {e}")
            self.vehicle = None

        # Final stats
        self.print_stats()
        print("[LiDAR] ✓ Shutdown complete")


# ─── Main Entry Point ──────────────────────────────────────────────────────────

def main():
    streamer = LidarStreamer()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n[LiDAR] ⏹ Ctrl+C received")
        streamer.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Steps 1-4
        streamer.connect()
        streamer.spawn_vehicle()
        streamer.attach_lidar()
        streamer.start_streaming()

        # Confirm LiDAR actor exists in the world
        actors = streamer.world.get_actors()
        lidar_actors = [a for a in actors if 'lidar' in a.type_id]
        print(f"\n[LiDAR] Verification: {len(lidar_actors)} LiDAR actor(s) in world:")
        for a in lidar_actors:
            print(f"  - {a.type_id} (id={a.id})")

        # Wait for first frame
        print("\n[LiDAR] Waiting for first point cloud...")
        wait_start = time.time()
        while streamer.get_latest() is None:
            if time.time() - wait_start > 10.0:
                print("[LiDAR] ❌ Timeout waiting for first LiDAR frame!")
                streamer.shutdown()
                return
            time.sleep(0.1)
        first = streamer.get_latest()
        print(f"[LiDAR] ✓ First frame received! shape={first['points'].shape}, dtype={first['points'].dtype}")

        # ── Main monitoring loop ──
        print(f"\n[LiDAR] Running for {DURATION}s (Ctrl+C to stop)...")
        print("=" * 70)

        start = time.time()
        last_stats = start

        while time.time() - start < DURATION:
            time.sleep(0.5)

            # Print stats periodically
            if time.time() - last_stats >= STATS_INTERVAL:
                streamer.print_stats()
                last_stats = time.time()

    except Exception as e:
        print(f"\n[LiDAR] ❌ Fatal error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        streamer.shutdown()


if __name__ == '__main__':
    main()
