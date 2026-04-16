"""
LiDAR Sensor Module for CARLA Autonomous Driving Simulation
============================================================

Wraps CARLA's ray-cast LiDAR sensor into a clean, production-ready Python class
that delivers timestamped numpy point clouds to downstream modules (e.g. PointNet,
VoxelNet, PointPillars) via a lock-free single-slot deque buffer.

Hardware equivalent: Velodyne HDL-64E  (64-channel, 10 Hz, ~1.1 M pts/sec)

Coordinate frame after conversion
-----------------------------------
   +X = forward (nose of vehicle)
   +Y = left
   +Z = up
   Origin = LiDAR mount point (roof centre, 2.4 m above ground)

   This is the standard ISO 8855 vehicle/ego-centric frame expected by all
   common 3D object-detection networks.

Typical downstream usage
------------------------
   from core.lidar_sensor import LidarSensor, LidarFrame

   lidar = LidarSensor(world, ego_vehicle)
   lidar.start()

   while running:
       frame: LidarFrame | None = lidar.get_latest()
       if frame and lidar.is_healthy:
           feed_to_network(frame.points)   # shape (N, 3), float32, ego frame

   lidar.destroy()
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass

import carla
import numpy as np

# ── Module-level logger ────────────────────────────────────────────────────────
# Inherits the root logger's handler/level; callers can configure independently:
#   logging.getLogger("lidar_sensor").setLevel(logging.DEBUG)
logger = logging.getLogger("lidar_sensor")


# ══════════════════════════════════════════════════════════════════════════════
#  LidarFrame  –  immutable, typed snapshot of one 360° scan
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LidarFrame:
    """
    One complete 360° LiDAR scan, ready for network inference.

    Fields
    ------
    frame_id   : CARLA simulation frame counter (monotonically increasing int).
                 Useful for synchronisation with camera/radar frames.

    timestamp  : Seconds since simulation start (float).
                 Compare consecutive timestamps to detect frame drops.

    points     : (N, 3) float32 array in vehicle frame (+X fwd, +Y left, +Z up).
                 N varies by scene density; typically 60 000–120 000 points.
                 Direct input for PointNet, VoxelNet, PointPillars — no further
                 coordinate transform needed.

    intensity  : (N,) float32 array, range [0.0, 1.0].
                 Surface reflectivity; higher = more reflective material.
                 Some networks (e.g. PointPainting) use this as a 4th channel.

    num_points : Convenience copy of len(points).  Use for health checks and
                 quick logging without touching the array object.
    """

    frame_id   : int
    timestamp  : float
    points     : np.ndarray   # shape (N, 3), float32
    intensity  : np.ndarray   # shape (N,),   float32
    num_points : int


# ══════════════════════════════════════════════════════════════════════════════
#  LidarSensor  –  main class
# ══════════════════════════════════════════════════════════════════════════════

class LidarSensor:
    """
    Production-grade CARLA LiDAR wrapper.

    Lifecycle
    ---------
    1. Instantiate  → blueprint is configured, sensor actor is spawned.
    2. .start()     → registers the callback; sensor begins pushing frames.
    3. .get_latest()→ call from your main loop to pull the newest frame.
    4. .destroy()   → stops the sensor and removes the CARLA actor.
                       Always call this (ideally in a try/finally block).

    Thread safety
    -------------
    CARLA invokes the callback on a dedicated sensor thread.
    The deque(maxlen=1) buffer is written by the sensor thread and read by the
    main thread.  Python's GIL makes single-slot deque append+read atomic
    for our purposes, so no explicit lock is needed.

    Performance contract
    --------------------
    Callback target: < 5 ms per frame.
    The hot path only does:
      • np.frombuffer  (zero-copy view)
      • two .copy() calls (unavoidable: frombuffer returns read-only memory)
      • one in-place negation (pts[:, 1] = -pts[:, 1])
      • dataclass construction (stack allocation)
    No Python-level loops, no dynamic memory allocation beyond the two copies.
    """

    # ── Blueprint / hardware constants ────────────────────────────────────────
    # Matching a Velodyne HDL-64E for 3D detection benchmark comparability.
    # Adjust points_per_second if you change channels or rotation_frequency:
    #   points_per_second = channels × rotation_frequency × points_per_line
    #   1_120_000         = 64       × 10               × ~1750

    BLUEPRINT_ID         = "sensor.lidar.ray_cast"

    # Sensor geometry
    CHANNELS             = 64          # vertical laser lines
    RANGE_M              = 100         # max reliable range in metres
    POINTS_PER_SECOND    = 1_120_000   # total points/sec across all channels
    ROTATION_FREQUENCY   = 10          # Hz — one full 360° per 100 ms
    UPPER_FOV_DEG        = 15.0        # degrees above horizontal
    LOWER_FOV_DEG        = -25.0       # degrees below horizontal (captures road)

    # Noise / dropoff — all disabled for clean simulation data.
    # To test detector robustness, set noise_stddev > 0 and dropoff_general_rate > 0.
    ATMOSPHERE_NOISE_SEED    = 0        # deterministic seed; 0 = no random dropout
    DROPOFF_GENERAL_RATE     = 0.0      # fraction of points randomly dropped
    DROPOFF_INTENSITY_LIMIT  = 0.0      # intensity threshold for dropoff
    DROPOFF_ZERO_INTENSITY   = 0.0      # fraction of surviving points set to 0
    NOISE_STDDEV             = 0.0      # Gaussian positional noise (metres)

    # Mount position: roof centre, 2.4 m above ground.
    # Matches Waymo/Cruise convention; keeps sensor above roof turbulence.
    MOUNT_TRANSFORM = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.4))

    # Health thresholds — tune to your scene / test environment
    MIN_HEALTHY_POINTS  = 10_000       # sparse below this → mount or sensor issue
    MAX_FRAME_GAP_S     = 0.15         # >150 ms between frames = probable drop

    def __init__(self, world: carla.World, ego_vehicle: carla.Vehicle) -> None:
        """
        Configure the sensor blueprint and spawn the CARLA actor.

        Parameters
        ----------
        world       : Active carla.World instance.
        ego_vehicle : The vehicle actor the sensor is attached to.
                      All returned point coordinates are relative to this vehicle.
        """
        self.world       = world
        self.ego_vehicle = ego_vehicle

        # ── Internal state ─────────────────────────────────────────────────
        # Single-slot buffer: the sensor callback always overwrites slot 0.
        # deque(maxlen=1) automatically discards the oldest entry, so the
        # main loop always sees the freshest frame with no blocking.
        self._buffer: collections.deque[LidarFrame] = collections.deque(maxlen=1)

        self._sensor_actor: carla.Actor | None = None  # set by start()

        # Health tracking
        self._last_timestamp: float | None  = None
        self._healthy: bool                 = False

        # Latency tracking for the __main__ demo (and optional monitoring)
        self._callback_times: list[float]   = []   # seconds, one entry per frame

        # ── Spawn actor ────────────────────────────────────────────────────
        self._sensor_actor = self._spawn_sensor()
        logger.info(
            "LidarSensor spawned (actor_id=%d) attached to vehicle %d",
            self._sensor_actor.id,
            ego_vehicle.id,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _spawn_sensor(self) -> carla.Actor:
        """
        Build the blueprint with all required attributes and spawn the actor.
        Returns the live carla.Actor; raises RuntimeError on failure.
        """
        bp_lib = self.world.get_blueprint_library()
        bp     = bp_lib.find(self.BLUEPRINT_ID)

        # ── Geometry ────────────────────────────────────────────────────────
        bp.set_attribute("channels",           str(self.CHANNELS))
        bp.set_attribute("range",              str(self.RANGE_M))
        bp.set_attribute("points_per_second",  str(self.POINTS_PER_SECOND))
        bp.set_attribute("rotation_frequency", str(self.ROTATION_FREQUENCY))
        bp.set_attribute("upper_fov",          str(self.UPPER_FOV_DEG))
        bp.set_attribute("lower_fov",          str(self.LOWER_FOV_DEG))

        # ── Noise / dropoff ─────────────────────────────────────────────────
        bp.set_attribute("atmosphere_noise_seed",   str(self.ATMOSPHERE_NOISE_SEED))
        bp.set_attribute("dropoff_general_rate",    str(self.DROPOFF_GENERAL_RATE))
        bp.set_attribute("dropoff_intensity_limit", str(self.DROPOFF_INTENSITY_LIMIT))
        bp.set_attribute("dropoff_zero_intensity",  str(self.DROPOFF_ZERO_INTENSITY))
        bp.set_attribute("noise_stddev",            str(self.NOISE_STDDEV))

        actor = self.world.spawn_actor(bp, self.MOUNT_TRANSFORM, attach_to=self.ego_vehicle)
        if actor is None:
            raise RuntimeError("CARLA failed to spawn LiDAR sensor actor.")
        return actor

    def _callback(self, data: carla.LidarMeasurement) -> None:
        """
        CARLA sensor callback — runs on CARLA's internal sensor thread.

        Hot path: keep this < 5 ms.  No I/O, no Python loops.

        Raw data layout
        ---------------
        data.raw_data is a bytes buffer of N×4 float32 values:
          [ x, y, z, intensity,  x, y, z, intensity, ... ]
        where CARLA uses a LEFT-handed coordinate system:
          +X = forward, +Y = left (but left-handed!), +Z = up

        Conversion to right-hand ISO 8855 vehicle frame
        ------------------------------------------------
        Only the Y axis needs flipping:
          pts[:, 1] *= -1
        After this, +Y genuinely points to the vehicle's left.
        """
        t0 = time.perf_counter()   # wall-clock start for latency measurement

        # ── 1. Parse raw bytes → numpy ───────────────────────────────────────
        # np.frombuffer gives a zero-copy READ-ONLY view of raw_data.
        # We must .copy() before any in-place modification.
        raw  = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)

        pts  = raw[:, :3].copy()   # (N, 3) — XYZ, mutable copy
        intn = raw[:, 3].copy()    # (N,)   — intensity, mutable copy

        # ── 2. Coordinate system: CARLA left-hand → ISO 8855 right-hand ─────
        # CARLA's Y axis increases to the LEFT but in a left-handed sense,
        # so negating it maps it to a standard right-hand +Y = left frame.
        pts[:, 1] = -pts[:, 1]

        # ── 3. Pack into frozen dataclass ────────────────────────────────────
        frame = LidarFrame(
            frame_id   = data.frame,
            timestamp  = data.timestamp,
            points     = pts,
            intensity  = intn,
            num_points = len(pts),
        )

        # ── 4. Write to single-slot buffer (non-blocking) ────────────────────
        self._buffer.append(frame)

        # ── 5. Health check ──────────────────────────────────────────────────
        self._run_health_check(frame)

        # ── 6. Record callback latency ───────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        self._callback_times.append(elapsed_ms)

        if elapsed_ms > 5.0:
            # Warn if we blow the 5 ms budget; this typically means the host
            # CPU is under load.  Consider moving heavy work out of the callback.
            logger.warning("LiDAR callback slow: %.2f ms (target < 5 ms)", elapsed_ms)

    def _run_health_check(self, frame: LidarFrame) -> None:
        """
        Update self._healthy based on point count and frame timing.
        Called from the sensor thread — keep it lightweight.
        """
        ok = True

        # ── Point count check ────────────────────────────────────────────────
        if frame.num_points < self.MIN_HEALTHY_POINTS:
            logger.warning(
                "LiDAR sparse — check sensor mount  "
                "(frame=%d, points=%d, threshold=%d)",
                frame.frame_id, frame.num_points, self.MIN_HEALTHY_POINTS,
            )
            ok = False

        # ── Frame-gap check ─────────────────────────────────────────────────
        if self._last_timestamp is not None:
            gap = frame.timestamp - self._last_timestamp
            if gap > self.MAX_FRAME_GAP_S:
                logger.warning(
                    "LiDAR frame drop detected  "
                    "(frame=%d, gap=%.3f s, threshold=%.3f s)",
                    frame.frame_id, gap, self.MAX_FRAME_GAP_S,
                )
                ok = False

        self._last_timestamp = frame.timestamp
        self._healthy = ok

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Register the data callback and begin receiving frames.
        Call this once after construction.
        The callback fires automatically at ROTATION_FREQUENCY (10 Hz).
        """
        if self._sensor_actor is None:
            raise RuntimeError("Sensor actor not initialised. Was __init__ called?")
        self._sensor_actor.listen(self._callback)
        logger.info("LidarSensor started (%.0f Hz)", self.ROTATION_FREQUENCY)

    def get_latest(self) -> LidarFrame | None:
        """
        Return the most-recently completed 360° scan, or None if no frame
        has arrived yet.

        This is a non-blocking O(1) read — safe to call every simulation tick
        from the main thread without any locking overhead.

        Returns
        -------
        LidarFrame  — frozen dataclass with .points (N×3 float32), .intensity,
                      .frame_id, .timestamp, .num_points
        None        — sensor hasn't delivered its first frame yet
        """
        return self._buffer[-1] if self._buffer else None

    @property
    def is_healthy(self) -> bool:
        """
        True if the last received frame passed all health checks:
          • num_points ≥ MIN_HEALTHY_POINTS  (10 000 by default)
          • frame gap   ≤ MAX_FRAME_GAP_S    (150 ms by default)

        Will be False until the very first frame arrives.
        Intended for use in upstream watchdog logic, e.g.:
            if not lidar.is_healthy:
                trigger_safe_stop()
        """
        return self._healthy

    def get_avg_callback_latency_ms(self) -> float:
        """
        Average wall-clock time spent inside _callback() across all frames
        received so far.  Returns 0.0 if no frames have been processed yet.
        Useful for performance profiling in the __main__ demo.
        """
        if not self._callback_times:
            return 0.0
        return sum(self._callback_times) / len(self._callback_times)

    def destroy(self) -> None:
        """
        Stop the sensor and remove the CARLA actor from the simulation.

        Always call this in a try/finally block to ensure cleanup even if the
        main loop throws an exception.  CARLA actors that are not destroyed
        persist in the simulator and accumulate across test runs.
        """
        if self._sensor_actor is not None:
            try:
                self._sensor_actor.stop()    # deregister callback first
                self._sensor_actor.destroy() # then remove actor from world
                logger.info("LidarSensor destroyed (actor_id=%d)", self._sensor_actor.id)
            except Exception as exc:
                # Actor may already be gone if the world was reset
                logger.warning("LidarSensor destroy error (ignored): %s", exc)
            finally:
                self._sensor_actor = None


# ══════════════════════════════════════════════════════════════════════════════
#  __main__  –  standalone smoke test
#
#  Usage:
#    cd <project_root>
#    python -m core.lidar_sensor          # or: python core/lidar_sensor.py
#
#  Requires a running CARLA server:  ./CarlaUE4.sh  (Linux/Mac)
#                                    CarlaUE4.exe   (Windows)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # ── Logging setup ────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── CARLA connection ─────────────────────────────────────────────────────
    # Default CARLA port is 2000; change if your server uses a different one.
    CARLA_HOST = "localhost"
    CARLA_PORT = 2000

    print(f"\nConnecting to CARLA at {CARLA_HOST}:{CARLA_PORT} …")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)   # seconds; raise if server is slow to respond

    # ── Load Town03 ──────────────────────────────────────────────────────────
    # Town03 has a variety of road types (highway, urban, roundabout) —
    # good for testing the full angular range of the LiDAR.
    world = client.load_world("Town03")
    print("World loaded:", world.get_map().name)

    # Use synchronous mode so each world.tick() advances exactly one frame.
    # This is essential for deterministic recording / dataset collection.
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 10.0   # 10 Hz = 100 ms per tick
    world.apply_settings(settings)

    # ── Spawn ego vehicle ────────────────────────────────────────────────────
    bp_lib        = world.get_blueprint_library()
    vehicle_bp    = bp_lib.find("vehicle.tesla.model3")
    spawn_points  = world.get_map().get_spawn_points()

    if not spawn_points:
        print("ERROR: no spawn points available in Town03. Exiting.")
        sys.exit(1)

    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    ego_vehicle.set_autopilot(True)
    print(f"Ego vehicle spawned  (actor_id={ego_vehicle.id})")

    # ── Attach LiDAR ─────────────────────────────────────────────────────────
    lidar = LidarSensor(world, ego_vehicle)
    lidar.start()

    # ── Run for 5 seconds ────────────────────────────────────────────────────
    RUN_SECONDS = 5
    frames_seen = 0
    start_wall  = time.time()

    print(f"\n{'─'*60}")
    print(f"{'Frame':>8}  {'Timestamp':>10}  {'Pts':>8}  {'Healthy':>8}")
    print(f"{'─'*60}")

    try:
        while time.time() - start_wall < RUN_SECONDS:
            world.tick()   # advance simulation by one fixed_delta_seconds step

            frame = lidar.get_latest()
            if frame is None:
                continue   # first tick: sensor hasn't fired yet

            frames_seen += 1
            print(
                f"{frame.frame_id:>8d}  "
                f"{frame.timestamp:>10.3f}  "
                f"{frame.num_points:>8d}  "
                f"{'OK' if lidar.is_healthy else 'WARN':>8}"
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # ── Always clean up ───────────────────────────────────────────────
        # This try/finally pattern is the correct idiom for CARLA resources.
        # Without it, actors accumulate across test runs and crash the server.
        print(f"\n{'─'*60}")
        print(f"Frames received    : {frames_seen}")
        print(f"Avg callback time  : {lidar.get_avg_callback_latency_ms():.3f} ms  (target < 5 ms)")
        print(f"{'─'*60}")

        lidar.destroy()
        ego_vehicle.destroy()
        print("Actors destroyed.")

        # Restore asynchronous mode so the CARLA server isn't left ticking
        # at a forced 10 Hz after the script exits.
        settings = world.get_settings()
        settings.synchronous_mode   = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        print("CARLA settings restored to async mode.")
