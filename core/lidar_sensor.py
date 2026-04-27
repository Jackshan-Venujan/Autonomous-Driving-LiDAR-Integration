"""
LiDAR Sensor — CARLA ray-cast LiDAR wrapper with thread-safe access.
"""

import threading
import numpy as np
import time


class LidarSensor:
    """Wraps a CARLA sensor.lidar.ray_cast actor and exposes the latest point cloud."""

    def __init__(self, world, vehicle, lidar_bp, transform):
        self._lock = threading.Lock()
        self._points: np.ndarray | None = None
        self._timestamp: float | None = None
        self._sensor = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
        self._sensor.listen(self._callback)
        print("✓ LiDAR sensor ready")

    # ------------------------------------------------------------------
    # Internal callback (runs on CARLA's sensor thread)
    # ------------------------------------------------------------------

    def _callback(self, data):
        # raw_data is a flat bytes buffer of float32: [x, y, z, intensity, ...]
        # Each point is 4 floats (16 bytes)
        raw = np.frombuffer(data.raw_data, dtype=np.float32)
        if raw.size % 4 != 0:
            return
        raw = raw.reshape(-1, 4)
        with self._lock:
            self._points = raw.copy()
            self._timestamp = data.timestamp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_latest(self):
        """Return (points_np, timestamp) or (None, None) if no data yet.

        points_np shape: (N, 4)  columns: [x, y, z, intensity]
        Coordinate frame: sensor-local  (+X=forward, +Y=right, +Z=up).
        """
        with self._lock:
            if self._points is None:
                return None, None
            return self._points.copy(), self._timestamp

    def is_ready(self) -> bool:
        with self._lock:
            return self._points is not None

    def destroy(self):
        try:
            if self._sensor is not None:
                self._sensor.stop()
                time.sleep(0.05)
                self._sensor.destroy()
                self._sensor = None
                print("✓ LiDAR sensor destroyed")
        except Exception as exc:
            print(f"⚠️  LiDAR sensor cleanup error: {exc}")
