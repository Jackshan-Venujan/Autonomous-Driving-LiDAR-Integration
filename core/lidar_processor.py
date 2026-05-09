"""
LiDAR Preprocessor — ground removal, range filter, ego-body filter.
"""

import numpy as np


class LidarProcessor:
    """Converts raw (N, 4) XYZІ point cloud to a clean (M, 3) XYZ array."""

    def __init__(
        self,
        ground_z_threshold: float = -1.4,
        min_range: float = 0.5,
        max_range: float = 50.0,
        ego_radius: float = 1.5,
    ):
        # Sensor sits at z=1.8 m above ground.  Points below this height
        # (sensor-local z < -1.4) are ground returns.
        self.ground_z_threshold = ground_z_threshold
        self.min_range = min_range
        self.max_range = max_range
        # Reject self-returns from the vehicle body within this horizontal radius.
        self.ego_radius = ego_radius

    def preprocess(self, points_xyzI: np.ndarray) -> np.ndarray:
        """Filter raw point cloud.

        Args:
            points_xyzI: numpy array shape (N, 4) — columns [x, y, z, intensity].

        Returns:
            numpy array shape (M, 3) — columns [x, y, z], dtype float32.
        """
        if points_xyzI is None or len(points_xyzI) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        xyz = points_xyzI[:, :3].astype(np.float32)

        # 1. Ground removal
        mask = xyz[:, 2] > self.ground_z_threshold
        xyz = xyz[mask]
        if len(xyz) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # 2. Horizontal range filter
        dist_xy = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        mask = (dist_xy >= self.min_range) & (dist_xy <= self.max_range)
        xyz = xyz[mask]
        dist_xy = dist_xy[mask]
        if len(xyz) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # 3. Ego-body filter (remove vehicle self-returns)
        mask = dist_xy > self.ego_radius
        xyz = xyz[mask]

        return xyz

    def snapshot_forward(
        self,
        points_xyz: np.ndarray,
        half_angle_deg: float = 1.0,
    ):
        """Simulate a non-rotating (stopped) LiDAR pointing straight ahead.

        Filters preprocessed points to those within ±half_angle_deg of the
        forward axis (atan2(y, x) = 0).  With the default 1° this approximates
        a single fixed-beam rangefinder firing forward.

        Args:
            points_xyz: (M, 3) output of preprocess().
            half_angle_deg: half-width of the forward cone in degrees (default 1°).

        Returns:
            (filtered_xyz, nearest_distance_m)
            filtered_xyz        — subset of input points inside the cone
            nearest_distance_m  — horizontal distance to the nearest surviving
                                  point, or None if the cone is empty.
        """
        if len(points_xyz) == 0:
            return np.zeros((0, 3), dtype=np.float32), None

        # Horizontal bearing from forward axis (+X). 0° = straight ahead.
        angles_deg = np.degrees(np.arctan2(points_xyz[:, 1], points_xyz[:, 0]))
        mask = np.abs(angles_deg) <= half_angle_deg
        filtered = points_xyz[mask]

        if len(filtered) == 0:
            return filtered, None

        dists = np.sqrt(filtered[:, 0] ** 2 + filtered[:, 1] ** 2)
        return filtered, float(np.min(dists))
