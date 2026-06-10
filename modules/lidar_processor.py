"""
LiDAR Processor Module
Parses CARLA ray_cast LiDAR point clouds, removes ground, clusters obstacles
with DBSCAN, and returns per-obstacle distance/angle/sector data.
"""

import numpy as np
import math
from typing import List, Dict, Optional

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not found. Install with: pip install scikit-learn")


class LidarProcessor:
    """
    Processes CARLA ray_cast LiDAR frames into obstacle descriptors.

    Coordinate frame (CARLA sensor frame):
        x = forward,  y = right,  z = up
    Ground is at z ≈ -sensor_height (sensor mounted at z=2.0 m on roof).
    """

    def __init__(
        self,
        ground_z_threshold: float = -1.5,
        self_hit_radius: float = 2.5,
        max_range: float = 50.0,
        eps: float = 0.5,
        min_samples: int = 5,
        min_cluster_points: int = 5,
    ):
        """
        Args:
            ground_z_threshold: Keep points with z > this value (sensor frame).
                                 Sensor at 2.0 m → threshold -1.5 keeps objects
                                 taller than ~0.5 m above the road surface.
            self_hit_radius:    Discard points closer than this to the sensor
                                 origin to eliminate ego-vehicle reflections.
            max_range:          Discard points beyond this horizontal distance.
            eps:                DBSCAN neighbourhood radius (metres).
            min_samples:        DBSCAN minimum points to form a core sample.
            min_cluster_points: Discard clusters with fewer points than this.
        """
        self.ground_z_threshold = ground_z_threshold
        self.self_hit_radius = self_hit_radius
        self.max_range = max_range
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_points = min_cluster_points

        self.last_points: Optional[np.ndarray] = None  # (N,3) xyz, ground-removed
        self._obstacles: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, raw_lidar_data) -> List[Dict]:
        """
        Full pipeline: raw CARLA LiDAR data → sorted list of obstacle dicts.

        Each dict contains:
            id          – cluster integer label
            distance    – horizontal distance to nearest cluster point (m)
            centroid    – (x, y, z) cluster centroid in sensor frame
            angle_deg   – bearing from ego front: 0=front, +90=right, ±180=rear
            bbox_3d     – (min_x, min_y, max_x, max_y) horizontal footprint
            point_count – number of LiDAR points in cluster
            sector      – 'front' | 'right' | 'rear' | 'left'
        """
        if not SKLEARN_AVAILABLE:
            self.last_points = None
            self._obstacles = []
            return []

        points = self._parse_point_cloud(raw_lidar_data)
        if points is None or len(points) == 0:
            self.last_points = None
            self._obstacles = []
            return []

        points = self._filter(points)
        if len(points) == 0:
            self.last_points = None
            self._obstacles = []
            return []

        self.last_points = points[:, :3].copy()

        labels = self._cluster(points)
        self._obstacles = self._build_obstacles(points, labels)
        return self._obstacles

    def get_nearest_obstacle(self, sector: str = 'all') -> Optional[Dict]:
        """Return closest obstacle, optionally restricted to a sector."""
        if not self._obstacles:
            return None
        if sector == 'all':
            return self._obstacles[0]  # list is sorted by distance
        matches = [o for o in self._obstacles if o['sector'] == sector]
        return matches[0] if matches else None

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _parse_point_cloud(self, raw_data) -> Optional[np.ndarray]:
        """Convert CARLA raw bytes → float32 Nx4 array (x, y, z, intensity)."""
        try:
            pts = np.frombuffer(raw_data.raw_data, dtype=np.float32)
            return pts.reshape(-1, 4)
        except Exception as e:
            print(f"[LidarProcessor] Parse error: {e}")
            return None

    def _filter(self, points: np.ndarray) -> np.ndarray:
        """Remove ground, ego-vehicle self-hits, and out-of-range points."""
        # Horizontal distance for range & self-hit checks
        dist_2d = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

        mask = (
            (points[:, 2] > self.ground_z_threshold)   # above ground
            & (dist_2d > self.self_hit_radius)          # not ego vehicle
            & (dist_2d < self.max_range)                # within sensor range
        )
        return points[mask]

    def _cluster(self, points: np.ndarray) -> np.ndarray:
        """Run DBSCAN on the XY plane; returns per-point cluster labels."""
        xy = points[:, :2]
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=1).fit(xy)
        return db.labels_

    def _build_obstacles(self, points: np.ndarray, labels: np.ndarray) -> List[Dict]:
        """Convert cluster labels into obstacle descriptor dicts."""
        obstacles: List[Dict] = []

        unique_labels = set(labels)
        unique_labels.discard(-1)  # -1 = DBSCAN noise

        for label in unique_labels:
            cluster = points[labels == label]
            if len(cluster) < self.min_cluster_points:
                continue

            x, y, z = cluster[:, 0], cluster[:, 1], cluster[:, 2]
            dist_2d = np.sqrt(x ** 2 + y ** 2)

            distance = float(dist_2d.min())
            cx, cy, cz = float(x.mean()), float(y.mean()), float(z.mean())

            angle_deg = math.degrees(math.atan2(cy, cx))
            sector = self._sector(angle_deg)

            obstacles.append({
                'id': int(label),
                'distance': distance,
                'centroid': (cx, cy, cz),
                'angle_deg': angle_deg,
                'bbox_3d': (float(x.min()), float(y.min()), float(x.max()), float(y.max())),
                'point_count': int(len(cluster)),
                'sector': sector,
            })

        obstacles.sort(key=lambda o: o['distance'])
        return obstacles

    @staticmethod
    def _sector(angle_deg: float) -> str:
        """Map bearing (degrees) to a named sector."""
        a = angle_deg
        if -45.0 <= a <= 45.0:
            return 'front'
        elif 45.0 < a <= 135.0:
            return 'right'
        elif a > 135.0 or a < -135.0:
            return 'rear'
        else:
            return 'left'
