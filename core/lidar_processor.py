"""
LiDAR Point-Cloud Perception Processor
=======================================
Pure-numpy + scikit-learn pipeline for obstacle detection from
CARLA standard LiDAR (sensor.lidar.ray_cast) point clouds.

Pipeline:
    1) ROI crop          — keep points in a rectangular volume around ego
    2) Ground removal    — z-threshold (fast) or RANSAC plane fit
    3) Clustering        — DBSCAN groups non-ground points into obstacles
    4) Feature extract   — centroid, AABB bounding box, point count, distances
    5) Sort by distance  — nearest obstacle first

No camera fusion, no ROS, no deep learning.
Phase-2 ready: each function is pure numpy in / dict out.

Dependencies:
    numpy
    scikit-learn    (pip install scikit-learn)
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[LidarProcessor] ⚠ scikit-learn not installed — clustering disabled")
    print("                   pip install scikit-learn")


# ─── Default Processing Config ─────────────────────────────────────────────────

PROC_CONFIG = {
    # ROI crop — axis-aligned bounding box (meters, in sensor frame)
    # Sensor frame: x=forward, y=left, z=up (CARLA LiDAR local frame)
    'roi_x_min': -10.0,       # behind ego
    'roi_x_max': 80.0,        # ahead of ego
    'roi_y_min': -20.0,       # right  (CARLA y: left-positive)
    'roi_y_max': 20.0,        # left
    'roi_z_min': -2.5,        # below sensor
    'roi_z_max': 3.0,         # above sensor

    # Ground removal
    'ground_method': 'z_threshold',  # 'z_threshold' or 'ransac'
    'ground_z_threshold': -1.5,      # points with z < this are ground (sensor frame)
    'ransac_distance': 0.2,          # RANSAC inlier distance (meters)
    'ransac_max_iter': 100,

    # DBSCAN clustering
    'cluster_eps': 1.5,              # max distance between neighbours (meters)
    'cluster_min_points': 8,         # minimum points per cluster
    'cluster_max_clusters': 50,      # cap to prevent over-segmentation

    # Distance / filtering
    'min_obstacle_points': 5,        # discard tiny clusters
    'max_obstacle_distance': 80.0,   # meters — discard far-away clusters
}


class LidarProcessor:
    """
    Stateless obstacle detection pipeline for Nx4 LiDAR point clouds.

    Typical usage:
        proc = LidarProcessor()
        result = proc.process(points_nx4)
        # result['obstacles'] = list of dicts, sorted by distance
        # result['ground_points'], result['non_ground_points'] etc.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = {**PROC_CONFIG, **(config or {})}
        self._dbscan = None
        if SKLEARN_AVAILABLE:
            self._dbscan = DBSCAN(
                eps=self.config['cluster_eps'],
                min_samples=self.config['cluster_min_points'],
                n_jobs=1,
            )
        print(f"[LidarProc] ✓ Initialized  "
              f"roi_x=[{self.config['roi_x_min']},{self.config['roi_x_max']}], "
              f"ground_z<{self.config['ground_z_threshold']}, "
              f"eps={self.config['cluster_eps']}, min_pts={self.config['cluster_min_points']}")

    # ── Main entry point ───────────────────────────────────────────────────

    def process(self, points: np.ndarray) -> Dict:
        """
        Full detection pipeline.

        Args:
            points: (N, 4) float32 — [x, y, z, intensity]

        Returns:
            dict with keys:
                'obstacles'          : list[dict] sorted by distance (nearest first)
                'raw_points'         : original input
                'roi_points'         : points after ROI crop
                'ground_points'      : ground points (Mx4)
                'non_ground_points'  : obstacle candidate points (Kx4)
                'cluster_labels'     : int array len K  (-1 = noise)
                'num_clusters'       : int
                'nearest_obstacle'   : dict or None
                'processing_time_ms' : float
        """
        t0 = time.perf_counter()

        result = {
            'raw_points': points,
            'roi_points': np.empty((0, 4), dtype=np.float32),
            'ground_points': np.empty((0, 4), dtype=np.float32),
            'non_ground_points': np.empty((0, 4), dtype=np.float32),
            'cluster_labels': np.array([], dtype=np.int32),
            'obstacles': [],
            'num_clusters': 0,
            'nearest_obstacle': None,
            'processing_time_ms': 0.0,
        }

        if points is None or len(points) == 0:
            return result

        # 1) ROI crop
        roi = self._roi_crop(points)
        result['roi_points'] = roi
        if len(roi) == 0:
            result['processing_time_ms'] = (time.perf_counter() - t0) * 1000
            return result

        # 2) Ground removal
        ground, non_ground = self._remove_ground(roi)
        result['ground_points'] = ground
        result['non_ground_points'] = non_ground
        if len(non_ground) < self.config['cluster_min_points']:
            result['processing_time_ms'] = (time.perf_counter() - t0) * 1000
            return result

        # 3) Clustering
        labels = self._cluster(non_ground[:, :3])  # cluster on xyz only
        result['cluster_labels'] = labels

        # 4) Extract per-cluster features
        obstacles = self._extract_obstacles(non_ground, labels)
        result['obstacles'] = obstacles
        result['num_clusters'] = len(obstacles)
        if obstacles:
            result['nearest_obstacle'] = obstacles[0]

        result['processing_time_ms'] = (time.perf_counter() - t0) * 1000
        return result

    # ── Pipeline stages ────────────────────────────────────────────────────

    def _roi_crop(self, pts: np.ndarray) -> np.ndarray:
        """Keep points inside axis-aligned bounding box."""
        c = self.config
        mask = (
            (pts[:, 0] >= c['roi_x_min']) & (pts[:, 0] <= c['roi_x_max']) &
            (pts[:, 1] >= c['roi_y_min']) & (pts[:, 1] <= c['roi_y_max']) &
            (pts[:, 2] >= c['roi_z_min']) & (pts[:, 2] <= c['roi_z_max'])
        )
        return pts[mask]

    def _remove_ground(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split into ground and non-ground points."""
        method = self.config.get('ground_method', 'z_threshold')
        if method == 'ransac' and SKLEARN_AVAILABLE:
            return self._ground_ransac(pts)
        return self._ground_z_threshold(pts)

    def _ground_z_threshold(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple z-cutoff ground removal."""
        z_thresh = self.config['ground_z_threshold']
        ground_mask = pts[:, 2] < z_thresh
        return pts[ground_mask], pts[~ground_mask]

    def _ground_ransac(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """RANSAC plane segmentation for ground removal (more robust on slopes)."""
        from sklearn.linear_model import RANSACRegressor
        X = pts[:, :2]         # x, y
        z = pts[:, 2]          # z
        ransac = RANSACRegressor(
            residual_threshold=self.config['ransac_distance'],
            max_trials=self.config['ransac_max_iter'],
        )
        try:
            ransac.fit(X, z)
            inlier_mask = ransac.inlier_mask_
            return pts[inlier_mask], pts[~inlier_mask]
        except Exception:
            # Fall back to z-threshold if RANSAC fails
            return self._ground_z_threshold(pts)

    def _cluster(self, xyz: np.ndarray) -> np.ndarray:
        """DBSCAN clustering on xyz coordinates. Returns labels (int array, -1=noise)."""
        if not SKLEARN_AVAILABLE or self._dbscan is None:
            # Fallback: treat everything as one cluster
            return np.zeros(len(xyz), dtype=np.int32)

        labels = self._dbscan.fit_predict(xyz)
        return labels.astype(np.int32)

    def _extract_obstacles(self, pts: np.ndarray, labels: np.ndarray) -> List[Dict]:
        """
        For each cluster, compute features and return sorted by distance.

        Each obstacle dict:
            'cluster_id'    : int
            'num_points'    : int
            'centroid'      : (cx, cy, cz)
            'bbox_min'      : (xmin, ymin, zmin)
            'bbox_max'      : (xmax, ymax, zmax)
            'bbox_size'     : (dx, dy, dz)
            'distance'      : float — Euclidean distance of centroid from origin (sensor)
            'nearest_dist'  : float — distance of closest point in cluster
            'points'        : Nx4 array (points belonging to this cluster)
        """
        unique_labels = np.unique(labels)
        obstacles = []
        min_pts = self.config['min_obstacle_points']
        max_dist = self.config['max_obstacle_distance']
        max_clusters = self.config['cluster_max_clusters']

        for label in unique_labels:
            if label == -1:  # noise
                continue

            cluster_mask = labels == label
            cluster_pts = pts[cluster_mask]

            if len(cluster_pts) < min_pts:
                continue

            xyz = cluster_pts[:, :3]
            centroid = xyz.mean(axis=0)
            bbox_min = xyz.min(axis=0)
            bbox_max = xyz.max(axis=0)
            bbox_size = bbox_max - bbox_min

            # Distances from sensor origin (0,0,0 in sensor frame)
            centroid_dist = np.linalg.norm(centroid)
            nearest_dist = np.min(np.linalg.norm(xyz, axis=1))

            if centroid_dist > max_dist:
                continue

            obstacles.append({
                'cluster_id': int(label),
                'num_points': len(cluster_pts),
                'centroid': tuple(centroid),
                'bbox_min': tuple(bbox_min),
                'bbox_max': tuple(bbox_max),
                'bbox_size': tuple(bbox_size),
                'distance': float(centroid_dist),
                'nearest_dist': float(nearest_dist),
                'points': cluster_pts,
            })

        # Sort by distance (nearest first)
        obstacles.sort(key=lambda o: o['distance'])

        # Cap cluster count
        if len(obstacles) > max_clusters:
            obstacles = obstacles[:max_clusters]

        return obstacles

    # ── Utility ────────────────────────────────────────────────────────────

    def update_config(self, **kwargs):
        """Update processing parameters at runtime."""
        self.config.update(kwargs)
        # Rebuild DBSCAN if clustering params changed
        if SKLEARN_AVAILABLE and ('cluster_eps' in kwargs or 'cluster_min_points' in kwargs):
            self._dbscan = DBSCAN(
                eps=self.config['cluster_eps'],
                min_samples=self.config['cluster_min_points'],
                n_jobs=1,
            )
            print(f"[LidarProc] DBSCAN updated: eps={self.config['cluster_eps']}, "
                  f"min_pts={self.config['cluster_min_points']}")

    @staticmethod
    def compute_aabb_corners(bbox_min: Tuple, bbox_max: Tuple) -> np.ndarray:
        """
        Compute 8 corners of an axis-aligned bounding box.

        Returns:
            (8, 3) array of corner points
        """
        xmin, ymin, zmin = bbox_min
        xmax, ymax, zmax = bbox_max
        return np.array([
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ], dtype=np.float64)

    @staticmethod
    def aabb_line_set_indices() -> List[List[int]]:
        """
        12 edges of an axis-aligned bounding box as index pairs.
        For use with Open3D LineSet.
        """
        return [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # top
            [0, 4], [1, 5], [2, 6], [3, 7],  # vertical
        ]
