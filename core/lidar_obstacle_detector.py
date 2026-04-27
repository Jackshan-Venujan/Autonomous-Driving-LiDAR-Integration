"""
LiDAR Obstacle Detector — DBSCAN clustering on filtered point cloud,
sector-based danger classification, and BEV visualisation canvas.
"""

import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from sklearn.cluster import DBSCAN
    _DBSCAN_AVAILABLE = True
except ImportError:
    _DBSCAN_AVAILABLE = False
    print("⚠️  sklearn not available — LiDAR clustering disabled")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LidarObstacle:
    centroid_x: float        # forward distance (m)  +X = forward
    centroid_y: float        # lateral distance (m)  +Y = right
    centroid_z: float
    distance: float          # horizontal distance sqrt(x²+y²)
    angle_deg: float         # angle from forward axis (+right, -left)
    sector: str              # 'front' | 'side_left' | 'side_right'
    point_count: int
    danger_level: str        # 'drive'|'cautious'|'slow'|'stop'|'emergency_stop'
    bbox_min_x: float = 0.0
    bbox_max_x: float = 0.0
    bbox_min_y: float = 0.0
    bbox_max_y: float = 0.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

ACTION_PRIORITY = {'drive': 0, 'cautious': 1, 'slow': 2, 'stop': 3, 'emergency_stop': 4}


class LidarObstacleDetector:
    """Clusters a preprocessed point cloud and classifies each cluster by danger level."""

    FRONT_ANGLE_DEG = 60.0   # ±60° from forward = front sector
    SIDE_ANGLE_DEG = 120.0   # 60°–120° = side sectors; >120° = rear (ignored)

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 3,
        base_emergency_dist: float = 3,         # 7.5
        base_stop_dist: float = 5.0,            # 15.0
        base_slow_dist: float = 10.0,           # 20.0
        base_cautious_dist: float = 15.0,       # 15.0
        speed_factor: float = 0.5,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.base_emergency_dist = base_emergency_dist
        self.base_stop_dist = base_stop_dist
        self.base_slow_dist = base_slow_dist
        self.base_cautious_dist = base_cautious_dist
        self.speed_factor = speed_factor

        self._last_obstacles: List[LidarObstacle] = []
        self._last_raw_points: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_thresholds(self, speed_kmh: float) -> dict:
        extra = max(0.0, speed_kmh) * self.speed_factor
        return {
            'emergency': self.base_emergency_dist + extra * 0.3,
            'stop':      self.base_stop_dist      + extra * 0.5,
            'slow':      self.base_slow_dist      + extra * 0.5,
            'cautious':  self.base_cautious_dist  + extra * 0.5,
        }

    def _classify_danger(self, distance: float, sector: str, thresholds: dict) -> str:
        if sector in ('side_left', 'side_right'):
            return 'cautious'   # sides never trigger stop
        # front sector
        if distance <= thresholds['emergency']:
            return 'emergency_stop'
        if distance <= thresholds['stop']:
            return 'stop'
        if distance <= thresholds['slow']:
            return 'slow'
        if distance <= thresholds['cautious']:
            return 'cautious'
        return 'drive'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, filtered_points: np.ndarray, vehicle_speed_kmh: float = 0.0) -> List[LidarObstacle]:
        """Run DBSCAN and classify clusters.

        Args:
            filtered_points: (M, 3) numpy array [x, y, z] from LidarProcessor.
            vehicle_speed_kmh: current ego speed for adaptive thresholds.

        Returns:
            List of LidarObstacle (front + side sectors; rear dropped).
        """
        self._last_obstacles = []
        self._last_raw_points = filtered_points

        if not _DBSCAN_AVAILABLE or len(filtered_points) < self.min_samples:
            return self._last_obstacles

        thresholds = self._get_thresholds(vehicle_speed_kmh)

        # Cluster on XY plane only (ignore height for grouping)
        xy = filtered_points[:, :2]
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(xy)

        for lbl in set(labels):
            if lbl == -1:
                continue  # noise

            mask = labels == lbl
            cluster = filtered_points[mask]

            cx = float(np.mean(cluster[:, 0]))
            cy = float(np.mean(cluster[:, 1]))
            cz = float(np.mean(cluster[:, 2]))
            dist = math.sqrt(cx ** 2 + cy ** 2)
            angle_deg = math.degrees(math.atan2(cy, cx))  # +right / -left

            abs_angle = abs(angle_deg)
            if abs_angle <= self.FRONT_ANGLE_DEG:
                sector = 'front'
            elif abs_angle <= self.SIDE_ANGLE_DEG:
                sector = 'side_right' if angle_deg > 0 else 'side_left'
            else:
                continue  # rear — skip

            danger = self._classify_danger(dist, sector, thresholds)

            self._last_obstacles.append(LidarObstacle(
                centroid_x=cx,
                centroid_y=cy,
                centroid_z=cz,
                distance=dist,
                angle_deg=angle_deg,
                sector=sector,
                point_count=int(np.sum(mask)),
                danger_level=danger,
                bbox_min_x=float(np.min(cluster[:, 0])),
                bbox_max_x=float(np.max(cluster[:, 0])),
                bbox_min_y=float(np.min(cluster[:, 1])),
                bbox_max_y=float(np.max(cluster[:, 1])),
            ))

        return self._last_obstacles

    def get_front_action(self) -> str:
        """Worst danger level among front-sector obstacles."""
        front = [o for o in self._last_obstacles if o.sector == 'front']
        if not front:
            return 'drive'
        return max(front, key=lambda o: ACTION_PRIORITY[o.danger_level]).danger_level

    def get_side_action(self) -> str:
        """'cautious' if any side obstacle exists, else 'drive'."""
        sides = [o for o in self._last_obstacles if o.sector in ('side_left', 'side_right')]
        return 'cautious' if sides else 'drive'

    def get_nearest_front(self) -> Optional[LidarObstacle]:
        front = [o for o in self._last_obstacles if o.sector == 'front']
        return min(front, key=lambda o: o.distance) if front else None

    # ------------------------------------------------------------------
    # BEV visualisation canvas
    # ------------------------------------------------------------------

    def render_bev(
        self,
        canvas_size: int = 400,
        meters_per_pixel: float = 0.25,
    ) -> np.ndarray:
        """Build a top-down bird's-eye-view image of the point cloud and clusters.

        Origin = ego vehicle at bottom-centre of canvas.
        +X (forward) points upward on the canvas.
        +Y (right) points to the right.

        Returns:
            (canvas_size, canvas_size, 3) uint8 BGR image.
        """
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        cx_px = canvas_size // 2        # pixel column for y=0 (centre)
        ego_row = canvas_size - 20      # pixel row for x=0 (ego vehicle)

        def world_to_px(x_m, y_m):
            """Convert sensor-frame (x=forward, y=right) to canvas pixel."""
            col = int(cx_px + y_m / meters_per_pixel)
            row = int(ego_row - x_m / meters_per_pixel)
            return col, row

        # --- Range rings ---
        for r_m in (10, 25, 50):
            r_px = int(r_m / meters_per_pixel)
            cv2.circle(canvas, (cx_px, ego_row), r_px, (60, 60, 60), 1)
            label_col = cx_px + r_px + 2
            label_row = ego_row
            if 0 <= label_col < canvas_size and 0 <= label_row < canvas_size:
                cv2.putText(canvas, f"{r_m}m", (label_col, label_row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

        # --- Raw points (white dots) ---
        if self._last_raw_points is not None and len(self._last_raw_points) > 0:
            for pt in self._last_raw_points[::3]:   # every 3rd point for speed
                col, row = world_to_px(pt[0], pt[1])
                if 0 <= col < canvas_size and 0 <= row < canvas_size:
                    canvas[row, col] = (200, 200, 200)

        # --- Cluster bounding boxes ---
        sector_colors = {
            'front':      (0, 0, 255),    # red
            'side_left':  (0, 215, 255),  # yellow
            'side_right': (0, 140, 255),  # orange
        }
        for obs in self._last_obstacles:
            color = sector_colors.get(obs.sector, (200, 200, 200))
            c1 = world_to_px(obs.bbox_max_x, obs.bbox_min_y)
            c2 = world_to_px(obs.bbox_min_x, obs.bbox_max_y)
            # Clamp to canvas
            x1, y1 = max(0, min(canvas_size - 1, c1[0])), max(0, min(canvas_size - 1, c1[1]))
            x2, y2 = max(0, min(canvas_size - 1, c2[0])), max(0, min(canvas_size - 1, c2[1]))
            if x1 != x2 and y1 != y2:
                cv2.rectangle(canvas, (min(x1, x2), min(y1, y2)),
                              (max(x1, x2), max(y1, y2)), color, 2)
            # Label: distance + danger
            lx, ly = world_to_px(obs.centroid_x, obs.centroid_y)
            label = f"{obs.distance:.1f}m {obs.danger_level.upper()[:4]}"
            cv2.putText(canvas, label, (max(0, lx - 30), max(10, ly - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # --- Ego vehicle rectangle ---
        ev_w, ev_h = 8, 14   # pixels
        ev_c, ev_r = cx_px, ego_row
        cv2.rectangle(canvas,
                      (ev_c - ev_w // 2, ev_r - ev_h),
                      (ev_c + ev_w // 2, ev_r + 4),
                      (255, 120, 0), -1)
        cv2.putText(canvas, "EGO", (ev_c - 10, ev_r + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 120, 0), 1)

        # --- Forward direction arrow ---
        cv2.arrowedLine(canvas, (cx_px, ego_row - 16),
                        (cx_px, ego_row - 36), (180, 180, 180), 1, tipLength=0.4)

        # --- Legend ---
        legend_y = 10
        for sector, color in sector_colors.items():
            cv2.putText(canvas, sector, (5, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            legend_y += 14

        return canvas
