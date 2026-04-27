"""
LiDAR-Camera Fusion Engine — conservative AND logic.

Strategy:
  • Camera detections are matched to LiDAR front-sector clusters by horizontal angle.
  • When matched: LiDAR distance is authoritative; camera provides the semantic class.
  • Camera-only detections: kept with their monocular distance estimate.
  • LiDAR-only front clusters: added as 'unknown_obstacle' with accurate distance.
  • Final action = max(camera_action, lidar_front_action) by danger priority.
"""

import math
from typing import List, Dict, Optional, Tuple

from core.lidar_obstacle_detector import LidarObstacle, ACTION_PRIORITY


def _higher_action(a: str, b: str) -> str:
    return a if ACTION_PRIORITY.get(a, 0) >= ACTION_PRIORITY.get(b, 0) else b


class LidarFusion:

    def __init__(self, angle_match_threshold_deg: float = 15.0):
        """
        Args:
            angle_match_threshold_deg: Max angular difference (degrees) between a camera
                detection's centre and a LiDAR cluster centroid for them to be considered
                the same obstacle.
        """
        self.angle_match_threshold = angle_match_threshold_deg

        # Exposed after each fuse() call for HUD logging
        self.last_camera_dist: Optional[float] = None
        self.last_lidar_dist: Optional[float] = None
        self.last_camera_action: str = 'drive'
        self.last_lidar_front_action: str = 'drive'
        self.last_fused_action: str = 'drive'

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _camera_angle(detection: Dict, img_width: int, focal_length_px: float) -> float:
        """Horizontal angle (degrees) of a camera-detection centre from the image's
        optical axis.  Positive = right of centre."""
        bbox = detection.get('bbox')
        if bbox is None:
            return 0.0
        cx = (bbox[0] + bbox[2]) / 2.0
        dx = cx - img_width / 2.0
        return math.degrees(math.atan2(dx, focal_length_px))

    # ------------------------------------------------------------------
    # Main fusion call
    # ------------------------------------------------------------------

    def fuse(
        self,
        camera_detections: List[Dict],
        lidar_obstacles: List[LidarObstacle],
        camera_action: str,
        vehicle_speed_kmh: float = 0.0,
        img_width: int = 1280,
        focal_length_px: float = 640.0,
    ) -> Tuple[List[Dict], str, Optional[Dict]]:
        """Fuse camera detections with LiDAR obstacles.

        Args:
            camera_detections: List of detection dicts from obstacle_detector.detect().
            lidar_obstacles:   List of LidarObstacle from lidar_obstacle_detector.detect().
            camera_action:     Action string from obstacle_detector.should_stop().
            vehicle_speed_kmh: Current ego speed.
            img_width:         Camera image width in pixels.
            focal_length_px:   Camera focal length in pixels.

        Returns:
            fused_detections: Each detection dict enriched with 'lidar_distance',
                              'lidar_danger', and 'fusion_method' keys.
            fused_action:     Conservative (worst-case) action string.
            nearest_fused:    The fused detection dict with the smallest effective
                              distance (for HUD display), or None.
        """
        self.last_camera_action = camera_action

        # Reset per-call state
        self.last_camera_dist = None
        self.last_lidar_dist = None

        front_obstacles = [o for o in lidar_obstacles if o.sector == 'front']
        matched_lidar_ids: set = set()
        fused_detections: List[Dict] = []

        # ----------------------------------------------------------
        # 1. Match camera detections → LiDAR front clusters
        # ----------------------------------------------------------
        for det in camera_detections:
            cam_angle = self._camera_angle(det, img_width, focal_length_px)
            cam_dist = det.get('distance')

            best_obs: Optional[LidarObstacle] = None
            best_diff = float('inf')
            for obs in front_obstacles:
                diff = abs(obs.angle_deg - cam_angle)
                if diff < self.angle_match_threshold and diff < best_diff:
                    best_diff = diff
                    best_obs = obs

            enriched = dict(det)  # shallow copy — preserve all camera fields

            if best_obs is not None:
                matched_lidar_ids.add(id(best_obs))
                enriched['lidar_distance'] = best_obs.distance
                enriched['lidar_danger'] = best_obs.danger_level
                enriched['fusion_method'] = 'FULL'

                # Track nearest LiDAR distance for HUD
                if self.last_lidar_dist is None or best_obs.distance < self.last_lidar_dist:
                    self.last_lidar_dist = best_obs.distance
            else:
                enriched['lidar_distance'] = None
                enriched['lidar_danger'] = None
                enriched['fusion_method'] = 'CAMERA_ONLY'

            # Track nearest camera distance for HUD
            if cam_dist is not None:
                if self.last_camera_dist is None or cam_dist < self.last_camera_dist:
                    self.last_camera_dist = cam_dist

            fused_detections.append(enriched)

        # ----------------------------------------------------------
        # 2. Unmatched LiDAR-only front clusters
        # ----------------------------------------------------------
        for obs in front_obstacles:
            if id(obs) in matched_lidar_ids:
                continue
            lidar_only: Dict = {
                'class': 'unknown_obstacle',
                'distance': obs.distance,
                'lidar_distance': obs.distance,
                'lidar_danger': obs.danger_level,
                'fusion_method': 'LIDAR_ONLY',
                'danger_level': obs.danger_level,
                'is_dangerous': obs.danger_level not in ('drive', 'cautious'),
                'confidence': 1.0,
                'bbox': None,
                'bbox_center': (0, 0),
                'in_lane': True,
                'sector': obs.sector,
                'angle_deg': obs.angle_deg,
            }
            fused_detections.append(lidar_only)

            if self.last_lidar_dist is None or obs.distance < self.last_lidar_dist:
                self.last_lidar_dist = obs.distance

        # ----------------------------------------------------------
        # 3. Conservative action fusion
        # ----------------------------------------------------------
        lidar_front_action = 'drive'
        if front_obstacles:
            lidar_front_action = max(
                (o.danger_level for o in front_obstacles),
                key=lambda a: ACTION_PRIORITY.get(a, 0),
            )
        self.last_lidar_front_action = lidar_front_action

        fused_action = _higher_action(camera_action, lidar_front_action)
        self.last_fused_action = fused_action

        # ----------------------------------------------------------
        # 4. Nearest fused obstacle for HUD / control decision
        # ----------------------------------------------------------
        nearest: Optional[Dict] = None
        nearest_dist = float('inf')
        for fd in fused_detections:
            # Prefer LiDAR distance when available (more accurate)
            eff_dist = fd.get('lidar_distance') or fd.get('distance') or float('inf')
            if eff_dist < nearest_dist:
                nearest_dist = eff_dist
                nearest = fd

        return fused_detections, fused_action, nearest
