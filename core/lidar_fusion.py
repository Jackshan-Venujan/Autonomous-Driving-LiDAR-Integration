"""
LiDAR-Camera Fusion Engine — conservative AND logic.

Matching strategy (in priority order):
  1. BBox projection  — project raw LiDAR points into camera frame, keep points
                        inside the YOLO bbox, take median depth (≥ 3 pts needed).
  2. Angle fallback   — if bbox projection yields < 3 pts, match camera detection
                        to the nearest front-sector LiDAR cluster by horizontal
                        angle (original approach, retained for sparse/distant cases).
  3. Camera-only      — no LiDAR match found; monocular distance kept.

Unmatched LiDAR-only front clusters are still appended as 'unknown_obstacle'.
Final action = max(camera_action, lidar_front_action) by danger priority.
"""

import math
from typing import List, Dict, Optional, Tuple

import numpy as np

from core.lidar_obstacle_detector import LidarObstacle, ACTION_PRIORITY
from core.lidar_camera_projector import LidarCameraProjector


def _higher_action(a: str, b: str) -> str:
    return a if ACTION_PRIORITY.get(a, 0) >= ACTION_PRIORITY.get(b, 0) else b


class LidarFusion:

    _CAM_STALE_MAX = 5    # evict a camera-only track after this many frames unseen
    _CAM_MATCH_PX  = 150  # max pixel distance for camera-only re-identification

    def __init__(self, angle_match_threshold_deg: float = 15.0):
        self.angle_match_threshold = angle_match_threshold_deg
        self.projector = LidarCameraProjector()  # defaults match sensor setup

        # Exposed after each fuse() call for HUD / metrics logging
        self.last_camera_dist: Optional[float] = None
        self.last_lidar_dist: Optional[float] = None
        self.last_camera_action: str = 'drive'
        self.last_lidar_front_action: str = 'drive'
        self.last_fused_action: str = 'drive'
        self.last_lidar_bbox_pts: int = 0  # point count from bbox projection

        # Camera-only obstacle tracker (persistent IDs for unmatched camera detections)
        self._cam_only_tracks: Dict[str, dict] = {}  # cam_id → {bbox_center, last_frame}
        self._next_cam_id: int = 0
        self._fuse_frame: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _match_or_create_cam_track(self, bbox_center: Tuple[int, int]) -> str:
        """Return a persistent camera-only obstacle ID for this bbox centre.

        Nearest-neighbour match in pixel space; creates a new ID when no
        existing track is within _CAM_MATCH_PX pixels.
        """
        best_id: Optional[str] = None
        best_dist = float('inf')
        for cid, ctrack in self._cam_only_tracks.items():
            dx = bbox_center[0] - ctrack['bbox_center'][0]
            dy = bbox_center[1] - ctrack['bbox_center'][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self._CAM_MATCH_PX and dist < best_dist:
                best_dist = dist
                best_id = cid

        if best_id is None:
            best_id = f'cam_{self._next_cam_id}'
            self._next_cam_id += 1

        self._cam_only_tracks[best_id] = {
            'bbox_center': bbox_center,
            'last_frame': self._fuse_frame,
        }
        return best_id

    @staticmethod
    def _camera_angle(detection: Dict, img_width: int, focal_length_px: float) -> float:
        """Horizontal angle (degrees) of a camera-detection centre from the
        image optical axis.  Positive = right of centre."""
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
        lidar_raw_points: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict], str, Optional[Dict]]:
        """Fuse camera detections with LiDAR obstacles.

        Args:
            camera_detections: List of detection dicts from obstacle_detector.detect().
            lidar_obstacles:   List of LidarObstacle from lidar_obstacle_detector.detect().
            camera_action:     Action string from obstacle_detector.should_stop().
            vehicle_speed_kmh: Current ego speed.
            img_width:         Camera image width in pixels.
            focal_length_px:   Camera focal length in pixels.
            lidar_raw_points:  (N, 3) preprocessed LiDAR XYZ for bbox projection.
                               When provided, bbox projection is tried first;
                               angle-based matching is used as fallback.

        Returns:
            fused_detections: Each detection dict enriched with 'lidar_distance',
                              'lidar_danger', 'fusion_method', and 'lidar_bbox_pts'.
            fused_action:     Conservative (worst-case) action string.
            nearest_fused:    The fused detection dict with the smallest effective
                              distance (for HUD display), or None.
        """
        self.last_camera_action = camera_action

        # Advance frame counter and evict stale camera-only tracks
        self._fuse_frame += 1
        for cid in list(self._cam_only_tracks.keys()):
            if self._fuse_frame - self._cam_only_tracks[cid]['last_frame'] > self._CAM_STALE_MAX:
                del self._cam_only_tracks[cid]

        # Reset per-call state
        self.last_camera_dist = None
        self.last_lidar_dist = None
        self.last_lidar_bbox_pts = 0

        front_obstacles = [o for o in lidar_obstacles if o.sector == 'front']
        matched_lidar_ids: set = set()
        fused_detections: List[Dict] = []

        have_raw = lidar_raw_points is not None and len(lidar_raw_points) > 0

        # ----------------------------------------------------------
        # 1. Match camera detections → LiDAR distance
        # ----------------------------------------------------------
        for det in camera_detections:
            cam_dist = det.get('distance')
            enriched = dict(det)

            bbox_dist: Optional[float] = None
            bbox_pts: int = 0
            fusion_method = 'CAMERA_ONLY'

            # --- Strategy A: BBox projection (preferred) ---
            if have_raw:
                bbox = det.get('bbox')
                if bbox is not None:
                    bbox_dist, bbox_pts = self.projector.filter_by_bbox(
                        lidar_raw_points, bbox, min_points=2
                    )

            cam_angle = self._camera_angle(det, img_width, focal_length_px)
            obstacle_id: Optional[str] = None

            if bbox_dist is not None:
                fusion_method = 'FULL_BBOX'
                enriched['lidar_distance'] = bbox_dist
                enriched['lidar_danger'] = self._classify_danger(bbox_dist, vehicle_speed_kmh)
                enriched['lidar_bbox_pts'] = bbox_pts
                if self.last_lidar_dist is None or bbox_dist < self.last_lidar_dist:
                    self.last_lidar_dist = bbox_dist
                if bbox_pts > self.last_lidar_bbox_pts:
                    self.last_lidar_bbox_pts = bbox_pts
                # Assign the LiDAR track ID whose angle best matches this camera detection
                best_angle_obs: Optional[LidarObstacle] = None
                best_angle_diff = float('inf')
                for obs in front_obstacles:
                    diff = abs(obs.angle_deg - cam_angle)
                    if diff < self.angle_match_threshold and diff < best_angle_diff:
                        best_angle_diff = diff
                        best_angle_obs = obs
                if best_angle_obs is not None:
                    obstacle_id = best_angle_obs.track_id
                    matched_lidar_ids.add(id(best_angle_obs))

            else:
                # --- Strategy B: Angle-based fallback (distance-aware) ---
                # Require both angular AND distance agreement, else the matched
                # cluster is a different physical object that just happens to
                # share an angular bearing with the camera detection.
                best_obs: Optional[LidarObstacle] = None
                best_diff = float('inf')
                for obs in front_obstacles:
                    ang_diff = abs(obs.angle_deg - cam_angle)
                    if ang_diff >= self.angle_match_threshold:
                        continue
                    if cam_dist is not None:
                        rel = abs(obs.distance - cam_dist) / max(cam_dist, 1.0)
                        if rel > 0.40:
                            continue
                    if ang_diff < best_diff:
                        best_diff = ang_diff
                        best_obs = obs

                if best_obs is not None:
                    matched_lidar_ids.add(id(best_obs))
                    fusion_method = 'FULL_ANGLE'
                    enriched['lidar_distance'] = best_obs.distance
                    enriched['lidar_danger'] = best_obs.danger_level
                    enriched['lidar_bbox_pts'] = 0
                    obstacle_id = best_obs.track_id
                    if self.last_lidar_dist is None or best_obs.distance < self.last_lidar_dist:
                        self.last_lidar_dist = best_obs.distance
                else:
                    enriched['lidar_distance'] = None
                    enriched['lidar_danger'] = None
                    enriched['lidar_bbox_pts'] = 0

            # Camera-only: assign persistent camera-side ID
            if obstacle_id is None:
                bbox_center = det.get('bbox_center', (0, 0))
                obstacle_id = self._match_or_create_cam_track(bbox_center)

            enriched['obstacle_id'] = obstacle_id
            enriched['fusion_method'] = fusion_method
            enriched['angle_deg'] = cam_angle

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
            # Only add as LIDAR_ONLY when bbox projection wasn't used (avoid duplicates)
            if have_raw:
                # bbox projection already consumed raw points; trust it over cluster
                pass
            lidar_only: Dict = {
                'class': 'unknown_obstacle',
                'distance': obs.distance,
                'lidar_distance': obs.distance,
                'lidar_danger': obs.danger_level,
                'lidar_bbox_pts': 0,
                'fusion_method': 'LIDAR_ONLY',
                'danger_level': obs.danger_level,
                'is_dangerous': obs.danger_level not in ('drive', 'cautious'),
                'confidence': 1.0,
                'bbox': None,
                'bbox_center': (0, 0),
                'in_lane': True,
                'sector': obs.sector,
                'angle_deg': obs.angle_deg,
                'obstacle_id': obs.track_id,
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
            # Safe-by-default fallback: camera distance is used when LiDAR didn't match.
            # Affects HUD/control only; per-obstacle log keeps sources separate.
            # See docs/SENSOR_COMPARISON.md before using this for accuracy comparisons.
            eff_dist = fd.get('lidar_distance') or fd.get('distance') or float('inf')
            if eff_dist < nearest_dist:
                nearest_dist = eff_dist
                nearest = fd

        return fused_detections, fused_action, nearest

    # ------------------------------------------------------------------
    # Danger classification (mirrors LidarObstacleDetector thresholds)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_danger(distance: float, speed_kmh: float) -> str:
        spd = speed_kmh
        if distance <= 3.0 + spd * 0.3 * 0.5:
            return 'emergency_stop'
        if distance <= 5.0 + spd * 0.5:
            return 'stop'
        if distance <= 10.0 + spd * 0.5:
            return 'slow'
        if distance <= 15.0 + spd * 0.5:
            return 'cautious'
        return 'drive'
