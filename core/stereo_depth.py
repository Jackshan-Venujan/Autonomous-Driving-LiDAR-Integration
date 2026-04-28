import numpy as np
from typing import List, Dict, Optional, Tuple


class StereoDepthEstimator:
    """
    Converts a disparity map (float32, pixels) to metric depth and associates
    depth readings with YOLO bounding boxes.

    Depth equation:  depth = (focal_length_px * baseline_m) / disparity_px

    The fuse() return signature is identical to LidarFusion.fuse() so that
    run_experiment.py is the only injection point — DrivingAgent is untouched.
    """

    # Speed-adaptive danger thresholds replicated from ObstacleDetector
    # to avoid a circular import.
    _THRESHOLDS = [
        ('emergency_stop', 10.0, 0.5),
        ('stop',           15.0, 1.0),
        ('warning',        20.0, 1.5),
        ('slowdown',       20.0, 2.0),
    ]

    def __init__(
        self,
        focal_length_px: float,
        baseline_m: float,
        min_depth_m: float = 0.5,
        max_depth_m: float = 100.0,
        depth_percentile: float = 20.0,
    ):
        """
        Args:
            focal_length_px:  Camera focal length in pixels (same for both cameras).
            baseline_m:       Inter-camera baseline in metres.
            min_depth_m:      Valid depth lower bound.
            max_depth_m:      Valid depth upper bound.
            depth_percentile: Percentile of valid bbox depth pixels used as the
                              representative distance. 20th percentile captures the
                              closest surface robustly while ignoring background noise.
        """
        self._focal = focal_length_px
        self._baseline = baseline_m
        self._min_depth = min_depth_m
        self._max_depth = max_depth_m
        self._depth_percentile = depth_percentile

        # Exposed for HUD parity with LidarFusion public attributes
        self.last_camera_dist: Optional[float] = None
        self.last_stereo_dist: Optional[float] = None
        self.last_camera_action: str = 'drive'
        self.last_stereo_action: str = 'drive'
        self.last_fused_action: str = 'drive'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def disparity_to_depth(self, disparity_map: np.ndarray) -> np.ndarray:
        """
        Convert disparity map (float32, pixels) to depth map (float32, metres).
        Invalid pixels (disparity <= 0) become np.nan.
        """
        depth = np.full_like(disparity_map, np.nan, dtype=np.float32)
        valid = disparity_map > 0
        depth[valid] = (self._focal * self._baseline) / disparity_map[valid]
        # Clip to valid range
        depth[(depth < self._min_depth) | (depth > self._max_depth)] = np.nan
        return depth

    def assign_depth_to_detections(
        self,
        detections: List[Dict],
        depth_map: np.ndarray,
    ) -> List[Dict]:
        """
        For each detection dict sample the depth map inside the bounding box
        and attach 'stereo_distance'. Returns new list of shallow-copied dicts.
        """
        enriched = []
        for det in detections:
            d = dict(det)
            bbox = d.get('bbox')
            if bbox is not None:
                depth_val = self._sample_depth_in_bbox(depth_map, bbox)
            else:
                depth_val = None
            d['stereo_distance'] = depth_val
            d['depth_source'] = 'STEREO'
            d['fusion_method'] = 'STEREO_FULL' if depth_val is not None else 'CAMERA_ONLY'
            enriched.append(d)
        return enriched

    def fuse(
        self,
        camera_detections: List[Dict],
        disparity_map: np.ndarray,
        camera_action: str,
        vehicle_speed_kmh: float = 0.0,
    ) -> Tuple[List[Dict], str, Optional[Dict]]:
        """
        High-level fusion — mirrors LidarFusion.fuse() return signature exactly:
            (fused_detections, fused_action, nearest_obstacle)

        Steps:
          1. Convert disparity to depth.
          2. Assign stereo depth to each YOLO detection.
          3. Override det['distance'] with stereo_distance when valid.
          4. Recompute danger level per detection.
          5. Fused action = worst-case across all detections.
        """
        depth_map = self.disparity_to_depth(disparity_map)
        enriched = self.assign_depth_to_detections(camera_detections, depth_map)

        fused_dets = []
        worst_action = 'drive'
        nearest: Optional[Dict] = None
        nearest_dist = float('inf')

        action_priority = {
            'drive': 0, 'slowdown': 1, 'warning': 2,
            'cautious': 2, 'slow': 2, 'stop': 3, 'emergency_stop': 4,
        }

        for det in enriched:
            d = dict(det)
            stereo_dist = d.get('stereo_distance')
            cam_dist = d.get('distance')

            if stereo_dist is not None:
                d['distance'] = stereo_dist
                action = self._compute_action(stereo_dist, vehicle_speed_kmh)
            elif cam_dist is not None:
                action = self._compute_action(cam_dist, vehicle_speed_kmh)
            else:
                action = 'drive'

            d['danger_level'] = action
            fused_dets.append(d)

            if action_priority.get(action, 0) > action_priority.get(worst_action, 0):
                worst_action = action

            use_dist = stereo_dist if stereo_dist is not None else cam_dist
            if use_dist is not None and use_dist < nearest_dist:
                nearest_dist = use_dist
                nearest = d

        # Update exposed HUD attributes
        camera_dists = [d.get('distance') for d in camera_detections if d.get('distance') is not None]
        stereo_dists = [d.get('stereo_distance') for d in enriched if d.get('stereo_distance') is not None]
        self.last_camera_dist = min(camera_dists) if camera_dists else None
        self.last_stereo_dist = min(stereo_dists) if stereo_dists else None
        self.last_camera_action = camera_action
        self.last_stereo_action = worst_action
        self.last_fused_action = worst_action

        return fused_dets, worst_action, nearest

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_action(self, distance: float, speed_kmh: float) -> str:
        """Replicate ObstacleDetector speed-adaptive threshold logic."""
        for action, base, speed_factor in self._THRESHOLDS:
            threshold = base + speed_factor * speed_kmh
            if distance <= threshold:
                return action
        return 'drive'

    def _sample_depth_in_bbox(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[float]:
        """
        Extract depth_percentile-th percentile of valid pixels inside bbox.
        Returns None if fewer than 10 valid pixels exist.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = depth_map.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = depth_map[y1:y2, x1:x2]
        valid = roi[np.isfinite(roi)]
        if len(valid) < 10:
            return None
        return float(np.percentile(valid, self._depth_percentile))
