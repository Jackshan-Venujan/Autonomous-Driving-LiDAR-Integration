"""
Rear Camera Perception Module
Object detection and lane detection for rear-facing camera.
Primary use: blind-spot monitoring during lane changes and overtaking.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from modules.lane_detector import LaneDetector
from detection.yolo_lane_filter import YOLOLaneFilter

# Zone boundaries (fraction of image width)
REAR_LEFT_ZONE_RATIO  = 0.33
REAR_RIGHT_ZONE_RATIO = 0.67

# Distance thresholds for lane-change advisories (metre;s)
BLIND_SPOT_WARN_M  = 40.0
BLIND_SPOT_DANGER_M = 20.0


class RearCameraProcessor:
    """
    Runs lane detection + YOLO object detection on the rear camera frame.

    The heavy lane-detection model (TuSimple ResNet-18) is loaded as a
    separate LaneDetector instance so its per-stream state (EMA, lost-lane
    timer, curve coefficients) is fully independent from the front camera.

    The ObstacleDetector (YOLO) is shared with the front camera – its
    detect() call is stateless so concurrent sequential use is safe.
    """

    def __init__(self, obstacle_detector, img_w: int = 1280, img_h: int = 720):
        """
        Args:
            obstacle_detector: Shared ObstacleDetector from DrivingAgent.
                               Saves ~40 MB GPU memory vs loading YOLO twice.
            img_w, img_h: Rear camera resolution (must match CARLA blueprint).
        """
        self.obstacle_detector = obstacle_detector
        self.img_w = img_w
        self.img_h = img_h

        # Independent lane detector — separate model instance loads its own
        # GPU weights (~245 MB).  Stateful fields (EMA, timers) are isolated.
        print("⏳ Loading rear camera lane detector...")
        self.lane_detector = LaneDetector()

        # Separate lane filter (cheap — no model weights)
        self.lane_filter = YOLOLaneFilter(img_width=img_w, img_height=img_h)
        self.lane_filter.use_triangular_roi = True
        self.lane_filter.fixed_lane_width_ratio = 0.22

        print("✓ Rear Camera Processor ready")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray, vehicle_speed_kmh: float = 0.0) -> Optional[Dict]:
        """
        Run perception pipeline on one rear camera frame.

        Returns:
            dict with keys:
                lane_data         – LaneDetector.detect() output
                all_detections    – raw YOLO detections
                lane_detections   – detections overlapping the rear lane mask
                left_zone         – detections in rear-left third of image
                right_zone        – detections in rear-right third of image
                center_zone       – detections in rear-centre third
                lane_change_safety – {'left': status, 'right': status, ...}
        """
        if image is None:
            return None

        lane_data = self.lane_detector.detect(image)

        all_detections, _ = self.obstacle_detector.detect(
            image, vehicle_speed_kmh=vehicle_speed_kmh
        )

        if lane_data and lane_data['filtered_lanes']:
            self.lane_filter.create_lane_mask_from_lanes(
                lane_data['filtered_lanes'],
                expansion_width=15,
                forward_extension=250
            )
            lane_detections = self.lane_filter.filter_detections_by_lane(
                all_detections, overlap_threshold=0.3
            )
        else:
            lane_detections = []

        left_zone, center_zone, right_zone = self._split_by_zone(all_detections)
        safety = self._assess_safety(left_zone, right_zone)

        return {
            'lane_data':          lane_data,
            'all_detections':     all_detections,
            'lane_detections':    lane_detections,
            'left_zone':          left_zone,
            'center_zone':        center_zone,
            'right_zone':         right_zone,
            'lane_change_safety': safety,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        Render rear camera frame with lanes, obstacles, zone dividers,
        and a lane-change safety panel at the bottom.
        """
        if image is None or result is None:
            return image

        vis = image.copy()

        # Title banner
        cv2.rectangle(vis, (0, 0), (self.img_w, 36), (20, 20, 60), -1)
        cv2.putText(vis, "REAR VIEW  |  BLIND SPOT MONITOR",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2)

        # Zone divider lines and labels
        lx = int(self.img_w * REAR_LEFT_ZONE_RATIO)
        rx = int(self.img_w * REAR_RIGHT_ZONE_RATIO)
        cv2.line(vis, (lx, 36), (lx, self.img_h), (70, 70, 70), 1)
        cv2.line(vis, (rx, 36), (rx, self.img_h), (70, 70, 70), 1)
        cv2.putText(vis, "LEFT",   (8,  54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
        cv2.putText(vis, "CENTER", (lx + 8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
        cv2.putText(vis, "RIGHT",  (rx + 8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

        # Detected lanes
        lane_data = result.get('lane_data')
        if lane_data and lane_data['filtered_lanes']:
            colors = [(0, 255, 0), (255, 80, 0), (0, 80, 255), (255, 255, 0)]
            for i, lane in enumerate(lane_data['filtered_lanes']):
                color = colors[i % len(colors)]
                for pt in lane:
                    cv2.circle(vis, tuple(pt), 3, color, -1)
                if len(lane) > 1:
                    cv2.polylines(vis, [np.array(lane, dtype=np.int32)], False, color, 2)

        # YOLO detections with danger-level colours
        _color = {
            'emergency': (0, 0, 255),
            'stop':      (0, 60, 255),
            'warning':   (0, 165, 255),
            'slowdown':  (0, 230, 230),
            'safe':      (0, 210, 0),
        }
        for det in result.get('all_detections', []):
            x1, y1, x2, y2 = det['bbox']
            col = _color.get(det.get('danger_level', 'safe'), (0, 210, 0))
            dist = det.get('distance')
            label = det['class'] + (f" {dist:.1f}m" if dist is not None else "")
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)

        # Info row (lanes detected, object count)
        info_y = self.img_h - 82
        lanes_n = lane_data['lanes_detected'] if lane_data else 0
        objs_n  = len(result.get('all_detections', []))
        cv2.putText(vis, f"Rear Lanes: {lanes_n}   Rear Objects: {objs_n}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (190, 190, 190), 1)

        # Safety panel
        self._draw_safety_panel(vis, result.get('lane_change_safety', {}))

        return vis

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_by_zone(self, detections: List[Dict]) -> Tuple[List, List, List]:
        """Partition detections into left / centre / right horizontal zones."""
        lx = int(self.img_w * REAR_LEFT_ZONE_RATIO)
        rx = int(self.img_w * REAR_RIGHT_ZONE_RATIO)
        left, centre, right = [], [], []
        for det in detections:
            cx = det['bbox_center'][0]
            if cx < lx:
                left.append(det)
            elif cx > rx:
                right.append(det)
            else:
                centre.append(det)
        return left, centre, right

    def _assess_safety(self, left_zone: List, right_zone: List) -> Dict:
        """
        Returns lane-change safety for left and right based on nearest vehicle
        in each zone.  Status: 'safe' | 'warning' | 'danger'
        """
        def _zone_status(vehicles):
            if not vehicles:
                return 'safe', None
            closest = min(vehicles, key=lambda d: d.get('distance') or float('inf'))
            dist = closest.get('distance')
            if dist is None:
                return 'warning', closest
            if dist <= BLIND_SPOT_DANGER_M:
                return 'danger', closest
            if dist <= BLIND_SPOT_WARN_M:
                return 'warning', closest
            return 'safe', closest

        ls, lv = _zone_status(left_zone)
        rs, rv = _zone_status(right_zone)
        return {'left': ls, 'left_vehicle': lv, 'right': rs, 'right_vehicle': rv}

    def _draw_safety_panel(self, vis: np.ndarray, safety: Dict):
        """Render the lane-change advisory strip at the bottom of the frame."""
        py = self.img_h - 46
        cv2.rectangle(vis, (0, py), (self.img_w, self.img_h), (15, 15, 15), -1)

        _col = {'safe': (0, 200, 0), 'warning': (0, 165, 255), 'danger': (0, 0, 220)}
        _lbl = {'safe': 'SAFE', 'warning': 'CAUTION', 'danger': 'DANGER'}

        # Left advisory
        ls = safety.get('left', 'safe')
        lv = safety.get('left_vehicle')
        ltxt = f"LANE CHG LEFT: {_lbl.get(ls, 'SAFE')}"
        if lv and lv.get('distance'):
            ltxt += f"  ({lv['distance']:.0f} m)"
        cv2.putText(vis, ltxt, (10, py + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, _col.get(ls, (200, 200, 200)), 2)

        # Right advisory
        rs = safety.get('right', 'safe')
        rv = safety.get('right_vehicle')
        rtxt = f"LANE CHG RIGHT: {_lbl.get(rs, 'SAFE')}"
        if rv and rv.get('distance'):
            rtxt += f"  ({rv['distance']:.0f} m)"
        cv2.putText(vis, rtxt, (self.img_w // 2 + 10, py + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, _col.get(rs, (200, 200, 200)), 2)
