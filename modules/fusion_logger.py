"""
Fusion Logger Module

Logs per-object fusion data (LiDAR + camera) to a CSV file at a
throttled rate of ~5 fps, regardless of the main loop frame rate.

CSV columns:
    timestamp          – wall-clock time (seconds since epoch, 6 decimal places)
    frame_id           – main-loop frame counter
    camera             – 'front' | 'rear'
    object_id          – rank within frame sorted by nearest distance (0 = nearest)
    class              – YOLO class label (e.g. 'car', 'person')
    confidence         – YOLO detection confidence [0–1]
    bbox_x1,bbox_y1    – top-left corner of bounding box (pixels)
    bbox_x2,bbox_y2    – bottom-right corner
    camera_distance_m  – monocular distance estimate (metres); '' if unavailable
    lidar_distance_m   – LiDAR-derived distance (metres);    '' if unavailable
    angle_deg          – bearing from ego front (degrees, +90=right); '' if unavailable
    lidar_point_count  – number of LiDAR points inside the bounding box
"""

import csv
import os
import time
from datetime import datetime
from typing import List, Dict, Optional


_CSV_HEADER = [
    'timestamp',
    'frame_id',
    'camera',
    'object_id',
    'class',
    'confidence',
    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
    'camera_distance_m',
    'lidar_distance_m',
    'angle_deg',
    'lidar_point_count',
]


class FusionLogger:
    """
    Throttled CSV logger for LiDAR-camera fusion detections.

    Usage:
        logger = FusionLogger(log_dir='logs')
        # Inside the main loop:
        logger.log_frame(frame_id, front_fused_dets, rear_fused_dets, lidar_obstacles)
    """

    TARGET_FPS    = 5
    LOG_INTERVAL  = 1.0 / TARGET_FPS   # 0.2 s between writes

    def __init__(self, log_dir: str = 'logs'):
        os.makedirs(log_dir, exist_ok=True)

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'fusion_log_{timestamp_str}.csv')

        self._file   = open(self.log_path, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(_CSV_HEADER)
        self._file.flush()

        self._last_log_time = 0.0   # epoch seconds of last written frame

        print(f"✓ FusionLogger: {self.log_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_frame(
        self,
        frame_id:        int,
        front_fused_dets: List[Dict],
        rear_fused_dets:  List[Dict],
        lidar_obstacles:  List[Dict],
    ) -> bool:
        """
        Write per-detection rows for this frame (if throttle allows).

        Args:
            frame_id         : current frame counter
            front_fused_dets : enriched detections from front camera (fuse_with_detections)
            rear_fused_dets  : enriched detections from rear camera
            lidar_obstacles  : raw obstacle list from LidarProcessor.process()

        Returns:
            True if rows were written, False if throttled.
        """
        now = time.time()
        if (now - self._last_log_time) < self.LOG_INTERVAL:
            return False
        self._last_log_time = now

        self._write_camera_dets(now, frame_id, 'front',
                                front_fused_dets, lidar_obstacles)
        self._write_camera_dets(now, frame_id, 'rear',
                                rear_fused_dets,  lidar_obstacles)

        self._file.flush()
        return True

    def close(self):
        """Flush and close the CSV file."""
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_camera_dets(
        self,
        timestamp:       float,
        frame_id:        int,
        camera:          str,
        fused_dets:      List[Dict],
        lidar_obstacles: List[Dict],
    ):
        """Write one row per detection for a given camera."""
        if not fused_dets:
            return

        # Sort by best available distance (camera or LiDAR), nearest first
        def sort_key(d):
            ld = d.get('lidar_distance')
            cd = d.get('distance')
            if ld is not None:
                return ld
            if cd is not None:
                return cd
            return 999.0

        sorted_dets = sorted(fused_dets, key=sort_key)

        for obj_id, det in enumerate(sorted_dets):
            x1, y1, x2, y2 = det.get('bbox', (0, 0, 0, 0))
            cls      = det.get('class', '')
            conf     = det.get('confidence', '')
            cam_d    = det.get('distance')        # monocular
            lid_d    = det.get('lidar_distance')  # LiDAR-derived
            npts     = det.get('lidar_point_count', 0)

            # Find angle_deg from LiDAR obstacles via nearest-centroid match
            angle_deg = self._find_angle(det, lidar_obstacles)

            self._writer.writerow([
                f'{timestamp:.6f}',
                frame_id,
                camera,
                obj_id,
                cls,
                f'{conf:.3f}' if isinstance(conf, float) else conf,
                int(x1), int(y1), int(x2), int(y2),
                f'{cam_d:.2f}' if cam_d is not None else '',
                f'{lid_d:.2f}' if lid_d is not None else '',
                f'{angle_deg:.1f}' if angle_deg is not None else '',
                npts,
            ])

    @staticmethod
    def _find_angle(
        det: Dict,
        lidar_obstacles: List[Dict],
        dist_tolerance: float = 5.0,
    ) -> Optional[float]:
        """
        Match a YOLO detection to the nearest LiDAR cluster (by distance)
        and return the cluster's angle_deg.

        Uses the LiDAR-derived distance if available, else the camera distance.
        Returns None if no cluster is within dist_tolerance metres.
        """
        ref_dist = det.get('lidar_distance') or det.get('distance')
        if ref_dist is None or not lidar_obstacles:
            return None

        best_angle = None
        best_diff  = dist_tolerance

        for obs in lidar_obstacles:
            diff = abs(obs['distance'] - ref_dist)
            if diff < best_diff:
                best_diff  = diff
                best_angle = obs['angle_deg']

        return best_angle
