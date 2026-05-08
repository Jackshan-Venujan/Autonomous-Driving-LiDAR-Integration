"""
Distance metrics logger — compares camera monocular distance, LiDAR-bbox
projected distance, and CARLA ground-truth distance frame by frame.

Outputs:
  distance_log_<timestamp>.csv     — live per-frame log (written incrementally)
  distance_summary_<timestamp>.csv — session summary written on close()
"""

import csv
import math
import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


class DistanceMetricsLogger:

    def __init__(self, output_dir: str = './metrics', window_size: int = 200):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        ts = time.strftime('%Y%m%d_%H%M%S')
        self._log_path = os.path.join(output_dir, f'distance_log_{ts}.csv')
        self._summary_path = os.path.join(output_dir, f'distance_summary_{ts}.csv')

        self._records: deque = deque(maxlen=window_size)
        self._frame = 0
        self._start_time = time.perf_counter()

        # Open streaming log file
        self._log_file = open(self._log_path, 'w', newline='')
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow([
            'frame', 'timestamp_s',
            'cam_dist_m', 'lidar_bbox_dist_m', 'gt_dist_m',
            'cam_error_m', 'lidar_error_m',
            'cam_latency_ms', 'lidar_latency_ms',
        ])
        self._log_file.flush()

    # ------------------------------------------------------------------

    def update(
        self,
        cam_dist: Optional[float],
        lidar_bbox_dist: Optional[float],
        gt_dist: Optional[float],
        cam_latency_ms: float = 0.0,
        lidar_latency_ms: float = 0.0,
    ) -> None:
        self._frame += 1
        ts = time.perf_counter() - self._start_time

        cam_err = abs(cam_dist - gt_dist) if (cam_dist is not None and gt_dist is not None) else None
        lid_err = abs(lidar_bbox_dist - gt_dist) if (lidar_bbox_dist is not None and gt_dist is not None) else None

        record = {
            'frame': self._frame,
            'ts': ts,
            'cam_dist': cam_dist,
            'lidar_dist': lidar_bbox_dist,
            'gt_dist': gt_dist,
            'cam_err': cam_err,
            'lid_err': lid_err,
            'cam_lat': cam_latency_ms,
            'lid_lat': lidar_latency_ms,
        }
        self._records.append(record)

        # Write to live CSV
        def _fmt(v):
            return f'{v:.4f}' if v is not None else ''

        self._log_writer.writerow([
            self._frame,
            f'{ts:.4f}',
            _fmt(cam_dist),
            _fmt(lidar_bbox_dist),
            _fmt(gt_dist),
            _fmt(cam_err),
            _fmt(lid_err),
            f'{cam_latency_ms:.2f}',
            f'{lidar_latency_ms:.2f}',
        ])
        self._log_file.flush()

    # ------------------------------------------------------------------

    def compute_metrics(self) -> dict:
        cam_errs = [r['cam_err'] for r in self._records if r['cam_err'] is not None]
        lid_errs = [r['lid_err'] for r in self._records if r['lid_err'] is not None]
        total = len(self._records)

        cam_dists = [r['cam_dist'] for r in self._records]
        lid_dists = [r['lidar_dist'] for r in self._records]
        gt_dists  = [r['gt_dist']   for r in self._records]

        cam_lats = [r['cam_lat'] for r in self._records]
        lid_lats = [r['lid_lat'] for r in self._records]

        def mae(errs):
            return float(np.mean(errs)) if errs else float('nan')

        def rmse(errs):
            return float(np.sqrt(np.mean(np.square(errs)))) if errs else float('nan')

        def mape(dists, gts):
            pcts = [
                abs(d - g) / g * 100.0
                for d, g in zip(dists, gts)
                if d is not None and g is not None and g > 0.5
            ]
            return float(np.mean(pcts)) if pcts else float('nan')

        def detection_rate(dists):
            valid = sum(1 for d in dists if d is not None)
            return valid / total if total > 0 else 0.0

        return {
            'cam_mae':            mae(cam_errs),
            'cam_rmse':           rmse(cam_errs),
            'cam_mape':           mape(cam_dists, gt_dists),
            'lidar_mae':          mae(lid_errs),
            'lidar_rmse':         rmse(lid_errs),
            'lidar_mape':         mape(lid_dists, gt_dists),
            'cam_detection_rate': detection_rate(cam_dists),
            'lidar_detection_rate': detection_rate(lid_dists),
            'cam_latency_mean':   float(np.mean(cam_lats)) if cam_lats else 0.0,
            'lidar_latency_mean': float(np.mean(lid_lats)) if lid_lats else 0.0,
            'n_samples':          total,
        }

    # ------------------------------------------------------------------

    def render_panel(self, width: int = 520, height: int = 340) -> np.ndarray:
        """Return a BGR OpenCV image showing the metrics comparison table."""
        m = self.compute_metrics()
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        font = cv2.FONT_HERSHEY_SIMPLEX
        WHITE  = (230, 230, 230)
        CYAN   = (255, 220, 0)
        GREEN  = (80, 220, 80)
        YELLOW = (0, 215, 255)
        RED    = (80, 80, 255)

        def _winner_color(cam_val, lid_val, lower_is_better=True):
            if math.isnan(cam_val) or math.isnan(lid_val):
                return WHITE, WHITE
            if lower_is_better:
                if lid_val < cam_val * 0.95:
                    return YELLOW, GREEN
                elif cam_val < lid_val * 0.95:
                    return GREEN, YELLOW
                else:
                    return YELLOW, YELLOW
            else:
                if lid_val > cam_val * 1.05:
                    return YELLOW, GREEN
                elif cam_val > lid_val * 1.05:
                    return GREEN, YELLOW
                else:
                    return YELLOW, YELLOW

        def _fmt(v):
            return f'{v:.3f}' if not math.isnan(v) else 'N/A'

        # Title
        cv2.putText(canvas, 'Distance Accuracy Metrics', (12, 28), font, 0.65, CYAN, 2)
        cv2.putText(canvas, f'Samples: {m["n_samples"]}', (340, 28), font, 0.55, WHITE, 1)

        # Header row
        cv2.putText(canvas, 'Metric',       (12,  58), font, 0.52, WHITE, 1)
        cv2.putText(canvas, 'Camera',       (220, 58), font, 0.52, WHITE, 1)
        cv2.putText(canvas, 'LiDAR-BBox',  (360, 58), font, 0.52, WHITE, 1)
        cv2.line(canvas, (10, 65), (width - 10, 65), (80, 80, 80), 1)

        rows = [
            ('MAE  (m)',         m['cam_mae'],            m['lidar_mae'],            True),
            ('RMSE (m)',         m['cam_rmse'],           m['lidar_rmse'],           True),
            ('MAPE (%)',         m['cam_mape'],           m['lidar_mape'],           True),
            ('Detect Rate',      m['cam_detection_rate'], m['lidar_detection_rate'], False),
            ('Latency (ms)',     m['cam_latency_mean'],   m['lidar_latency_mean'],   True),
        ]

        y = 90
        for label, cv, lv, lib in rows:
            cc, lc = _winner_color(cv, lv, lib)
            cv2.putText(canvas, label,   (12,  y), font, 0.52, WHITE,  1)
            cv2.putText(canvas, _fmt(cv), (220, y), font, 0.52, cc,    2)
            cv2.putText(canvas, _fmt(lv), (360, y), font, 0.52, lc,    2)
            y += 38

        cv2.line(canvas, (10, y + 2), (width - 10, y + 2), (80, 80, 80), 1)
        y += 22

        # Winner declaration
        cam_mae = m['cam_mae']
        lid_mae = m['lidar_mae']
        if math.isnan(cam_mae) and math.isnan(lid_mae):
            winner_text = 'Insufficient data'
            w_color = WHITE
        elif math.isnan(cam_mae):
            winner_text = 'Winner: LIDAR-BBOX (no cam data)'
            w_color = GREEN
        elif math.isnan(lid_mae):
            winner_text = 'Winner: CAMERA (no lidar-bbox data)'
            w_color = GREEN
        elif lid_mae < cam_mae * 0.95:
            pct = (cam_mae - lid_mae) / cam_mae * 100
            winner_text = f'Winner: LIDAR-BBOX  ({pct:.1f}% lower MAE)'
            w_color = GREEN
        elif cam_mae < lid_mae * 0.95:
            pct = (lid_mae - cam_mae) / lid_mae * 100
            winner_text = f'Winner: CAMERA  ({pct:.1f}% lower MAE)'
            w_color = GREEN
        else:
            winner_text = 'Too close to call (< 5% diff)'
            w_color = YELLOW

        cv2.putText(canvas, winner_text, (12, y + 18), font, 0.55, w_color, 2)

        y += 50
        cv2.putText(canvas, f'Log: {os.path.basename(self._log_path)}',
                    (12, y), font, 0.40, (120, 120, 120), 1)

        return canvas

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush live log and write session summary CSV."""
        self._log_file.flush()
        self._log_file.close()

        m = self.compute_metrics()

        def _fmt(v):
            return f'{v:.4f}' if not math.isnan(v) else 'N/A'

        with open(self._summary_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['metric', 'camera', 'lidar_bbox'])
            w.writerow(['MAE_m',            _fmt(m['cam_mae']),            _fmt(m['lidar_mae'])])
            w.writerow(['RMSE_m',           _fmt(m['cam_rmse']),           _fmt(m['lidar_rmse'])])
            w.writerow(['MAPE_pct',         _fmt(m['cam_mape']),           _fmt(m['lidar_mape'])])
            w.writerow(['detection_rate',   _fmt(m['cam_detection_rate']), _fmt(m['lidar_detection_rate'])])
            w.writerow(['mean_latency_ms',  _fmt(m['cam_latency_mean']),   _fmt(m['lidar_latency_mean'])])
            w.writerow(['n_samples',        m['n_samples'],                m['n_samples']])

            cam_mae = m['cam_mae']
            lid_mae = m['lidar_mae']
            if math.isnan(cam_mae) and math.isnan(lid_mae):
                winner = 'N/A'
            elif math.isnan(cam_mae):
                winner = 'LIDAR-BBOX'
            elif math.isnan(lid_mae):
                winner = 'CAMERA'
            elif lid_mae < cam_mae * 0.95:
                winner = 'LIDAR-BBOX'
            elif cam_mae < lid_mae * 0.95:
                winner = 'CAMERA'
            else:
                winner = 'TIE'
            w.writerow(['winner', winner, '-'])

        print(f'[Metrics] Per-frame log  : {self._log_path}')
        print(f'[Metrics] Session summary: {self._summary_path}')
