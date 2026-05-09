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
            'cam_dist_m', 'lidar_bbox_dist_m', 'lidar_snap_dist_m', 'gt_dist_m',
            'cam_error_m', 'lidar_error_m', 'lidar_snap_error_m',
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
        lidar_snapshot_dist: Optional[float] = None,
    ) -> None:
        self._frame += 1
        ts = time.perf_counter() - self._start_time

        cam_err  = abs(cam_dist - gt_dist)            if (cam_dist is not None            and gt_dist is not None) else None
        lid_err  = abs(lidar_bbox_dist - gt_dist)     if (lidar_bbox_dist is not None     and gt_dist is not None) else None
        snap_err = abs(lidar_snapshot_dist - gt_dist) if (lidar_snapshot_dist is not None and gt_dist is not None) else None

        record = {
            'frame': self._frame,
            'ts': ts,
            'cam_dist': cam_dist,
            'lidar_dist': lidar_bbox_dist,
            'snap_dist': lidar_snapshot_dist,
            'gt_dist': gt_dist,
            'cam_err': cam_err,
            'lid_err': lid_err,
            'snap_err': snap_err,
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
            _fmt(lidar_snapshot_dist),
            _fmt(gt_dist),
            _fmt(cam_err),
            _fmt(lid_err),
            _fmt(snap_err),
            f'{cam_latency_ms:.2f}',
            f'{lidar_latency_ms:.2f}',
        ])
        self._log_file.flush()

    # ------------------------------------------------------------------

    def compute_metrics(self) -> dict:
        cam_errs  = [r['cam_err']  for r in self._records if r['cam_err']  is not None]
        lid_errs  = [r['lid_err']  for r in self._records if r['lid_err']  is not None]
        snap_errs = [r['snap_err'] for r in self._records if r['snap_err'] is not None]
        total = len(self._records)

        cam_dists  = [r['cam_dist']   for r in self._records]
        lid_dists  = [r['lidar_dist'] for r in self._records]
        snap_dists = [r['snap_dist']  for r in self._records]
        gt_dists   = [r['gt_dist']    for r in self._records]

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
            'cam_mae':              mae(cam_errs),
            'cam_rmse':             rmse(cam_errs),
            'cam_mape':             mape(cam_dists, gt_dists),
            'lidar_mae':            mae(lid_errs),
            'lidar_rmse':           rmse(lid_errs),
            'lidar_mape':           mape(lid_dists, gt_dists),
            'snap_mae':             mae(snap_errs),
            'snap_rmse':            rmse(snap_errs),
            'snap_mape':            mape(snap_dists, gt_dists),
            'cam_detection_rate':   detection_rate(cam_dists),
            'lidar_detection_rate': detection_rate(lid_dists),
            'snap_detection_rate':  detection_rate(snap_dists),
            'cam_latency_mean':     float(np.mean(cam_lats)) if cam_lats else 0.0,
            'lidar_latency_mean':   float(np.mean(lid_lats)) if lid_lats else 0.0,
            'n_samples':            total,
        }

    # ------------------------------------------------------------------

    def render_panel(self, width: int = 700, height: int = 340) -> np.ndarray:
        """Return a BGR OpenCV image showing the metrics comparison table."""
        m = self.compute_metrics()
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        font = cv2.FONT_HERSHEY_SIMPLEX
        WHITE  = (230, 230, 230)
        CYAN   = (255, 220, 0)
        GREEN  = (80, 220, 80)
        YELLOW = (0, 215, 255)

        # Column x positions
        C_LABEL = 12
        C_CAM   = 200
        C_BBOX  = 370
        C_SNAP  = 540

        def _best3(a, b, c, lower_is_better=True):
            """Return color triple (cam, bbox, snap): GREEN=best, YELLOW=other."""
            vals = [a, b, c]
            valid = [(v, i) for i, v in enumerate(vals) if not math.isnan(v)]
            if not valid:
                return WHITE, WHITE, WHITE
            best_idx = min(valid, key=lambda x: x[0])[1] if lower_is_better \
                       else max(valid, key=lambda x: x[0])[1]
            colors = []
            for i, v in enumerate(vals):
                if math.isnan(v):
                    colors.append(WHITE)
                elif i == best_idx:
                    colors.append(GREEN)
                else:
                    colors.append(YELLOW)
            return tuple(colors)

        def _fmt(v):
            return f'{v:.3f}' if not math.isnan(v) else 'N/A'

        # Title
        cv2.putText(canvas, 'Distance Accuracy Metrics', (12, 28), font, 0.65, CYAN, 2)
        cv2.putText(canvas, f'Samples: {m["n_samples"]}', (500, 28), font, 0.55, WHITE, 1)

        # Header row
        cv2.putText(canvas, 'Metric',        (C_LABEL, 58), font, 0.50, WHITE, 1)
        cv2.putText(canvas, 'Camera',        (C_CAM,   58), font, 0.50, WHITE, 1)
        cv2.putText(canvas, 'LiDAR-BBox',   (C_BBOX,  58), font, 0.50, WHITE, 1)
        cv2.putText(canvas, 'LiDAR-Snap',   (C_SNAP,  58), font, 0.50, WHITE, 1)
        cv2.line(canvas, (10, 65), (width - 10, 65), (80, 80, 80), 1)

        rows = [
            ('MAE  (m)',    m['cam_mae'],            m['lidar_mae'],            m['snap_mae'],            True),
            ('RMSE (m)',    m['cam_rmse'],           m['lidar_rmse'],           m['snap_rmse'],           True),
            ('MAPE (%)',    m['cam_mape'],           m['lidar_mape'],           m['snap_mape'],           True),
            ('Detect Rate', m['cam_detection_rate'], m['lidar_detection_rate'], m['snap_detection_rate'], False),
            ('Latency(ms)', m['cam_latency_mean'],   m['lidar_latency_mean'],   float('nan'),             True),
        ]

        y = 90
        for label, cv_val, lv, sv, lib in rows:
            cc, lc, sc = _best3(cv_val, lv, sv, lib)
            cv2.putText(canvas, label,        (C_LABEL, y), font, 0.50, WHITE, 1)
            cv2.putText(canvas, _fmt(cv_val), (C_CAM,   y), font, 0.50, cc,   2)
            cv2.putText(canvas, _fmt(lv),     (C_BBOX,  y), font, 0.50, lc,   2)
            cv2.putText(canvas, _fmt(sv),     (C_SNAP,  y), font, 0.50, sc,   2)
            y += 38

        cv2.line(canvas, (10, y + 2), (width - 10, y + 2), (80, 80, 80), 1)
        y += 22

        # Winner declaration (three-way by MAE)
        maes = {
            'CAMERA':      m['cam_mae'],
            'LIDAR-BBOX':  m['lidar_mae'],
            'LIDAR-SNAP':  m['snap_mae'],
        }
        valid_maes = {k: v for k, v in maes.items() if not math.isnan(v)}
        if not valid_maes:
            winner_text = 'Insufficient data'
            w_color = WHITE
        elif len(valid_maes) == 1:
            winner_text = f'Winner: {list(valid_maes.keys())[0]} (only valid source)'
            w_color = GREEN
        else:
            best_name  = min(valid_maes, key=valid_maes.get)
            second_mae = sorted(valid_maes.values())[1]
            best_mae   = valid_maes[best_name]
            if best_mae < second_mae * 0.95:
                pct = (second_mae - best_mae) / second_mae * 100
                winner_text = f'Winner: {best_name}  ({pct:.1f}% lower MAE)'
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
            w.writerow(['metric', 'camera', 'lidar_bbox', 'lidar_snap'])
            w.writerow(['MAE_m',           _fmt(m['cam_mae']),            _fmt(m['lidar_mae']),   _fmt(m['snap_mae'])])
            w.writerow(['RMSE_m',          _fmt(m['cam_rmse']),           _fmt(m['lidar_rmse']),  _fmt(m['snap_rmse'])])
            w.writerow(['MAPE_pct',        _fmt(m['cam_mape']),           _fmt(m['lidar_mape']),  _fmt(m['snap_mape'])])
            w.writerow(['detection_rate',  _fmt(m['cam_detection_rate']), _fmt(m['lidar_detection_rate']), _fmt(m['snap_detection_rate'])])
            w.writerow(['mean_latency_ms', _fmt(m['cam_latency_mean']),   _fmt(m['lidar_latency_mean']),   'N/A'])
            w.writerow(['n_samples',       m['n_samples'],                m['n_samples'],         m['n_samples']])

            maes = {'CAMERA': m['cam_mae'], 'LIDAR-BBOX': m['lidar_mae'], 'LIDAR-SNAP': m['snap_mae']}
            valid = {k: v for k, v in maes.items() if not math.isnan(v)}
            if not valid:
                winner = 'N/A'
            else:
                best = min(valid, key=valid.get)
                second = sorted(valid.values())[1] if len(valid) > 1 else valid[best] * 2
                winner = best if valid[best] < second * 0.95 else 'TIE'
            w.writerow(['winner', winner, '-', '-'])

        print(f'[Metrics] Per-frame log  : {self._log_path}')
        print(f'[Metrics] Session summary: {self._summary_path}')
