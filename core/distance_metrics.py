"""
Distance metrics logger — compares camera monocular distance, LiDAR-bbox
projected distance, and CARLA ground-truth distance frame by frame.

Outputs:
  distance_log_<timestamp>.csv     — live per-frame aggregate log (one row/frame)
  obstacle_log_<timestamp>.csv     — live per-obstacle log (one row/obstacle/frame)
  distance_summary_<timestamp>.csv — session summary written on close()
"""

import csv
import math
import os
import time
from collections import deque
from typing import Dict, List, Optional

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

        # Open streaming aggregate log file (one row per frame)
        self._log_file = open(self._log_path, 'w', newline='')
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow([
            'frame', 'timestamp_s',
            'cam_dist_m', 'lidar_bbox_dist_m', 'gt_dist_m',
            'cam_error_m', 'lidar_error_m',
            'cam_latency_ms', 'lidar_latency_ms',
        ])
        self._log_file.flush()

        # Open per-obstacle log file (one row per obstacle per frame)
        self._obs_log_path = os.path.join(output_dir, f'obstacle_log_{ts}.csv')
        self._obs_file = open(self._obs_log_path, 'w', newline='')
        self._obs_writer = csv.writer(self._obs_file)
        self._obs_writer.writerow([
            'frame', 'timestamp_s',
            'obstacle_id', 'detection_source',
            'cam_dist_m', 'lidar_dist_m', 'gt_dist_m',
            'cam_error_m', 'lidar_error_m',
            'cam_rel_error_pct', 'lidar_rel_error_pct',
            'cam_within_1m', 'lidar_within_1m',
            'cam_latency_ms', 'lidar_latency_ms',
            'danger_level', 'angle_deg', 'sector', 'match_method',
        ])
        self._obs_file.flush()
        # In-memory store for per-obstacle records (unbounded — needed for summary stats)
        self._obs_records: List[dict] = []

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

    def log_obstacles(
        self,
        frame_n: int,
        fused_detections: List[Dict],
        gt_dist: Optional[float] = None,
        cam_latency_ms: float = 0.0,
        lidar_latency_ms: float = 0.0,
    ) -> None:
        """Write one CSV row per obstacle for cross-sensor comparison.

        Each row contains the independent camera distance and LiDAR distance
        for the same physical obstacle (identified by obstacle_id), alongside
        ground-truth and derived error metrics.
        """
        ts = time.perf_counter() - self._start_time

        def _f(v) -> str:
            return f'{v:.4f}' if v is not None else ''

        for det in fused_detections:
            obstacle_id  = det.get('obstacle_id', '')
            match_method = det.get('fusion_method', '')

            # Determine detection source from match method
            if match_method in ('FULL_BBOX', 'FULL_ANGLE'):
                detection_source = 'BOTH'
            elif match_method == 'LIDAR_ONLY':
                detection_source = 'LIDAR_ONLY'
            else:
                detection_source = 'CAM_ONLY'

            # Raw per-sensor distances (never fused)
            cam_dist   = det.get('distance')         # camera monocular estimate
            lidar_dist = det.get('lidar_distance')   # LiDAR cluster distance

            # Error metrics (only when GT is available)
            cam_err   = abs(cam_dist   - gt_dist) if (cam_dist   is not None and gt_dist is not None) else None
            lidar_err = abs(lidar_dist - gt_dist) if (lidar_dist is not None and gt_dist is not None) else None

            cam_rel_pct   = (cam_err   / gt_dist * 100.0) if (cam_err   is not None and gt_dist > 0) else None
            lidar_rel_pct = (lidar_err / gt_dist * 100.0) if (lidar_err is not None and gt_dist > 0) else None

            cam_w1m   = (1 if cam_err   <= 1.0 else 0) if cam_err   is not None else ''
            lidar_w1m = (1 if lidar_err <= 1.0 else 0) if lidar_err is not None else ''

            danger_level = det.get('danger_level') or det.get('lidar_danger') or ''
            angle_deg    = det.get('angle_deg')
            sector       = det.get('sector', '')

            record = {
                'frame':            frame_n,
                'ts':               ts,
                'obstacle_id':      obstacle_id,
                'detection_source': detection_source,
                'cam_dist':         cam_dist,
                'lidar_dist':       lidar_dist,
                'gt_dist':          gt_dist,
                'cam_err':          cam_err,
                'lidar_err':        lidar_err,
                'cam_rel_pct':      cam_rel_pct,
                'lidar_rel_pct':    lidar_rel_pct,
                'cam_w1m':          cam_w1m,
                'lidar_w1m':        lidar_w1m,
                'cam_lat':          cam_latency_ms,
                'lidar_lat':        lidar_latency_ms,
                'danger_level':     danger_level,
                'angle_deg':        angle_deg,
                'sector':           sector,
                'match_method':     match_method,
            }
            self._obs_records.append(record)

            self._obs_writer.writerow([
                frame_n,
                f'{ts:.4f}',
                obstacle_id,
                detection_source,
                _f(cam_dist),
                _f(lidar_dist),
                _f(gt_dist),
                _f(cam_err),
                _f(lidar_err),
                _f(cam_rel_pct),
                _f(lidar_rel_pct),
                cam_w1m,
                lidar_w1m,
                f'{cam_latency_ms:.2f}',
                f'{lidar_latency_ms:.2f}',
                danger_level,
                f'{angle_deg:.2f}' if isinstance(angle_deg, (int, float)) else '',
                sector,
                match_method,
            ])
        self._obs_file.flush()

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
        """Flush live logs and write session summary CSV."""
        self._log_file.flush()
        self._log_file.close()

        self._obs_file.flush()
        self._obs_file.close()

        m = self.compute_metrics()

        def _fmt(v):
            return f'{v:.4f}' if not math.isnan(v) else 'N/A'

        # --- Per-obstacle summary stats ---
        obs = self._obs_records
        n_obs = len(obs)

        both_rows    = [r for r in obs if r['detection_source'] == 'BOTH']
        cam_only_rows  = [r for r in obs if r['detection_source'] == 'CAM_ONLY']
        lidar_only_rows = [r for r in obs if r['detection_source'] == 'LIDAR_ONLY']

        cam_errs_obs  = [r['cam_err']   for r in obs if r['cam_err']   is not None]
        lidar_errs_obs = [r['lidar_err'] for r in obs if r['lidar_err'] is not None]

        def _pct95(errs):
            return float(np.percentile(errs, 95)) if errs else float('nan')

        def _within1m_pct(w_key):
            vals = [r[w_key] for r in obs if isinstance(r[w_key], int)]
            return float(np.mean(vals) * 100.0) if vals else float('nan')

        frames_with_cam   = len({r['frame'] for r in obs if r['cam_dist']   is not None})
        frames_with_lidar = len({r['frame'] for r in obs if r['lidar_dist'] is not None})
        frames_with_both  = len({r['frame'] for r in both_rows})
        total_frames = len({r['frame'] for r in obs}) if obs else 0

        def _frate(n):
            return f'{n / total_frames:.4f}' if total_frames else 'N/A'

        with open(self._summary_path, 'w', newline='') as f:
            w = csv.writer(f)

            # --- Aggregate frame-level metrics (existing) ---
            w.writerow(['metric', 'camera', 'lidar_bbox'])
            w.writerow(['MAE_m',            _fmt(m['cam_mae']),            _fmt(m['lidar_mae'])])
            w.writerow(['RMSE_m',           _fmt(m['cam_rmse']),           _fmt(m['lidar_rmse'])])
            w.writerow(['MAPE_pct',         _fmt(m['cam_mape']),           _fmt(m['lidar_mape'])])
            w.writerow(['detection_rate',   _fmt(m['cam_detection_rate']), _fmt(m['lidar_detection_rate'])])
            w.writerow(['mean_latency_ms',  _fmt(m['cam_latency_mean']),   _fmt(m['lidar_latency_mean'])])
            w.writerow(['n_frame_samples',  m['n_samples'],                m['n_samples']])

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
            w.writerow(['MAE_winner', winner, '-'])

            # --- Per-obstacle metrics (new) ---
            w.writerow([])
            w.writerow(['--- per-obstacle metrics ---', '', ''])
            w.writerow(['metric', 'camera', 'lidar'])
            w.writerow(['obstacle_MAE_m',
                        _fmt(float(np.mean(cam_errs_obs))  if cam_errs_obs  else float('nan')),
                        _fmt(float(np.mean(lidar_errs_obs)) if lidar_errs_obs else float('nan'))])
            w.writerow(['obstacle_RMSE_m',
                        _fmt(float(np.sqrt(np.mean(np.square(cam_errs_obs))))  if cam_errs_obs  else float('nan')),
                        _fmt(float(np.sqrt(np.mean(np.square(lidar_errs_obs)))) if lidar_errs_obs else float('nan'))])
            w.writerow(['obstacle_95th_pct_error_m', _fmt(_pct95(cam_errs_obs)), _fmt(_pct95(lidar_errs_obs))])
            w.writerow(['within_1m_pct',             _fmt(_within1m_pct('cam_w1m')), _fmt(_within1m_pct('lidar_w1m'))])
            w.writerow(['n_obstacle_rows',           n_obs, n_obs])

            # Detection source breakdown
            w.writerow([])
            w.writerow(['--- detection source rates (per obstacle row) ---', '', ''])
            w.writerow(['both_detected_rate',  _frate(frames_with_both),  '-'])
            w.writerow(['cam_detection_rate',  _frate(frames_with_cam),   '-'])
            w.writerow(['lidar_detection_rate', _frate(frames_with_lidar), '-'])
            w.writerow(['cam_only_obstacle_rate',   f'{len(cam_only_rows)  / n_obs:.4f}' if n_obs else 'N/A', '-'])
            w.writerow(['lidar_only_obstacle_rate', f'{len(lidar_only_rows) / n_obs:.4f}' if n_obs else 'N/A', '-'])
            w.writerow(['note_cam_only_rate',   '≈ camera FP / LiDAR FN rate', ''])
            w.writerow(['note_lidar_only_rate', '≈ camera FN / LiDAR FP rate', ''])

        print(f'[Metrics] Per-frame log   : {self._log_path}')
        print(f'[Metrics] Per-obstacle log: {self._obs_log_path}')
        print(f'[Metrics] Session summary : {self._summary_path}')
