"""
Dataset Analysis & Statistics Reporter — Task 07
==================================================

Analyses a recorded ADAS session and prints a comprehensive report:
  - Session metadata summary
  - Frame coverage and drop rate
  - Obstacle statistics (class distribution, distance, severity)
  - Pipeline latency summary (per-stage percentiles)
  - File size breakdown
  - Dataset quality indicators

Usage:
    python export_stats.py recordings/session_20240115_143022
    python export_stats.py recordings/session_20240115_143022 --json
    python export_stats.py recordings/session_20240115_143022 --csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Core analysis functions
# ══════════════════════════════════════════════════════════════════════════════

def collect_stats(session_dir: Path) -> dict:
    """
    Walk a session directory and collect aggregate statistics.

    Returns a dict with all statistics. Does not print anything.
    """
    from lidar_dataset import discover_frames, load_metadata

    metadata = load_metadata(session_dir)
    frames   = discover_frames(session_dir)
    n_frames = len(frames)

    if n_frames == 0:
        return {'error': 'No valid frames found', 'n_frames': 0}

    # ── Per-frame accumulators ─────────────────────────────────────────────
    timestamps        : List[float] = []
    n_raw_pts_list    : List[int]   = []
    n_filt_pts_list   : List[int]   = []
    n_obstacles_list  : List[int]   = []
    n_confirmed_list  : List[int]   = []
    distances         : List[float] = []
    ttc_values        : List[float] = []
    class_counts      : Counter     = Counter()
    severity_counts   : Counter     = Counter()
    fusion_counts     : Counter     = Counter()
    sector_counts     : Counter     = Counter()

    timing_preproc    : List[float] = []
    timing_cluster    : List[float] = []
    timing_tracking   : List[float] = []
    timing_fusion     : List[float] = []
    timing_total      : List[float] = []

    file_sizes        : Dict[str, List[int]] = defaultdict(list)

    # Ego state
    speeds_ms : List[float] = []

    for frame_dir in frames:
        fid_str = frame_dir.name

        # ── Pipeline stats ─────────────────────────────────────────────────
        stats_path = frame_dir / 'pipeline_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                ps = json.load(f)
            timestamps.append(float(ps.get('timestamp', 0.0)))

            lf = ps.get('lidar_frame', {})
            n_raw_pts_list.append(int(lf.get('n_raw_points', 0)))
            n_filt_pts_list.append(int(lf.get('n_filtered_pts', 0)))

            tm = ps.get('timing_ms', {})
            timing_preproc.append(float(tm.get('lidar_preproc', 0.0)))
            timing_cluster.append(float(tm.get('clustering', 0.0)))
            timing_tracking.append(float(tm.get('tracking', 0.0)))
            timing_fusion.append(float(tm.get('fusion', 0.0)))
            timing_total.append(float(tm.get('total_pipeline', 0.0)))

            fs_stats = ps.get('fusion_stats', {})
            fusion_counts['FULL']       += int(fs_stats.get('n_full', 0))
            fusion_counts['LIDAR_ONLY'] += int(fs_stats.get('n_lidar_only', 0))
            fusion_counts['CAMERA_ONLY']+= int(fs_stats.get('n_camera_only', 0))

        # ── Obstacles ──────────────────────────────────────────────────────
        obs_path = frame_dir / 'obstacles.json'
        if obs_path.exists():
            with open(obs_path, 'r') as f:
                obs_data = json.load(f)
            obs_list = obs_data.get('obstacles', [])
            n_obstacles_list.append(len(obs_list))
            n_confirmed = sum(1 for o in obs_list if o.get('is_confirmed', False))
            n_confirmed_list.append(n_confirmed)

            for obs in obs_list:
                d = obs.get('min_distance', -1.0)
                if d >= 0:
                    distances.append(float(d))
                ttc = obs.get('ttc_seconds')
                if ttc is not None and ttc < 60.0:
                    ttc_values.append(float(ttc))
                class_counts[obs.get('fused_class', 'unknown')] += 1
                severity_counts[obs.get('severity', 'SAFE')]    += 1
                fusion_counts_obs = obs.get('fusion_method', 'LIDAR_ONLY')
                sector_counts[obs.get('sector', 'UNKNOWN')]     += 1

        # ── Ego state ──────────────────────────────────────────────────────
        ego_path = frame_dir / 'ego_state.json'
        if ego_path.exists():
            with open(ego_path, 'r') as f:
                ego = json.load(f)
            speeds_ms.append(float(ego.get('speed_ms', 0.0)))

        # ── File sizes ─────────────────────────────────────────────────────
        for fname in ['lidar_raw.npy', 'lidar_filtered.npy',
                      'camera_rgb.jpg', 'radar_bev.png',
                      'obstacles.json', 'pipeline_stats.json']:
            fp = frame_dir / fname
            if fp.exists():
                file_sizes[fname].append(fp.stat().st_size)

    # ── Aggregate ──────────────────────────────────────────────────────────

    def _percentiles(arr: List[float], pcts=(50, 90, 95, 99)) -> dict:
        if not arr:
            return {f'p{p}': 0.0 for p in pcts}
        a = np.array(arr, dtype=np.float64)
        return {f'p{p}': round(float(np.percentile(a, p)), 2) for p in pcts}

    def _basic(arr: List[float]) -> dict:
        if not arr:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
        a = np.array(arr, dtype=np.float64)
        return {
            'mean' : round(float(a.mean()), 3),
            'std'  : round(float(a.std()),  3),
            'min'  : round(float(a.min()),  3),
            'max'  : round(float(a.max()),  3),
            'count': len(arr),
        }

    def _size_mb(sizes: List[int]) -> dict:
        if not sizes:
            return {'mean_kb': 0.0, 'total_mb': 0.0}
        a = np.array(sizes)
        return {
            'mean_kb' : round(float(a.mean()) / 1024, 1),
            'total_mb': round(float(a.sum())  / (1024**2), 1),
        }

    duration_s = 0.0
    if timestamps:
        duration_s = max(timestamps) - min(timestamps)

    stats_path = session_dir / 'session_stats.json'
    session_stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            session_stats = json.load(f)

    return {
        'session_id'      : metadata.get('session_id', session_dir.name),
        'carla_map'       : metadata.get('carla_map', 'unknown'),
        'recorder_version': metadata.get('recorder_version', 'unknown'),

        'frames': {
            'total'          : n_frames,
            'duration_s'     : round(duration_s, 1),
            'effective_hz'   : round(n_frames / max(duration_s, 1), 2),
            'frames_saved'   : session_stats.get('frames_saved', n_frames),
            'frames_dropped' : session_stats.get('frames_dropped', 0),
            'drop_rate_pct'  : session_stats.get('drop_rate_pct', 0.0),
        },

        'lidar': {
            'raw_points'     : _basic(n_raw_pts_list),
            'filtered_points': _basic(n_filt_pts_list),
            'filter_ratio'   : (
                round(np.mean([f / max(r, 1)
                               for r, f in zip(n_raw_pts_list, n_filt_pts_list)]), 4)
                if n_raw_pts_list else 0.0
            ),
        },

        'obstacles': {
            'total_detections': sum(n_obstacles_list),
            'per_frame'       : _basic(n_obstacles_list),
            'confirmed_ratio' : (
                round(sum(n_confirmed_list) / max(sum(n_obstacles_list), 1), 4)
            ),
            'distance_m'      : _basic(distances),
            'ttc_s'           : _basic(ttc_values),
            'class_counts'    : dict(class_counts),
            'severity_counts' : dict(severity_counts),
            'fusion_counts'   : dict(fusion_counts),
            'sector_counts'   : dict(sector_counts),
        },

        'ego': {
            'speed_ms': _basic(speeds_ms),
        },

        'timing_ms': {
            'lidar_preproc': {**_basic(timing_preproc), **_percentiles(timing_preproc)},
            'clustering'   : {**_basic(timing_cluster),  **_percentiles(timing_cluster)},
            'tracking'     : {**_basic(timing_tracking), **_percentiles(timing_tracking)},
            'fusion'       : {**_basic(timing_fusion),   **_percentiles(timing_fusion)},
            'total'        : {**_basic(timing_total),    **_percentiles(timing_total)},
        },

        'file_sizes': {
            fname: _size_mb(sizes)
            for fname, sizes in file_sizes.items()
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Report printing
# ══════════════════════════════════════════════════════════════════════════════

def print_report(stats: dict) -> None:
    """Print a human-readable statistics report to stdout."""
    SEP  = '=' * 70
    SEP2 = '-' * 70

    def _row(label: str, value: str) -> None:
        print(f"  {label:<32} {value}")

    print(SEP)
    print(f"  ADAS Dataset Report — Task 07")
    print(SEP)
    _row("Session",         stats.get('session_id', '?'))
    _row("Map",             stats.get('carla_map', '?'))
    _row("Recorder version",stats.get('recorder_version', '?'))
    print()

    # ── Frames ────────────────────────────────────────────────────────────
    fr = stats.get('frames', {})
    print("  FRAMES")
    print(SEP2)
    _row("Total frames",    str(fr.get('total', 0)))
    _row("Duration",        f"{fr.get('duration_s', 0):.1f} s")
    _row("Effective rate",  f"{fr.get('effective_hz', 0):.2f} Hz")
    _row("Frames saved",    str(fr.get('frames_saved', 0)))
    _row("Frames dropped",  f"{fr.get('frames_dropped', 0)} ({fr.get('drop_rate_pct', 0):.1f}%)")
    print()

    # ── LiDAR ─────────────────────────────────────────────────────────────
    ld = stats.get('lidar', {})
    print("  LIDAR POINTS")
    print(SEP2)
    raw  = ld.get('raw_points', {})
    filt = ld.get('filtered_points', {})
    _row("Raw points (mean)",      f"{raw.get('mean', 0):,.0f}")
    _row("Filtered points (mean)", f"{filt.get('mean', 0):,.0f}")
    _row("Filter ratio (mean)",    f"{ld.get('filter_ratio', 0):.3f}")
    print()

    # ── Obstacles ─────────────────────────────────────────────────────────
    ob = stats.get('obstacles', {})
    print("  OBSTACLES")
    print(SEP2)
    _row("Total detections",     str(ob.get('total_detections', 0)))
    pf = ob.get('per_frame', {})
    _row("Per-frame mean",       f"{pf.get('mean', 0):.2f}")
    _row("Per-frame max",        str(pf.get('max', 0)))
    _row("Confirmed ratio",      f"{ob.get('confirmed_ratio', 0):.3f}")
    dist = ob.get('distance_m', {})
    _row("Distance mean",        f"{dist.get('mean', 0):.1f} m")
    _row("Distance min",         f"{dist.get('min', 0):.1f} m")
    ttc = ob.get('ttc_s', {})
    _row("TTC mean",             f"{ttc.get('mean', 0):.2f} s")
    print()
    print("  Class distribution:")
    for cls, cnt in sorted(ob.get('class_counts', {}).items()):
        _row(f"  {cls}", str(cnt))
    print("  Severity distribution:")
    for sev, cnt in ob.get('severity_counts', {}).items():
        _row(f"  {sev}", str(cnt))
    print("  Fusion method:")
    for method, cnt in ob.get('fusion_counts', {}).items():
        _row(f"  {method}", str(cnt))
    print()

    # ── Timing ────────────────────────────────────────────────────────────
    tm = stats.get('timing_ms', {})
    print("  PIPELINE TIMING (ms)")
    print(SEP2)
    print(f"  {'Stage':<18} {'mean':>7} {'p50':>7} {'p90':>7} {'p95':>7} {'p99':>7}")
    print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for stage in ['lidar_preproc', 'clustering', 'tracking', 'fusion', 'total']:
        s = tm.get(stage, {})
        print(f"  {stage:<18} "
              f"{s.get('mean',0):>7.1f} "
              f"{s.get('p50',0):>7.1f} "
              f"{s.get('p90',0):>7.1f} "
              f"{s.get('p95',0):>7.1f} "
              f"{s.get('p99',0):>7.1f}")
    print()

    # ── File sizes ────────────────────────────────────────────────────────
    fs = stats.get('file_sizes', {})
    if fs:
        print("  FILE SIZES")
        print(SEP2)
        total_mb = 0.0
        for fname, sz in sorted(fs.items()):
            mean_kb  = sz.get('mean_kb', 0.0)
            total_file_mb = sz.get('total_mb', 0.0)
            total_mb += total_file_mb
            _row(fname, f"{mean_kb:.1f} KB/frame  ({total_file_mb:.1f} MB total)")
        _row("TOTAL", f"{total_mb:.1f} MB")
        print()

    print(SEP)


def export_csv(stats: dict, path: Path) -> None:
    """Export per-stage timing stats to CSV."""
    import csv
    timing = stats.get('timing_ms', {})
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['stage', 'mean', 'std', 'min', 'max', 'p50', 'p90', 'p95', 'p99'])
        for stage, s in timing.items():
            writer.writerow([
                stage,
                s.get('mean', 0), s.get('std', 0),
                s.get('min',  0), s.get('max', 0),
                s.get('p50',  0), s.get('p90', 0),
                s.get('p95',  0), s.get('p99', 0),
            ])


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _main() -> None:
    parser = argparse.ArgumentParser(
        description='Analyse an ADAS dataset recording session.'
    )
    parser.add_argument('session_dir', help='Path to session directory')
    parser.add_argument('--json', action='store_true',
                        help='Print stats as JSON instead of human-readable')
    parser.add_argument('--csv', metavar='PATH',
                        help='Export timing stats to CSV file')
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        print(f"ERROR: {session_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Analysing {session_dir} ...")
    stats = collect_stats(session_dir)

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
    else:
        print_report(stats)

    if args.csv:
        export_csv(stats, Path(args.csv))
        print(f"Timing CSV written to {args.csv}")


if __name__ == '__main__':
    _main()
