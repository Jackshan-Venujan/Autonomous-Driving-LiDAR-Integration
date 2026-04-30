"""
Evaluation script to compare camera-only vs LiDAR-only obstacle detection.

Usage (example):
  python evaluation/obstacle_evaluation.py --camera camera_outputs_dir --lidar lidar_outputs_dir --gt gt.jsonl --out results

Inputs:
- camera outputs JSONL (camera_detections.jsonl) or directory containing JSONL files
- lidar outputs JSONL (lidar_obstacles.jsonl) or directory
- ground-truth JSONL: lines with frame_id, gt_id, class, bbox (optional), centroid (optional), distance (optional)

Outputs:
- results/summary.csv
- results/metrics.json
- optional plots if matplotlib available
"""

import os
import json
import argparse
import math
from collections import defaultdict


def _convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(v) for v in obj]
    
    # Convert numpy types
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.int_, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    
    return obj

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


# ----------------------------- IO helpers ---------------------------------

def load_jsonl_file(path):
    recs = []
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def find_jsonl_in_dir(d):
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.jsonl')]
    return files


def load_records(path_or_dir):
    """Load JSONL records from path or directory."""
    recs = []
    if os.path.isdir(path_or_dir):
        files = find_jsonl_in_dir(path_or_dir)
        for f in files:
            recs.extend(load_jsonl_file(f))
    elif os.path.isfile(path_or_dir):
        recs = load_jsonl_file(path_or_dir)
    else:
        raise FileNotFoundError(path_or_dir)
    return recs


# ----------------------------- Matching helpers ---------------------------

def iou_xyxy(boxA, boxB):
    if not boxA or not boxB:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def match_camera_detections(dets, gts, iou_thresh=0.5):
    """Greedy match detections to GT by IoU. Returns lists of matched pairs (det, gt), unmatched_dets, unmatched_gts."""
    matched = []
    unmatched_dets = []
    unmatched_gts = gts.copy()

    # sort dets by confidence descending
    dets_sorted = sorted(dets, key=lambda d: float(d.get('confidence', 0.0)), reverse=True)

    for d in dets_sorted:
        best_iou = 0.0
        best_gt = None
        for gt in unmatched_gts:
            i = iou_xyxy(d.get('bbox', []), gt.get('bbox', []))
            if i > best_iou:
                best_iou = i
                best_gt = gt
        if best_iou >= iou_thresh and best_gt is not None:
            matched.append((d, best_gt))
            unmatched_gts.remove(best_gt)
        else:
            unmatched_dets.append(d)
    return matched, unmatched_dets, unmatched_gts


def match_lidar_detections(dets, gts, dist_thresh=1.5):
    """Greedy match LiDAR obstacles to GT by centroid 2D distance.
    Dets and GTs must have 'centroid' (x,y,...) or both have 'distance' and an angle (less robust).
    """
    matched = []
    unmatched_dets = []
    unmatched_gts = gts.copy()

    # If centroids available, use them
    for d in dets:
        dx = d.get('centroid')
        if dx is None:
            continue
        best_dist = float('inf')
        best_gt = None
        for gt in unmatched_gts:
            gx = gt.get('centroid')
            if gx is None:
                continue
            dist = math.hypot(dx[0] - gx[0], dx[1] - gx[1])
            if dist < best_dist:
                best_dist = dist
                best_gt = gt
        if best_gt is not None and best_dist <= dist_thresh:
            matched.append((d, best_gt))
            unmatched_gts.remove(best_gt)
        else:
            unmatched_dets.append(d)

    # Any dets without centroid treated as unmatched
    # Remaining unmatched_gts are FN
    return matched, unmatched_dets, unmatched_gts


# ----------------------------- Metrics -----------------------------------

def compute_detection_counts(matched_pairs, unmatched_dets, unmatched_gts):
    TP = len(matched_pairs)
    FP = len(unmatched_dets)
    FN = len(unmatched_gts)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {'TP': TP, 'FP': FP, 'FN': FN, 'precision': precision, 'recall': recall, 'f1': f1}


def compute_distance_stats(errors):
    if not errors:
        return {}
    import math
    arr = list(errors)
    n = len(arr)
    mae = sum(abs(e) for e in arr) / n
    mse = sum(e * e for e in arr) / n
    rmse = math.sqrt(mse)
    median = sorted(abs(e) for e in arr)[n // 2]
    bias = sum(arr) / n
    within_05 = sum(1 for e in arr if abs(e) <= 0.5) / n
    within_1 = sum(1 for e in arr if abs(e) <= 1.0) / n
    within_2 = sum(1 for e in arr if abs(e) <= 2.0) / n
    return {
        'count': n,
        'mae': mae,
        'rmse': rmse,
        'median_abs_error': median,
        'bias': bias,
        'pct_within_0.5m': within_05,
        'pct_within_1m': within_1,
        'pct_within_2m': within_2,
    }


# ----------------------------- Runner ------------------------------------

def run_evaluation(camera_path, lidar_path, gt_path, out_dir, iou_thresh=0.5, lidar_dist_thresh=1.5):
    _ensure_out(out_dir)

    camera_recs = load_records(camera_path) if camera_path is not None else []
    lidar_recs = load_records(lidar_path) if lidar_path is not None else []
    gt_recs = load_records(gt_path) if gt_path is not None else []

    # Group by frame
    camera_by_frame = defaultdict(list)
    lidar_by_frame = defaultdict(list)
    gt_by_frame = defaultdict(list)

    for r in camera_recs:
        camera_by_frame[r.get('frame_id')].append(r)
    for r in lidar_recs:
        lidar_by_frame[r.get('frame_id')].append(r)
    for r in gt_recs:
        gt_by_frame[r.get('frame_id')].append(r)

    all_frames = set(list(camera_by_frame.keys()) + list(lidar_by_frame.keys()) + list(gt_by_frame.keys()))

    # Accumulators
    cam_tp = cam_fp = cam_fn = 0
    lidar_tp = lidar_fp = lidar_fn = 0
    cam_distance_errors = []
    lidar_distance_errors = []

    per_frame_summary = []

    for frame in sorted(all_frames):
        cams = camera_by_frame.get(frame, [])
        lids = lidar_by_frame.get(frame, [])
        gts = gt_by_frame.get(frame, [])

        cam_matched, cam_unmatched_dets, cam_unmatched_gts = match_camera_detections(cams, gts, iou_thresh=iou_thresh)
        lidar_matched, lidar_unmatched_dets, lidar_unmatched_gts = match_lidar_detections(lids, gts, dist_thresh=lidar_dist_thresh)

        cam_counts = compute_detection_counts(cam_matched, cam_unmatched_dets, cam_unmatched_gts)
        lid_counts = compute_detection_counts(lidar_matched, lidar_unmatched_dets, lidar_unmatched_gts)

        cam_tp += cam_counts['TP']; cam_fp += cam_counts['FP']; cam_fn += cam_counts['FN']
        lidar_tp += lid_counts['TP']; lidar_fp += lid_counts['FP']; lidar_fn += lid_counts['FN']

        # distance errors
        for (d, gt) in cam_matched:
            if d.get('distance') is not None and gt.get('distance') is not None:
                cam_distance_errors.append(float(d['distance']) - float(gt['distance']))
        for (d, gt) in lidar_matched:
            if d.get('distance') is not None and gt.get('distance') is not None:
                lidar_distance_errors.append(float(d['distance']) - float(gt['distance']))

        per_frame_summary.append({
            'frame_id': frame,
            'cam_TP': cam_counts['TP'], 'cam_FP': cam_counts['FP'], 'cam_FN': cam_counts['FN'],
            'lidar_TP': lid_counts['TP'], 'lidar_FP': lid_counts['FP'], 'lidar_FN': lid_counts['FN']
        })

    cam_metrics = {
        'TP': cam_tp, 'FP': cam_fp, 'FN': cam_fn,
    }
    lid_metrics = {
        'TP': lidar_tp, 'FP': lidar_fp, 'FN': lidar_fn,
    }

    cam_perf = compute_detection_counts([], [None] * cam_fp, [None] * cam_fn) if (cam_tp + cam_fp + cam_fn) == 0 else None
    # compute precision/recall properly
    cam_precision = cam_tp / (cam_tp + cam_fp) if (cam_tp + cam_fp) > 0 else 0.0
    cam_recall = cam_tp / (cam_tp + cam_fn) if (cam_tp + cam_fn) > 0 else 0.0
    cam_f1 = (2 * cam_precision * cam_recall / (cam_precision + cam_recall)) if (cam_precision + cam_recall) > 0 else 0.0

    lidar_precision = lidar_tp / (lidar_tp + lidar_fp) if (lidar_tp + lidar_fp) > 0 else 0.0
    lidar_recall = lidar_tp / (lidar_tp + lidar_fn) if (lidar_tp + lidar_fn) > 0 else 0.0
    lidar_f1 = (2 * lidar_precision * lidar_recall / (lidar_precision + lidar_recall)) if (lidar_precision + lidar_recall) > 0 else 0.0

    cam_dist_stats = compute_distance_stats(cam_distance_errors)
    lidar_dist_stats = compute_distance_stats(lidar_distance_errors)

    summary = {
        'camera': {
            'TP': cam_tp, 'FP': cam_fp, 'FN': cam_fn,
            'precision': cam_precision, 'recall': cam_recall, 'f1': cam_f1,
            'distance_stats': cam_dist_stats
        },
        'lidar': {
            'TP': lidar_tp, 'FP': lidar_fp, 'FN': lidar_fn,
            'precision': lidar_precision, 'recall': lidar_recall, 'f1': lidar_f1,
            'distance_stats': lidar_dist_stats
        },
        'frames_evaluated': len(all_frames)
    }

    # Save outputs
    out_csv = os.path.join(out_dir, 'per_frame_summary.csv')
    if pd is not None:
        pd.DataFrame(per_frame_summary).to_csv(out_csv, index=False)
    else:
        import csv
        with open(out_csv, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(per_frame_summary[0].keys()) if per_frame_summary else [])
            writer.writeheader()
            writer.writerows(per_frame_summary)

    out_json = os.path.join(out_dir, 'summary.json')
    with open(out_json, 'w') as fh:
        json.dump(_convert_to_serializable(summary), fh, indent=2)

    # Plots
    if _HAS_MATPLOTLIB:
        if cam_distance_errors:
            plt.figure()
            plt.hist(cam_distance_errors, bins=40)
            plt.title('Camera distance error (m)')
            plt.xlabel('error (m)')
            plt.savefig(os.path.join(out_dir, 'camera_distance_error_hist.png'))
            plt.close()
        if lidar_distance_errors:
            plt.figure()
            plt.hist(lidar_distance_errors, bins=40)
            plt.title('LiDAR distance error (m)')
            plt.xlabel('error (m)')
            plt.savefig(os.path.join(out_dir, 'lidar_distance_error_hist.png'))
            plt.close()

    print(f"Saved per-frame CSV to: {out_csv}")
    print(f"Saved summary JSON to: {out_json}")

    return summary


def _ensure_out(path):
    os.makedirs(path, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--camera', required=False, help='Camera outputs JSONL file or directory (camera_detections.jsonl)')
    p.add_argument('--lidar', required=False, help='LiDAR outputs JSONL file or directory (lidar_obstacles.jsonl)')
    p.add_argument('--gt', required=True, help='Ground-truth JSONL file or directory')
    p.add_argument('--out', default='evaluation_results', help='Output directory')
    p.add_argument('--iou', type=float, default=0.5, help='IoU threshold for camera matching')
    p.add_argument('--lidar-dist', type=float, default=1.5, help='Centroid distance threshold (m) for LiDAR matching')
    args = p.parse_args()

    summary = run_evaluation(args.camera, args.lidar, args.gt, args.out, iou_thresh=args.iou, lidar_dist_thresh=args.lidar_dist)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
