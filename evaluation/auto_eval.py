#!/usr/bin/env python3
"""
Automated Evaluation: Compare camera & LiDAR detections against CARLA ground truth.

Usage:
  python evaluation/auto_eval.py
  
  Or after running main.py:
  python evaluation/auto_eval.py --camera output/eval_reports/camera_detections.jsonl --lidar output/eval_reports/lidar_obstacles.jsonl --gt output/ground_truth/carla_ground_truth.jsonl
"""

import argparse
import json
import os
from pathlib import Path


def run_auto_eval(camera_path=None, lidar_path=None, gt_path=None):
    """Run evaluation with automatically detected paths."""
    
    # Auto-detect paths
    if camera_path is None:
        camera_path = 'output/eval_reports/camera_detections.jsonl'
    if lidar_path is None:
        lidar_path = 'output/eval_reports/lidar_obstacles.jsonl'
    if gt_path is None:
        gt_path = 'output/ground_truth/carla_ground_truth.jsonl'
    
    out_dir = 'evaluation_results_with_gt'
    
    # Check if files exist
    missing = []
    for path, name in [(camera_path, 'Camera'), (lidar_path, 'LiDAR'), (gt_path, 'Ground Truth')]:
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")
    
    if missing:
        print("❌ Missing files:")
        for m in missing:
            print(f"   {m}")
        print("\n💡 Make sure you ran: python main.py [wait 1-2 min] [press Q]")
        return False
    
    print("\n" + "="*80)
    print("  AUTOMATED EVALUATION: Camera vs LiDAR vs CARLA Ground Truth")
    print("="*80)
    print(f"\n📁 Using detections from:")
    print(f"   Camera: {camera_path}")
    print(f"   LiDAR:  {lidar_path}")
    print(f"   GT:     {gt_path}")
    
    # Count records
    with open(camera_path) as f:
        cam_count = sum(1 for _ in f)
    with open(lidar_path) as f:
        lidar_count = sum(1 for _ in f)
    with open(gt_path) as f:
        gt_count = sum(1 for _ in f)
    
    print(f"\n📊 Data summary:")
    print(f"   Camera detections: {cam_count}")
    print(f"   LiDAR obstacles: {lidar_count}")
    print(f"   GT objects: {gt_count}")
    
    # Run evaluation
    import sys
    sys.path.insert(0, os.path.dirname(__file__).rstrip('/'))
    from obstacle_evaluation import run_evaluation
    
    print(f"\n▶️  Running evaluation (IoU threshold 0.5, LiDAR dist threshold 1.5m)...")
    summary = run_evaluation(
        camera_path=camera_path,
        lidar_path=lidar_path,
        gt_path=gt_path,
        out_dir=out_dir,
        iou_thresh=0.5,
        lidar_dist_thresh=1.5
    )
    
    # Print results
    print("\n" + "="*80)
    print("  RESULTS")
    print("="*80)
    
    print(f"\n📷 CAMERA PERFORMANCE:")
    cam = summary['camera']
    print(f"   TP (correct): {cam['TP']}")
    print(f"   FP (false alarms): {cam['FP']}")
    print(f"   FN (missed): {cam['FN']}")
    print(f"   Precision: {cam['precision']:.1%} (out of {cam['TP'] + cam['FP']} detected)")
    print(f"   Recall: {cam['recall']:.1%} (found {cam['TP']} of {cam['TP'] + cam['FN']})")
    print(f"   F1: {cam['f1']:.3f}")
    
    dist_cam = cam.get('distance_stats', {})
    if dist_cam and dist_cam.get('count', 0) > 0:
        print(f"\n   Distance accuracy (on {dist_cam['count']} matched objects):")
        print(f"     MAE: {dist_cam.get('mae', 'N/A'):.2f}m")
        print(f"     RMSE: {dist_cam.get('rmse', 'N/A'):.2f}m")
        print(f"     Bias: {dist_cam.get('bias', 'N/A'):+.2f}m")
        print(f"     % within 1m: {dist_cam.get('pct_within_1m', 0):.1%}")
        print(f"     % within 2m: {dist_cam.get('pct_within_2m', 0):.1%}")
    
    print(f"\n🔴 LIDAR PERFORMANCE:")
    lidar = summary['lidar']
    print(f"   TP (correct): {lidar['TP']}")
    print(f"   FP (false alarms): {lidar['FP']}")
    print(f"   FN (missed): {lidar['FN']}")
    print(f"   Precision: {lidar['precision']:.1%} (out of {lidar['TP'] + lidar['FP']} detected)")
    print(f"   Recall: {lidar['recall']:.1%} (found {lidar['TP']} of {lidar['TP'] + lidar['FN']})")
    print(f"   F1: {lidar['f1']:.3f}")
    
    dist_lidar = lidar.get('distance_stats', {})
    if dist_lidar and dist_lidar.get('count', 0) > 0:
        print(f"\n   Distance accuracy (on {dist_lidar['count']} matched objects):")
        print(f"     MAE: {dist_lidar.get('mae', 'N/A'):.2f}m")
        print(f"     RMSE: {dist_lidar.get('rmse', 'N/A'):.2f}m")
        print(f"     Bias: {dist_lidar.get('bias', 'N/A'):+.2f}m")
        print(f"     % within 1m: {dist_lidar.get('pct_within_1m', 0):.1%}")
        print(f"     % within 2m: {dist_lidar.get('pct_within_2m', 0):.1%}")
    
    # Comparison
    print(f"\n⚖️  COMPARISON:")
    if cam['recall'] > lidar['recall']:
        diff = (cam['recall'] - lidar['recall']) * 100
        print(f"   ✅ Camera detects more objects (+{diff:.1f}% recall)")
    elif lidar['recall'] > cam['recall']:
        diff = (lidar['recall'] - cam['recall']) * 100
        print(f"   ✅ LiDAR detects more objects (+{diff:.1f}% recall)")
    
    if cam['precision'] > lidar['precision']:
        diff = (cam['precision'] - lidar['precision']) * 100
        print(f"   ✅ Camera has fewer false alarms (+{diff:.1f}% precision)")
    elif lidar['precision'] > cam['precision']:
        diff = (lidar['precision'] - cam['precision']) * 100
        print(f"   ✅ LiDAR has fewer false alarms (+{diff:.1f}% precision)")
    
    print(f"\n📁 Results saved to: {out_dir}/")
    print(f"   - summary.json (full metrics)")
    print(f"   - per_frame_summary.csv (per-frame breakdown)")
    if os.path.exists(f'{out_dir}/camera_distance_error_hist.png'):
        print(f"   - camera_distance_error_hist.png")
    if os.path.exists(f'{out_dir}/lidar_distance_error_hist.png'):
        print(f"   - lidar_distance_error_hist.png")
    
    print("="*80)
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Auto-evaluate detections against CARLA GT')
    p.add_argument('--camera', default=None, help='Path to camera detections JSONL')
    p.add_argument('--lidar', default=None, help='Path to LiDAR detections JSONL')
    p.add_argument('--gt', default=None, help='Path to ground truth JSONL')
    args = p.parse_args()
    
    success = run_auto_eval(args.camera, args.lidar, args.gt)
    exit(0 if success else 1)
