"""
Smoke test: generate synthetic detection and GT data, run evaluation.
"""
import os
import json
import tempfile

def create_test_data(out_dir):
    """Create minimal synthetic detection and GT JSONL files."""
    os.makedirs(out_dir, exist_ok=True)

    # Synthetic camera detections
    cam_file = os.path.join(out_dir, 'camera_detections.jsonl')
    cam_records = [
        {'frame_id': 'frame_001', 'detection_id': 'cam-001', 'sensor': 'camera', 'class': 'car', 
         'class_id': 2, 'bbox': [100, 150, 200, 300], 'bbox_center': [150, 225], 
         'distance': 9.5, 'confidence': 0.95, 'is_dangerous': True, 'danger_level': 'stop', 'in_lane': True},
        {'frame_id': 'frame_001', 'detection_id': 'cam-002', 'sensor': 'camera', 'class': 'car', 
         'class_id': 2, 'bbox': [250, 200, 350, 350], 'bbox_center': [300, 275], 
         'distance': 15.0, 'confidence': 0.87, 'is_dangerous': False, 'danger_level': 'warning', 'in_lane': True},
        
        {'frame_id': 'frame_002', 'detection_id': 'cam-003', 'sensor': 'camera', 'class': 'car', 
         'class_id': 2, 'bbox': [120, 160, 220, 310], 'bbox_center': [170, 235], 
         'distance': 8.9, 'confidence': 0.92, 'is_dangerous': True, 'danger_level': 'stop', 'in_lane': True},
        # False positive in frame_002
        {'frame_id': 'frame_002', 'detection_id': 'cam-004', 'sensor': 'camera', 'class': 'person', 
         'class_id': 0, 'bbox': [400, 100, 450, 250], 'bbox_center': [425, 175], 
         'distance': 25.0, 'confidence': 0.6, 'is_dangerous': False, 'danger_level': 'safe', 'in_lane': False},
    ]
    with open(cam_file, 'w') as fh:
        for rec in cam_records:
            fh.write(json.dumps(rec) + '\n')

    # Synthetic LiDAR obstacles
    lidar_file = os.path.join(out_dir, 'lidar_obstacles.jsonl')
    lidar_records = [
        {'frame_id': 'frame_001', 'detection_id': 'lidar-001', 'sensor': 'lidar', 
         'centroid': [15.0, -0.5, 0.0], 'distance': 15.02, 'angle_deg': -1.9, 
         'sector': 'front', 'point_count': 45, 'danger_level': 'cautious', 
         'bbox_min_x': 12.0, 'bbox_max_x': 18.0, 'bbox_min_y': -2.0, 'bbox_max_y': 1.0},
        
        {'frame_id': 'frame_002', 'detection_id': 'lidar-002', 'sensor': 'lidar', 
         'centroid': [14.2, -0.3, 0.0], 'distance': 14.2, 'angle_deg': -1.2, 
         'sector': 'front', 'point_count': 52, 'danger_level': 'cautious', 
         'bbox_min_x': 11.5, 'bbox_max_x': 17.0, 'bbox_min_y': -1.8, 'bbox_max_y': 1.2},
    ]
    with open(lidar_file, 'w') as fh:
        for rec in lidar_records:
            fh.write(json.dumps(rec) + '\n')

    # Synthetic ground truth
    gt_file = os.path.join(out_dir, 'ground_truth.jsonl')
    gt_records = [
        {'frame_id': 'frame_001', 'gt_id': 'gt-001', 'class': 'car', 
         'bbox': [105, 155, 205, 305], 'centroid': [14.8, -0.4, 0.0], 'distance': 10.0},
        {'frame_id': 'frame_001', 'gt_id': 'gt-002', 'class': 'car', 
         'bbox': [255, 205, 355, 355], 'centroid': [28.0, 1.2, 0.0], 'distance': 28.2},
        
        {'frame_id': 'frame_002', 'gt_id': 'gt-003', 'class': 'car', 
         'bbox': [125, 165, 225, 315], 'centroid': [14.0, -0.2, 0.0], 'distance': 14.0},
    ]
    with open(gt_file, 'w') as fh:
        for rec in gt_records:
            fh.write(json.dumps(rec) + '\n')

    return cam_file, lidar_file, gt_file


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/auto_drive/auto_drive/codes-phase-II/Phase-II-testing-lidar/Autonomous-Driving-LiDAR-Integration')
    
    from evaluation.obstacle_evaluation import run_evaluation

    test_data_dir = '/tmp/obstacle_eval_test'
    out_dir = '/tmp/obstacle_eval_results'
    
    print("Creating synthetic test data...")
    cam_file, lidar_file, gt_file = create_test_data(test_data_dir)
    print(f"  Camera: {cam_file}")
    print(f"  LiDAR: {lidar_file}")
    print(f"  GT: {gt_file}")
    
    print("\nRunning evaluation...")
    summary = run_evaluation(
        camera_path=cam_file,
        lidar_path=lidar_file,
        gt_path=gt_file,
        out_dir=out_dir,
        iou_thresh=0.5,
        lidar_dist_thresh=2.0
    )
    
    print("\n=== EVALUATION RESULTS ===")
    import json
    print(json.dumps(summary, indent=2))
    
    print(f"\nOutputs saved to: {out_dir}")
    print("Files:")
    for f in os.listdir(out_dir):
        print(f"  - {f}")
