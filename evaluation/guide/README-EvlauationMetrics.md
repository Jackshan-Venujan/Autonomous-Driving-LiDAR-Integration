Quick usage

1) Export detections from your running detectors (example):

```python
from evaluation.exporters import export_camera_detections_jsonl, export_lidar_obstacles_jsonl

# camera: after running ObstacleDetector.detect(image)
# detections, annotated = obstacle_detector.detect(img)
# export_camera_detections_jsonl(detections, out_dir='output/eval/camera', frame_id='frame_0001')

# lidar: after running LidarObstacleDetector.detect(points)
# obstacles = lidar_detector.detect(points)
# export_lidar_obstacles_jsonl(obstacles, out_dir='output/eval/lidar', frame_id='frame_0001')
```

2) Run evaluation (requires a GT JSONL with fields: frame_id, gt_id, class, bbox (optional), centroid (optional), distance (optional)):

```bash
python evaluation/obstacle_evaluation.py \
  --camera output/eval/camera \
  --lidar output/eval/lidar \
  --gt path/to/gt.jsonl \
  --out evaluation_results \
  --iou 0.5 \
  --lidar-dist 1.5
```

3) Results written to `evaluation_results/`: `per_frame_summary.csv`, `summary.json`, and optional plots if matplotlib is installed.

Notes:
- If your GT lacks a `distance` field, distance/localization stats will be skipped.
- Camera matching uses 2D IoU on image bboxes; LiDAR uses 2D centroid distance in meters. Adjust thresholds as needed.
