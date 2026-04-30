# CARLA Ground Truth Evaluation Workflow

## Overview

Automatically extract ground truth from CARLA simulation API during `main.py` execution. This captures all actor positions every frame, then compares them against camera and LiDAR detections.

## Quick Start (3 steps)

### Step 1: Run Simulation with Auto GT Extraction
```bash
python main.py
# [Wait 1-2 minutes, drive/navigate as desired]
# [Press Q to exit]
```

This creates three files:
- `output/eval_reports/camera_detections.jsonl` — all camera detections
- `output/eval_reports/lidar_obstacles.jsonl` — all LiDAR obstacles  
- `output/ground_truth/carla_ground_truth.jsonl` — auto-extracted GT from CARLA

### Step 2: Run Automated Evaluation
```bash
python evaluation/auto_eval.py
```

This automatically finds the three files above and runs evaluation.

### Step 3: Review Results
```bash
cat evaluation_results_with_gt/summary.json | python -m json.tool
```

## What Gets Recorded

**Camera detection example:**
```json
{"frame_id": "frame_0001", "detection_id": "cam-001", "sensor": "camera", "class": "car", "distance": 20.1, "confidence": 0.92, "danger_level": "stop"}
```

**LiDAR obstacle example:**
```json
{"frame_id": "frame_0001", "detection_id": "lidar-001", "sensor": "lidar", "centroid": [20.0, 0.3, 0.0], "distance": 20.02, "sector": "front", "point_count": 45}
```

**CARLA Ground Truth example:**
```json
{"frame_id": "frame_0001", "gt_id": "vehicle.tesla.model3-502", "class": "car", "centroid": [20.0, 0.2, 0.0], "distance": 20.01, "actor_id": 502}
```

## What It Measures

### Detection Metrics
- **Recall** — % of GT objects detected (higher = better for safety)
- **Precision** — % of detections correct (higher = fewer false alarms)
- **F1** — harmonic mean balancing recall & precision

### Distance/Localization Metrics (on correct detections only)
- **MAE** — mean absolute error (average distance estimation error)
- **RMSE** — root mean square error (penalizes outliers)
- **Bias** — systematic over/under-estimate
- **% within 1m / 2m** — how many estimates are close?

## Example Output

```
================================================================================
  AUTOMATED EVALUATION: Camera vs LiDAR vs CARLA Ground Truth
================================================================================

📊 Data summary:
   Camera detections: 1243
   LiDAR obstacles: 45
   GT objects: 1850

▶️  Running evaluation (IoU threshold 0.5, LiDAR dist threshold 1.5m)...

================================================================================
  RESULTS
================================================================================

📷 CAMERA PERFORMANCE:
   TP (correct): 1100
   FP (false alarms): 143
   FN (missed): 750
   Precision: 88.5% (out of 1243 detected)
   Recall: 59.5% (found 1100 of 1850)
   F1: 0.714

   Distance accuracy (on 1100 matched objects):
     MAE: 1.23m
     RMSE: 1.89m
     Bias: -0.12m (slight underestimate)
     % within 1m: 62.3%
     % within 2m: 84.1%

🔴 LIDAR PERFORMANCE:
   TP (correct): 35
   FP (false alarms): 10
   FN (missed): 1815
   Precision: 77.8% (out of 45 detected)
   Recall: 1.9% (found 35 of 1850)
   F1: 0.037

   Distance accuracy (on 35 matched objects):
     MAE: 0.34m
     RMSE: 0.45m
     Bias: +0.08m (slight overestimate)
     % within 1m: 97.1%
     % within 2m: 100.0%

⚖️  COMPARISON:
   ✅ Camera detects more objects (+57.6% recall)
   ✅ Camera has better precision (+10.7%)

📁 Results saved to: evaluation_results_with_gt/
   - summary.json (full metrics)
   - per_frame_summary.csv (per-frame breakdown)
   - camera_distance_error_hist.png
   - lidar_distance_error_hist.png
```

## How It Works

### GT Extraction (`CarlaGTExtractor`)

Each frame:
1. Get all actors from `world.get_actors()`
2. Convert actor 3D position to ego-relative coordinates:
   - X = forward (m)
   - Y = right (m)
   - Z = height above ground (m)
3. Compute distance = √(x² + y²)
4. Estimate 2D image bounding box using camera model
5. Save frame GT to JSONL

### Matching

**Camera ↔ GT:**
- Match by 2D image IoU (intersection-over-union)
- Threshold: 0.5 (adjustable: `--iou 0.5`)

**LiDAR ↔ GT:**
- Match by 3D centroid distance
- Threshold: 1.5m (adjustable: `--lidar-dist 1.5`)

**Distance Error Metrics:**
- Only computed on matched (TP) pairs
- FN and FP detections excluded from distance stats

## Advanced: Stratified Analysis

Evaluate by distance range:

```bash
# Filter GT to near objects only (< 20m)
jq 'select(.distance < 20)' output/ground_truth/carla_ground_truth.jsonl > gt_near.jsonl

# Re-evaluate
python evaluation/obstacle_evaluation.py \
  --camera output/eval_reports/camera_detections.jsonl \
  --gt gt_near.jsonl \
  --out results_near
```

Repeat for far objects (20-50m) to see range-dependent weaknesses.

## Troubleshooting

**Q: Auto-eval says files not found**
- Make sure you ran `python main.py` and exited with Q
- Check that `output/eval_reports/` and `output/ground_truth/` exist
- Files might be 0 bytes if simulation crashed; re-run

**Q: LiDAR detection rate very low (like above example)**
- Check LiDAR sensor config in main.py (range, FOV, channels)
- Check LiDAR preprocessor thresholds (`ground_z_threshold`, `ego_radius`)
- Run with LiDAR BEV visualization [P] to see if points are captured
- See [core/lidar_processor.py](../core/lidar_processor.py) for filtering parameters

**Q: Distance MAE very high for camera**
- Camera focal length calibration might be off
- Try adjusting `focal_length` in [modules/obstacle_detector.py](../modules/obstacle_detector.py)
- Verify assumed object heights in `OBJECT_HEIGHTS` dict match real objects

**Q: All detections are FP (false positives)**
- Ground truth extraction might be failing (check CARLA actors on screen)
- Thresholds too strict; try `--iou 0.3` or `--lidar-dist 2.5`
- Check simulation is actually spawning objects

## Next Steps

1. **Improve sensor** with weakest performance
   - Low recall → relax detection threshold
   - High FP → tighten threshold
   - Bad distance → recalibrate camera, check LiDAR mounting

2. **Sensor fusion** — combine strengths:
   ```python
   # Use camera for near objects, LiDAR for far
   if distance < 15m:
       trust = camera_detection
   else:
       trust = lidar_detection
   ```

3. **Analyze failure modes**:
   - Use per-frame CSV to find problem scenarios
   - Check object class (small person vs large truck)
   - Correlate with CARLA weather conditions

## Files Modified

- `main.py` — integrated GT extraction + live reporting
- `evaluation/carla_gt_extractor.py` — NEW: extracts actor data from CARLA
- `evaluation/auto_eval.py` — NEW: runs full evaluation with nice formatting
- `EVALUATION_GUIDE.md` — comprehensive manual (see main evaluation docs)
