# Evaluation Framework: Camera vs LiDAR Obstacle Detection

## Overview

A complete evaluation framework to compare **camera-only** vs **LiDAR-only** obstacle detection performance in your autonomous driving system. The framework tracks detections at runtime and supports post-hoc analysis with ground-truth data.

### Key Components

1. **Live Reporter** (`evaluation/live_reporter.py`) — Runs during simulation to track detections and show live comparison stats
2. **Exporters** (`evaluation/exporters.py`) — Convert in-memory detections to standardized JSONL format
3. **Batch Evaluator** (`evaluation/obstacle_evaluation.py`) — Offline evaluation script to match detections to GT and compute accuracy metrics

---

## Quick Start

### Step 1: Run Main.py with Live Reporting

```bash
python main.py
```

**What happens:**
- Live reporter automatically starts and tracks all camera and LiDAR detections
- Every 300 frames, a live stats report is printed to console  
- On exit (Ctrl+C or pressing Q), final reports are saved:
  - `output/eval_reports/live_report.json` — aggregate statistics
  - `output/eval_reports/camera_detections.jsonl` — all camera detections
  - `output/eval_reports/lidar_obstacles.jsonl` — all LiDAR detections
  - **Sensor Comparison Summary** printed to console

Example output:
```
======================================================================
  LIVE EVALUATION REPORT
======================================================================
  Frames processed: 1500
  Elapsed time: 60.2s (24.9 fps)

  CAMERA:
    Total detections: 1243
    Dangerous: 89
    Avg per frame: 0.83

  LIDAR:
    Total detections: 1876
    Dangerous: 156
    Avg per frame: 1.25

  Latest Frame: frame_1500
    Camera detections: 2
    LiDAR detections: 3
======================================================================
```

### Step 2 (Optional): Run Batch Evaluation with Ground-Truth

If you have ground-truth annotations (JSONL format), run:

```bash
python evaluation/obstacle_evaluation.py \
  --camera output/eval_reports/camera_detections.jsonl \
  --lidar output/eval_reports/lidar_obstacles.jsonl \
  --gt path/to/ground_truth.jsonl \
  --out evaluation_results \
  --iou 0.5 \
  --lidar-dist 1.5
```

**Parameters:**
- `--iou 0.5` — intersection-over-union threshold for camera matching (default: 0.5)
- `--lidar-dist 1.5` — centroid distance threshold in meters for LiDAR matching (default: 1.5)

**Output files:**
- `evaluation_results/summary.json` — complete metrics (TP/FP/FN, precision, recall, F1, distance errors)
- `evaluation_results/per_frame_summary.csv` — per-frame statistics
- `evaluation_results/*.png` — distance error histograms (if matplotlib available)

---

## Metrics Explained

### Detection Metrics (on GT matches)

- **TP (True Positives)** — Detections correctly matched to GT objects
- **FP (False Positives)** — Detections with no GT match (spurious)
- **FN (False Negatives)** — GT objects with no detection
- **Precision** = TP / (TP + FP) — Of all positives predicted, how many were correct?
- **Recall** = TP / (TP + FN) — Of all actual positives, how many did we detect?
- **F1** = 2 × (Precision × Recall) / (Precision + Recall) — Harmonic mean balancing precision/recall

### Distance/Localization Metrics (on matched TP pairs only)

- **MAE** = Mean Absolute Error (m) — Average magnitude of distance estimation error
- **RMSE** = Root Mean Square Error (m) — Penalizes large outliers more
- **Median Absolute Error** (m) — Robust to outliers
- **Bias** = Mean Signed Error (m) — Is detector systematically over/under-estimating distance?
- **% within 0.5/1.0/2.0 m** — What fraction of detections are within N meters of GT?

---

## Ground-Truth Format

For batch evaluation, prepare a JSONL file with one GT object per line:

```json
{
  "frame_id": "frame_001",
  "gt_id": "gt-001",
  "class": "car",
  "bbox": [100, 150, 200, 300],
  "centroid": [14.8, -0.4, 0.0],
  "distance": 10.0
}
```

**Required fields:**
- `frame_id` — must match detection frame_id
- `gt_id` — unique identifier

**Optional fields:**
- `bbox` — [x1, y1, x2, y2] ints in image coords (for camera matching via IoU)
- `centroid` — [x, y, z] floats in 3D world coords (for LiDAR matching via distance)
- `distance` — ground-truth distance in meters (enables distance/localization error stats)
- `class` — object class name

**Matching Rules:**
- **Camera detections** → GT matched by 2D image IoU (configurable, default 0.5)
- **LiDAR obstacles** → GT matched by 2D centroid distance in meters (configurable, default 1.5 m)

---

## Crucial Metrics Comparison

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Recall** | Fraction of obstacles detected | Safety — missing obstacles is dangerous |
| **Precision** | Fraction of detections correct | False alarms → wasted control effort, jerky driving |
| **Distance MAE** | Avg distance estimation error | Collision avoidance — wrong distance → wrong braking |
| **Distance Bias** | Systematic over/under-estimate | If biased, can be calibrated; if random, indicates poor measurement |
| **% within 2m** | Accuracy at short range | Safety-critical: most accidents happen <5m away |
| **Detection at range** | Detection rate vs. object distance | Shows sensor effective range (camera fades at 30+m, LiDAR stable 0–50m) |

---

## Example Analysis Workflow

1. **Run** `main.py` for your test scenario (e.g., 5 min of highway driving)
2. **Extract** detections: `output/eval_reports/{camera,lidar}_detections.jsonl`
3. **Manually annotate** ~100 frames with bounding boxes and/or distances if GT unavailable
4. **Run** batch evaluator with GT
5. **Review** `summary.json` and CSV for:
   - Which sensor has higher recall? (detects more)
   - Which has better distance accuracy?
   - Where do they disagree? (use per-frame CSV to find problematic frames)
6. **Fusion insight**: Can fusing both sensors improve both metrics?

---

## Advanced: Stratified Evaluation

To evaluate metrics by distance range, object size, or road segments, manually filter the JSONL files and run separate evaluations:

```bash
# Evaluate only near objects (< 15m)
jq 'select(.distance < 15)' ground_truth.jsonl > gt_near.jsonl
python evaluation/obstacle_evaluation.py --camera ... --gt gt_near.jsonl --out results_near

# Evaluate only far objects (15-50m)
jq 'select(.distance >= 15 and .distance <= 50)' ground_truth.jsonl > gt_far.jsonl
python evaluation/obstacle_evaluation.py --camera ... --gt gt_far.jsonl --out results_far
```

Compare `results_near/summary.json` and `results_far/summary.json` to see if one sensor is better at specific ranges.

---

## Implementation Notes

### Live Reporter Integration

In `main.py`, after each frame:
```python
result = self.agent.process_frame(self.camera_data)
camera_dets = result['obstacle_data'].get('all_detections', [])
lidar_obs = result['lidar_obstacles']
self.eval_reporter.log_frame_detections(frame_count, camera_dets, lidar_obs)
```

### Detection Schema (JSONL)

**Camera detection:**
```json
{
  "frame_id": "frame_001",
  "detection_id": "cam-001",
  "sensor": "camera",
  "class": "car",
  "distance": 9.5,
  "confidence": 0.95,
  "danger_level": "stop",
  "in_lane": true
}
```

**LiDAR obstacle:**
```json
{
  "frame_id": "frame_001",
  "detection_id": "lidar-001",
  "sensor": "lidar",
  "centroid": [15.0, -0.5, 0.0],
  "distance": 15.02,
  "sector": "front",
  "point_count": 45,
  "danger_level": "cautious"
}
```

---

## Troubleshooting

**Q: No detections in JSONL files?**
- Check that detection JSONL files exist and are readable with `head -5 output/eval_reports/camera_detections.jsonl`
- Ensure vehicle is actually detecting objects (check console for detection counts)

**Q: Batch evaluator hangs?**
- Large GT files can be slow; try filtering to first 100 frames: `head -100 gt.jsonl > gt_small.jsonl`
- Distance error computation disabled if GT lacks `distance` field (only TP/FP/FN computed)

**Q: Plots not saving?**
- matplotlib may not be installed; install with `pip install matplotlib`
- If still missing, stats are still saved to JSON/CSV

**Q: How do I handle multiple objects of same class in one frame?**
- Evaluator does greedy matching (highest confidence first for camera, closest for LiDAR)
- Manual post-processing recommended for complex scenes

---

## Future Enhancements

- [ ] MOTA/MOTP (tracking metrics) if temporal consistency needed
- [ ] Fusion evaluation (compare fused vs. sensor-only)
- [ ] Per-class metrics
- [ ] Occlusion-aware evaluation (harder GT vs. easy GT)
- [ ] Interactive visualization of TP/FP/FN per frame
