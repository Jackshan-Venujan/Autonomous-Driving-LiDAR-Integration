# Task 05 — Camera–LiDAR Fusion Layer

## Overview

Camera–LiDAR late-fusion layer that combines semantically-rich YOLO detections
with geometrically-precise Kalman-tracked obstacles to produce a unified obstacle
list where every entry has both **WHAT** (semantic class) and **WHERE** (3D position).

### Why fusion

| Sensor alone | Strength | Weakness |
|---|---|---|
| Camera (YOLO) | Semantic class (car/person/cyclist) | No 3D — distance unknown |
| LiDAR (Task 04) | Accurate 3D geometry and distance | Class from size heuristic only |
| **Fused** | Both WHAT + WHERE with best accuracy | — |

Real-world impact: a black vehicle on a dark road returns very few LiDAR points
(low reflectivity) but is clearly visible to the camera. A pedestrian behind a
bush is invisible to the camera but detected as a vertical LiDAR cluster. Fusion
ensures neither case causes a safety miss.

---

## Files

| File | Purpose |
|---|---|
| `fusion_calibration.py` | `CameraCalibration` (K matrix, FOV) + `FusionExtrinsics` (LiDAR→Camera transform) |
| `lidar_fusion.py` | `SensorFusion`, `FusionConfig`, `FusedObstacle`, `FusionResult`, `YoloDetection` |
| `test_lidar_fusion.py` | 15 unit + integration tests |

---

## Quick Start

```python
from core.fusion_calibration import CameraCalibration, FusionExtrinsics
from core.lidar_fusion import SensorFusion, FusionConfig, YoloDetection

# Offline / unit-test mode (no CARLA required)
cal  = CameraCalibration.from_params(image_width=1280, image_height=720, fov_deg=90.0)
extr = FusionExtrinsics.from_offsets(
    lidar_xyz=(0.0, 0.0, 2.4),    # LiDAR mount: roof centre
    camera_xyz=(1.5, 0.0, 2.1),   # Camera mount: windshield
)
fusion = SensorFusion(cal, extr)

# --- Camera thread (30 fps) ---
def camera_callback(yolo_results, timestamp):
    dets = [
        YoloDetection(
            class_name=r.class_name,
            confidence=r.confidence,
            bbox_xyxy=r.bbox_xyxy,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        for r in yolo_results
    ]
    fusion.push_camera_frame(dets, timestamp)

# --- LiDAR thread (10 Hz) ---
def lidar_tick(tracked_obstacles, frame_id, timestamp):
    result = fusion.fuse(tracked_obstacles, frame_id, timestamp)

    for fo in result.fused_obstacles:
        print(f"[{fo.fusion_method}] {fo.fused_class} "
              f"dist={fo.min_distance:.1f}m  "
              f"ttc={fo.ttc_seconds:.1f}s  "
              f"sev={fo.severity}")
```

### Live CARLA

```python
cal  = CameraCalibration.from_carla_blueprint(cam_bp, 1280, 720)
extr = FusionExtrinsics.from_carla_transforms(
    lidar_sensor.get_transform(),
    camera_sensor.get_transform(),
)
```

---

## Output Contract — `FusedObstacle`

| Field | Type | Notes |
|---|---|---|
| `track_id` | int | Persistent Kalman ID; -1 for CAMERA_ONLY |
| `fusion_method` | str | `'FULL'` / `'LIDAR_ONLY'` / `'CAMERA_ONLY'` |
| `fused_class` | str | `vehicle` / `pedestrian` / `cyclist` / `structure` / `unknown` |
| `camera_confidence` | float | YOLO confidence; 0.0 for LIDAR_ONLY |
| `fused_confidence` | float | Weighted fusion of camera + LiDAR confidence |
| `center_xyz` | (3,) float32 | Kalman-filtered 3D position (vehicle frame) |
| `min_distance` | float | EMA-smoothed surface distance (m); **-1.0 = unknown** |
| `velocity_ms` | float | Radial closing speed (m/s) |
| `ttc_seconds` | float | Time-to-collision; `inf` if not approaching |
| `severity` | str | `BRAKE` / `WARN` / `SAFE` |
| `camera_bbox_xyxy` | tuple or None | YOLO pixel bbox `[x1,y1,x2,y2]` |
| `projected_bbox_xyxy` | tuple or None | LiDAR projected pixel bbox (for HUD overlay) |
| `match_iou` | float | IoU quality score for NN data filtering |
| `kalman_state` | (6,) float64 | `[px,py,pz,vx,vy,vz]` for Task 07 NN training |

### Fusion method meaning

```
FULL         → LiDAR track + YOLO detection matched (best quality — use for NN training)
LIDAR_ONLY   → LiDAR track, no camera match (geometry accurate; class from size heuristic)
CAMERA_ONLY  → Camera detection, no LiDAR (min_distance = -1.0; distance unknown)
```

---

## FusionConfig — Key Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_camera_age_s` | 0.08 | Drop stale camera frames older than 80 ms |
| `iou_threshold` | 0.15 | Min IoU for LiDAR-camera bbox match |
| `min_projected_points` | 3 | Min cluster points that must project into image |
| `camera_conf_weight` | 0.55 | Camera weight in fused confidence |
| `lidar_conf_weight` | 0.45 | LiDAR weight in fused confidence |

### Why IoU threshold is 0.15 (not 0.5)

Cross-sensor projection errors (calibration uncertainty, partial clusters,
centroid vs object-centre offset) cause IoU < 0.5 even for correct matches.
0.15 catches valid matches at range > 30 m where projection is imprecise.
0.10 accepts too many false matches. 0.25 rejects valid far-range matches.

---

## Coordinate Systems

```
LiDAR / vehicle frame (ISO 8855 right-hand, after Task 01 Y-flip):
  +X = forward   +Y = left   +Z = up

Camera frame (OpenCV pinhole):
  +X = image right   +Y = image down   +Z = optical axis (depth/forward)

CARLA uses left-hand (+Y = right). FusionExtrinsics applies R_fix:
  Camera X  =  CARLA camera Y
  Camera Y  = -CARLA camera Z
  Camera Z  =  CARLA camera X
```

---

## Projection Pipeline

```
Cluster point (px, py, pz) in LiDAR frame
    |
    v  T_lidar_cam (4x4 homogeneous transform)
    |
Camera frame point (Px, Py, Pz)
    |
    v  Perspective projection (pinhole model)
    |
  u = f * (Px / Pz) + cx
  v = f * (Py / Pz) + cy
    |
    v  Image bounds filter (+ margin)
    |
2D projected bbox [x1, y1, x2, y2]
    |
    v  IoU with YOLO bbox
    |
Match / no match
```

---

## Modified Files (Tasks 01–04)

Task 05 required two backward-compatible additions to earlier modules:

**`lidar_clusterer.py` — `Obstacle` dataclass:**
```python
points : Optional[np.ndarray] = None
# (N, 3) float32 cluster points — stored for Task 05 projection.
```

**`lidar_tracker.py` — `TrackedObstacle` dataclass:**
```python
cluster_points : Optional[np.ndarray] = None
# Carried from track.last_detection.points for Task 05 _project_obstacle().
```

Both additions use `None` defaults so all existing Task 01–04 code is unaffected.

---

## Test Results

```
Results: 15 passed, 0 failed / 15 total
Benchmark: mean=0.65 ms  p95=0.81 ms  budget=8.0 ms
```

| Test | Description |
|---|---|
| T01 | CameraCalibration focal length formula |
| T02 | K matrix shape and values |
| T03 | is_in_image boundary conditions |
| T04 | FusionExtrinsics axis convention (CARLA to OpenCV) |
| T05 | transform_points single point |
| T06 | YoloDetection properties |
| T07 | No camera frame -> all LIDAR_ONLY |
| T08 | Stale camera frame -> all LIDAR_ONLY |
| T09 | FULL fusion with synthetic projected cluster (IoU=0.433) |
| T10 | CAMERA_ONLY for unmatched YOLO detections |
| T11 | Output sorted by min_distance; CAMERA_ONLY last |
| T12 | FusionResult.fusion_quality() |
| T13 | FusedObstacle.to_dict() JSON serialisable |
| T14 | Thread safety (push from background thread) |
| T15 | Performance: 50 frames x 10 tracks x 5 dets |

---

## Downstream Usage

| Task | Consumes |
|---|---|
| Task 06 — HUD | `FusedObstacle.{fused_class, min_distance, ttc_seconds, severity, projected_bbox_xyxy, camera_bbox_xyxy, sector, bearing_deg}` |
| Task 07 — NN Export | `FusedObstacle.{kalman_state, kalman_uncertainty, camera_bbox_norm, fused_class, match_iou}` — filter by `fusion_method == 'FULL'` for highest quality labels |
