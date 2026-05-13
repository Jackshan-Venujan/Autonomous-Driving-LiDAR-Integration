# LiDAR Architecture

This document describes the LiDAR sensing pipeline of the project: how raw
LiDAR scans flow from the CARLA simulator, through preprocessing, clustering,
and fusion with the camera, and finally drive the vehicle's control decisions.

---

## 1. High-level pipeline

```
   ┌─────────────────────────────┐
   │      CARLA Simulator        │
   │   (LiDAR on car roof)       │
   │   → shoots laser beams      │
   └──────────────┬──────────────┘
                  │  raw points (x, y, z, intensity)
                  ▼
   ┌─────────────────────────────┐
   │   1. LiDAR SENSOR           │
   │   core/lidar_sensor.py      │
   │                             │
   │   "Grab the latest scan"    │
   └──────────────┬──────────────┘
                  │  (N points)
                  ▼
   ┌─────────────────────────────┐
   │   2. LiDAR PROCESSOR        │
   │   core/lidar_processor.py   │
   │                             │
   │   • Remove ground           │
   │   • Keep 0.5 m – 50 m       │
   │   • Remove ego car          │
   └──────────────┬──────────────┘
                  │  (clean points)
                  ▼
   ┌─────────────────────────────┐
   │   3. OBSTACLE DETECTOR      │
   │   lidar_obstacle_detector.py│
   │                             │
   │   • Group points (DBSCAN)   │
   │   • Find each obstacle's    │
   │     distance & angle        │
   │   • Track over time         │
   │   • Decide danger level:    │
   │       drive / cautious /    │
   │       slow / stop /         │
   │       emergency_stop        │
   └──────────────┬──────────────┘
                  │  list of obstacles
                  ▼
   ┌─────────────────────────────┐
   │   4. FUSION                 │
   │   core/lidar_fusion.py      │
   │                             │
   │   Combine with camera:      │
   │   • match by bbox / angle   │
   │   • pick worst action       │
   │     (camera vs LiDAR)       │
   │                             │
   │   helper:                   │
   │   lidar_camera_projector.py │
   │   (projects 3D → 2D image)  │
   └──────────────┬──────────────┘
                  │  fused action +
                  │  fused obstacles
                  ▼
   ┌─────────────────────────────┐
   │   DRIVING AGENT             │
   │   modules/driving_agent.py  │
   │                             │
   │   → throttle / brake / steer│
   └─────────────────────────────┘
```

### One-sentence summary of each stage

1. **Sensor** — reads raw 3D points from CARLA's LiDAR.
2. **Processor** — throws away ground, far/near junk, and the car itself.
3. **Obstacle Detector** — clusters points into objects and rates how dangerous each one is.
4. **Fusion** — matches LiDAR objects to camera objects and picks the safer driving decision.
5. **Driving Agent** — turns the decision into actual throttle/brake/steer.

---

## 2. Component-by-component detail

### 2.1 `core/lidar_sensor.py` — `LidarSensor`
- Thread-safe wrapper around CARLA's ray-cast LiDAR (20 Hz).
- **Input:** raw sensor callback (bytes buffer).
- **Output:** `(N, 4)` array `[x, y, z, intensity]`.
- Key methods:
  - `_callback()` — unpacks raw bytes → float32 reshape.
  - `get_latest()` — thread-safe copy delivered to the caller.

### 2.2 `core/lidar_processor.py` — `LidarProcessor`
- Cleans up the raw cloud before clustering.
- **Input:** `(N, 4)` XYZI.
- **Output:** `(M, 3)` XYZ where `M < N`.
- Stages:
  - Ground removal — drop points with `z ≤ -1.4 m`.
  - Range filter — keep points with `0.5 m ≤ dist ≤ 50 m`.
  - Ego-body reject — drop points within `1.5 m` (the car itself).

### 2.3 `core/lidar_obstacle_detector.py` — `LidarObstacleDetector`
- Turns a cloud of points into a list of tracked obstacles with danger levels.
- **Input:** `(M, 3)` XYZ + `vehicle_speed_kmh`.
- **Output:** `List[LidarObstacle]` with `track_id`, `distance`, `angle_deg`,
  `sector`, `danger_level`.
- Stages:
  - DBSCAN clustering (`eps=0.5`, `min_samples=3`) on the XY plane.
  - Centroid + bearing calculation: `angle_deg = atan2(cy, cx)`.
  - Sector assignment — front (±30°), side_left/right, rear (skipped).
  - Temporal track matching with a ±10° angle window.
  - Point-history accumulation over the last 5 frames for centroid stability.
  - Asymmetric EMA — fast escalation, damped de-escalation.
  - Danger hysteresis — escalation immediate, de-escalation requires 3 frames.
  - Track eviction after 3 stale frames.
- Danger levels (in order of severity):
  `drive → cautious → slow → stop → emergency_stop`.

### 2.4 `core/lidar_camera_projector.py` — `LidarCameraProjector`
*See section 3 below for a deep dive — this is the bridge between the 3D
LiDAR world and the 2D camera image.*

### 2.5 `core/lidar_fusion.py` — `LidarFusion`
- Conservative fusion of camera detections and LiDAR clusters.
- **Input:** `camera_detections`, `lidar_obstacles`, `camera_action`,
  `vehicle_speed_kmh`, image width, focal length, raw LiDAR points.
- **Output:** `(fused_detections, fused_action, nearest_obstacle)`.
- For each camera detection, fusion tries three strategies in order:
  - **Strategy A — BBox projection** (preferred): projects raw LiDAR points
    into the camera image via `LidarCameraProjector.filter_by_bbox()` and
    takes the median depth of points inside the bbox.
  - **Strategy B — Angle fallback:** matches by horizontal angle (±15°)
    AND distance agreement (within 40% relative error).
  - **Strategy C — Camera only:** no LiDAR support; keeps camera distance.
- Unmatched front-sector LiDAR clusters are appended as `LIDAR_ONLY`
  detections.
- **Action fusion is conservative:** the fused action is
  `max(camera_action, lidar_action)` by `ACTION_PRIORITY`
  (`drive < cautious < slow < stop < emergency_stop`).
- Camera distance and LiDAR distance are stored in **separate fields**
  (`distance` and `lidar_distance`) on every detection — they are never
  mixed, so per-sensor accuracy comparisons remain clean.

---

## 3. What `lidar_camera_projector.py` is doing

The LiDAR sees the world in **3D** (points with X, Y, Z in metres).
The camera sees the world in **2D** (pixels on an image).
This file is the **translator** between them.

### 3.1 The core problem it solves

The LiDAR and the camera sit in different places on the ego vehicle:

```
                ┌──── LiDAR ────┐    (x=2.0, z=1.8, no tilt)
                │   z=1.8 m     │
                │      ●        │
   roof ───────┤      ↑ 0.4 m   │
                │      ↓        │
                │   z=1.4 m  ●  │
                │       ↘ -15° │    (camera tilted down 15°)
                └──── Camera ───┘
```

So a 3D LiDAR point isn't directly comparable to a camera pixel — you have to:
1. Account for the **0.4 m height difference**.
2. Account for the **camera being tilted 15° downward**.
3. Then squash 3D → 2D using the pinhole camera model.

### 3.2 Two functions, two jobs

#### `project_points(lidar_xyz)` — lines 57–117
*"Turn 3D LiDAR points into pixel positions on the camera image."*

| Step | What it does | Line |
|------|--------------|------|
| 1 | Shift Z up by 0.4 m (camera sits below LiDAR) | 84 |
| 2 | Rotate by +15° to undo the camera's downward tilt | 90–92 |
| 3 | Throw away points **behind** the camera | 95 |
| 4 | Pinhole projection: `u = fx·(y/x) + cx`, `v = fy·(−z/x) + cy` | 103–104 |
| 5 | Throw away points that fall **outside the image** (1280×720) | 107–110 |

**Output:** for each surviving LiDAR point → `(u, v, depth in metres)`.

#### `filter_by_bbox(lidar_xyz, bbox)` — lines 123–163
*"How far away is the object inside this camera bounding box, according to LiDAR?"*

This is the function the fusion stage actually uses:

```
   Camera sees a car  →  YOLO draws a bbox around it  →  ?
                                                         │
                                                         ▼
              filter_by_bbox(lidar_points, bbox):
                  • project ALL LiDAR points into the image
                  • keep only the ones that fall INSIDE the bbox
                    (with a small 15% padding for safety)
                  • take the MEDIAN depth of those points
                  → that's the LiDAR distance to the car
```

The **median** is used (not mean) so a couple of stray points don't ruin the
estimate. If fewer than 2 LiDAR points land inside the bbox, the function
returns `None` — meaning "LiDAR can't say."

### 3.3 Why this matters for the experiment

This file is the **bridge** that lets us compare camera distance vs LiDAR
distance for the **same object**:

```
   YOLO bbox of car  ──► camera distance (from pinhole / stereo)
        │
        └──► filter_by_bbox() ──► LiDAR distance (median of points in bbox)

   Both numbers refer to the SAME car, so they can be compared
   directly against ground truth.
```

Without this projector, we would only have two unrelated lists of distances —
there would be no way to say "this camera reading and this LiDAR reading are
about the same car."

---

## 4. Data flow per frame (sequence)

```
   main.py                       driving_agent.py:380           core/*
   ───────────                   ──────────────────             ──────
   while running:
      cam_img = camera.get()
      agent.process_frame(cam_img)
            │
            ├──► lidar_sensor.get_latest()  ───────►  (N,4) raw_pts
            │
            ├──► lidar_processor.preprocess(raw_pts)
            │                              ───────►  (M,3) filtered_pts
            │
            ├──► lidar_obstacle_detector.detect(
            │       filtered_pts,
            │       vehicle_speed_kmh)     ───────►  List[LidarObstacle]
            │
            ├──► obstacle_detector.detect(cam_img)
            │                              ───────►  camera_detections,
            │                                        camera_action
            │
            ├──► lidar_fusion.fuse(
            │       camera_detections,
            │       lidar_obstacles,
            │       camera_action,
            │       lidar_raw_points=filtered_pts)
            │       │
            │       │  internally calls
            │       │  ↳ LidarCameraProjector.filter_by_bbox()
            │       │     for Strategy A
            │       │
            │       └─►  fused_detections,
            │            fused_action,
            │            nearest_obstacle
            │
            ├──► distance_metrics.update(
            │       cam_dist        = fusion.last_camera_dist,
            │       lidar_bbox_dist = fusion.last_lidar_dist,
            │       gt_dist         = ground_truth_dist)
            │
            └──► control: throttle / brake / steer
                          driven by obstacle_action
```

---

## 5. Key thresholds at a glance

| Stage | Param | Default | File |
|-------|-------|---------|------|
| Processor | `ground_z_threshold` | -1.4 m | `core/lidar_processor.py` |
| Processor | `min_range` | 0.5 m | `core/lidar_processor.py` |
| Processor | `max_range` | 50.0 m | `core/lidar_processor.py` |
| Processor | `ego_radius` | 1.5 m | `core/lidar_processor.py` |
| Detector | `eps` (DBSCAN) | 0.5 m | `core/lidar_obstacle_detector.py` |
| Detector | `min_samples` | 3 | `core/lidar_obstacle_detector.py` |
| Detector | `base_emergency_dist` | 3 m | `core/lidar_obstacle_detector.py` |
| Detector | `base_stop_dist` | 5 m | `core/lidar_obstacle_detector.py` |
| Detector | `base_slow_dist` | 10 m | `core/lidar_obstacle_detector.py` |
| Detector | `base_cautious_dist` | 15 m | `core/lidar_obstacle_detector.py` |
| Projector | `image_width × image_height` | 1280 × 720 | `core/lidar_camera_projector.py` |
| Projector | `fov_deg` | 90° | `core/lidar_camera_projector.py` |
| Projector | `cam_pitch_deg` | −15° | `core/lidar_camera_projector.py` |
| Projector | `lidar_z_above_cam` | 0.4 m | `core/lidar_camera_projector.py` |
| Fusion | `angle_match_threshold_deg` | 15° | `core/lidar_fusion.py` |
| Fusion (Strategy A) | bbox padding | ±15% | `core/lidar_fusion.py` |
| Fusion (Strategy A) | `min_points` per bbox | 2 | `core/lidar_fusion.py` |
