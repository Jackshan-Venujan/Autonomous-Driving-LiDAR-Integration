# Sensor Comparison: Camera vs LiDAR Distance Accuracy

This document explains a runtime behavior of the fusion engine that affects how
distance numbers should be interpreted when comparing camera-monocular distance
against LiDAR-projected distance. **Read this before drawing conclusions from
any aggregate distance metrics.**

---

## The "safe by default" camera fallback

The fusion engine prioritizes **driving safety over experimental purity**. When
a camera detection cannot be paired with a LiDAR cluster, the camera-derived
distance is *not* discarded — it is kept and used to drive HUD, danger
classification, and control. This is correct for an autonomous vehicle (a
degraded estimate beats a missing estimate when an object is in front of you),
but it means a naive read of certain logs will make camera look more available
or more accurate than it actually is.

A camera detection becomes "unmatched" when **both** of the following are true:

- LiDAR bbox projection finds **fewer than 3** raw points inside the camera
  bounding box ([core/lidar_fusion.py:160-162](../core/lidar_fusion.py#L160-L162)).
- No LiDAR cluster is within the **15° horizontal angle gate**
  ([core/lidar_fusion.py:34](../core/lidar_fusion.py#L34),
  [core/lidar_fusion.py:189-206](../core/lidar_fusion.py#L189-L206)).

When this happens:

| Where | What happens | Citation |
| --- | --- | --- |
| The detection dict | `lidar_distance = None`, `fusion_method = 'CAMERA_ONLY'`. The camera `distance` field stays. | [core/lidar_fusion.py:207-210](../core/lidar_fusion.py#L207-L210) |
| Nearest-obstacle / HUD / control | `eff_dist = lidar_distance or distance` — silently uses camera distance. | [core/lidar_fusion.py:278](../core/lidar_fusion.py#L278) |
| Frame-level aggregate log | `lidar_bbox_dist_m` is empty for that frame, but `cam_dist_m` is still written. | [core/distance_metrics.py:39-44](../core/distance_metrics.py#L39-L44), [core/driving_agent.py:415-421](../core/driving_agent.py#L415-L421) |
| Per-obstacle log | Row is tagged `detection_source = CAM_ONLY`, `match_method = CAMERA_ONLY`, `lidar_dist_m` is empty. **Clean.** | [core/distance_metrics.py:51-60](../core/distance_metrics.py#L51-L60), [core/distance_metrics.py:182-192](../core/distance_metrics.py#L182-L192) |

A second, related effect is on the LiDAR side: when a LiDAR cluster is **not**
matched to any camera detection, the engine emits a synthetic `LIDAR_ONLY`
detection that copies the LiDAR distance into both the camera `distance` field
and `lidar_distance` ([core/lidar_fusion.py:236-253](../core/lidar_fusion.py#L236-L253)).
In the per-obstacle log this appears as `cam_dist_m == lidar_dist_m` — the
camera value in those rows is **not** a real camera measurement.

---

## Which log to use for the experiment

| File | Has fallback bias? | Use for accuracy comparison? |
| --- | --- | --- |
| `metrics/obstacle_log_*.csv` | No — every row is stamped with `detection_source` and `match_method` | **Yes**, filter `detection_source == 'BOTH'` |
| `metrics/distance_log_*.csv` (frame aggregate) | Yes — camera distance written even on frames with no LiDAR match | No |
| `metrics/distance_summary_*.csv` | Derived from the frame aggregate | No |

A `BOTH` row is the only row where:

- a single physical obstacle was detected by the camera in this frame,
- LiDAR also confirmed it (either via FULL_BBOX with ≥3 projected points, or
  via FULL_ANGLE within 15°),
- `cam_dist_m` is a true camera-monocular estimate, and
- `lidar_dist_m` is a true LiDAR cluster distance,
- both measured **independently** of each other.

Rows with `detection_source ∈ {CAM_ONLY, LIDAR_ONLY}` represent disagreement
about whether an obstacle exists at all — they are not valid head-to-head
distance comparisons and should be excluded from accuracy metrics. (You can
still report them separately as detection-rate / coverage metrics.)

---

## How to analyze

```python
import pandas as pd

df = pd.read_csv("metrics/obstacle_log_<timestamp>.csv")

# Keep only frames where both sensors independently measured the same obstacle.
both = df[df["detection_source"] == "BOTH"].copy()

# Per-obstacle absolute error against ground truth.
both["cam_err_m"]   = (both["cam_dist_m"]   - both["gt_dist_m"]).abs()
both["lidar_err_m"] = (both["lidar_dist_m"] - both["gt_dist_m"]).abs()

print(both[["cam_err_m", "lidar_err_m"]].describe())
print("LiDAR wins:", (both["lidar_err_m"] < both["cam_err_m"]).mean())
```

For coverage/availability (a separate question from accuracy), look at the
`detection_source` value counts on the unfiltered dataframe.

---

## Why this is not "fixed" in code

Disabling the camera fallback would make the vehicle blind whenever the LiDAR
match fails (sparse returns at distance, occlusion, narrow objects), which is
unsafe. The right separation is: keep the fallback in the fusion/control path,
and use the `detection_source` flag in the per-obstacle log to filter cleanly
at analysis time.
