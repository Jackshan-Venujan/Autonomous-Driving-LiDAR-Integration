This is a sub branch from New-LiDAR-Obstacle-Detector Branch upon the lane detection accuracy reduced significantly.

Current Issues in this commit
1. LiDAR measured distance is not stable, DRIVE and SLOW changes frequently causing unstable decision making. I think giving a threshold value 
2. LiDAR detection only works on automatic mode. not in manual mode.
3. Traffic Light model is not accurate, even without the traffic light, it is detecting


**Plan: Fix LiDAR Obstacle Distance Instability**
Context
Distance readings for a static front obstacle are jumping between 6m and 20m every few frames. This swing crosses three threshold boundaries — stop (5m), slow (10m), cautious (15m) — causing the danger level and the vehicle's stop/drive decision to oscillate. The vehicle alternates between braking hard and resuming full speed even though nothing is moving.

Root Cause: LidarObstacleDetector has zero temporal memory. Every call to detect() runs DBSCAN fresh and reports the raw centroid distance. A 32-channel LiDAR hits a different subset of surface facets each scan rotation, so the cluster centroid drifts ±7m per frame on a static object. No filtering exists anywhere in the LiDAR pipeline.

Steering already has this fix (lateral_error_alpha=0.3 EMA in driving_agent.py:609-613). The LiDAR pipeline needs equivalent treatment.

How Temporal Memory Fixes This
Two complementary memory mechanisms are combined:

Memory Type 1 — Obstacle State Tracks (self._tracks)
A dictionary that remembers each tracked obstacle across frames — its smoothed distance, last known angle, and danger level hysteresis state. Updated each frame via EMA. This is lightweight (one dict entry per obstacle) and solves the action oscillation.

Memory Type 2 — Point Cloud Accumulation (_tracks[id]['point_history'])
A sliding window (deque) that stores the raw XYZ cluster points from the last N frames for each tracked obstacle. When computing the centroid, all accumulated points are stacked and averaged together rather than just the single-frame cluster. This gives 3–5× more points per obstacle → far more stable centroid geometry.

Why point cloud accumulation also enables future 3D bounding boxes: The accumulated points span Z (height) across multiple scan rings and frames. Taking min_z / max_z over the accumulated point window gives accurate obstacle height extents. These are stored in the LidarObstacle dataclass as bbox_min_z / bbox_max_z, providing the data needed to render true 3D bounding boxes in a future 3D LiDAR visualizer (e.g. Open3D) without any further changes to the detector.

Files to Modify
Primary file: core/lidar_obstacle_detector.py

LidarObstacle dataclass also gains two new fields (bbox_min_z, bbox_max_z) to carry 3D height data for future visualization. These are optional fields with default 0.0 so no call site changes are needed.

Implementation Plan
1. Extend imports (line 10)
from typing import List, Optional, Dict
from collections import deque
2. Add bbox_min_z / bbox_max_z to LidarObstacle dataclass (lines 34–37)
bbox_min_z: float = 0.0   # lowest Z across accumulated point history (obstacle bottom)
bbox_max_z: float = 0.0   # highest Z across accumulated point history (obstacle top)
These fields store the 3D height extent of the obstacle computed from accumulated point cloud history. Existing code that constructs LidarObstacle without these fields continues to work because they have defaults.

3. Add temporal state to __init__ (after line 72)
# Temporal filtering state
self._EMA_ALPHA: float = 0.3          # matches lateral_error_alpha in driving_agent
self._ANGLE_WINDOW: float = 10.0      # deg tolerance for cross-frame obstacle matching
self._DEESC_REQUIRED: int = 3         # frames at lower level before de-escalating (150ms at 20Hz)
self._MAX_STALE: int = 3              # frames before dropping an unmatched track
self._POINT_HISTORY_LEN: int = 5      # frames of raw cluster points to accumulate per track

# Track dict keyed by string ID. Each entry holds:
#   sector, angle_deg, smooth_dist, committed_action,
#   deesc_candidate, deesc_count, stale,
#   point_history (deque of np.ndarray — raw XYZ points from last N frames)
self._tracks: Dict[str, dict] = {}
self._next_track_id: int = 0

# Hysteresis state for the overall front action (no-obstacle → drive)
self._committed_front_action: str = 'drive'
self._no_front_obs_count: int = 0
4. Add four helper methods (after _classify_danger, before detect)
_make_track_id() — returns next unique track ID string.

_match_to_track(sector, angle_deg) — linear scan over _tracks; returns ID of closest track in same sector within _ANGLE_WINDOW degrees, else None. Uses angle only (not distance) because distance is the noisy variable being corrected.

_apply_asymmetric_ema(raw_dist, prev_smooth) — returns min(raw_dist, alpha*raw + (1-alpha)*prev_smooth). The min() means: if obstacle is closer than history thinks → accept immediately (safety-first). If farther away → damp with EMA. This gives zero-latency escalation and smoothed de-escalation in one operation.

_apply_action_hysteresis(track, new_action) — mutates the track dict, returns committed action:

If ACTION_PRIORITY[new_action] >= ACTION_PRIORITY[committed] → commit immediately (escalation).
Otherwise → increment deesc_count; commit only after _DEESC_REQUIRED consecutive frames at the same lower level.
5. Modify detect() (lines 105–165)
Before cluster loop — add seen_track_ids: set = set()

Inside cluster loop, replace lines 148–163 with:

# Match cluster to existing track (by sector + angle proximity)
tid = self._match_to_track(sector, angle_deg)
if tid is None:
    # New obstacle: create a fresh track with point history
    tid = self._make_track_id()
    self._tracks[tid] = {
        'sector': sector, 'angle_deg': angle_deg,
        'smooth_dist': dist,
        'committed_action': 'drive',
        'deesc_candidate': None, 'deesc_count': 0, 'stale': 0,
        'point_history': deque(maxlen=self._POINT_HISTORY_LEN),
    }
else:
    track = self._tracks[tid]
    track['angle_deg'] = angle_deg
    track['stale'] = 0

seen_track_ids.add(tid)
track = self._tracks[tid]

# ── Point cloud accumulation ──────────────────────────────────────────
# Append this frame's cluster points to the sliding window.
track['point_history'].append(cluster)

# Stack all accumulated frames into one array for stable centroid + extents.
# For a new track (1 frame) this equals cluster; for mature tracks (≥5 frames)
# this is 5x the number of points → much more stable centroid geometry.
accumulated = np.vstack(track['point_history'])

acc_cx = float(np.mean(accumulated[:, 0]))
acc_cy = float(np.mean(accumulated[:, 1]))
acc_cz = float(np.mean(accumulated[:, 2]))
acc_dist = math.sqrt(acc_cx ** 2 + acc_cy ** 2)

# ── EMA on accumulated centroid distance ──────────────────────────────
track['smooth_dist'] = self._apply_asymmetric_ema(acc_dist, track['smooth_dist'])
smoothed_dist = track['smooth_dist']

# 3D bounding box extents from the accumulated point cloud
acc_min_z = float(np.min(accumulated[:, 2]))
acc_max_z = float(np.max(accumulated[:, 2]))

danger_raw = self._classify_danger(smoothed_dist, sector, thresholds)
danger = self._apply_action_hysteresis(track, danger_raw)

self._last_obstacles.append(LidarObstacle(
    centroid_x=acc_cx, centroid_y=acc_cy, centroid_z=acc_cz,
    distance=smoothed_dist,          # EMA-smoothed accumulated centroid distance
    angle_deg=angle_deg,
    sector=sector,
    point_count=len(accumulated),    # total accumulated points (grows to N*cluster_size)
    danger_level=danger,             # hysteresis-committed level
    bbox_min_x=float(np.min(accumulated[:, 0])),
    bbox_max_x=float(np.max(accumulated[:, 0])),
    bbox_min_y=float(np.min(accumulated[:, 1])),
    bbox_max_y=float(np.max(accumulated[:, 1])),
    bbox_min_z=acc_min_z,            # 3D height extent — ready for future 3D viz
    bbox_max_z=acc_max_z,
))
After cluster loop, before return — evict stale tracks (their point_history is discarded automatically since they are deleted) and update committed front action:

# Evict tracks not seen this frame
for tid in list(self._tracks.keys()):
    if tid not in seen_track_ids:
        self._tracks[tid]['stale'] += 1
        if self._tracks[tid]['stale'] >= self._MAX_STALE:
            del self._tracks[tid]

# Update committed front action with hysteresis
front_obs = [o for o in self._last_obstacles if o.sector == 'front']
if front_obs:
    self._no_front_obs_count = 0
    self._committed_front_action = max(
        front_obs, key=lambda o: ACTION_PRIORITY[o.danger_level]
    ).danger_level
else:
    self._no_front_obs_count += 1
    if self._no_front_obs_count >= self._DEESC_REQUIRED:
        self._committed_front_action = 'drive'
6. Modify get_front_action() (lines 167–172)
Replace body with:

return self._committed_front_action
All other public methods (get_side_action, get_nearest_front, render_bev) are unchanged. The BEV display automatically shows smoothed distances since it reads self._last_obstacles.

Why These Parameter Values
Parameter	Value	Reason
_EMA_ALPHA	0.3	Matches existing lateral_error_alpha in driving_agent.py. Gives ~140ms time constant at 20Hz.
_ANGLE_WINDOW	10°	LiDAR centroid angle drift on a static obstacle is <5° per frame at 10m range. 10° gives a safe margin without cross-matching adjacent-lane obstacles.
_DEESC_REQUIRED	3	150ms at 20Hz — below human reaction time (250ms), filters single/double-frame misreads.
_MAX_STALE	3	150ms stale tolerance handles single-frame occlusion without creating ghost tracks.
_POINT_HISTORY_LEN	5	Accumulates 5 frames of point cloud per obstacle (250ms at 20Hz). Gives 3–5× more points per cluster for a stable centroid. Also wide enough to capture full Z extent of the obstacle across multiple scan rings.
Asymmetric EMA	min(raw, ema)	Zero-latency escalation for new close obstacles; damped de-escalation for apparent distance increases.
How Point Cloud Memory Enables Future 3D Bounding Boxes
Once bbox_min_z and bbox_max_z are stored in LidarObstacle, any 3D visualizer can draw accurate box edges around the obstacle using the 6 values: bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y, bbox_min_z, bbox_max_z. The current BEV renderer ignores Z (it is 2D), so no change to render_bev() is needed now. A future Open3D or PCL visualizer can read these fields directly from _last_obstacles and draw 3D wireframe boxes — no detector changes required at that point.

Verification
Quick unit test (no CARLA needed):

from core.lidar_obstacle_detector import LidarObstacleDetector
import numpy as np

det = LidarObstacleDetector()

def cluster_at(dist, n=20):
    pts = np.random.randn(n, 3) * 0.3
    pts[:, 0] += dist
    return pts.astype(np.float32)

# Simulate 6m/20m alternating (the reported bug)
for i, d in enumerate([6, 20, 6, 20, 6, 20, 6, 20]):
    obs = det.detect(cluster_at(d), 0.0)
    print(f"raw={d}m  smooth={obs[0].distance:.1f}m  action={det.get_front_action()}")

# Expected: action stays 'slow' or 'stop'; distance stays between 8-14m range
# NOT jumping between 'drive' and 'stop' every frame
In CARLA: Watch the BEV overlay and console output. The LIDAR=Xm(action) column should show a monotonically-changing distance value and a stable action label for a parked vehicle.

Regression check:

detect() signature unchanged — no call-site changes needed
get_front_action() still returns a string from ACTION_PRIORITY
render_bev() unchanged — now displays smoothed values automatically
lidar_fusion.py and driving_agent.py require no changes
