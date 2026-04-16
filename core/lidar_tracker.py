"""
Multi-Object LiDAR Tracker — Task 04
======================================

Receives a ClusterResult (List[Obstacle]) from LidarClusterer every 100 ms
and produces a temporally-stable list of TrackedObstacle objects with:

  - Persistent track IDs across frames (survives brief occlusions up to 0.8 s)
  - Kalman-smoothed 3D positions (noise-free distance measurements)
  - Per-obstacle 3D velocity vectors (metres/second, world frame)
  - Radial closing speed and time-to-collision (TTC)
  - EMA-smoothed safety-critical min_distance
  - Sector assignment for 360-degree situational awareness
  - Pre-computed severity level for HUD (Task 06)
  - Full Kalman state exported for NN training (Task 07)

Pipeline per frame
------------------
  1. Predict all active tracks forward by dt (Kalman predict step)
  2. Build cost matrix: predicted positions vs new centroids (Euclidean + type penalty)
  3. Hungarian matching (scipy.optimize.linear_sum_assignment)
  4. Update matched tracks (Kalman update + EMA smoothing)
  5. Create new tracks for unmatched detections
  6. Increment lost_frames for unmatched tracks
  7. Delete tracks exceeding max_lost_frames
  8. Build TrackedObstacle output list, sort by min_distance

Safety-critical design decisions
---------------------------------
  - Raw DBSCAN IDs change every frame: Kalman tracking assigns persistent IDs.
  - LiDAR min_distance jitters +-0.3 m per frame: EMA suppresses spurious brakes.
  - TTC uses smoothed min_distance, not centroid: always correct for collision geometry.
  - Velocity is world-frame (ego motion corrected): stationary objects read ~0 m/s.
  - Track confirmed only after min_hits_to_confirm frames: suppresses ghost tracks.

Coordinate frame (inherited from Tasks 01-03)
----------------------------------------------
  +X = forward    +Y = left    +Z = up
  Origin = LiDAR roof mount (2.4 m above ground)
  All distances: metres.  All speeds: m/s.  All angles: degrees.

Install dependencies
--------------------
  pip install filterpy scipy
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    # When imported as part of the `core` package
    from core.kalman_filter import ObjectKalmanFilter
    from core.lidar_clusterer import Obstacle, ClusterResult
except ImportError:
    # When run directly as a script: python core/lidar_tracker.py
    from kalman_filter import ObjectKalmanFilter
    from lidar_clusterer import Obstacle, ClusterResult

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][Tracker] %(message)s',
    datefmt='%H:%M:%S',
)


# ══════════════════════════════════════════════════════════════════════════════
#  TrackerConfig  —  all tunable parameters in one frozen, serialisable object
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrackerConfig:
    """
    All tunable tracker parameters with physical justification.

    Every value has an inline comment explaining its meaning and how to tune.
    Serialisable to/from JSON for recording-session metadata.
    """

    # ── Kalman filter — process noise Q matrix diagonal ──────────────────────

    q_pos: float = 0.08
    # Position process noise variance (m^2).
    # Models how much position deviates per step due to unmodelled dynamics.
    # Lower = smoother position but slower to react to sudden manoeuvres.
    # 0.08 m^2 -> std 0.28 m/step. Good for city traffic at 10 Hz.
    # Tune UP to 0.25 for aggressive drivers; DOWN to 0.02 for slow pedestrians.

    q_vel: float = 0.8
    # Velocity process noise variance ((m/s)^2).
    # Models unmodelled acceleration (braking, cornering).
    # Higher than q_pos because velocity changes faster than position.
    # 0.8 (m/s)^2 -> std 0.89 m/s/step.
    # A car braking at 5 m/s^2 changes velocity by 0.5 m/s per 10 Hz step.
    # Tune UP to 3.0 for highway merge scenarios.

    # ── Kalman filter — measurement noise R matrix diagonal ──────────────────

    r_pos_xy: float = 0.25
    # XY position measurement noise variance (m^2).
    # LiDAR centroid accuracy: +-0.3-0.5 m due to partial occlusion and
    # DBSCAN centroid estimation error.
    # 0.25 m^2 -> std 0.5 m. Pessimistic for near objects; good at 30-50 m.
    # DO NOT reduce below 0.05 — causes filter divergence on jittery clusters.

    r_pos_z: float = 0.15
    # Z position measurement noise variance (m^2).
    # Z centroid is more stable than XY (64-channel fixed vertical resolution).
    # 0.15 m^2 -> std 0.39 m.

    # ── Initial state covariance P matrix ────────────────────────────────────

    p_pos_init: float = 2.0
    # Initial position uncertainty (m^2) for new tracks.
    # std = sqrt(2.0) = 1.41 m. Converges to ~R after 3-5 updates.

    p_vel_init: float = 5.0
    # Initial velocity uncertainty ((m/s)^2) for new tracks.
    # std = sqrt(5.0) = 2.24 m/s. Covers initial velocity range 0-5 m/s.

    # ── Data association — Hungarian matching ────────────────────────────────

    max_assoc_dist: float = 3.5
    # Maximum 3D Euclidean distance (m) to allow a track-detection match.
    # At 10 Hz, 126 km/h (35 m/s) vehicle moves 3.5 m per frame.
    # 3.5 m covers all urban and suburban speeds with margin.
    # Reduce to 2.0 for parking lot use (slower objects, denser scenes).

    type_match_penalty: float = 2.0
    # Cost added if detection type != track type (e.g., vehicle vs pedestrian).
    # Soft penalty — discourages cross-type matches without blocking them.
    # Set to 0.0 to disable type-based cost adjustment.

    # ── Track lifecycle ───────────────────────────────────────────────────────

    min_hits_to_confirm: int = 2
    # Consecutive matched frames needed before a track is 'confirmed'.
    # Suppresses ghost tracks from 1-2 frame DBSCAN noise clusters.
    # 2 hits = 0.2 s delay. Objects at 80 m take >2 s to close at 140 km/h.
    # Reduce to 1 for maximum reactivity (more false tracks expected).

    max_lost_frames: int = 8
    # Consecutive frames without detection before track deletion.
    # At 10 Hz: 8 frames = 0.8 s. A vehicle occluded by a truck at city
    # speeds (~50 km/h) is typically hidden for 0.5-0.8 s.
    # Reduce to 3 for aggressive deletion; increase to 15 for highway use.

    # ── Distance smoothing ────────────────────────────────────────────────────

    dist_ema_alpha: float = 0.35
    # EMA alpha for min_distance smoothing.
    # Formula: smooth = alpha * new + (1-alpha) * prev
    # 0.35: responds to 0.5 m step change in ~4 frames (0.4 s).
    # CRITICAL: Laggy distance = delayed braking. Do not over-smooth.
    # Recommended range: 0.3-0.5 for safety-critical distance reporting.

    # ── TTC computation ───────────────────────────────────────────────────────

    ttc_min_closing_speed: float = 0.3
    # Minimum radial closing speed (m/s) to compute TTC.
    # Below this: object is stationary or receding — TTC = inf.
    # Prevents noise TTC values like 300 s from near-stationary objects.

    ttc_max_value: float = 60.0
    # TTC cap (seconds). Beyond 60 s is irrelevant for ADAS decisions.

    # ── Velocity reliability ──────────────────────────────────────────────────

    velocity_min_age: int = 3
    # Minimum track age before velocity is considered reliable.
    # Kalman initialises velocity from position changes; first 3 frames
    # have high P_vel uncertainty. Before this age: velocity_ms = 0, ttc = inf.

    # ── Performance limits ────────────────────────────────────────────────────

    max_tracks: int = 60
    # Maximum simultaneous tracks. Beyond this, stale low-confidence tracks
    # are pruned to prevent O(N^2) cost matrix blowup.

    max_proc_ms: float = 5.0
    # Hard budget for tracker.update() per frame (ms).
    # Hungarian for 60 tracks x 30 detections: ~0.5 ms.
    # Kalman predict+update for 60 tracks: ~1.5 ms.
    # Total budget leaves ~3 ms headroom for lifecycle + output assembly.
    # NOTE: Windows adds ~4-8 ms overhead vs Linux due to numpy allocator and
    # OS scheduler granularity. Expect ~8-12 ms mean on Windows; ~3-5 ms on Linux.
    # Raise to 15.0 when running on Windows in production to suppress false warnings.

    log_every_n_frames: int = 100
    # Print INFO-level summary every N frames.

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  TrackedObstacle  —  primary output contract for downstream tasks
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedObstacle:
    """
    A single tracked obstacle with temporally-stable ID, smoothed distance,
    velocity estimate, and time-to-collision.

    This is the PRIMARY OUTPUT CONTRACT consumed by:
      Task 05 — sensor fusion
      Task 06 — HUD display
      Task 07 — NN training data export

    Coordinate frame: ISO 8855 vehicle RH frame (inherited from Tasks 01-03).
    All distances: metres. All speeds: m/s. All angles: degrees.
    NumPy arrays: float32 unless noted (float64 for Kalman internals).
    """

    # ── Track identity ────────────────────────────────────────────────────────
    track_id:    int
    # Persistent ID across frames. Survives occlusions up to max_lost_frames.
    # Assigned on first detection; never reused. Use for temporal linking in NN.

    frame_id:    int
    # Frame in which this output was generated.

    timestamp:   float
    # Simulation timestamp (seconds since sim start) of the current frame.

    age_frames:  int
    # Total frames this track has existed (including lost frames).
    # Reliable velocity at age_frames >= velocity_min_age.

    lost_frames: int
    # Consecutive frames without a matching detection.
    # 0 = matched in current frame. >0 = coasting on Kalman prediction.

    is_confirmed: bool
    # True when age_frames >= min_hits_to_confirm.
    # Only confirmed tracks are used for HUD and safety decisions.

    # ── Classification ────────────────────────────────────────────────────────
    type:       str
    # Inherited from last matched Obstacle.type.
    # 'vehicle' | 'pedestrian' | 'cyclist' | 'structure' | 'unknown'

    confidence: float
    # EMA-smoothed Obstacle.confidence over track lifetime.
    # Prevents flickering on HUD display.

    # ── Kalman-smoothed geometry ──────────────────────────────────────────────
    center_xyz:  np.ndarray   # (3,) float32 — Kalman-filtered AABB centroid
    extent_lwh:  np.ndarray   # (3,) float32 — EMA-smoothed bounding box [L, W, H]
    heading_deg: float        # EMA-smoothed PCA heading [-180, 180]

    # ── Distance (safety-critical) ────────────────────────────────────────────
    min_distance: float
    # EMA-smoothed minimum surface distance (metres).
    # USE THIS FIELD FOR ALL SAFETY DECISIONS:
    #   - Collision warning thresholds
    #   - Emergency braking trigger
    #   - TTC computation denominator
    # DO NOT use centroid_distance for safety — it underestimates danger.

    centroid_distance: float
    # Distance to Kalman-smoothed centroid (metres). sqrt(cx^2 + cy^2).
    # Use for path planning and NN anchor regression.

    # ── Motion estimation ─────────────────────────────────────────────────────
    velocity_xyz: np.ndarray        # (3,) float32 — WORLD frame [vx, vy, vz] m/s
    # World velocity = Kalman ego-frame velocity + ego_velocity_xyz.
    # Stationary parked car reads ~[0, 0, 0] m/s.
    # Use for trajectory prediction NNs which expect world-frame labels.

    velocity_ego_frame: np.ndarray  # (3,) float32 — EGO frame [vx, vy, vz] m/s
    # Raw Kalman output. Positive vx = moving away from ego front.
    # Use for TTC computation and collision zone assessment.

    velocity_ms: float
    # Radial CLOSING speed (m/s).
    # Positive = approaching ego. Negative = receding.
    # Computed as: -dot(vel_ego, unit_vec_to_obstacle)

    speed_ms: float
    # Absolute scalar speed in world frame (m/s).
    # np.linalg.norm(velocity_xyz).
    # Static threshold: < 0.5 m/s. Moving vehicle: > 2.0 m/s.

    # ── Time-to-collision ─────────────────────────────────────────────────────
    ttc_seconds: float
    # TTC = min_distance / velocity_ms   (if closing speed > ttc_min_closing_speed)
    # TTC = float('inf')                 (if receding or near-stationary)
    # TTC = min(TTC, ttc_max_value)      (capped at 60 s)
    # PRIMARY TRIGGER for safety actions:
    #   ttc < 1.5 s  -> EMERGENCY BRAKE
    #   ttc < 3.0 s  -> HARD BRAKE
    #   ttc < 5.0 s  -> WARN + soft brake
    #   ttc >= 5.0 s -> SAFE

    # ── Bearing and sector ────────────────────────────────────────────────────
    bearing_deg: float
    # Azimuth to obstacle: 0=front, 90=left, -90=right, +-180=rear.

    sector: str
    # 45-degree sector name for 360-degree awareness.
    # One of: FRONT | FRONT_LEFT | LEFT | REAR_LEFT |
    #         REAR  | REAR_RIGHT | RIGHT | FRONT_RIGHT

    # ── Kalman internals for NN export ────────────────────────────────────────
    kalman_state: np.ndarray        # (6,) float64 [px,py,pz,vx,vy,vz]
    kalman_uncertainty: np.ndarray  # (6,) float64 — diagonal of P matrix

    # ── Pre-computed severity ─────────────────────────────────────────────────
    severity: str
    # 'BRAKE' | 'WARN' | 'SAFE'
    # Computed at update time so HUD (Task 06) doesn't re-implement logic.

    # ── Cluster points for Task 05 Camera-LiDAR fusion ────────────────────────
    cluster_points: Optional[np.ndarray] = None
    # (N, 3) float32 cluster points in vehicle frame from the last matched
    # Task 03 Obstacle. Carried here so Task 05 SensorFusion._project_obstacle()
    # can project onto the image plane without coupling to TrackState internals.
    # None for tracks with no matched detection in the current frame (coasting).

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dict for Task 07 export."""
        return {
            'track_id':           self.track_id,
            'frame_id':           self.frame_id,
            'timestamp':          round(self.timestamp, 4),
            'age_frames':         self.age_frames,
            'lost_frames':        self.lost_frames,
            'is_confirmed':       self.is_confirmed,
            'type':               self.type,
            'confidence':         round(float(self.confidence), 4),
            'center_xyz':         [round(float(v), 4) for v in self.center_xyz],
            'extent_lwh':         [round(float(v), 4) for v in self.extent_lwh],
            'heading_deg':        round(float(self.heading_deg), 2),
            'min_distance':       round(float(self.min_distance), 4),
            'centroid_distance':  round(float(self.centroid_distance), 4),
            'velocity_xyz':       [round(float(v), 4) for v in self.velocity_xyz],
            'velocity_ego_frame': [round(float(v), 4) for v in self.velocity_ego_frame],
            'velocity_ms':        round(float(self.velocity_ms), 4),
            'speed_ms':           round(float(self.speed_ms), 4),
            'ttc_seconds':        (round(float(self.ttc_seconds), 3)
                                   if self.ttc_seconds != float('inf') else None),
            'bearing_deg':        round(float(self.bearing_deg), 2),
            'sector':             self.sector,
            'severity':           self.severity,
            'kalman_state':       [round(float(v), 6) for v in self.kalman_state],
            'kalman_uncertainty': [round(float(v), 6) for v in self.kalman_uncertainty],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  TrackState  —  internal mutable state for one tracked obstacle
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackState:
    """
    Internal mutable state for one tracked obstacle.

    Managed exclusively by MultiObjectTracker. NOT exported to downstream.
    One instance per tracked obstacle; holds the Kalman filter and all
    running statistics updated every frame.
    """
    track_id:       int
    kf:             ObjectKalmanFilter   # Kalman filter instance
    type:           str                  # last detected object type

    age_frames:     int   = 0    # total frames this track has existed
    lost_frames:    int   = 0    # consecutive frames without detection
    hits:           int   = 0    # consecutive matched frames

    confidence_ema: float = 0.0  # EMA-smoothed Obstacle.confidence
    min_dist_ema:   float = 0.0  # EMA-smoothed min_distance (safety-critical)
    heading_ema:    float = 0.0  # EMA-smoothed heading_deg

    extent_ema: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )   # EMA-smoothed extent_lwh (stabilises bounding box jitter)

    last_detection: Optional[Obstacle] = None  # most recent matched Obstacle


# ══════════════════════════════════════════════════════════════════════════════
#  MultiObjectTracker  —  main tracker class
# ══════════════════════════════════════════════════════════════════════════════

class MultiObjectTracker:
    """
    Multi-object tracker using 6D Kalman filters + Hungarian data association.

    Receives a ClusterResult every frame and maintains a set of persistent
    tracks, each with a Kalman filter for smooth 3D position and velocity.

    Core algorithm (executed in update()):
        1.  Predict all tracks forward by dt (dead-reckoning)
        2.  Build cost matrix: predicted_pos vs detection_centroid + type penalty
        3.  Hungarian matching via scipy.optimize.linear_sum_assignment
        4.  Update matched tracks (Kalman + EMA)
        5.  Create new tracks for unmatched detections
        6.  Increment lost_frames for unmatched tracks
        7.  Delete tracks exceeding max_lost_frames
        8.  Prune to max_tracks if necessary
        9.  Build and sort TrackedObstacle output list

    Thread safety:
        update() is NOT thread-safe. Call from a single processing thread.
        Use an external threading.Lock if sharing between threads.

    Usage:
        tracker = MultiObjectTracker(TrackerConfig())
        for cluster_result in pipeline:
            tracked = tracker.update(cluster_result, ego_velocity_xyz)
            for t in tracked:
                safety_system.process(t)
    """

    # Ordered clockwise from FRONT (bearing 0 deg).
    # Each sector spans 45 degrees, centred on its cardinal/intercardinal angle.
    SECTOR_NAMES: List[str] = [
        'FRONT',       #   0 deg centroid
        'FRONT_LEFT',  #  45 deg
        'LEFT',        #  90 deg
        'REAR_LEFT',   # 135 deg
        'REAR',        # 180 deg
        'REAR_RIGHT',  # -135 deg (225)
        'RIGHT',       # -90 deg (270)
        'FRONT_RIGHT', # -45 deg (315)
    ]

    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self._cfg         = config
        self._tracks:     Dict[int, TrackState] = {}
        self._next_id:    int   = 0
        self._frame_count: int  = 0
        self._last_ts:    float = -1.0

        logger.info(
            "MultiObjectTracker initialised | "
            "max_assoc=%.1fm | max_lost=%d | min_hits=%d | ema_alpha=%.2f",
            config.max_assoc_dist,
            config.max_lost_frames,
            config.min_hits_to_confirm,
            config.dist_ema_alpha,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        result:            ClusterResult,
        ego_velocity_xyz:  np.ndarray,   # (3,) float64, world frame m/s
    ) -> List[TrackedObstacle]:
        """
        Process one frame of obstacle detections and return tracked obstacles.

        Args:
            result:           ClusterResult from LidarClusterer.cluster().
            ego_velocity_xyz: Ego vehicle world-frame velocity (3,) float64.
                              Obtain from: np.array([v.x, v.y, v.z])
                              where v = vehicle.get_velocity() in CARLA.
                              Used to convert Kalman ego-relative velocity to
                              world-frame velocity for stationary-object correction.

        Returns:
            List[TrackedObstacle] sorted by min_distance ascending.
            Includes both confirmed and tentative tracks (check is_confirmed).
            Lost tracks (lost_frames > 0) are included until max_lost_frames.

        Performance contract:
            Must complete in < config.max_proc_ms (default 5 ms).
        """
        cfg        = self._cfg
        t_start    = time.perf_counter()
        detections = result.obstacles
        timestamp  = result.timestamp
        frame_id   = result.frame_id

        # ── Step 0: Compute dt ────────────────────────────────────────────────
        # dt varies because CARLA can drop simulation ticks.
        # Guard against first frame (last_ts not set) and implausible values.
        if self._last_ts < 0.0:
            dt = 0.1   # first frame: assume 10 Hz nominal rate
        else:
            dt = timestamp - self._last_ts
            dt = max(0.01, min(1.0, dt))   # clamp: 10 ms to 1 s
        self._last_ts = timestamp

        # ── Step 1: Predict all tracks ────────────────────────────────────────
        # Run predict() for EVERY track, matched or not.
        # Unmatched tracks coast on dead-reckoning (Kalman prediction only).
        for track in self._tracks.values():
            track.kf.predict(dt)
            track.age_frames += 1

        # ── Step 2: Build cost matrix ─────────────────────────────────────────
        track_ids        = list(self._tracks.keys())
        n_tracks         = len(track_ids)
        n_dets           = len(detections)
        matched_track_ids = set()
        matched_det_ids   = set()

        if n_tracks > 0 and n_dets > 0:
            cost = self._build_cost_matrix(track_ids, detections, cfg)

            # ── Step 3: Hungarian matching ────────────────────────────────────
            # linear_sum_assignment minimises total cost over all row-column pairs.
            # Runs in O(N^3) but for N <= 60 this is < 0.5 ms.
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                # Reject pairs whose cost exceeds the hard distance gate.
                # This is NOT a duplicate of the cost_matrix guard — we must
                # reject here too because Hungarian may assign high-cost pairs
                # when the number of tracks exceeds the number of detections.
                if cost[r, c] > cfg.max_assoc_dist + cfg.type_match_penalty + 1.0:
                    continue

                tid = track_ids[r]
                matched_track_ids.add(tid)
                matched_det_ids.add(c)

                # ── Step 4: Update matched tracks ─────────────────────────────
                self._update_track(self._tracks[tid], detections[c], cfg)

        # ── Step 5: Create new tracks for unmatched detections ────────────────
        for i, det in enumerate(detections):
            if i not in matched_det_ids:
                self._create_track(det)

        # ── Step 6: Increment lost_frames for unmatched tracks ────────────────
        for tid in track_ids:
            if tid not in matched_track_ids:
                self._tracks[tid].lost_frames += 1

        # ── Step 7: Delete stale tracks ───────────────────────────────────────
        stale = [
            tid for tid, t in self._tracks.items()
            if t.lost_frames > cfg.max_lost_frames
        ]
        for tid in stale:
            logger.debug(
                "Track %d deleted after %d lost frames.", tid, cfg.max_lost_frames
            )
            del self._tracks[tid]

        # ── Step 8: Prune to max_tracks ───────────────────────────────────────
        if len(self._tracks) > cfg.max_tracks:
            self._prune_tracks(cfg)

        # ── Step 9: Build output ──────────────────────────────────────────────
        output: List[TrackedObstacle] = []
        for track in self._tracks.values():
            tracked_obs = self._build_output(
                track, frame_id, timestamp, ego_velocity_xyz, cfg
            )
            output.append(tracked_obs)

        output.sort(key=lambda t: t.min_distance)

        # ── Timing check ──────────────────────────────────────────────────────
        total_ms = (time.perf_counter() - t_start) * 1_000.0
        if total_ms > cfg.max_proc_ms:
            logger.warning(
                "Frame %d: tracker %.1f ms exceeds budget %.0f ms.",
                frame_id, total_ms, cfg.max_proc_ms,
            )

        self._frame_count += 1
        if self._frame_count % cfg.log_every_n_frames == 0:
            n_confirmed = sum(1 for t in output if t.is_confirmed)
            logger.info(
                "[Frame %d] tracks=%d confirmed=%d dets=%d dt=%.1fms lat=%.1fms",
                frame_id, len(self._tracks), n_confirmed,
                n_dets, dt * 1000.0, total_ms,
            )

        return output

    @property
    def track_count(self) -> int:
        """Total number of active tracks (confirmed + tentative)."""
        return len(self._tracks)

    @property
    def confirmed_count(self) -> int:
        """Number of confirmed tracks (hits >= min_hits_to_confirm)."""
        return sum(
            1 for t in self._tracks.values()
            if t.hits >= self._cfg.min_hits_to_confirm
        )

    @property
    def active_track_ids(self) -> List[int]:
        """List of all active track IDs."""
        return list(self._tracks.keys())

    def reset(self) -> None:
        """Clear all tracks and reset state (e.g. after a scene change)."""
        self._tracks.clear()
        self._next_id    = 0
        self._frame_count = 0
        self._last_ts    = -1.0
        logger.info("MultiObjectTracker reset.")

    # ── Private: Cost matrix ───────────────────────────────────────────────────

    def _build_cost_matrix(
        self,
        track_ids:  List[int],
        detections: List[Obstacle],
        cfg:        TrackerConfig,
    ) -> np.ndarray:
        """
        Build (n_tracks x n_detections) cost matrix for Hungarian matching.

        Cost = 3D Euclidean distance(predicted_position, detection_centroid)
               + type_match_penalty if types differ.

        Pairs where distance > max_assoc_dist are assigned a large sentinel
        cost (max_assoc_dist + type_match_penalty + 100) to effectively block
        the match while keeping the matrix valid for Hungarian.

        WHY EUCLIDEAN NOT MAHALANOBIS:
            Mahalanobis uses the Kalman P matrix to weight by uncertainty —
            theoretically better. In practice, for N=60 tracks x 30 detections,
            it requires 1800 3x3 matrix inversions per frame (~2 ms overhead).
            Euclidean achieves equivalent matching quality when Q and R are
            well-tuned, which they are for our typical urban LiDAR scene.

        Args:
            track_ids:  list of active track IDs (rows)
            detections: list of Obstacle from Task 03 (columns)
            cfg:        TrackerConfig

        Returns:
            cost: np.ndarray shape (n_tracks, n_detections), float64
        """
        BLOCK_COST = cfg.max_assoc_dist + cfg.type_match_penalty + 100.0
        cost = np.full((len(track_ids), len(detections)), BLOCK_COST, dtype=np.float64)

        for i, tid in enumerate(track_ids):
            track    = self._tracks[tid]
            pred_pos = track.kf.position   # (3,) float64 — predicted centroid

            for j, det in enumerate(detections):
                det_pos = det.center_xyz.astype(np.float64)
                dist    = float(np.linalg.norm(pred_pos - det_pos))

                if dist > cfg.max_assoc_dist:
                    # Hard gate: do not match this pair.
                    continue

                c = dist

                # Soft type penalty: discourages vehicle<->pedestrian swaps
                # but does not fully block them (in case type flickers).
                if track.type != det.type and det.type != 'unknown':
                    c += cfg.type_match_penalty

                cost[i, j] = c

        return cost

    # ── Private: Track update ──────────────────────────────────────────────────

    def _update_track(
        self,
        track: TrackState,
        det:   Obstacle,
        cfg:   TrackerConfig,
    ) -> None:
        """
        Update a matched track with a new detection.

        Runs Kalman measurement update, then applies EMA smoothing to:
          - min_distance  (safety-critical; NOT Kalman-filtered)
          - bounding box extent (stabilises noisy cluster size)
          - heading (PCA heading fluctuates; EMA damps it)
          - confidence (prevents HUD flickering)
          - object type (accepts new type if not 'unknown')

        min_distance is NOT Kalman-filtered because the Kalman filter tracks
        the CENTROID, not the nearest surface point. Nearest-point distance
        requires a separate EMA filter.

        Args:
            track: TrackState to update in-place.
            det:   Matching Obstacle from Task 03.
            cfg:   TrackerConfig.
        """
        # Kalman measurement update with AABB centroid
        track.kf.update(det.center_xyz.astype(np.float64))

        # Reset lost counter; increment hit counter
        track.lost_frames = 0
        track.hits       += 1

        # Update type if new detection has a concrete type
        if det.type != 'unknown':
            track.type = det.type

        alpha = cfg.dist_ema_alpha

        # EMA: min_distance — safety-critical smoothing
        # Initialise on first detection to avoid alpha-blend from 0.
        if track.min_dist_ema == 0.0:
            track.min_dist_ema = det.min_distance
        else:
            track.min_dist_ema = (alpha * det.min_distance
                                  + (1.0 - alpha) * track.min_dist_ema)

        # EMA: bounding box extent
        if track.extent_ema.sum() == 0.0:
            track.extent_ema = det.extent_lwh.copy()
        else:
            track.extent_ema = (alpha * det.extent_lwh
                                + (1.0 - alpha) * track.extent_ema)

        # EMA: heading (linear with wraparound handling)
        # Linear EMA is valid for heading because typical heading change is
        # << 10 deg/frame. Wraparound handled explicitly at +-180 deg boundary.
        if track.age_frames <= 1:
            track.heading_ema = det.heading_deg
        else:
            diff = det.heading_deg - track.heading_ema
            if diff >  180.0: diff -= 360.0
            if diff < -180.0: diff += 360.0
            track.heading_ema += alpha * diff
            # Normalise result to [-180, 180]
            track.heading_ema = ((track.heading_ema + 180.0) % 360.0) - 180.0

        # EMA: confidence
        if track.confidence_ema == 0.0:
            track.confidence_ema = det.confidence
        else:
            track.confidence_ema = (alpha * det.confidence
                                    + (1.0 - alpha) * track.confidence_ema)

        track.last_detection = det

    # ── Private: New track creation ────────────────────────────────────────────

    def _create_track(self, det: Obstacle) -> None:
        """
        Create a new TrackState for a first-seen detection.

        Args:
            det: Obstacle that did not match any active track.
        """
        cfg = self._cfg
        tid = self._next_id
        self._next_id += 1

        kf = ObjectKalmanFilter(
            initial_position=det.center_xyz.astype(np.float64),
            config=cfg,
        )

        track = TrackState(
            track_id      = tid,
            kf            = kf,
            type          = det.type,
            age_frames    = 1,
            lost_frames   = 0,
            hits          = 1,
            confidence_ema= det.confidence,
            min_dist_ema  = det.min_distance,
            extent_ema    = det.extent_lwh.copy(),
            heading_ema   = det.heading_deg,
            last_detection= det,
        )
        self._tracks[tid] = track

        logger.debug(
            "New track %d: %s at %.1fm bearing=%.0fdeg",
            tid, det.type, det.min_distance, det.bearing_deg,
        )

    # ── Private: Prune excess tracks ──────────────────────────────────────────

    def _prune_tracks(self, cfg: TrackerConfig) -> None:
        """
        Remove the lowest-priority tracks when total count exceeds max_tracks.

        Priority order (deleted first):
          1. Most lost frames (long-absent tracks)
          2. Lowest confidence EMA (uncertain tracks)

        Keeps confirmed, recently-active, high-confidence tracks.
        """
        n_excess = len(self._tracks) - cfg.max_tracks
        if n_excess <= 0:
            return

        # Sort by: lost_frames DESC (most stale first), confidence ASC (least confident first)
        sorted_tracks = sorted(
            self._tracks.items(),
            key=lambda kv: (-kv[1].lost_frames, kv[1].confidence_ema),
        )
        to_delete = [tid for tid, _ in sorted_tracks[:n_excess]]
        for tid in to_delete:
            del self._tracks[tid]
            logger.debug("Track %d pruned (max_tracks=%d reached).", tid, cfg.max_tracks)

    # ── Private: Output construction ───────────────────────────────────────────

    def _build_output(
        self,
        track:            TrackState,
        frame_id:         int,
        timestamp:        float,
        ego_velocity_xyz: np.ndarray,   # (3,) float64, world frame
        cfg:              TrackerConfig,
    ) -> TrackedObstacle:
        """
        Construct a TrackedObstacle from a TrackState.

        Computes:
          - World-frame velocity (Kalman ego-velocity + ego_velocity_xyz)
          - Radial closing speed (dot product of ego-frame velocity and unit_vec)
          - TTC (min_distance / closing_speed, capped)
          - Sector and bearing from Kalman-smoothed position
          - Severity level (BRAKE / WARN / SAFE)
          - Kalman state and uncertainty diagonal for NN export

        Args:
            track:            TrackState with latest Kalman state.
            frame_id:         Current CARLA frame ID.
            timestamp:        Current simulation timestamp (s).
            ego_velocity_xyz: Ego world velocity (3,) float64.
            cfg:              TrackerConfig.

        Returns:
            TrackedObstacle ready for Task 05/06/07.
        """
        # ── Kalman-smoothed position (ego frame) ──────────────────────────────
        pos_f64    = track.kf.position          # (3,) float64
        center_xyz = pos_f64.astype(np.float32)

        # ── Velocity in ego frame ─────────────────────────────────────────────
        # Kalman estimates velocity RELATIVE to the ego sensor.
        # First velocity_min_age frames: Kalman is still initialising from
        # position differences — estimate is noisy. We zero it out below.
        vel_ego_f64 = track.kf.velocity         # (3,) float64, ego-relative

        # ── World-frame velocity ──────────────────────────────────────────────
        # world_velocity = ego_relative_velocity + ego_world_velocity
        # Physical meaning: if ego drives at [10, 0, 0] m/s and Kalman reports
        # obstacle moving at [-10, 0, 0] in ego frame, it's stationary in world.
        vel_world_f64 = vel_ego_f64 + ego_velocity_xyz.astype(np.float64)

        # Zero velocity for young tracks — unreliable before velocity_min_age frames
        if track.age_frames < cfg.velocity_min_age:
            vel_ego_f64   = np.zeros(3, dtype=np.float64)
            vel_world_f64 = np.zeros(3, dtype=np.float64)

        velocity_ego_frame = vel_ego_f64.astype(np.float32)
        velocity_xyz       = vel_world_f64.astype(np.float32)
        speed_ms           = float(np.linalg.norm(vel_world_f64))

        # ── Radial closing speed ──────────────────────────────────────────────
        # Unit vector from ego origin to obstacle centroid (XY plane only).
        # Z component excluded because closing speed is measured in the
        # horizontal plane (relevant for braking, not vertical separation).
        dist_2d = float(np.hypot(pos_f64[0], pos_f64[1]))
        if dist_2d < 0.1:
            # Degenerate case: obstacle is essentially at sensor origin.
            unit_vec = np.array([1.0, 0.0, 0.0])
        else:
            unit_vec = np.array([
                pos_f64[0] / dist_2d,
                pos_f64[1] / dist_2d,
                0.0,
            ])

        # Closing speed = -dot(vel_ego, unit_vec)
        # Sign convention: unit_vec points AWAY from ego (outward).
        # If obstacle moves toward ego, its velocity opposes unit_vec.
        # dot product < 0 -> negate to get positive closing speed.
        velocity_ms = float(-np.dot(vel_ego_f64, unit_vec))

        # ── Time-to-collision ─────────────────────────────────────────────────
        smooth_dist = track.min_dist_ema   # EMA-smoothed safety distance

        if (track.age_frames >= cfg.velocity_min_age
                and velocity_ms > cfg.ttc_min_closing_speed):
            raw_ttc = smooth_dist / velocity_ms
            ttc     = min(raw_ttc, cfg.ttc_max_value)
        else:
            ttc = float('inf')

        # ── Geometry ──────────────────────────────────────────────────────────
        centroid_distance = float(np.hypot(pos_f64[0], pos_f64[1]))
        bearing_deg       = float(np.degrees(np.arctan2(pos_f64[1], pos_f64[0])))
        sector            = self._bearing_to_sector(bearing_deg)

        # ── Severity ──────────────────────────────────────────────────────────
        severity = self._compute_severity(smooth_dist, ttc)

        # ── Confirmation ──────────────────────────────────────────────────────
        is_confirmed = track.hits >= cfg.min_hits_to_confirm

        # ── Kalman internals for NN ────────────────────────────────────────────
        kalman_state       = track.kf.get_state_vector()
        kalman_uncertainty = track.kf.get_covariance_diagonal()

        return TrackedObstacle(
            track_id           = track.track_id,
            frame_id           = frame_id,
            timestamp          = timestamp,
            age_frames         = track.age_frames,
            lost_frames        = track.lost_frames,
            is_confirmed       = is_confirmed,
            type               = track.type,
            confidence         = float(track.confidence_ema),
            center_xyz         = center_xyz,
            extent_lwh         = track.extent_ema.copy(),
            heading_deg        = float(track.heading_ema),
            min_distance       = float(smooth_dist),
            centroid_distance  = centroid_distance,
            velocity_xyz       = velocity_xyz,
            velocity_ego_frame = velocity_ego_frame,
            velocity_ms        = velocity_ms,
            speed_ms           = speed_ms,
            ttc_seconds        = ttc,
            bearing_deg        = bearing_deg,
            sector             = sector,
            severity           = severity,
            kalman_state       = kalman_state,
            kalman_uncertainty = kalman_uncertainty,
            cluster_points     = (track.last_detection.points
                                  if track.last_detection is not None else None),
        )

    # ── Private: Bearing -> sector ─────────────────────────────────────────────

    @staticmethod
    def _bearing_to_sector(bearing_deg: float) -> str:
        """
        Map a bearing angle (degrees) to a named 45-degree sector.

        Bearing convention (from TrackedObstacle.bearing_deg):
            0 deg   = FRONT    (straight ahead)
            90 deg  = LEFT
           -90 deg  = RIGHT
          +-180 deg = REAR

        Sectors are 45 deg wide, centred on cardinal/intercardinal angles.
        Formula: round(bearing / 45) % 8 maps to the SECTOR_NAMES index.

        Args:
            bearing_deg: Azimuth angle in degrees [-180, 180].

        Returns:
            One of: FRONT, FRONT_LEFT, LEFT, REAR_LEFT,
                    REAR, REAR_RIGHT, RIGHT, FRONT_RIGHT.
        """
        # round() to nearest 45-degree step, then take modulo 8 for circular wrap.
        # Examples:
        #   0   -> round(0/45)=0   % 8 = 0 -> FRONT
        #   45  -> round(1)=1      % 8 = 1 -> FRONT_LEFT
        #   90  -> round(2)=2      % 8 = 2 -> LEFT
        #   180 -> round(4)=4      % 8 = 4 -> REAR
        #  -135 -> round(-3)=-3    % 8 = 5 -> REAR_RIGHT
        #  -90  -> round(-2)=-2    % 8 = 6 -> RIGHT
        #  -45  -> round(-1)=-1    % 8 = 7 -> FRONT_RIGHT
        sector_names = [
            'FRONT', 'FRONT_LEFT', 'LEFT', 'REAR_LEFT',
            'REAR',  'REAR_RIGHT', 'RIGHT', 'FRONT_RIGHT',
        ]
        idx = int(round(bearing_deg / 45.0)) % 8
        return sector_names[idx]

    # ── Private: Severity ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_severity(dist: float, ttc: float) -> str:
        """
        Compute pre-computed severity level based on distance and TTC.

        Thresholds:
          BRAKE: dist < 6 m  OR  ttc < 3.0 s   (imminent collision)
          WARN:  dist < 15 m OR  ttc < 5.0 s   (caution zone)
          SAFE:  otherwise                      (no action required)

        The distance thresholds provide a fallback when TTC is infinity
        (stationary objects: TTC = inf but they still block the path at close range).

        Physical justification:
          - 6 m at city speed (50 km/h = 13.9 m/s): ~0.43 s to contact -> BRAKE
          - 15 m at 50 km/h: ~1.1 s -> WARN
          - TTC < 3.0 s: < 3 typical reaction+brake time -> BRAKE
          - TTC < 5.0 s: approaching but still time to warn -> WARN

        Args:
            dist: EMA-smoothed min_distance (metres).
            ttc:  Computed TTC (seconds). float('inf') if not approaching.

        Returns:
            'BRAKE' | 'WARN' | 'SAFE'
        """
        if dist < 6.0 or (ttc != float('inf') and ttc < 3.0):
            return 'BRAKE'
        if dist < 15.0 or (ttc != float('inf') and ttc < 5.0):
            return 'WARN'
        return 'SAFE'


# ══════════════════════════════════════════════════════════════════════════════
#  CARLA integration demo
#  Run with:  python core/lidar_tracker.py --demo
# ══════════════════════════════════════════════════════════════════════════════

def _run_carla_demo() -> None:
    """
    Full pipeline demo: LiDAR -> preprocess -> cluster -> track.

    Spawns a Tesla Model3 in Town03, drives for 15 seconds, and prints a
    live tracking table showing track ID, type, smoothed distance, velocity,
    TTC, and severity for the 5 nearest confirmed tracks.

    Requires a running CARLA server on localhost:2000.
    """
    import carla

    try:
        from core.lidar_sensor       import LidarSensor
        from core.lidar_preprocessor import LidarPreprocessor, PreprocessConfig
        from core.lidar_clusterer    import LidarClusterer, ClusterConfig
    except ImportError:
        from lidar_sensor       import LidarSensor
        from lidar_preprocessor import LidarPreprocessor, PreprocessConfig
        from lidar_clusterer    import LidarClusterer, ClusterConfig

    CARLA_HOST  = "localhost"
    CARLA_PORT  = 2000
    RUN_SECONDS = 15

    print(f"\nConnecting to CARLA {CARLA_HOST}:{CARLA_PORT} ...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.load_world("Town03")

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    bp_lib       = world.get_blueprint_library()
    ego_bp       = bp_lib.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    ego_vehicle  = world.spawn_actor(ego_bp, spawn_points[0])
    ego_vehicle.set_autopilot(True)

    lidar    = LidarSensor(world, ego_vehicle)
    prep     = LidarPreprocessor(PreprocessConfig(), enforce_budget=False)
    clusterer = LidarClusterer(ClusterConfig())
    tracker  = MultiObjectTracker(TrackerConfig())
    lidar.start()

    print(f"Running tracker demo for {RUN_SECONDS} s ...")
    print(f"{'TrkID':>5} {'Type':>10} {'Dist(m)':>8} {'Vel(m/s)':>9} "
          f"{'TTC(s)':>7} {'Sector':>12} {'Sev':>6}")
    print("-" * 60)

    start       = time.time()
    frame_count = 0

    try:
        while time.time() - start < RUN_SECONDS:
            world.tick()
            raw = lidar.get_latest()
            if raw is None:
                continue

            cloud    = prep.process(raw)
            result   = clusterer.cluster(cloud)

            # Ego velocity from CARLA (world frame)
            v           = ego_vehicle.get_velocity()
            ego_vel_xyz = np.array([v.x, v.y, v.z], dtype=np.float64)

            tracked = tracker.update(result, ego_vel_xyz)
            frame_count += 1

            # Print top-5 confirmed tracks
            confirmed = [t for t in tracked if t.is_confirmed][:5]
            for t in confirmed:
                ttc_str = f"{t.ttc_seconds:.1f}" if t.ttc_seconds != float('inf') else "inf"
                print(
                    f"{t.track_id:>5} {t.type:>10} {t.min_distance:>8.1f} "
                    f"{t.velocity_ms:>+9.2f} {ttc_str:>7} {t.sector:>12} {t.severity:>6}"
                )
            if confirmed:
                print()

    finally:
        lidar.destroy()
        ego_vehicle.destroy()
        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        print(f"\nDemo complete: {frame_count} frames | "
              f"{tracker.track_count} active tracks | "
              f"{tracker.confirmed_count} confirmed.")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    if '--demo' in sys.argv:
        _run_carla_demo()
    else:
        print("Usage: python core/lidar_tracker.py --demo")
        print("       --demo  : Run full CARLA integration demo (requires server)")
        sys.exit(0)
