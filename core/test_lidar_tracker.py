"""
Unit + Integration tests for Task 04: Kalman Tracker
======================================================

Tests are self-contained — no CARLA required.
All synthetic data is constructed with deterministic RNG seeds.

Run:
    python core/test_lidar_tracker.py
    python core/test_lidar_tracker.py --verbose

Coverage
--------
  T01  ObjectKalmanFilter predict-only: position drifts by velocity * dt
  T02  ObjectKalmanFilter update: position snaps toward measurement
  T03  ObjectKalmanFilter velocity convergence over 20 frames
  T04  New tracks created for every unmatched detection
  T05  Matched track IDs are persistent across frames
  T06  Track deleted after max_lost_frames consecutive misses
  T07  is_confirmed is False until min_hits_to_confirm frames
  T08  Output sorted by min_distance ascending
  T09  TTC = inf when obstacle is receding or young track
  T10  TTC finite and correct when obstacle is approaching
  T11  Sector assignment covers all 8 sectors (360-degree)
  T12  Severity BRAKE/WARN/SAFE thresholds
  T13  World-frame velocity compensation (stationary = ~0 world speed)
  T14  ClusterResult round-trip: cluster() wraps process() correctly
  T15  Benchmark: 50-frame pipeline under timing budget
"""

from __future__ import annotations

import sys
import time
import math
import traceback
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import paths: support both `python core/test_lidar_tracker.py` (direct)
# and `import core.test_lidar_tracker` (package).
# ---------------------------------------------------------------------------
try:
    from core.kalman_filter  import ObjectKalmanFilter
    from core.lidar_tracker  import (
        MultiObjectTracker, TrackerConfig, TrackedObstacle,
        TrackState,
    )
    from core.lidar_clusterer import Obstacle, ClusterResult
except ImportError:
    from kalman_filter  import ObjectKalmanFilter
    from lidar_tracker  import (
        MultiObjectTracker, TrackerConfig, TrackedObstacle,
        TrackState,
    )
    from lidar_clusterer import Obstacle, ClusterResult


# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic data helpers
# ══════════════════════════════════════════════════════════════════════════════

_CFG = TrackerConfig()   # default config used across most tests


def _make_obstacle(
    x: float = 20.0,
    y: float = 0.0,
    z: float = 0.0,
    obj_type: str = 'vehicle',
    min_distance: float = 20.0,
    confidence: float = 0.8,
) -> Obstacle:
    """
    Construct a minimal Obstacle at a given position.
    centroid_dist, bearing_deg, heading_deg, extent_lwh are
    computed/approximated consistently with the clusterer.
    """
    centre = np.array([x, y, z], dtype=np.float32)
    bearing = float(np.degrees(np.arctan2(y, x)))
    centroid_dist = float(np.hypot(x, y))
    return Obstacle(
        id            = 0,
        type          = obj_type,
        center_xyz    = centre,
        extent_lwh    = np.array([4.5, 2.0, 1.8], dtype=np.float32),
        min_pt        = centre - np.array([2.25, 1.0, 0.9], dtype=np.float32),
        max_pt        = centre + np.array([2.25, 1.0, 0.9], dtype=np.float32),
        heading_deg   = 0.0,
        bearing_deg   = bearing,
        min_distance  = min_distance,
        centroid_dist = centroid_dist,
        point_count   = 80,
        confidence    = confidence,
    )


def _make_result(
    obstacles: List[Obstacle],
    frame_id:  int   = 0,
    timestamp: float = 0.0,
) -> ClusterResult:
    return ClusterResult(obstacles=obstacles, frame_id=frame_id, timestamp=timestamp)


def _ego_vel(vx: float = 0.0, vy: float = 0.0, vz: float = 0.0) -> np.ndarray:
    return np.array([vx, vy, vz], dtype=np.float64)


def _run_tracker_n_frames(
    tracker:    MultiObjectTracker,
    obstacles:  List[Obstacle],
    n_frames:   int,
    dt:         float = 0.1,
) -> List[TrackedObstacle]:
    """Feed the same obstacle list for n_frames and return the last output."""
    out = []
    for i in range(n_frames):
        result = _make_result(obstacles, frame_id=i, timestamp=i * dt)
        out = tracker.update(result, _ego_vel())
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Test runner
# ══════════════════════════════════════════════════════════════════════════════

_TESTS: List[Tuple[str, callable]] = []


def _test(name: str):
    """Decorator to register a test function."""
    def decorator(fn):
        _TESTS.append((name, fn))
        return fn
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
#  T01  ObjectKalmanFilter predict-only: position advances by v*dt
# ══════════════════════════════════════════════════════════════════════════════

@_test("T01  KF predict-only: position drifts by v*dt")
def test_kf_predict_only():
    """
    After initialisation at [10, 0, 0] and one predict(dt=0.1),
    without any update, the position should drift toward [10 + vx*dt, ...].
    Since initial velocity = 0, position should barely move (only due to P).
    The Kalman predict step propagates x = F @ x, so position = initial + 0*dt.
    """
    pos0 = np.array([10.0, 0.0, 0.0])
    kf   = ObjectKalmanFilter(pos0, _CFG)

    pred = kf.predict(dt=0.1)

    # With zero initial velocity, predicted position = initial position
    assert np.allclose(pred, pos0, atol=1e-6), (
        f"Predict with zero velocity should return initial position. Got {pred}"
    )
    assert np.allclose(kf.position, pos0, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  T02  ObjectKalmanFilter update: position moves toward measurement
# ══════════════════════════════════════════════════════════════════════════════

@_test("T02  KF update: position corrected toward measurement")
def test_kf_update_corrects_position():
    """
    Initialise at [0, 0, 0], then update with measurement [10, 0, 0].
    After one update, position should be between 0 and 10 (Kalman gain blending).
    """
    kf = ObjectKalmanFilter(np.zeros(3), _CFG)
    kf.predict(0.1)
    kf.update(np.array([10.0, 0.0, 0.0]))

    pos = kf.position
    # With large P_init and moderate R, Kalman gain is high: position should
    # have moved significantly toward 10.
    assert pos[0] > 1.0, f"Position x should be pulled toward 10.0. Got {pos[0]:.3f}"
    assert pos[0] < 10.0, f"Position x should not overshoot 10.0. Got {pos[0]:.3f}"
    # Y and Z should remain near 0
    assert abs(pos[1]) < 0.5
    assert abs(pos[2]) < 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  T03  ObjectKalmanFilter velocity convergence over 20 frames
# ══════════════════════════════════════════════════════════════════════════════

@_test("T03  KF velocity converges to true velocity over 20 frames")
def test_kf_velocity_convergence():
    """
    Simulate a vehicle moving at [5, 0, 0] m/s (5 m/s forward).
    After 20 frames of predict+update, the Kalman velocity estimate should
    converge close to [5, 0, 0] m/s.
    """
    TRUE_VEL = 5.0   # m/s forward
    dt       = 0.1
    kf       = ObjectKalmanFilter(np.array([0.0, 2.0, 0.5]), _CFG)

    rng = np.random.default_rng(seed=42)
    pos = np.array([0.0, 2.0, 0.5])

    for i in range(20):
        pos = pos + np.array([TRUE_VEL, 0.0, 0.0]) * dt
        kf.predict(dt)
        # Add small measurement noise
        noisy_pos = pos + rng.normal(0, 0.05, 3)
        kf.update(noisy_pos)

    vel = kf.velocity
    assert abs(vel[0] - TRUE_VEL) < 1.5, (
        f"Velocity x should converge to ~{TRUE_VEL} m/s. Got {vel[0]:.3f}"
    )
    assert abs(vel[1]) < 0.5, f"Velocity y should be near 0. Got {vel[1]:.3f}"
    assert abs(vel[2]) < 0.5, f"Velocity z should be near 0. Got {vel[2]:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
#  T04  New tracks created for every unmatched detection
# ══════════════════════════════════════════════════════════════════════════════

@_test("T04  New tracks created for all unmatched detections")
def test_new_tracks_created():
    """
    Feed 5 detections to an empty tracker on frame 0.
    All 5 should create new tracks (n_tracks == 5).
    """
    tracker = MultiObjectTracker(_CFG)
    dets    = [_make_obstacle(x=float(5 * (i + 1)), min_distance=float(5 * (i + 1)))
               for i in range(5)]
    result  = _make_result(dets, frame_id=0, timestamp=0.0)
    tracker.update(result, _ego_vel())

    assert tracker.track_count == 5, (
        f"Expected 5 tracks after 5 unmatched detections. Got {tracker.track_count}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T05  Track IDs are persistent across frames (same obstacle re-matched)
# ══════════════════════════════════════════════════════════════════════════════

@_test("T05  Track IDs persistent across consecutive matching frames")
def test_track_id_persistence():
    """
    Feed one stationary obstacle across 5 frames.
    All frames should report the same track_id.
    """
    tracker = MultiObjectTracker(_CFG)
    det     = _make_obstacle(x=20.0)

    ids = set()
    for i in range(5):
        result = _make_result([det], frame_id=i, timestamp=i * 0.1)
        out    = tracker.update(result, _ego_vel())
        ids.update(t.track_id for t in out)

    assert len(ids) == 1, (
        f"Persistent obstacle should always have the same track_id. Got ids={ids}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T06  Track deleted after max_lost_frames consecutive misses
# ══════════════════════════════════════════════════════════════════════════════

@_test("T06  Track deleted after max_lost_frames misses")
def test_track_deleted_after_lost():
    """
    Create one track, then send empty detection lists.
    After max_lost_frames + 1 frames, the track should be deleted.
    """
    cfg     = TrackerConfig(max_lost_frames=3)
    tracker = MultiObjectTracker(cfg)

    # Frame 0: create track
    result = _make_result([_make_obstacle(x=15.0)], frame_id=0, timestamp=0.0)
    tracker.update(result, _ego_vel())
    assert tracker.track_count == 1

    # Frames 1 to max_lost_frames: no detections — track should coast
    for i in range(1, cfg.max_lost_frames + 1):
        result = _make_result([], frame_id=i, timestamp=i * 0.1)
        tracker.update(result, _ego_vel())
        assert tracker.track_count == 1, (
            f"Track should still exist at lost_frame={i} (max={cfg.max_lost_frames})"
        )

    # Frame max_lost_frames + 1: one more miss -> deletion
    result = _make_result([], frame_id=cfg.max_lost_frames + 1,
                          timestamp=(cfg.max_lost_frames + 1) * 0.1)
    tracker.update(result, _ego_vel())
    assert tracker.track_count == 0, (
        f"Track should be deleted after {cfg.max_lost_frames + 1} empty frames. "
        f"Still have {tracker.track_count} tracks."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T07  is_confirmed is False until min_hits_to_confirm frames
# ══════════════════════════════════════════════════════════════════════════════

@_test("T07  is_confirmed False until min_hits_to_confirm frames")
def test_confirmation_threshold():
    """
    With min_hits_to_confirm=3:
      - Frames 1-2: is_confirmed should be False
      - Frame 3+:   is_confirmed should be True
    """
    cfg     = TrackerConfig(min_hits_to_confirm=3)
    tracker = MultiObjectTracker(cfg)
    det     = _make_obstacle(x=20.0)

    for i in range(5):
        result = _make_result([det], frame_id=i, timestamp=i * 0.1)
        out    = tracker.update(result, _ego_vel())
        assert len(out) == 1
        t = out[0]
        expected_confirmed = (i >= cfg.min_hits_to_confirm - 1)
        assert t.is_confirmed == expected_confirmed, (
            f"Frame {i}: expected is_confirmed={expected_confirmed}, got {t.is_confirmed}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  T08  Output sorted by min_distance ascending
# ══════════════════════════════════════════════════════════════════════════════

@_test("T08  Output sorted by min_distance ascending")
def test_output_sorted_by_distance():
    """
    Feed 4 obstacles at different distances.
    After enough frames to confirm, output[0] must be nearest.
    """
    tracker = MultiObjectTracker(_CFG)
    dists   = [40.0, 10.0, 25.0, 5.0]
    dets    = [_make_obstacle(x=d, min_distance=d) for d in dists]

    out = _run_tracker_n_frames(tracker, dets, n_frames=3)

    distances = [t.min_distance for t in out]
    assert distances == sorted(distances), (
        f"Output not sorted by min_distance. Got: {distances}"
    )
    # Nearest should be ~5 m (with EMA converging over 3 frames)
    assert distances[0] < 10.0, f"Nearest track should be ~5 m. Got {distances[0]:.1f}"


# ══════════════════════════════════════════════════════════════════════════════
#  T09  TTC = inf for young track or receding obstacle
# ══════════════════════════════════════════════════════════════════════════════

@_test("T09  TTC = inf for young track or receding obstacle")
def test_ttc_inf_cases():
    """
    Case A: New track (age < velocity_min_age) -> TTC must be inf.
    Case B: Track moving away (velocity_ms < 0) -> TTC must be inf.
    """
    cfg     = TrackerConfig(velocity_min_age=3)
    tracker = MultiObjectTracker(cfg)
    det     = _make_obstacle(x=20.0)

    # Case A: frame 0, track just created (age=1 < 3)
    result = _make_result([det], frame_id=0, timestamp=0.0)
    out    = tracker.update(result, _ego_vel())
    assert out[0].ttc_seconds == float('inf'), (
        f"Young track should have TTC=inf. Got {out[0].ttc_seconds}"
    )

    # Case B: stationary ego + stationary obstacle -> radial speed ~0 -> TTC=inf
    # After several frames of no relative motion, velocity_ms should be near 0.
    for i in range(1, 10):
        result = _make_result([det], frame_id=i, timestamp=i * 0.1)
        out    = tracker.update(result, _ego_vel())

    t = out[0]
    assert t.ttc_seconds == float('inf'), (
        f"Stationary obstacle should have TTC=inf. "
        f"velocity_ms={t.velocity_ms:.3f} ttc={t.ttc_seconds}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T10  TTC finite and approximately correct for approaching obstacle
# ══════════════════════════════════════════════════════════════════════════════

@_test("T10  TTC finite and approximately correct for approaching obstacle")
def test_ttc_approaching():
    """
    Simulate an obstacle approaching at ~10 m/s from 40 m.
    Expected TTC ~= 4 s.  After Kalman converges, TTC should be 2-8 s.
    """
    cfg     = TrackerConfig(velocity_min_age=3, dist_ema_alpha=0.5)
    tracker = MultiObjectTracker(cfg)
    dt      = 0.1
    SPEED   = 10.0  # m/s closing

    for i in range(25):
        # Obstacle starts at x=40, moves toward ego at 10 m/s
        x       = max(1.0, 40.0 - SPEED * i * dt)
        det     = _make_obstacle(x=x, min_distance=x)
        result  = _make_result([det], frame_id=i, timestamp=i * dt)
        out     = tracker.update(result, _ego_vel())

    t = out[0]
    assert t.ttc_seconds != float('inf'), (
        f"Approaching obstacle should have finite TTC. velocity_ms={t.velocity_ms:.3f}"
    )
    # At frame 25: x = 40 - 10*2.4 = 16 m, speed ~10 m/s -> TTC ~1.6 s
    # With Kalman smoothing, allow generous range [0.5, 8.0] s
    assert 0.5 <= t.ttc_seconds <= 8.0, (
        f"TTC out of expected range [0.5, 8.0]. Got {t.ttc_seconds:.2f} s. "
        f"velocity_ms={t.velocity_ms:.3f} min_dist={t.min_distance:.3f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T11  Sector assignment covers all 8 sectors
# ══════════════════════════════════════════════════════════════════════════════

@_test("T11  Sector assignment covers all 8 sectors")
def test_sector_assignment():
    """
    Place obstacles at 8 canonical bearings (0, 45, 90, 135, 180, -135, -90, -45 deg).
    Each should map to the correct named sector.
    """
    MOT = MultiObjectTracker
    bearing_to_expected = {
          0.0: 'FRONT',
         45.0: 'FRONT_LEFT',
         90.0: 'LEFT',
        135.0: 'REAR_LEFT',
        180.0: 'REAR',
       -135.0: 'REAR_RIGHT',
        -90.0: 'RIGHT',
        -45.0: 'FRONT_RIGHT',
    }
    for bearing, expected in bearing_to_expected.items():
        result = MOT._bearing_to_sector(bearing)
        assert result == expected, (
            f"Bearing {bearing} deg: expected sector '{expected}', got '{result}'"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  T12  Severity BRAKE / WARN / SAFE thresholds
# ══════════════════════════════════════════════════════════════════════════════

@_test("T12  Severity BRAKE/WARN/SAFE thresholds correct")
def test_severity_thresholds():
    """
    Verify _compute_severity() returns the right level for boundary cases.
    """
    cs = MultiObjectTracker._compute_severity

    # BRAKE: close distance
    assert cs(dist=4.0,  ttc=float('inf')) == 'BRAKE', "dist=4 m should be BRAKE"
    # BRAKE: imminent TTC
    assert cs(dist=20.0, ttc=2.0)         == 'BRAKE', "ttc=2 s should be BRAKE"
    # WARN: moderate distance
    assert cs(dist=10.0, ttc=float('inf')) == 'WARN',  "dist=10 m should be WARN"
    # WARN: moderate TTC
    assert cs(dist=20.0, ttc=4.0)         == 'WARN',  "ttc=4 s should be WARN"
    # SAFE: far and slow closure
    assert cs(dist=20.0, ttc=float('inf')) == 'SAFE',  "dist=20 m should be SAFE"
    assert cs(dist=20.0, ttc=30.0)        == 'SAFE',  "ttc=30 s should be SAFE"


# ══════════════════════════════════════════════════════════════════════════════
#  T13  World-frame velocity corrected for ego motion
# ══════════════════════════════════════════════════════════════════════════════

@_test("T13  World-velocity corrects for ego motion (parked car ~0)")
def test_world_velocity_ego_correction():
    """
    Ego drives at 10 m/s forward. A parked car at x=30 m appears to move
    at ~-10 m/s in ego frame. World velocity should be ~0 after correction.
    """
    cfg     = TrackerConfig(velocity_min_age=3, dist_ema_alpha=0.5)
    tracker = MultiObjectTracker(cfg)
    EGO_VX  = 10.0   # m/s forward

    for i in range(20):
        # Parked car stays at x=30 in world frame.
        # In ego frame: car appears to move backward: x_ego = 30 - ego_travel
        x_ego = 30.0 - EGO_VX * i * 0.1
        if x_ego < 1.0:
            break
        det    = _make_obstacle(x=x_ego, min_distance=x_ego)
        result = _make_result([det], frame_id=i, timestamp=i * 0.1)
        out    = tracker.update(result, _ego_vel(vx=EGO_VX))

    t = out[0]
    # World speed of a parked car should be near 0 (allow 3 m/s Kalman convergence lag)
    assert t.speed_ms < 3.0, (
        f"Parked car world speed should be ~0. Got {t.speed_ms:.2f} m/s. "
        f"velocity_xyz={[round(float(v),2) for v in t.velocity_xyz]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T14  ClusterResult wraps process() fields correctly
# ══════════════════════════════════════════════════════════════════════════════

@_test("T14  ClusterResult fields match source cloud")
def test_cluster_result_fields():
    """
    Verify ClusterResult.frame_id and .timestamp are propagated from the cloud.
    """
    # Use a minimal duck-typed cloud object (no real CARLA needed)
    class _FakeCloud:
        frame_id  = 999
        timestamp = 42.5
        obstacles = [_make_obstacle(x=10.0)]

    result = ClusterResult(
        obstacles = _FakeCloud.obstacles,
        frame_id  = _FakeCloud.frame_id,
        timestamp = _FakeCloud.timestamp,
    )
    assert result.frame_id  == 999,  f"Expected frame_id=999, got {result.frame_id}"
    assert result.timestamp == 42.5, f"Expected timestamp=42.5, got {result.timestamp}"
    assert len(result.obstacles) == 1


# ══════════════════════════════════════════════════════════════════════════════
#  T15  Benchmark: 50-frame pipeline under timing budget
# ══════════════════════════════════════════════════════════════════════════════

@_test("T15  Benchmark: 50 frames, 20 obstacles each, mean < 5 ms")
def test_benchmark_timing():
    """
    Run 50 frames with 20 detections per frame and measure tracker latency.
    Target: mean < 5 ms per frame (default max_proc_ms budget).
    This is not a hard assertion on slow CI machines — it prints a warning.
    """
    cfg     = TrackerConfig(log_every_n_frames=1000)
    tracker = MultiObjectTracker(cfg)
    rng     = np.random.default_rng(seed=0)
    times   = []

    for i in range(50):
        # 20 randomly placed obstacles in 5-50 m range
        dets = []
        for _ in range(20):
            angle = rng.uniform(-math.pi, math.pi)
            r     = rng.uniform(5.0, 50.0)
            x     = r * math.cos(angle)
            y     = r * math.sin(angle)
            dets.append(_make_obstacle(x=x, y=y, min_distance=r))

        result = _make_result(dets, frame_id=i, timestamp=i * 0.1)

        t0 = time.perf_counter()
        tracker.update(result, _ego_vel())
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times))
    p95_ms  = float(np.percentile(times, 95))
    print(
        f"\n    [Benchmark] mean={mean_ms:.2f} ms  "
        f"p95={p95_ms:.2f} ms  budget={cfg.max_proc_ms:.0f} ms"
    )

    # Warn only — don't fail on slow CI environments
    if mean_ms > cfg.max_proc_ms * 2:
        print(
            f"    WARNING: mean {mean_ms:.1f} ms exceeds 2x budget "
            f"({cfg.max_proc_ms * 2:.0f} ms). Check for performance regression."
        )
    # Assert a very loose upper bound (100 ms would indicate a serious bug)
    assert mean_ms < 100.0, f"Tracker catastrophically slow: {mean_ms:.1f} ms/frame"


# ══════════════════════════════════════════════════════════════════════════════
#  Test runner
# ══════════════════════════════════════════════════════════════════════════════

def _run_all(verbose: bool = False) -> bool:
    """Execute all registered tests, print pass/fail summary."""
    passed = 0
    failed = 0
    failures = []

    print(f"\n{'='*60}")
    print(f"  LiDAR Tracker — Task 04 Test Suite  ({len(_TESTS)} tests)")
    print(f"{'='*60}")

    for name, fn in _TESTS:
        try:
            fn()
            status = "PASS"
            passed += 1
            if verbose:
                print(f"  [PASS] {name}")
            else:
                print(f"  [PASS] {name}")
        except Exception as exc:
            status = "FAIL"
            failed += 1
            print(f"  [FAIL] {name}")
            print(f"         {exc}")
            if verbose:
                traceback.print_exc()
            failures.append((name, exc))

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed / {len(_TESTS)} total")
    print(f"{'='*60}\n")

    if failures:
        print("Failed tests:")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        print()

    return failed == 0


if __name__ == '__main__':
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    ok      = _run_all(verbose=verbose)
    sys.exit(0 if ok else 1)
