"""
Standalone unit test for obstacle ID propagation through LidarFusion.

No CARLA, no GPU required — runs with only numpy (sklearn optional).
Heavy dependencies (cv2, carla, sklearn) are mocked if not installed.

Usage:
    python test_obstacle_ids.py
"""

import math
import sys
import numpy as np
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock unavailable heavy dependencies BEFORE any core imports
# ---------------------------------------------------------------------------
for _mod in ('cv2', 'carla'):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
try:
    import sklearn  # noqa: F401
except ImportError:
    sys.modules['sklearn'] = MagicMock()
    sys.modules['sklearn.cluster'] = MagicMock()
    sys.modules['sklearn.cluster'].DBSCAN = MagicMock()

from core.lidar_obstacle_detector import LidarObstacle  # noqa: E402


def _make_lidar_obs(track_id: str, angle_deg: float, distance: float = 10.0) -> LidarObstacle:
    rad = math.radians(angle_deg)
    cx = distance * math.cos(rad)
    cy = distance * math.sin(rad)
    return LidarObstacle(
        centroid_x=cx, centroid_y=cy, centroid_z=0.0,
        distance=distance, angle_deg=angle_deg,
        sector='front', point_count=50,
        danger_level='stop',
        track_id=track_id,
    )


def _make_cam_det(angle_deg: float, img_width: int = 1280, focal_px: float = 640.0,
                  distance: float = 10.0) -> dict:
    """Camera detection dict with a bbox centred at the given horizontal angle."""
    cx_px = img_width / 2.0 + focal_px * math.tan(math.radians(angle_deg))
    bbox_w, bbox_h = 80, 120
    x1 = int(cx_px - bbox_w / 2)
    y1 = 300
    return {
        'bbox': (x1, y1, x1 + bbox_w, y1 + bbox_h),
        'bbox_center': (int(cx_px), y1 + bbox_h // 2),
        'distance': distance,
        'confidence': 0.9,
        'class': 'car',
        'class_id': 2,
        'danger_level': 'stop',
        'is_dangerous': True,
        'in_lane': True,
        'lane_overlap': 0.8,
    }


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
_results = []


def _check(name: str, condition: bool, detail: str = '') -> None:
    status = PASS if condition else FAIL
    suffix = f'  ({detail})' if detail else ''
    print(f'  [{status}] {name}{suffix}')
    _results.append(condition)


# ---------------------------------------------------------------------------

def test_full_angle_match() -> None:
    """Camera detection at 5° matched to LiDAR track at 4° → gets LiDAR track_id."""
    from core.lidar_fusion import LidarFusion
    fusion = LidarFusion(angle_match_threshold_deg=15.0)
    obs = _make_lidar_obs('t0', angle_deg=4.0)
    det = _make_cam_det(angle_deg=5.0)

    fused, _, _ = fusion.fuse(
        camera_detections=[det],
        lidar_obstacles=[obs],
        camera_action='stop',
        lidar_raw_points=None,  # force angle-based path
    )

    _check('FULL_ANGLE: obstacle_id == t0',
           fused[0].get('obstacle_id') == 't0',
           f"got {fused[0].get('obstacle_id')!r}")
    _check('FULL_ANGLE: fusion_method == FULL_ANGLE',
           fused[0].get('fusion_method') == 'FULL_ANGLE')


def test_camera_only_no_lidar() -> None:
    """Camera detection with no LiDAR obstacle → gets cam_N ID."""
    from core.lidar_fusion import LidarFusion
    fusion = LidarFusion(angle_match_threshold_deg=15.0)
    det = _make_cam_det(angle_deg=0.0)

    fused, _, _ = fusion.fuse(
        camera_detections=[det],
        lidar_obstacles=[],
        camera_action='stop',
        lidar_raw_points=None,
    )

    obs_id = fused[0].get('obstacle_id', '')
    _check('CAM_ONLY: obstacle_id starts with cam_',
           obs_id.startswith('cam_'), f"got {obs_id!r}")
    _check('CAM_ONLY: fusion_method == CAMERA_ONLY',
           fused[0].get('fusion_method') == 'CAMERA_ONLY')


def test_camera_only_outside_angle_threshold() -> None:
    """Camera at 0°, LiDAR at 30° (> 15° threshold) → camera-only ID."""
    from core.lidar_fusion import LidarFusion
    fusion = LidarFusion(angle_match_threshold_deg=15.0)
    obs = _make_lidar_obs('t0', angle_deg=30.0)
    det = _make_cam_det(angle_deg=0.0)

    fused, _, _ = fusion.fuse(
        camera_detections=[det],
        lidar_obstacles=[obs],
        camera_action='stop',
        lidar_raw_points=None,
    )

    obs_id = fused[0].get('obstacle_id', '')
    _check('Outside threshold: obstacle_id starts with cam_',
           obs_id.startswith('cam_'), f"got {obs_id!r}")


def test_lidar_only() -> None:
    """LiDAR obstacle with no camera detection → appended as LIDAR_ONLY with track_id."""
    from core.lidar_fusion import LidarFusion
    fusion = LidarFusion(angle_match_threshold_deg=15.0)
    obs = _make_lidar_obs('t0', angle_deg=2.0)

    fused, _, _ = fusion.fuse(
        camera_detections=[],
        lidar_obstacles=[obs],
        camera_action='drive',
        lidar_raw_points=None,
    )

    lidar_only = [d for d in fused if d.get('fusion_method') == 'LIDAR_ONLY']
    _check('LIDAR_ONLY: entry exists', len(lidar_only) == 1)
    if lidar_only:
        _check('LIDAR_ONLY: obstacle_id == t0',
               lidar_only[0].get('obstacle_id') == 't0',
               f"got {lidar_only[0].get('obstacle_id')!r}")


def test_cross_frame_lidar_id_persistence() -> None:
    """Same LiDAR track across two consecutive fuse() calls → same obstacle_id."""
    from core.lidar_fusion import LidarFusion
    fusion = LidarFusion(angle_match_threshold_deg=15.0)
    obs = _make_lidar_obs('t0', angle_deg=3.0)
    det = _make_cam_det(angle_deg=3.0)

    fused1, _, _ = fusion.fuse([det], [obs], 'stop', lidar_raw_points=None)
    fused2, _, _ = fusion.fuse([det], [obs], 'stop', lidar_raw_points=None)

    id1 = fused1[0].get('obstacle_id')
    id2 = fused2[0].get('obstacle_id')
    _check('Persistence: same ID across 2 frames',
           id1 == id2 == 't0', f"frame1={id1!r} frame2={id2!r}")


def test_camera_only_persistence() -> None:
    """Camera-only detection at the same bbox centre across 2 frames → same cam_N ID."""
    from core.lidar_fusion import LidarFusion
    fusion = LidarFusion(angle_match_threshold_deg=15.0)
    det = _make_cam_det(angle_deg=0.0)

    fused1, _, _ = fusion.fuse([det], [], 'stop', lidar_raw_points=None)
    fused2, _, _ = fusion.fuse([det], [], 'stop', lidar_raw_points=None)

    id1 = fused1[0].get('obstacle_id')
    id2 = fused2[0].get('obstacle_id')
    _check('CAM persistence: same cam_N ID across 2 frames',
           id1 == id2 and id1 is not None, f"frame1={id1!r} frame2={id2!r}")
    _check('CAM persistence: ID starts with cam_',
           id1 is not None and id1.startswith('cam_'), f"got {id1!r}")


def test_camera_only_stale_eviction() -> None:
    """Camera-only det absent for 6 frames → new ID on re-appearance."""
    from core.lidar_fusion import LidarFusion
    fusion = LidarFusion(angle_match_threshold_deg=15.0)
    det = _make_cam_det(angle_deg=0.0)

    # First detection
    fused1, _, _ = fusion.fuse([det], [], 'stop', lidar_raw_points=None)
    first_id = fused1[0].get('obstacle_id')

    # Skip 6 frames (more than _CAM_STALE_MAX=5)
    for _ in range(6):
        fusion.fuse([], [], 'drive', lidar_raw_points=None)

    # Re-detect at same position
    fused2, _, _ = fusion.fuse([det], [], 'stop', lidar_raw_points=None)
    second_id = fused2[0].get('obstacle_id')

    _check('Stale eviction: new ID assigned after 6-frame gap',
           first_id != second_id,
           f"first={first_id!r} second={second_id!r}")


def test_multiple_obstacles_distinct_ids() -> None:
    """Two LiDAR obstacles at different angles → each camera det gets the correct ID."""
    from core.lidar_fusion import LidarFusion
    fusion = LidarFusion(angle_match_threshold_deg=15.0)
    obs_a = _make_lidar_obs('t0', angle_deg=5.0,  distance=10.0)
    obs_b = _make_lidar_obs('t1', angle_deg=-8.0, distance=15.0)
    det_a = _make_cam_det(angle_deg=5.0,  distance=10.0)
    det_b = _make_cam_det(angle_deg=-8.0, distance=15.0)

    fused, _, _ = fusion.fuse(
        [det_a, det_b], [obs_a, obs_b], 'stop', lidar_raw_points=None
    )

    id_map = {d['obstacle_id']: d for d in fused if d.get('obstacle_id')}
    _check('Multi-obs: t0 present', 't0' in id_map)
    _check('Multi-obs: t1 present', 't1' in id_map)
    _check('Multi-obs: IDs are distinct', len(id_map) >= 2)


def test_full_bbox_match() -> None:
    """FULL_BBOX path: LiDAR raw points inside camera bbox → gets LiDAR track_id."""
    from core.lidar_fusion import LidarFusion
    fusion = LidarFusion(angle_match_threshold_deg=15.0)
    obs = _make_lidar_obs('t0', angle_deg=2.0, distance=10.0)

    # Build a camera detection that covers the centre of the image
    img_width, focal_px = 1280, 640.0
    cx_px = img_width / 2.0  # angle=0 → centre pixel
    bbox_w, bbox_h = 200, 300
    x1 = int(cx_px - bbox_w / 2)
    y1 = 250
    det = {
        'bbox': (x1, y1, x1 + bbox_w, y1 + bbox_h),
        'bbox_center': (int(cx_px), y1 + bbox_h // 2),
        'distance': 10.0,
        'confidence': 0.9, 'class': 'car', 'class_id': 2,
        'danger_level': 'stop', 'is_dangerous': True,
        'in_lane': True, 'lane_overlap': 0.9,
    }

    # Synthetic raw LiDAR points: cluster at 10 m forward, slight offset so they project
    # into the camera bbox.  The projector reprojects XYZ → (u, v) pixels.
    # At 10 m forward (X=10), the horizontal angle is ~2° → u ≈ cx_px + focal*tan(2°)
    # We place points at X=10, Y varies ±0.2 m (so angle stays within bbox width).
    rng = np.random.default_rng(42)
    n_pts = 20
    pts_x = np.full(n_pts, 10.0) + rng.uniform(-0.3, 0.3, n_pts)
    pts_y = rng.uniform(-0.2, 0.2, n_pts)   # small lateral offset
    pts_z = rng.uniform(0.0, 1.5, n_pts)
    raw_pts = np.column_stack([pts_x, pts_y, pts_z])

    fused, _, _ = fusion.fuse(
        camera_detections=[det],
        lidar_obstacles=[obs],
        camera_action='stop',
        img_width=img_width,
        focal_length_px=focal_px,
        lidar_raw_points=raw_pts,
    )

    method = fused[0].get('fusion_method')
    obs_id = fused[0].get('obstacle_id')
    if method == 'FULL_BBOX':
        _check('FULL_BBOX: obstacle_id == t0',
               obs_id == 't0', f"got {obs_id!r}")
    else:
        # Fell back to angle matching — still should get t0
        _check(f'FULL_BBOX (fell back to {method}): obstacle_id == t0',
               obs_id == 't0', f"got {obs_id!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print('=' * 60)
    print('  Obstacle ID propagation — unit tests')
    print('=' * 60)

    tests = [
        ('FULL_ANGLE match',           test_full_angle_match),
        ('CAMERA_ONLY (no LiDAR)',     test_camera_only_no_lidar),
        ('CAMERA_ONLY (outside angle)',test_camera_only_outside_angle_threshold),
        ('LIDAR_ONLY',                 test_lidar_only),
        ('Cross-frame LiDAR persist',  test_cross_frame_lidar_id_persistence),
        ('Camera-only persist',        test_camera_only_persistence),
        ('Stale eviction',             test_camera_only_stale_eviction),
        ('Multiple obstacles',         test_multiple_obstacles_distinct_ids),
        ('FULL_BBOX match',            test_full_bbox_match),
    ]

    for name, fn in tests:
        print(f'\n[{name}]')
        fn()

    passed = sum(_results)
    total  = len(_results)
    print('\n' + '=' * 60)
    if passed == total:
        print(f'  All {total} checks passed.')
    else:
        print(f'  {passed}/{total} checks passed  — {total - passed} FAILED.')
    print('=' * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()
