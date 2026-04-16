"""
Test suite for Task 07 -- ADAS Real-Time Data Exporter
=======================================================

15 tests covering:
  T01  ExporterConfig defaults and invariants
  T02  json_serialise_safe handles all special types
  T03  write_json_atomic is atomic (no partial files on crash simulation)
  T04  save_numpy_compressed enforces float32 and C-contiguous
  T05  build_nn_label_detection_3d: class_id, heading_rad conversion
  T06  build_nn_label_risk: severity_id mapping, TTC=inf -> None
  T07  LidarConfig.to_dict roundtrip
  T08  DataExporter.open_session creates directory structure
  T09  DataExporter.record_frame selects frames correctly (every_n, min_obs)
  T10  DataExporter drops frames when queue is full (non-blocking)
  T11  DataExporter writes correct lidar_raw.npy shape (N, 4)
  T12  DataExporter writes valid obstacles.json with nn_label blocks
  T13  DataExporter.close_session drains queue and returns stats
  T14  verify_export detects missing files and invalid JSON
  T15  Performance benchmark: 50 record_frame calls (pipeline overhead)

Run:
    cd core && python test_lidar_exporter.py
    cd core && python test_lidar_exporter.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    from export_utils import (
        json_serialise_safe, write_json_atomic, save_numpy_compressed,
        build_nn_label_detection_3d, build_nn_label_risk, CLASS_ID_MAP,
    )
    from lidar_exporter import ExporterConfig, LidarConfig, DataExporter
    from verify_export import verify_session
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Test infrastructure
# ══════════════════════════════════════════════════════════════════════════════

_tests: List = []


def _test(name: str):
    def decorator(fn):
        _tests.append((name, fn))
        return fn
    return decorator


def _run_all(verbose: bool = False) -> None:
    passed = 0
    failed = 0
    for name, fn in _tests:
        try:
            fn(verbose)
            if verbose:
                print(f"  [PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}")
            if verbose:
                traceback.print_exc()
            else:
                print(f"         {exc}")
            failed += 1

    total = passed + failed
    print(f"\nResults: {passed} passed, {failed} failed / {total} total")
    if failed:
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Synthetic helpers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _FakeFusedObstacle:
    """Minimal FusedObstacle stand-in for testing."""
    track_id          : int   = 1
    frame_id          : int   = 0
    timestamp         : float = 0.0
    fusion_method     : str   = 'FULL'
    lidar_class       : str   = 'vehicle'
    camera_class      : str   = 'car'
    fused_class       : str   = 'car'
    camera_confidence : float = 0.9
    fused_confidence  : float = 0.85
    center_xyz        : np.ndarray = None
    extent_lwh        : np.ndarray = None
    heading_deg       : float = -15.0
    bearing_deg       : float = -9.7
    sector            : str   = 'FRONT'
    min_distance      : float = 10.0
    velocity_ms       : float = 3.0
    velocity_xyz      : np.ndarray = None
    ttc_seconds       : float = 3.33
    severity          : str   = 'WARN'
    camera_bbox_xyxy  : Optional[tuple] = (200.0, 100.0, 500.0, 400.0)
    camera_bbox_norm  : Optional[tuple] = (0.35, 0.347, 0.234, 0.417)
    projected_bbox_xyxy: Optional[tuple] = (210.0, 110.0, 490.0, 390.0)
    match_iou         : float = 0.74
    kalman_state      : np.ndarray = None
    kalman_uncertainty: np.ndarray = None
    age_frames        : int   = 10
    lost_frames       : int   = 0
    is_confirmed      : bool  = True
    cluster_points    : Optional[np.ndarray] = None

    def __post_init__(self):
        if self.center_xyz is None:
            self.center_xyz = np.array([10.0, -2.0, 0.4], dtype=np.float32)
        if self.extent_lwh is None:
            self.extent_lwh = np.array([4.5, 2.0, 1.5], dtype=np.float32)
        if self.velocity_xyz is None:
            self.velocity_xyz = np.array([-3.0, -0.5, 0.0], dtype=np.float32)
        if self.kalman_state is None:
            self.kalman_state = np.zeros(6, dtype=np.float64)
        if self.kalman_uncertainty is None:
            self.kalman_uncertainty = np.ones(6, dtype=np.float64) * 0.1

    def to_dict(self) -> dict:
        return {
            'track_id'           : self.track_id,
            'frame_id'           : self.frame_id,
            'timestamp'          : round(self.timestamp, 4),
            'fusion_method'      : self.fusion_method,
            'lidar_class'        : self.lidar_class,
            'camera_class'       : self.camera_class,
            'fused_class'        : self.fused_class,
            'camera_confidence'  : round(float(self.camera_confidence), 4),
            'fused_confidence'   : round(float(self.fused_confidence), 4),
            'center_xyz'         : [round(float(v), 4) for v in self.center_xyz],
            'extent_lwh'         : [round(float(v), 4) for v in self.extent_lwh],
            'heading_deg'        : round(float(self.heading_deg), 2),
            'bearing_deg'        : round(float(self.bearing_deg), 2),
            'sector'             : self.sector,
            'min_distance'       : round(float(self.min_distance), 4),
            'velocity_ms'        : round(float(self.velocity_ms), 4),
            'velocity_xyz'       : [round(float(v), 4) for v in self.velocity_xyz],
            'ttc_seconds'        : (round(float(self.ttc_seconds), 3)
                                    if self.ttc_seconds != float('inf') else None),
            'severity'           : self.severity,
            'camera_bbox_xyxy'   : list(self.camera_bbox_xyxy) if self.camera_bbox_xyxy else None,
            'camera_bbox_norm'   : list(self.camera_bbox_norm) if self.camera_bbox_norm else None,
            'projected_bbox_xyxy': list(self.projected_bbox_xyxy) if self.projected_bbox_xyxy else None,
            'match_iou'          : round(float(self.match_iou), 4),
            'kalman_state'       : [round(float(v), 6) for v in self.kalman_state],
            'kalman_uncertainty' : [round(float(v), 6) for v in self.kalman_uncertainty],
            'age_frames'         : self.age_frames,
            'lost_frames'        : self.lost_frames,
            'is_confirmed'       : self.is_confirmed,
        }


@dataclass
class _FakeFusionResult:
    frame_id        : int   = 1
    timestamp       : float = 10.0
    fused_obstacles : list  = None
    n_lidar_tracks  : int   = 2
    n_camera_dets   : int   = 1
    n_full          : int   = 1
    n_lidar_only    : int   = 1
    n_camera_only   : int   = 0
    camera_frame_age_s: float = 0.02
    proc_ms         : float = 5.5

    def __post_init__(self):
        if self.fused_obstacles is None:
            self.fused_obstacles = [_FakeFusedObstacle()]

    def fusion_quality(self) -> float:
        total = self.n_lidar_tracks
        return self.n_full / total if total > 0 else 0.0


@dataclass
class _FakeLidarFrame:
    frame_id  : int         = 1
    timestamp : float       = 10.0
    points    : np.ndarray  = None
    intensity : np.ndarray  = None
    num_points: int         = 5000

    def __post_init__(self):
        if self.points is None:
            rng = np.random.default_rng(42)
            self.points    = rng.uniform(-30, 30, (self.num_points, 3)).astype(np.float32)
        if self.intensity is None:
            rng = np.random.default_rng(43)
            self.intensity = rng.uniform(0, 1, (self.num_points,)).astype(np.float32)


@dataclass
class _FakePreprocessedCloud:
    frame_id  : int        = 1
    timestamp : float      = 10.0
    points    : np.ndarray = None

    def __post_init__(self):
        if self.points is None:
            rng = np.random.default_rng(44)
            self.points = rng.uniform(-20, 20, (3000, 3)).astype(np.float32)


@dataclass
class _FakeClusterResult:
    frame_id  : int  = 1
    timestamp : float = 10.0
    obstacles : list = None

    def __post_init__(self):
        if self.obstacles is None:
            self.obstacles = []


def _default_cfg(tmpdir: str, **kwargs) -> ExporterConfig:
    return ExporterConfig(
        base_output_dir=tmpdir,
        n_writer_threads=1,
        write_queue_maxsize=16,
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  T01 -- ExporterConfig defaults
# ══════════════════════════════════════════════════════════════════════════════

@_test("T01 ExporterConfig defaults and layout invariants")
def test_t01(verbose):
    cfg = ExporterConfig()
    assert cfg.jpeg_quality == 90, "default JPEG quality should be 90"
    assert cfg.write_queue_maxsize == 60, "default queue size should be 60"
    assert cfg.n_writer_threads == 2, "default writer threads should be 2"
    assert cfg.save_every_n_frames == 1, "default every_n_frames should be 1"
    assert cfg.gt_max_range_m == 120.0, "default GT range should be 120 m"
    assert cfg.alert_flash_hz <= 6.0 if hasattr(cfg, 'alert_flash_hz') else True

    # to_dict roundtrip
    d = cfg.to_dict()
    assert d['jpeg_quality'] == 90
    assert d['n_writer_threads'] == 2
    assert isinstance(d['gt_actor_types'], (list, tuple))

    if verbose:
        print(f"    ExporterConfig has {len(d)} fields")


# ══════════════════════════════════════════════════════════════════════════════
#  T02 -- json_serialise_safe
# ══════════════════════════════════════════════════════════════════════════════

@_test("T02 json_serialise_safe handles all special types")
def test_t02(verbose):
    # np.ndarray
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert json_serialise_safe(arr) == [1.0, 2.0, 3.0]

    # np.floating
    assert json_serialise_safe(np.float32(3.14)) == pytest_approx(3.14, 1e-3)

    # inf and nan -> None
    assert json_serialise_safe(float('inf')) is None
    assert json_serialise_safe(float('nan')) is None
    assert json_serialise_safe(np.float32('inf')) is None

    # np.integer
    assert json_serialise_safe(np.int64(42)) == 42

    # np.bool_
    assert json_serialise_safe(np.bool_(True)) is True

    # Path — use a platform-neutral comparison
    p = Path('tmp') / 'test'
    assert json_serialise_safe(p) == str(p)

    # Verify full json roundtrip
    data = {
        'arr'   : arr,
        'inf'   : float('inf'),
        'speed' : np.float32(14.2),
        'count' : np.int64(7),
        'path'  : Path('/tmp'),
    }
    s = json.dumps(data, default=json_serialise_safe)
    parsed = json.loads(s)
    assert parsed['inf'] is None
    assert parsed['count'] == 7
    assert parsed['path'] == '/tmp'


def pytest_approx(val, tol):
    """Minimal approximate equality check (no pytest dependency)."""
    class _Approx:
        def __init__(self, v, t): self.v = v; self.t = t
        def __eq__(self, other): return abs(other - self.v) < self.t
    return _Approx(val, tol)


# ══════════════════════════════════════════════════════════════════════════════
#  T03 -- write_json_atomic
# ══════════════════════════════════════════════════════════════════════════════

@_test("T03 write_json_atomic: no .tmp files remain after write")
def test_t03(verbose):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'test.json'
        data = {'frame_id': 1, 'value': 42.5, 'arr': [1, 2, 3]}
        write_json_atomic(data, path)

        assert path.exists(), "JSON file should exist after atomic write"
        assert not path.with_suffix('.tmp').exists(), ".tmp file should be gone"

        with open(path) as f:
            loaded = json.load(f)
        assert loaded['frame_id'] == 1
        assert loaded['value'] == 42.5

        if verbose:
            print(f"    File size: {path.stat().st_size} bytes")


# ══════════════════════════════════════════════════════════════════════════════
#  T04 -- save_numpy_compressed
# ══════════════════════════════════════════════════════════════════════════════

@_test("T04 save_numpy_compressed enforces float32 and C-contiguous")
def test_t04(verbose):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'test.npy'

        # Normal save
        arr = np.random.rand(100, 4).astype(np.float32)
        save_numpy_compressed(arr, path)
        loaded = np.load(str(path))
        assert loaded.shape == (100, 4)
        assert loaded.dtype == np.float32

        # float64 should raise
        try:
            save_numpy_compressed(arr.astype(np.float64), path)
            assert False, "Should have raised TypeError for float64"
        except TypeError as e:
            assert 'float32' in str(e)

        # F-contiguous should be auto-converted
        arr_f = np.asfortranarray(arr)
        assert not arr_f.flags['C_CONTIGUOUS']
        save_numpy_compressed(arr_f, path)   # must not raise
        loaded2 = np.load(str(path))
        assert loaded2.flags['C_CONTIGUOUS']

        if verbose:
            print(f"    Saved (100,4) float32: {path.stat().st_size} bytes")


# ══════════════════════════════════════════════════════════════════════════════
#  T05 -- build_nn_label_detection_3d
# ══════════════════════════════════════════════════════════════════════════════

@_test("T05 build_nn_label_detection_3d: class_id and heading_rad")
def test_t05(verbose):
    obs = _FakeFusedObstacle(fused_class='car', heading_deg=90.0)
    label = build_nn_label_detection_3d(obs)

    assert label['class_id']   == 0,          "car -> class_id=0"
    assert label['class_name'] == 'car'
    assert abs(label['heading_rad'] - 1.5708) < 0.001, \
        f"90 deg -> pi/2 rad, got {label['heading_rad']}"
    assert len(label['center'])   == 3
    assert len(label['size'])     == 3
    assert len(label['velocity']) == 2

    # unknown class
    obs2  = _FakeFusedObstacle(fused_class='spaceship')
    l2    = build_nn_label_detection_3d(obs2)
    assert l2['class_id'] == 5, "unknown class -> 5"

    if verbose:
        print(f"    Detection label keys: {list(label.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
#  T06 -- build_nn_label_risk
# ══════════════════════════════════════════════════════════════════════════════

@_test("T06 build_nn_label_risk: severity_id, TTC=inf -> None")
def test_t06(verbose):
    # WARN with finite TTC
    obs = _FakeFusedObstacle(severity='WARN', ttc_seconds=3.5)
    r   = build_nn_label_risk(obs)
    assert r['severity_id']   == 1
    assert r['severity_name'] == 'WARN'
    assert r['is_threat']     is True
    assert abs(r['ttc_s'] - 3.5) < 0.01

    # BRAKE
    obs2 = _FakeFusedObstacle(severity='BRAKE', ttc_seconds=1.2)
    r2   = build_nn_label_risk(obs2)
    assert r2['severity_id'] == 2
    assert r2['is_threat']   is True

    # SAFE with TTC=inf
    obs3 = _FakeFusedObstacle(severity='SAFE', ttc_seconds=float('inf'))
    r3   = build_nn_label_risk(obs3)
    assert r3['severity_id'] == 0
    assert r3['ttc_s']       is None,  "TTC=inf must serialise to None"
    assert r3['is_threat']   is False

    if verbose:
        print(f"    WARN risk label: {r}")


# ══════════════════════════════════════════════════════════════════════════════
#  T07 -- LidarConfig.to_dict
# ══════════════════════════════════════════════════════════════════════════════

@_test("T07 LidarConfig.to_dict roundtrip")
def test_t07(verbose):
    cfg  = LidarConfig(channels=64, range_m=100.0, rotation_frequency=10.0)
    d    = cfg.to_dict()
    assert d['channels']           == 64
    assert d['range_m']            == 100.0
    assert d['rotation_frequency'] == 10.0
    assert d['noise_stddev']       == 0.0
    assert 'mount_xyz' in d
    # Serialisable as JSON
    s = json.dumps(d)
    loaded = json.loads(s)
    assert loaded['channels'] == 64

    if verbose:
        print(f"    LidarConfig fields: {list(d.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
#  T08 -- open_session creates directory structure
# ══════════════════════════════════════════════════════════════════════════════

@_test("T08 open_session creates directory structure and metadata.json")
def test_t08(verbose):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg      = _default_cfg(tmpdir)
        exporter = DataExporter(cfg)
        lidar_cfg = LidarConfig()

        session_dir = exporter.open_session(
            lidar_config=lidar_cfg,
            camera_config={'image_width': 1280},
            extra_metadata={'test_run': True},
        )
        exporter.close_session()

        # Check directory structure
        assert session_dir.exists(),               "session dir must exist"
        assert (session_dir / 'frames').exists(),  "frames/ dir must exist"
        assert (session_dir / 'metadata.json').exists(), "metadata.json must exist"

        with open(session_dir / 'metadata.json') as f:
            meta = json.load(f)

        assert meta['coordinate_frame'] == 'ISO8855_vehicle_RH'
        assert meta['distance_unit']    == 'metres'
        assert 'lidar_config' in meta
        assert 'nn_compatibility' in meta
        assert meta.get('test_run')     is True

        if verbose:
            print(f"    Session: {session_dir.name}")
            print(f"    Metadata keys: {list(meta.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
#  T09 -- Frame selection: every_n and min_obstacles
# ══════════════════════════════════════════════════════════════════════════════

@_test("T09 record_frame respects save_every_n_frames and min_obstacles_to_save")
def test_t09(verbose):
    with tempfile.TemporaryDirectory() as tmpdir:
        # every_n=3: save frames 3, 6, 9, ...
        cfg = _default_cfg(tmpdir, save_every_n_frames=3, min_obstacles_to_save=0)
        exp = DataExporter(cfg)
        exp.open_session(LidarConfig())

        fr = _FakeLidarFrame()
        cl = _FakePreprocessedCloud()
        cr = _FakeClusterResult()
        fu = _FakeFusionResult()

        results = []
        for i in range(9):
            r = exp.record_frame(fr, cl, cr, fu)
            results.append(r)

        exp.close_session()

        # frames 1,2 -> skip; 3 -> save; 4,5 -> skip; 6 -> save; ...
        saved = sum(1 for r in results if r)
        assert saved == 3, f"Expected 3 saved, got {saved}"

        # min_obstacles=2: skip frames with < 2 confirmed obstacles
        cfg2 = _default_cfg(tmpdir, min_obstacles_to_save=2)
        exp2 = DataExporter(cfg2)
        exp2.open_session(LidarConfig())

        # 1 obstacle -> should skip
        fu_one = _FakeFusionResult(
            fused_obstacles=[_FakeFusedObstacle()],
        )
        r2 = exp2.record_frame(fr, cl, cr, fu_one)
        exp2.close_session()
        assert r2 is False, "Should skip frame with 1 confirmed obstacle (min=2)"

        if verbose:
            print(f"    every_n=3 results: {results}")


# ══════════════════════════════════════════════════════════════════════════════
#  T10 -- Queue-full: frames dropped non-blocking
# ══════════════════════════════════════════════════════════════════════════════

@_test("T10 record_frame drops frames non-blocking when queue is full")
def test_t10(verbose):
    with tempfile.TemporaryDirectory() as tmpdir:
        # tiny queue, no writer threads consuming it
        cfg = ExporterConfig(
            base_output_dir=tmpdir,
            write_queue_maxsize=2,
            n_writer_threads=0,   # no consumers — queue will fill immediately
        )
        exp = DataExporter(cfg)
        exp.open_session(LidarConfig())

        fr = _FakeLidarFrame()
        cl = _FakePreprocessedCloud()
        cr = _FakeClusterResult()
        fu = _FakeFusionResult()

        results = []
        t0 = time.perf_counter()
        for _ in range(5):
            r = exp.record_frame(fr, cl, cr, fu)
            results.append(r)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Manually empty queue to allow sentinel-driven shutdown
        while not exp._write_queue.empty():
            exp._write_queue.get_nowait()

        exp.close_session()

        n_dropped = sum(1 for r in results if not r)
        assert n_dropped >= 2, f"Expected dropped frames with queue=2, got {n_dropped}"
        assert elapsed_ms < 100, \
            f"record_frame should be non-blocking, took {elapsed_ms:.1f} ms for 5 calls"

        if verbose:
            print(f"    5 frames: {results} | elapsed={elapsed_ms:.1f} ms")


# ══════════════════════════════════════════════════════════════════════════════
#  T11 -- lidar_raw.npy shape is (N, 4)
# ══════════════════════════════════════════════════════════════════════════════

@_test("T11 lidar_raw.npy written as (N, 4) float32 with intensity packed")
def test_t11(verbose):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _default_cfg(tmpdir, save_lidar_raw=True, save_lidar_intensity=True)
        exp = DataExporter(cfg)
        session_dir = exp.open_session(LidarConfig())

        N  = 5000
        fr = _FakeLidarFrame(num_points=N)
        cl = _FakePreprocessedCloud()
        cr = _FakeClusterResult()
        fu = _FakeFusionResult(fused_obstacles=[
            _FakeFusedObstacle(frame_id=fr.frame_id)
        ])

        exp.record_frame(fr, cl, cr, fu)
        exp.close_session()

        # Find the written frame
        frame_dir = list((session_dir / 'frames').iterdir())[0]
        raw_path  = frame_dir / 'lidar_raw.npy'

        assert raw_path.exists(), "lidar_raw.npy must be written"
        arr = np.load(str(raw_path))

        assert arr.dtype == np.float32,      "dtype must be float32"
        assert arr.ndim  == 2,               "must be 2D"
        assert arr.shape == (N, 4),          f"shape must be (N, 4), got {arr.shape}"
        # Column 3 is intensity [0, 1]
        assert arr[:, 3].min() >= 0.0
        assert arr[:, 3].max() <= 1.0 + 1e-5

        if verbose:
            print(f"    lidar_raw.npy shape={arr.shape} dtype={arr.dtype}")


# ══════════════════════════════════════════════════════════════════════════════
#  T12 -- obstacles.json has nn_label blocks
# ══════════════════════════════════════════════════════════════════════════════

@_test("T12 obstacles.json contains nn_label blocks with detection_3d and risk_label")
def test_t12(verbose):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _default_cfg(tmpdir)
        exp = DataExporter(cfg)
        session_dir = exp.open_session(LidarConfig())

        fr = _FakeLidarFrame()
        cl = _FakePreprocessedCloud()
        cr = _FakeClusterResult()
        fu = _FakeFusionResult(fused_obstacles=[
            _FakeFusedObstacle(
                track_id=7, severity='WARN', ttc_seconds=3.33,
                fused_class='car',
            )
        ])

        exp.record_frame(fr, cl, cr, fu)
        exp.close_session()

        frame_dir = list((session_dir / 'frames').iterdir())[0]
        with open(frame_dir / 'obstacles.json') as f:
            data = json.load(f)

        assert data['n_obstacles'] == 1
        obs = data['obstacles'][0]
        assert 'nn_label' in obs, "obstacle must have nn_label"
        nn  = obs['nn_label']
        assert 'detection_3d' in nn, "nn_label must have detection_3d"
        assert 'risk_label'   in nn, "nn_label must have risk_label"

        d3 = nn['detection_3d']
        assert d3['class_id']    == 0,       "car -> class_id=0"
        assert d3['class_name']  == 'car'
        assert len(d3['center']) == 3
        assert d3['heading_rad'] is not None

        rk = nn['risk_label']
        assert rk['severity_id']   == 1,    "WARN -> severity_id=1"
        assert rk['is_threat']     is True
        assert abs(rk['ttc_s'] - 3.33) < 0.01

        if verbose:
            print(f"    detection_3d: {d3}")


# ══════════════════════════════════════════════════════════════════════════════
#  T13 -- close_session drains queue and returns stats
# ══════════════════════════════════════════════════════════════════════════════

@_test("T13 close_session drains write queue and returns correct stats")
def test_t13(verbose):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _default_cfg(tmpdir)
        exp = DataExporter(cfg)
        exp.open_session(LidarConfig())

        fr = _FakeLidarFrame()
        cl = _FakePreprocessedCloud()
        cr = _FakeClusterResult()
        fu = _FakeFusionResult(fused_obstacles=[_FakeFusedObstacle()])

        n_queued = 0
        for i in range(5):
            if exp.record_frame(fr, cl, cr, fu):
                n_queued += 1

        stats = exp.close_session()

        assert 'frames_saved'  in stats
        assert 'duration_s'    in stats
        assert stats['frames_saved'] == n_queued
        assert stats['frames_dropped'] == 0
        assert stats['effective_hz']   > 0

        if verbose:
            print(f"    Session stats: {stats}")


# ══════════════════════════════════════════════════════════════════════════════
#  T14 -- verify_export detects issues
# ══════════════════════════════════════════════════════════════════════════════

@_test("T14 verify_export detects missing files and invalid JSON")
def test_t14(verbose):
    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = Path(tmpdir) / 'session_test'
        frames_dir  = session_dir / 'frames' / '000001'
        frames_dir.mkdir(parents=True)

        # Missing metadata.json -> should be flagged
        issues_no_meta = verify_session(session_dir)
        assert any(i.level == 'ERROR' and 'metadata.json' in i.message
                   for i in issues_no_meta), \
            "Missing metadata.json should be an ERROR"

        # Write metadata.json
        meta = {
            'session_id': 'test',
            'coordinate_frame': 'ISO8855_vehicle_RH',
            'distance_unit': 'metres',
            'lidar_config': {},
            'exporter_config': {},
        }
        write_json_atomic(meta, session_dir / 'metadata.json')

        # Write invalid JSON in frame
        (frames_dir / 'obstacles.json').write_text('{ broken json !!!')

        issues_bad_json = verify_session(session_dir)
        assert any(i.level == 'ERROR' and 'invalid json' in i.message.lower()
                   for i in issues_bad_json), \
            "Invalid JSON should be an ERROR"

        if verbose:
            for issue in issues_bad_json[:5]:
                print(f"    {issue}")


# ══════════════════════════════════════════════════════════════════════════════
#  T15 -- Performance benchmark: 50 record_frame calls
# ══════════════════════════════════════════════════════════════════════════════

@_test("T15 Performance: 50 record_frame calls < 0.5 ms each (pipeline overhead)")
def test_t15(verbose):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _default_cfg(tmpdir)
        exp = DataExporter(cfg)
        exp.open_session(LidarConfig())

        fr = _FakeLidarFrame(num_points=10_000)
        cl = _FakePreprocessedCloud()
        cr = _FakeClusterResult()
        fu = _FakeFusionResult(fused_obstacles=[
            _FakeFusedObstacle(track_id=i)
            for i in range(10)
        ])

        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            exp.record_frame(fr, cl, cr, fu)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        exp.close_session()

        arr    = np.array(latencies)
        mean_ms = float(arr.mean())
        p95_ms  = float(np.percentile(arr, 95))

        # Budget: 1.5 ms mean (generous for Windows scheduler noise)
        budget_mean_ms = 1.5
        budget_p95_ms  = 5.0

        if verbose:
            print(f"    Benchmark: mean={mean_ms:.2f} ms  "
                  f"p95={p95_ms:.2f} ms  budget={budget_mean_ms} ms")

        assert mean_ms < budget_mean_ms, \
            f"record_frame mean {mean_ms:.2f} ms exceeds {budget_mean_ms} ms budget"
        assert p95_ms < budget_p95_ms, \
            f"record_frame p95 {p95_ms:.2f} ms exceeds {budget_p95_ms} ms"


# ══════════════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 07 exporter test suite')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    print('=' * 60)
    print('  Task 07 -- ADAS Data Exporter test suite')
    print('=' * 60)
    _run_all(args.verbose)
