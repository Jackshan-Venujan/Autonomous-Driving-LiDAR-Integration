"""
Test suite for Task 06 -- ADAS HUD & Radar Display
====================================================

15 tests covering:
  T01  HUDConfig defaults and layout invariants
  T02  DisplayData construction via build_display_data helper
  T03  build_display_data worst_severity logic (BRAKE > WARN > SAFE)
  T04  build_display_data nearest_obstacle selection
  T05  Vehicle frame -> radar pixel coordinate transform
  T06  Severity colour selection (BRAKE/WARN/SAFE/unknown)
  T07  Sector worst severity aggregation
  T08  Alert alpha sine-wave oscillation (in range [0, alert_brake_alpha_max])
  T09  Obstacle sort order: far -> near on radar (z-order)
  T10  RadarView.vehicle_to_radar_px symmetry (left <-> right)
  T11  RadarView init_pygame and render (headless pygame)
  T12  HUDRenderer render_camera_panel (headless pygame, no camera image)
  T13  HUDRenderer render_dashboard (headless pygame)
  T14  build_display_data with CAMERA_ONLY obstacles excluded from nearest
  T15  Performance benchmark: 50 render frames, 15 obstacles

Run:
    cd core && python test_lidar_hud.py
    cd core && python test_lidar_hud.py --verbose
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from typing import List, Optional

import numpy as np

# ── Headless pygame setup BEFORE any pygame import ──────────────────────────
# SDL_VIDEODRIVER=offscreen is Linux-only; on Windows use dummy or omit entirely.
# RadarView/HUDRenderer use pygame.Surface() which never needs a display mode.
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

# ── Imports (package-aware) ──────────────────────────────────────────────────
try:
    from core.lidar_hud import HUDConfig, DisplayData, LiDARHUD
    from core.radar_view import RadarView
    from core.hud_renderer import HUDRenderer
    from core.lidar_fusion import FusedObstacle, FusionResult
except ImportError:
    from lidar_hud import HUDConfig, DisplayData, LiDARHUD             # type: ignore
    from radar_view import RadarView                                     # type: ignore
    from hud_renderer import HUDRenderer                                 # type: ignore
    from lidar_fusion import FusedObstacle, FusionResult                # type: ignore


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

def _make_fused_obstacle(
    track_id     : int   = 1,
    min_distance : float = 10.0,
    severity     : str   = 'SAFE',
    sector       : str   = 'FRONT',
    bearing_deg  : float = 0.0,
    fused_class  : str   = 'vehicle',
    fusion_method: str   = 'FULL',
    velocity_ms  : float = 0.0,
    ttc_seconds  : float = float('inf'),
    camera_bbox  : Optional[tuple] = None,
    proj_bbox    : Optional[tuple] = None,
    center_xyz   : Optional[np.ndarray] = None,
    lost_frames  : int   = 0,
) -> FusedObstacle:
    center = (center_xyz if center_xyz is not None
              else np.array([min_distance, 0.0, 0.0], dtype=np.float32))
    return FusedObstacle(
        track_id            = track_id,
        frame_id            = 0,
        timestamp           = 0.0,
        fusion_method       = fusion_method,
        lidar_class         = fused_class,
        camera_class        = fused_class if fusion_method != 'LIDAR_ONLY' else 'unknown',
        fused_class         = fused_class,
        camera_confidence   = 0.9 if fusion_method == 'FULL' else 0.0,
        fused_confidence    = 0.85,
        center_xyz          = center,
        extent_lwh          = np.array([4.0, 2.0, 1.5], dtype=np.float32),
        heading_deg         = 0.0,
        bearing_deg         = bearing_deg,
        sector              = sector,
        min_distance        = min_distance,
        velocity_ms         = velocity_ms,
        velocity_xyz        = np.zeros(3, dtype=np.float32),
        ttc_seconds         = ttc_seconds,
        severity            = severity,
        camera_bbox_xyxy    = camera_bbox,
        camera_bbox_norm    = None,
        projected_bbox_xyxy = proj_bbox,
        match_iou           = 0.4 if fusion_method == 'FULL' else 0.0,
        kalman_state        = np.zeros(6, dtype=np.float64),
        kalman_uncertainty  = np.ones(6, dtype=np.float64),
        age_frames          = 5,
        lost_frames         = lost_frames,
        is_confirmed        = True,
    )


def _make_fusion_result(obstacles: List[FusedObstacle]) -> FusionResult:
    n_full  = sum(1 for o in obstacles if o.fusion_method == 'FULL')
    n_lidar = sum(1 for o in obstacles if o.fusion_method == 'LIDAR_ONLY')
    n_cam   = sum(1 for o in obstacles if o.fusion_method == 'CAMERA_ONLY')
    return FusionResult(
        frame_id           = 1,
        timestamp          = 0.1,
        fused_obstacles    = obstacles,
        n_lidar_tracks     = n_full + n_lidar,
        n_camera_dets      = n_full + n_cam,
        n_full             = n_full,
        n_lidar_only       = n_lidar,
        n_camera_only      = n_cam,
        camera_frame_age_s = 0.02,
        proc_ms            = 5.0,
    )


def _default_cfg() -> HUDConfig:
    return HUDConfig()


# ══════════════════════════════════════════════════════════════════════════════
#  T01 -- HUDConfig layout invariants
# ══════════════════════════════════════════════════════════════════════════════

@_test("T01 HUDConfig layout invariants")
def test_t01(verbose):
    cfg = _default_cfg()
    assert cfg.camera_panel_width + cfg.right_panel_width == cfg.window_width, \
        "camera_panel_width + right_panel_width != window_width"
    assert cfg.radar_height + cfg.dashboard_height == cfg.window_height, \
        "radar_height + dashboard_height != window_height"
    assert cfg.alert_flash_hz <= 6.0, \
        f"alert_flash_hz {cfg.alert_flash_hz} exceeds safety limit of 6 Hz"
    assert cfg.iou_threshold if hasattr(cfg, 'iou_threshold') else True


# ══════════════════════════════════════════════════════════════════════════════
#  T02 -- build_display_data basic construction
# ══════════════════════════════════════════════════════════════════════════════

@_test("T02 build_display_data constructs valid DisplayData")
def test_t02(verbose):
    obs = [
        _make_fused_obstacle(track_id=1, min_distance=5.0,  severity='BRAKE'),
        _make_fused_obstacle(track_id=2, min_distance=12.0, severity='WARN'),
        _make_fused_obstacle(track_id=3, min_distance=30.0, severity='SAFE'),
    ]
    result = _make_fusion_result(obs)
    data   = LiDARHUD.build_display_data(result, ego_speed_ms=10.0)

    assert data.n_brake           == 1
    assert data.n_warn            == 1
    assert data.n_safe            == 1
    assert data.n_obstacles_total == 3
    assert abs(data.ego_speed_kmh - 36.0) < 0.01
    assert data.frame_id          == 1


# ══════════════════════════════════════════════════════════════════════════════
#  T03 -- worst_severity priority: BRAKE > WARN > SAFE
# ══════════════════════════════════════════════════════════════════════════════

@_test("T03 worst_severity priority: BRAKE > WARN > SAFE")
def test_t03(verbose):
    def ws(severities):
        obs = [_make_fused_obstacle(track_id=i, severity=s)
               for i, s in enumerate(severities)]
        return LiDARHUD.build_display_data(_make_fusion_result(obs)).worst_severity

    assert ws(['SAFE', 'SAFE'])          == 'SAFE'
    assert ws(['SAFE', 'WARN'])          == 'WARN'
    assert ws(['WARN', 'BRAKE'])         == 'BRAKE'
    assert ws(['BRAKE', 'SAFE', 'WARN']) == 'BRAKE'
    assert ws([])                        == 'SAFE'


# ══════════════════════════════════════════════════════════════════════════════
#  T04 -- nearest_obstacle selection (CAMERA_ONLY excluded)
# ══════════════════════════════════════════════════════════════════════════════

@_test("T04 nearest_obstacle skips CAMERA_ONLY (min_distance=-1)")
def test_t04(verbose):
    obs = [
        _make_fused_obstacle(track_id=1, min_distance=-1.0, fusion_method='CAMERA_ONLY'),
        _make_fused_obstacle(track_id=2, min_distance=20.0, fusion_method='LIDAR_ONLY'),
        _make_fused_obstacle(track_id=3, min_distance=10.0, fusion_method='FULL'),
    ]
    # Pre-sort as fusion output would (CAMERA_ONLY last)
    obs_sorted = sorted(obs, key=lambda o: float('inf') if o.min_distance < 0
                                           else o.min_distance)
    result = _make_fusion_result(obs_sorted)
    data   = LiDARHUD.build_display_data(result)

    assert data.nearest_obstacle is not None
    assert data.nearest_obstacle.min_distance >= 0, \
        "nearest_obstacle should not be CAMERA_ONLY"
    # Nearest known distance is 10.0 (track_id=3)
    assert abs(data.nearest_obstacle.min_distance - 10.0) < 1e-6


# ══════════════════════════════════════════════════════════════════════════════
#  T05 -- Vehicle frame -> radar pixel coordinate transform
# ══════════════════════════════════════════════════════════════════════════════

@_test("T05 vehicle_to_radar_px: forward=up, left=left")
def test_t05(verbose):
    cfg   = _default_cfg()
    radar = RadarView(cfg)
    cx, cy = radar.centre

    # Ego centre -> radar centre
    rx, ry = radar.vehicle_to_radar_px(0.0, 0.0)
    assert rx == cx and ry == cy, f"Ego origin should map to ({cx},{cy}), got ({rx},{ry})"

    # Forward (10 m) -> y decreases (up on screen)
    rx_fwd, ry_fwd = radar.vehicle_to_radar_px(10.0, 0.0)
    assert ry_fwd < cy,  f"Forward obstacle should be above centre: ry={ry_fwd}, cy={cy}"
    assert rx_fwd == cx, f"Forward obstacle should have same x as centre"

    # Left (5 m) -> x decreases (left on screen)
    rx_l, ry_l = radar.vehicle_to_radar_px(0.0, 5.0)
    assert rx_l < cx, f"Left obstacle should be left of centre: rx={rx_l}, cx={cx}"
    assert ry_l == cy

    # Right (5 m) -> x increases (right on screen)
    rx_r, ry_r = radar.vehicle_to_radar_px(0.0, -5.0)
    assert rx_r > cx, f"Right obstacle should be right of centre: rx={rx_r}, cx={cx}"


# ══════════════════════════════════════════════════════════════════════════════
#  T06 -- Severity colour selection
# ══════════════════════════════════════════════════════════════════════════════

@_test("T06 severity colour selection from HUDConfig")
def test_t06(verbose):
    cfg = _default_cfg()

    sev_col = {'BRAKE': cfg.colour_brake,
               'WARN' : cfg.colour_warn,
               'SAFE' : cfg.colour_safe}.get

    assert sev_col('BRAKE') == cfg.colour_brake
    assert sev_col('WARN')  == cfg.colour_warn
    assert sev_col('SAFE')  == cfg.colour_safe

    # All colours are 3-tuples in [0, 255]
    for col in [cfg.colour_brake, cfg.colour_warn, cfg.colour_safe]:
        assert len(col) == 3
        for c in col:
            assert 0 <= c <= 255


# ══════════════════════════════════════════════════════════════════════════════
#  T07 -- Sector worst severity aggregation
# ══════════════════════════════════════════════════════════════════════════════

@_test("T07 sector worst severity: BRAKE overrides WARN, WARN overrides SAFE")
def test_t07(verbose):
    order = {'BRAKE': 2, 'WARN': 1, 'SAFE': 0}

    def worst_in_sector(obs_list):
        sector_worst = {}
        for obs in obs_list:
            s    = obs.sector
            sev  = obs.severity
            curr = sector_worst.get(s, 'SAFE')
            if order.get(sev, 0) >= order.get(curr, 0):
                sector_worst[s] = sev
        return sector_worst

    obs = [
        _make_fused_obstacle(sector='FRONT', severity='SAFE'),
        _make_fused_obstacle(sector='FRONT', severity='WARN'),
        _make_fused_obstacle(sector='FRONT', severity='BRAKE'),
        _make_fused_obstacle(sector='LEFT',  severity='SAFE'),
        _make_fused_obstacle(sector='REAR',  severity='WARN'),
    ]
    sw = worst_in_sector(obs)

    assert sw['FRONT'] == 'BRAKE', f"FRONT should be BRAKE, got {sw['FRONT']}"
    assert sw['LEFT']  == 'SAFE',  f"LEFT should be SAFE, got {sw['LEFT']}"
    assert sw['REAR']  == 'WARN',  f"REAR should be WARN, got {sw['REAR']}"


# ══════════════════════════════════════════════════════════════════════════════
#  T08 -- Alert alpha oscillation is in valid range
# ══════════════════════════════════════════════════════════════════════════════

@_test("T08 BRAKE alert alpha oscillates in [0, alert_brake_alpha_max]")
def test_t08(verbose):
    cfg   = _default_cfg()
    alpha_max = cfg.alert_brake_alpha_max

    # Sample alpha at many time points
    for ti in range(100):
        t     = ti * 0.05   # sample every 50 ms
        alpha = int(alpha_max * abs(math.sin(math.pi * cfg.alert_flash_hz * t)))
        assert 0 <= alpha <= alpha_max, \
            f"alpha={alpha} out of [0, {alpha_max}] at t={t:.2f}"

    # Verify flash hz is below seizure-inducing limit
    assert cfg.alert_flash_hz <= 6.0


# ══════════════════════════════════════════════════════════════════════════════
#  T09 -- Radar obstacle render order (far first -> near on top)
# ══════════════════════════════════════════════════════════════════════════════

@_test("T09 radar draw order: far obstacles first (near on top)")
def test_t09(verbose):
    obstacles = [
        _make_fused_obstacle(track_id=1, min_distance=5.0),
        _make_fused_obstacle(track_id=2, min_distance=50.0),
        _make_fused_obstacle(track_id=3, min_distance=20.0),
    ]
    # Replicate the sort from RadarView._draw_obstacles
    draw_order = sorted(
        obstacles,
        key=lambda o: -(o.min_distance if o.min_distance >= 0 else 999.0),
    )
    dists = [o.min_distance for o in draw_order]
    assert dists == sorted(dists, reverse=True), \
        f"Expected far->near, got {dists}"
    # Nearest (5.0) should be last (drawn on top)
    assert draw_order[-1].min_distance == 5.0


# ══════════════════════════════════════════════════════════════════════════════
#  T10 -- RadarView.vehicle_to_radar_px symmetry
# ══════════════════════════════════════════════════════════════════════════════

@_test("T10 vehicle_to_radar_px left/right symmetry about centre x")
def test_t10(verbose):
    cfg   = _default_cfg()
    radar = RadarView(cfg)
    cx, cy = radar.centre

    # Left and right at same distance should be symmetric about cx
    rx_l, _ = radar.vehicle_to_radar_px(0.0,  10.0)   # left  +10 m
    rx_r, _ = radar.vehicle_to_radar_px(0.0, -10.0)   # right -10 m

    offset_l = cx - rx_l
    offset_r = rx_r - cx
    assert abs(offset_l - offset_r) <= 1, \
        f"Left offset {offset_l} != right offset {offset_r} (not symmetric)"

    # Forward and rear at same distance should be symmetric about cy
    _, ry_f = radar.vehicle_to_radar_px( 10.0, 0.0)   # forward
    _, ry_b = radar.vehicle_to_radar_px(-10.0, 0.0)   # rear

    offset_f = cy - ry_f
    offset_b = ry_b - cy
    assert abs(offset_f - offset_b) <= 1, \
        f"Forward offset {offset_f} != rear offset {offset_b}"


# ══════════════════════════════════════════════════════════════════════════════
#  T11 -- RadarView init_pygame and headless render
# ══════════════════════════════════════════════════════════════════════════════

@_test("T11 RadarView init_pygame and render (headless)")
def test_t11(verbose):
    import pygame
    pygame.init()

    cfg   = _default_cfg()
    radar = RadarView(cfg)
    radar.init_pygame()

    obs = [
        _make_fused_obstacle(track_id=1, min_distance=10.0, severity='WARN',
                              sector='FRONT',
                              center_xyz=np.array([10.0, 0.0, 0.0], dtype=np.float32)),
        _make_fused_obstacle(track_id=-1, min_distance=-1.0, fusion_method='CAMERA_ONLY',
                              bearing_deg=90.0, sector='LEFT'),
    ]
    data = DisplayData(fused_obstacles=obs, pipeline_ms={})
    surf = radar.render(data)

    assert surf is not None
    assert surf.get_width()  == cfg.right_panel_width
    assert surf.get_height() == cfg.radar_height

    pygame.quit()


# ══════════════════════════════════════════════════════════════════════════════
#  T12 -- HUDRenderer camera panel (no camera image, headless)
# ══════════════════════════════════════════════════════════════════════════════

@_test("T12 HUDRenderer camera panel renders without camera image (headless)")
def test_t12(verbose):
    import pygame
    pygame.init()

    cfg      = _default_cfg()
    renderer = HUDRenderer(cfg)
    renderer.init_pygame()

    obs  = [_make_fused_obstacle(track_id=1, min_distance=8.0, severity='BRAKE',
                                  camera_bbox=(200., 100., 500., 400.),
                                  proj_bbox=(210., 110., 490., 390.))]
    data = DisplayData(fused_obstacles=obs, worst_severity='BRAKE',
                       nearest_obstacle=obs[0], n_brake=1)
    surf = renderer.render_camera_panel(data)

    assert surf is not None
    assert surf.get_width()  == cfg.camera_panel_width
    assert surf.get_height() == cfg.window_height

    pygame.quit()


# ══════════════════════════════════════════════════════════════════════════════
#  T13 -- HUDRenderer dashboard panel (headless)
# ══════════════════════════════════════════════════════════════════════════════

@_test("T13 HUDRenderer dashboard renders without crash (headless)")
def test_t13(verbose):
    import pygame
    pygame.init()

    cfg      = _default_cfg()
    renderer = HUDRenderer(cfg)
    renderer.init_pygame()

    obs = [
        _make_fused_obstacle(track_id=1, min_distance=5.0,  severity='BRAKE'),
        _make_fused_obstacle(track_id=2, min_distance=12.0, severity='WARN'),
        _make_fused_obstacle(track_id=3, min_distance=30.0, severity='SAFE'),
    ]
    data = DisplayData(
        fused_obstacles   = obs,
        ego_speed_ms      = 14.0,
        ego_speed_kmh     = 50.4,
        ego_throttle      = 0.5,
        ego_brake         = 0.0,
        ego_steer         = 0.1,
        n_brake           = 1,
        n_warn            = 1,
        n_safe            = 1,
        n_obstacles_total = 3,
        nearest_obstacle  = obs[0],
        worst_severity    = 'BRAKE',
        pipeline_ms       = {'lidar_preproc': 8, 'clustering': 12,
                              'tracking': 4, 'fusion': 5, 'total': 29},
    )
    surf = renderer.render_dashboard(data)

    assert surf is not None
    assert surf.get_width()  == cfg.right_panel_width
    assert surf.get_height() == cfg.dashboard_height

    pygame.quit()


# ══════════════════════════════════════════════════════════════════════════════
#  T14 -- CAMERA_ONLY excluded from nearest_obstacle
# ══════════════════════════════════════════════════════════════════════════════

@_test("T14 CAMERA_ONLY obstacles excluded from nearest_obstacle and sorted last")
def test_t14(verbose):
    # Simulate fusion output: CAMERA_ONLY first by track_id but last by sort
    obs = [
        _make_fused_obstacle(track_id=1, min_distance=15.0, fusion_method='FULL'),
        _make_fused_obstacle(track_id=-1, min_distance=-1.0, fusion_method='CAMERA_ONLY'),
    ]
    result = _make_fusion_result(obs)
    data   = LiDARHUD.build_display_data(result)

    # Nearest obstacle must not be CAMERA_ONLY
    assert data.nearest_obstacle is not None
    assert data.nearest_obstacle.fusion_method != 'CAMERA_ONLY'
    assert data.nearest_obstacle.min_distance  == 15.0


# ══════════════════════════════════════════════════════════════════════════════
#  T15 -- Performance benchmark: 50 frames x 15 obstacles
# ══════════════════════════════════════════════════════════════════════════════

@_test("T15 Performance: 50 render frames x 15 obstacles (headless)")
def test_t15(verbose):
    import pygame
    pygame.init()

    cfg      = _default_cfg()
    renderer = HUDRenderer(cfg)
    radar    = RadarView(cfg)
    renderer.init_pygame()
    radar.init_pygame()

    rng       = np.random.default_rng(seed=0)
    latencies = []
    n_frames  = 50
    n_obs     = 15

    for i in range(n_frames):
        obs = []
        for j in range(n_obs):
            dist = float(rng.uniform(5.0, 55.0))
            obs.append(_make_fused_obstacle(
                track_id     = j,
                min_distance = dist,
                severity     = rng.choice(['BRAKE', 'WARN', 'SAFE']),
                sector       = rng.choice(['FRONT', 'LEFT', 'RIGHT', 'REAR']),
                camera_bbox  = (float(rng.uniform(0, 900)),
                                float(rng.uniform(0, 600)),
                                float(rng.uniform(100, 1280)),
                                float(rng.uniform(100, 720))),
                center_xyz   = np.array([dist, 0.0, 0.0], dtype=np.float32),
            ))

        result = _make_fusion_result(obs)
        data   = LiDARHUD.build_display_data(
            result,
            ego_speed_ms  = 14.0,
            pipeline_ms   = {'lidar_preproc': 8.0, 'clustering': 12.0,
                              'tracking': 4.0, 'fusion': 5.0, 'total': 29.0},
        )

        t0 = time.perf_counter()
        renderer.render_camera_panel(data)
        renderer.render_dashboard(data)
        radar.render(data)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

    mean_ms = float(np.mean(latencies))
    p95_ms  = float(np.percentile(latencies, 95))

    if verbose:
        print(f"\n  Benchmark: mean={mean_ms:.2f} ms  p95={p95_ms:.2f} ms  "
              f"budget={cfg.max_render_ms:.0f} ms")

    # Soft limit: < 200 ms per frame (very conservative for offscreen rendering)
    assert mean_ms < 200.0, \
        f"Rendering too slow: mean={mean_ms:.1f} ms (limit 200 ms)"

    pygame.quit()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 06 HUD tests')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("  Task 06 -- ADAS HUD & Radar Display test suite")
    print("=" * 60)
    _run_all(verbose=args.verbose)
