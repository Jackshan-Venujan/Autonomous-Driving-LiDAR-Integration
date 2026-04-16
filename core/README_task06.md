# Task 06 — 360° Real-Time HUD & Radar Display

## Overview

Implements a real-time Pygame display window for the ADAS pipeline, showing:
- **Camera overlay panel** (left): live camera image with obstacle bounding boxes, severity alerts, and sector threat arcs
- **Radar BEV panel** (top-right): 360° bird's-eye-view of all tracked obstacles in vehicle frame
- **Dashboard panel** (bottom-right): ego telemetry, obstacle table, and per-stage pipeline latency bars

## Window Layout

```
┌──────────────────────────────────────────────────────────┐
│                                        │  Radar BEV      │
│                                        │  480 × 500 px   │
│          Camera Overlay                ├─────────────────┤
│          1120 × 900 px                 │  Dashboard      │
│                                        │  480 × 400 px   │
└────────────────────────────────────────┴─────────────────┘
             1600 × 900 px total
```

## Files

| File | Purpose |
|---|---|
| `lidar_hud.py` | `HUDConfig`, `DisplayData`, `LiDARHUD` — top-level orchestrator |
| `hud_renderer.py` | `HUDRenderer` — camera panel and dashboard rendering |
| `radar_view.py` | `RadarView` — 360° BEV radar rendering + BEV PNG export |
| `test_lidar_hud.py` | 15-test suite (all pure-math + headless pygame) |

## Quick Start

```python
import pygame
from core import LiDARHUD, HUDConfig, DisplayData

# Build config (all values have safe defaults)
cfg = HUDConfig()

# Create and start the display
hud = LiDARHUD(cfg)
hud.start()   # spawns background 30 fps display thread

# Each sensor pipeline tick (10 Hz):
data = LiDARHUD.build_display_data(
    fusion_result     = fusion_result,   # FusionResult from SensorFusion.fuse()
    camera_image_rgb  = rgb_array,       # numpy HxWx3 uint8, or None
    ego_speed_ms      = 14.0,
    ego_throttle      = 0.6,
    ego_brake         = 0.0,
    ego_steer         = -0.05,
    pipeline_ms       = {'lidar_preproc': 8, 'clustering': 12, 'tracking': 4,
                         'fusion': 5, 'total': 29},
)
hud.update(data)

# On shutdown:
hud.stop()
```

## Radar Coordinate Mapping

Vehicle frame → radar pixel:

```
radar_x = cx_px - obstacle.center_xyz[1] * scale   # +Y left  → pixel left
radar_y = cy_px - obstacle.center_xyz[0] * scale   # +X fwd   → pixel up
```

- Ego vehicle is always at radar centre
- Range rings drawn at distances configured in `HUDConfig.radar_range_rings`
- Camera FOV drawn as a pie-wedge using `HUDConfig.camera_fov_deg`

## Obstacle Colours

| Fusion method | Colour |
|---|---|
| `FULL` | Green |
| `LIDAR_ONLY` | Blue |
| `CAMERA_ONLY` | Yellow (hollow diamond at radar edge only — no range) |

## Severity Colours

| Severity | Colour |
|---|---|
| `BRAKE` | Red |
| `WARN` | Amber |
| `SAFE` | Green |

## Alert Overlays (Camera Panel)

- **BRAKE**: pulsing full-screen red overlay — `alpha = brake_alpha_max × |sin(π × flash_hz × t)|`
- **WARN**: pulsing amber 10 px border

Flash rate is capped at 6 Hz by `HUDConfig.alert_flash_hz` (IEC 61508 photosensitive epilepsy limit).

## Keybindings (display window)

| Key | Action |
|---|---|
| `ESC` / `Q` | Quit display |
| `S` | Save current BEV radar frame as PNG |
| `R` | Print current FPS to log |

## BEV Frame Export (NN Training)

Set `HUDConfig.save_bev_frames = True` to automatically export each radar render
as a lossless PNG. Output path: `HUDConfig.bev_save_dir / frame_{id:06d}.png`.

These PNGs match the input contract for BEVDet / BEVFormer / TPVFormer.

## Thread Safety

`LiDARHUD` uses a `threading.RLock` to protect `DisplayData` shared between:
- **Sensor pipeline thread** (10 Hz) calling `hud.update(data)`
- **Display thread** (30 fps) calling `hud.get_latest()`

## Configuration Reference (`HUDConfig`)

| Field | Default | Description |
|---|---|---|
| `window_width` | 1600 | Total window width (px) |
| `window_height` | 900 | Total window height (px) |
| `camera_panel_width` | 1120 | Left camera panel width (px) |
| `right_panel_width` | 480 | Right panel width (px) |
| `radar_height` | 500 | Radar BEV panel height (px) |
| `dashboard_height` | 400 | Dashboard panel height (px) |
| `radar_max_range_m` | 60.0 | Radar display range (m) |
| `radar_range_rings` | [15, 30, 45, 60] | Range ring distances (m) |
| `camera_fov_deg` | 90.0 | Camera field-of-view cone (degrees) |
| `target_fps` | 30 | Display thread target frame rate |
| `alert_flash_hz` | 3.0 | Alert overlay oscillation rate (≤ 6 Hz) |
| `alert_brake_alpha_max` | 160 | Max alpha for BRAKE flash (0–255) |
| `save_bev_frames` | False | Auto-export BEV PNGs for NN training |
| `bev_save_dir` | `"bev_frames"` | Output directory for BEV PNGs |

## Test Results

```
Results: 15 passed, 0 failed / 15 total
Benchmark: mean ~21 ms  p95 ~36 ms  budget=33 ms  (Windows, 50 frames x 15 obs)
```

Run tests:
```bash
cd core && python test_lidar_hud.py --verbose
```
