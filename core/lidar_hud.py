"""
ADAS HUD Manager — Task 06
============================

Top-level HUD module. Provides:

  HUDConfig    — all display parameters (window size, colours, FPS)
  DisplayData  — shared snapshot written by pipeline, read by renderer
  LiDARHUD     — Pygame window manager, coordinates RadarView + HUDRenderer

Architecture
------------
  Sensor pipeline thread (10 Hz):
      result  = fusion.fuse(tracked_obs, frame_id, timestamp)
      display = LiDARHUD.build_display_data(result, camera_img, ego_state)
      hud.update(display)

  Display thread (30 fps, managed by LiDARHUD.run_display_loop):
      data = hud.get_latest()        (thread-safe read)
      cam  = renderer.render_camera_panel(data)
      dash = renderer.render_dashboard(data)
      radar= radar_view.render(data)
      blit to screen

Layout
------
  +-------------------------------+----------+
  |                               |  RADAR   |
  |    Camera + Overlay           |  BEV     |
  |    (camera_panel_width wide)  |  480x500 |
  |                               +----------+
  |                               | DASHBOARD|
  |                               |  480x400 |
  +-------------------------------+----------+
  window_width=1600, window_height=900

Keybindings
-----------
  ESC / Q : quit
  S       : save BEV frame immediately
  R       : reset display statistics
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    from core.lidar_fusion import FusedObstacle, FusionResult
    from core.hud_renderer import HUDRenderer
    from core.radar_view   import RadarView
except ImportError:
    from lidar_fusion import FusedObstacle, FusionResult  # type: ignore
    from hud_renderer import HUDRenderer                   # type: ignore
    from radar_view   import RadarView                     # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][HUD] %(message)s',
    datefmt='%H:%M:%S',
)


# ══════════════════════════════════════════════════════════════════════════════
#  HUDConfig
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HUDConfig:
    """
    All display parameters in one frozen, serialisable dataclass.

    Every pixel, colour, and timing threshold is configurable.
    Designed so the CARLA integration script only needs to change HUDConfig
    — no modifications to renderer internals required.
    """

    # ── Window layout ─────────────────────────────────────────────────────────
    window_width        : int   = 1600
    window_height       : int   = 900
    camera_panel_width  : int   = 1120
    right_panel_width   : int   = 480
    radar_height        : int   = 500
    dashboard_height    : int   = 400

    # ── Radar geometry ────────────────────────────────────────────────────────
    radar_max_range_m   : float         = 60.0
    radar_range_rings   : Tuple         = (15.0, 30.0, 45.0, 60.0)

    # ── Target frame rates ────────────────────────────────────────────────────
    display_fps         : int   = 30

    # ── Severity colours (RGB) ────────────────────────────────────────────────
    colour_brake        : Tuple = (220,  50,  50)
    colour_warn         : Tuple = (240, 180,  40)
    colour_safe         : Tuple = ( 50, 200, 110)
    colour_unknown      : Tuple = (150, 150, 150)

    # ── Fusion method colours ─────────────────────────────────────────────────
    colour_full_fusion  : Tuple = ( 50, 200, 110)
    colour_lidar_only   : Tuple = ( 60, 120, 220)
    colour_camera_only  : Tuple = (220, 200,  40)

    # ── Object type colours ───────────────────────────────────────────────────
    colour_vehicle      : Tuple = ( 80, 160, 255)
    colour_pedestrian   : Tuple = (255, 160,  50)
    colour_cyclist      : Tuple = (160, 255, 160)
    colour_structure    : Tuple = (180, 140, 255)
    colour_unknown_type : Tuple = (160, 160, 160)

    # ── Background colours ────────────────────────────────────────────────────
    colour_bg_main      : Tuple = ( 10,  12,  18)
    colour_bg_radar     : Tuple = (  8,  12,  28)
    colour_bg_dashboard : Tuple = ( 12,  16,  24)
    colour_radar_ring   : Tuple = ( 40,  55,  90)
    colour_radar_grid   : Tuple = ( 30,  40,  65)
    colour_ego          : Tuple = (255, 255, 255)

    # ── Text colours ──────────────────────────────────────────────────────────
    colour_text_primary : Tuple = (220, 220, 220)
    colour_text_muted   : Tuple = (130, 130, 140)
    colour_text_label   : Tuple = (100, 120, 160)

    # ── Typography ────────────────────────────────────────────────────────────
    font_size_large     : int   = 20
    font_size_medium    : int   = 15
    font_size_small     : int   = 12
    font_size_alert     : int   = 36

    # ── Camera overlay ────────────────────────────────────────────────────────
    overlay_box_thickness       : int   = 2
    overlay_label_bg_alpha      : int   = 180
    overlay_ttc_warn_threshold  : float = 3.0
    overlay_ttc_brake_threshold : float = 1.5
    sector_arc_thickness        : int   = 6

    # ── Alert system ─────────────────────────────────────────────────────────
    alert_flash_hz              : float = 4.0
    # Max 6 Hz — photosensitive epilepsy safety limit (IEC 61508)
    alert_pulse_hz              : float = 2.0
    alert_brake_alpha_max       : int   = 160

    # ── Radar obstacle rendering ──────────────────────────────────────────────
    radar_min_box_px            : int   = 5
    radar_velocity_arrow_scale  : float = 1.5
    radar_lost_alpha            : int   = 100
    sector_indicator_size       : int   = 20

    # ── BEV frame export for NN ───────────────────────────────────────────────
    save_bev_frames             : bool  = False
    # Disabled by default — enable for NN data collection runs.
    bev_save_path               : str   = 'bev_frames'

    # ── Performance ───────────────────────────────────────────────────────────
    max_render_ms               : float = 33.0
    log_every_n_frames          : int   = 300

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
#  DisplayData
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DisplayData:
    """
    Snapshot of all data needed for one render frame.

    Written by the sensor pipeline (10 Hz) via LiDARHUD.update().
    Read by the display loop (30 fps) via LiDARHUD.get_latest().

    Between LiDAR ticks, the renderer reuses the previous snapshot.
    This decouples render rate (30 fps) from sensor rate (10 Hz).
    """
    frame_id             : int   = 0
    timestamp            : float = 0.0

    # From Task 05 SensorFusion
    fused_obstacles      : List[FusedObstacle]   = field(default_factory=list)
    fusion_result        : Optional[object]       = None

    # Camera image: (H, W, 3) uint8 RGB. None until first camera frame.
    camera_image_rgb     : Optional[np.ndarray]  = None

    # Ego vehicle state
    ego_speed_ms         : float = 0.0
    ego_speed_kmh        : float = 0.0
    ego_heading_deg      : float = 0.0
    ego_throttle         : float = 0.0
    ego_brake            : float = 0.0
    ego_steer            : float = 0.0

    # Pipeline timing (milliseconds per stage)
    pipeline_ms          : Dict[str, float] = field(default_factory=lambda: {
        'lidar_preproc': 0.0,
        'clustering'   : 0.0,
        'tracking'     : 0.0,
        'fusion'       : 0.0,
        'total'        : 0.0,
    })

    # Summary stats (pre-computed for fast rendering)
    n_obstacles_total    : int                       = 0
    n_brake              : int                       = 0
    n_warn               : int                       = 0
    n_safe               : int                       = 0
    nearest_obstacle     : Optional[FusedObstacle]   = None
    worst_severity       : str                        = 'SAFE'
    bev_frame_id         : int                        = 0


# ══════════════════════════════════════════════════════════════════════════════
#  LiDARHUD
# ══════════════════════════════════════════════════════════════════════════════

class LiDARHUD:
    """
    Top-level HUD manager. Owns the Pygame window and coordinates rendering.

    Usage:
        hud = LiDARHUD(HUDConfig())
        hud.start()                    # starts display loop in background thread

        # In sensor pipeline (10 Hz):
        display = LiDARHUD.build_display_data(fusion_result, camera_img, ...)
        hud.update(display)

        # To run in foreground (blocking):
        hud.run_display_loop()
    """

    def __init__(self, config: HUDConfig = HUDConfig()) -> None:
        self._cfg       = config
        self._data      = DisplayData()
        self._lock      = threading.RLock()
        self._running   = False
        self._thread    : Optional[threading.Thread] = None
        self._renderer  = HUDRenderer(config)
        self._radar     = RadarView(config)
        self._frame_count = 0
        self._fps_history : collections.deque = collections.deque(maxlen=60)

        logger.info(
            "LiDARHUD created | %dx%d | fps=%d | save_bev=%s",
            config.window_width, config.window_height,
            config.display_fps, config.save_bev_frames,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Thread-safe data update
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, data: DisplayData) -> None:
        """
        Thread-safe update of the display snapshot from the sensor pipeline.

        Call this at every LiDAR tick (10 Hz) from the pipeline thread.
        The display loop will pick up the latest snapshot on its next frame.

        Args:
            data: DisplayData built from the latest fusion result.
        """
        with self._lock:
            self._data = data

    def get_latest(self) -> DisplayData:
        """
        Thread-safe read of the latest DisplayData snapshot.

        Returns a reference to the current snapshot. Callers must not
        modify the returned object — it may be read by multiple renderers.
        """
        with self._lock:
            return self._data

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the display loop in a background daemon thread.

        The thread exits when stop() is called or the window is closed.
        """
        if self._running:
            logger.warning("LiDARHUD.start() called while already running")
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self.run_display_loop, daemon=True, name='lidar-hud'
        )
        self._thread.start()
        logger.info("LiDARHUD display thread started")

    def stop(self) -> None:
        """
        Signal the display loop to stop and wait for the thread to exit.
        """
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("LiDARHUD stopped")

    @property
    def is_running(self) -> bool:
        """True if the display loop thread is alive."""
        return self._running and (self._thread is not None
                                  and self._thread.is_alive())

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Main display loop
    # ──────────────────────────────────────────────────────────────────────────

    def run_display_loop(self) -> None:
        """
        Pygame display loop at config.display_fps (default 30 fps).

        Rendering pipeline per frame:
          1. Handle pygame events (quit, keybindings)
          2. Read latest DisplayData (thread-safe)
          3. Render camera panel  (HUDRenderer)
          4. Render dashboard     (HUDRenderer)
          5. Render radar BEV     (RadarView)
          6. Composite onto window surface
          7. pygame.display.flip()
          8. (Optional) Save BEV frame to disk
          9. Enforce frame rate via pygame.time.Clock

        Keybindings:
          ESC / Q : quit
          S       : save BEV frame immediately (regardless of save_bev_frames)
          R       : log FPS stats
        """
        if not _PYGAME_AVAILABLE:
            logger.error("pygame not installed — HUD unavailable. pip install pygame")
            return

        cfg = self._cfg
        pygame.init()
        screen = pygame.display.set_mode((cfg.window_width, cfg.window_height))
        pygame.display.set_caption("ADAS LiDAR HUD — Task 06")
        clock = pygame.time.Clock()

        self._renderer.init_pygame()
        self._radar.init_pygame()

        logger.info("LiDARHUD display loop started")

        while self._running:
            # ── Event handling ────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        self._running = False

                    elif event.key == pygame.K_s:
                        data = self.get_latest()
                        radar_s = self._radar.render(data)
                        self._radar.save_bev_frame(
                            radar_s, data.bev_frame_id, cfg.bev_save_path
                        )
                        logger.info("BEV frame %d saved manually", data.bev_frame_id)

                    elif event.key == pygame.K_r:
                        fps = (np.mean(list(self._fps_history))
                               if self._fps_history else 0.0)
                        logger.info("FPS=%.1f  frames=%d", fps, self._frame_count)

            # ── Render ────────────────────────────────────────────────────────
            t_start = time.perf_counter()

            data       = self.get_latest()
            cam_surf   = self._renderer.render_camera_panel(data)
            dash_surf  = self._renderer.render_dashboard(data)
            radar_surf = self._radar.render(data)

            # Composite
            screen.fill(cfg.colour_bg_main)
            screen.blit(cam_surf,   (0, 0))
            screen.blit(radar_surf, (cfg.camera_panel_width, 0))
            screen.blit(dash_surf,  (cfg.camera_panel_width, cfg.radar_height))

            pygame.display.flip()

            render_ms = (time.perf_counter() - t_start) * 1000.0
            if render_ms > cfg.max_render_ms:
                logger.debug("Frame %d: render %.1f ms > budget %.0f ms",
                             self._frame_count, render_ms, cfg.max_render_ms)

            # ── BEV save ──────────────────────────────────────────────────────
            if cfg.save_bev_frames:
                self._radar.save_bev_frame(
                    radar_surf, data.bev_frame_id, cfg.bev_save_path
                )

            # ── Stats ─────────────────────────────────────────────────────────
            self._frame_count += 1
            actual_fps = clock.get_fps()
            if actual_fps > 0:
                self._fps_history.append(actual_fps)

            if self._frame_count % cfg.log_every_n_frames == 0:
                avg_fps = (np.mean(list(self._fps_history))
                           if self._fps_history else 0.0)
                logger.info("[HUD] frame=%d  fps=%.1f  render=%.1f ms",
                            self._frame_count, avg_fps, render_ms)

            clock.tick(cfg.display_fps)

        pygame.quit()
        logger.info("LiDARHUD display loop ended after %d frames", self._frame_count)

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: DisplayData factory helper
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_display_data(
        fusion_result    : FusionResult,
        camera_image_rgb : Optional[np.ndarray] = None,
        ego_speed_ms     : float                = 0.0,
        ego_heading_deg  : float                = 0.0,
        ego_throttle     : float                = 0.0,
        ego_brake        : float                = 0.0,
        ego_steer        : float                = 0.0,
        pipeline_ms      : Optional[Dict[str, float]] = None,
    ) -> DisplayData:
        """
        Build a DisplayData snapshot from pipeline outputs.

        This is the recommended entry point for the sensor pipeline thread.
        Pre-computes summary statistics so the render thread doesn't need to.

        Args:
            fusion_result   : FusionResult from Task 05 SensorFusion.fuse().
            camera_image_rgb: (H, W, 3) uint8 RGB numpy array or None.
            ego_speed_ms    : Ego vehicle speed in m/s.
            ego_heading_deg : Ego heading (degrees, world frame, 0=north).
            ego_throttle    : [0, 1]
            ego_brake       : [0, 1]
            ego_steer       : [-1, 1]
            pipeline_ms     : Dict of stage latencies in milliseconds.

        Returns:
            DisplayData ready to pass to LiDARHUD.update().
        """
        obstacles = fusion_result.fused_obstacles

        n_brake = sum(1 for o in obstacles if o.severity == 'BRAKE')
        n_warn  = sum(1 for o in obstacles if o.severity == 'WARN')
        n_safe  = sum(1 for o in obstacles if o.severity == 'SAFE')

        # Nearest obstacle with known distance (obstacles already sorted by dist)
        nearest: Optional[FusedObstacle] = None
        for o in obstacles:
            if o.min_distance >= 0:
                nearest = o
                break

        worst_sev = 'SAFE'
        if n_brake > 0:
            worst_sev = 'BRAKE'
        elif n_warn > 0:
            worst_sev = 'WARN'

        return DisplayData(
            frame_id          = fusion_result.frame_id,
            timestamp         = fusion_result.timestamp,
            fused_obstacles   = obstacles,
            fusion_result     = fusion_result,
            camera_image_rgb  = camera_image_rgb,
            ego_speed_ms      = ego_speed_ms,
            ego_speed_kmh     = ego_speed_ms * 3.6,
            ego_heading_deg   = ego_heading_deg,
            ego_throttle      = ego_throttle,
            ego_brake         = ego_brake,
            ego_steer         = ego_steer,
            pipeline_ms       = pipeline_ms or {},
            n_obstacles_total = len(obstacles),
            n_brake           = n_brake,
            n_warn            = n_warn,
            n_safe            = n_safe,
            nearest_obstacle  = nearest,
            worst_severity    = worst_sev,
            bev_frame_id      = fusion_result.frame_id,
        )
