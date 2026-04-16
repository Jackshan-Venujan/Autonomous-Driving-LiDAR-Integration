"""
HUD Camera Overlay & Dashboard Renderer — Task 06
===================================================

Renders two Pygame panels:

  Camera panel  (left, camera_panel_width x window_height):
    - CARLA camera image as background
    - Bounding boxes for all obstacles with camera detections
    - Projected LiDAR bbox overlay (orange)
    - Distance + semantic class labels
    - TTC warning text for close obstacles
    - Sector threat arcs at image edges
    - Fullscreen BRAKE alert flash / WARN border pulse

  Dashboard panel  (right panel, lower section):
    - Ego speed readout (large)
    - Ego control readouts (throttle, brake, steer)
    - Severity summary bar (BRAKE/WARN/SAFE counts)
    - Nearest obstacle highlight
    - Top-5 obstacle table
    - Pipeline latency breakdown
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from core.lidar_fusion import FusedObstacle, FusionResult
except ImportError:
    from lidar_fusion import FusedObstacle, FusionResult  # type: ignore

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][HUD] %(message)s',
    datefmt='%H:%M:%S',
)


# ══════════════════════════════════════════════════════════════════════════════
#  HUDRenderer
# ══════════════════════════════════════════════════════════════════════════════

class HUDRenderer:
    """
    Renders the camera overlay panel and numeric dashboard panel.

    Thread safety: all render methods should be called from the display
    thread only. DisplayData is read-only inside render calls.
    """

    def __init__(self, config: 'HUDConfig') -> None:
        self._cfg    = config
        self._width  = config.camera_panel_width
        self._height = config.window_height

        # Image scale state (updated each frame when camera image is present)
        self._img_scale    : float = 1.0
        self._img_y_offset : int   = 0
        self._img_src_w    : int   = 1280   # assumed CARLA camera width
        self._img_src_h    : int   = 720    # assumed CARLA camera height

        # Fonts (initialised after pygame.init())
        self._font_large  : Optional[pygame.font.Font] = None
        self._font_med    : Optional[pygame.font.Font] = None
        self._font_small  : Optional[pygame.font.Font] = None
        self._font_alert  : Optional[pygame.font.Font] = None

        # Surfaces
        self._cam_surface  : Optional[pygame.Surface] = None
        self._dash_surface : Optional[pygame.Surface] = None

    def init_pygame(self) -> None:
        """Initialise fonts and surfaces. Call after pygame.init()."""
        cfg = self._cfg
        self._font_large  = pygame.font.SysFont('monospace', cfg.font_size_large, bold=True)
        self._font_med    = pygame.font.SysFont('monospace', cfg.font_size_medium)
        self._font_small  = pygame.font.SysFont('monospace', cfg.font_size_small)
        self._font_alert  = pygame.font.SysFont('monospace', cfg.font_size_alert, bold=True)
        self._cam_surface  = pygame.Surface((self._width, self._height))
        self._dash_surface = pygame.Surface(
            (cfg.right_panel_width, cfg.dashboard_height)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Camera panel
    # ──────────────────────────────────────────────────────────────────────────

    def render_camera_panel(self, data: 'DisplayData') -> 'pygame.Surface':
        """
        Render the camera image with all obstacle overlays.

        Returns:
            pygame.Surface (camera_panel_width x window_height).
        """
        cfg  = self._cfg
        surf = self._cam_surface
        surf.fill(cfg.colour_bg_main)

        # Camera image (or placeholder)
        if data.camera_image_rgb is not None:
            self._blit_camera_image(surf, data.camera_image_rgb)
        else:
            self._draw_no_camera_placeholder(surf, cfg)

        # Threat sector arcs at image edges
        self._draw_sector_arcs(surf, data.fused_obstacles, cfg)

        # Obstacle bounding boxes + labels
        self._draw_obstacle_overlays(surf, data, cfg)

        # Alert overlays (BRAKE flash, WARN pulse)
        self._draw_alert_overlay(surf, data, cfg)

        return surf

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Dashboard panel
    # ──────────────────────────────────────────────────────────────────────────

    def render_dashboard(self, data: 'DisplayData') -> 'pygame.Surface':
        """
        Render the numeric dashboard panel.

        Returns:
            pygame.Surface (right_panel_width x dashboard_height).
        """
        cfg  = self._cfg
        surf = self._dash_surface
        surf.fill(cfg.colour_bg_dashboard)

        W = cfg.right_panel_width
        y = 10

        def hline(yy: int) -> int:
            pygame.draw.line(surf, cfg.colour_radar_ring, (10, yy), (W - 10, yy), 1)
            return yy + 8

        y = hline(y)

        # ── Ego speed ─────────────────────────────────────────────────────────
        spd_lbl  = self._font_alert.render(f"{data.ego_speed_kmh:.0f}", True,
                                            cfg.colour_text_primary)
        unit_lbl = self._font_med.render("km/h", True, cfg.colour_text_muted)
        surf.blit(spd_lbl,  (W // 2 - spd_lbl.get_width()  // 2, y))
        y += spd_lbl.get_height()
        surf.blit(unit_lbl, (W // 2 - unit_lbl.get_width() // 2, y))
        y += unit_lbl.get_height() + 6

        # ── Ego controls ──────────────────────────────────────────────────────
        ctrl = (f"THR:{data.ego_throttle:.2f}  "
                f"BRK:{data.ego_brake:.2f}  "
                f"STR:{data.ego_steer:+.2f}")
        clbl = self._font_small.render(ctrl, True, cfg.colour_text_muted)
        surf.blit(clbl, (W // 2 - clbl.get_width() // 2, y))
        y += clbl.get_height() + 8

        y = hline(y)

        # ── Severity summary bar ──────────────────────────────────────────────
        y = self._draw_severity_bar(surf, data, cfg, y, W)
        y += 8

        y = hline(y)

        # ── Nearest obstacle ──────────────────────────────────────────────────
        y = self._draw_nearest_obstacle(surf, data, cfg, y, W)
        y += 8

        y = hline(y)

        # ── Top-5 table ───────────────────────────────────────────────────────
        y = self._draw_obstacle_table(surf, data, cfg, y, W)
        y += 4

        y = hline(y)

        # ── Latency breakdown ─────────────────────────────────────────────────
        self._draw_latency_breakdown(surf, data, cfg, y, W)

        return surf

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Camera panel helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _blit_camera_image(
        self,
        surf    : 'pygame.Surface',
        img_rgb : np.ndarray,
    ) -> None:
        """
        Scale and blit camera image onto the camera panel surface.

        Scales image to fill panel width, maintaining aspect ratio.
        Updates self._img_scale and self._img_y_offset for bbox transform.

        Args:
            surf   : Target surface (camera_panel_width x window_height).
            img_rgb: (H, W, 3) uint8 RGB numpy array.
        """
        H, W = img_rgb.shape[:2]
        self._img_src_w = W
        self._img_src_h = H

        target_w = self._width
        target_h = int(H * target_w / W)
        self._img_scale    = target_w / W
        self._img_y_offset = max(0, (target_h - self._height) // 2)

        # Convert numpy (H, W, 3) -> pygame Surface: transpose to (W, H, 3)
        pg_surf = pygame.surfarray.make_surface(img_rgb.transpose(1, 0, 2))
        scaled  = pygame.transform.scale(pg_surf, (target_w, target_h))

        # Crop to panel height (centre-crop if taller)
        surf.blit(scaled, (0, 0), (0, self._img_y_offset, target_w, self._height))

    def _draw_no_camera_placeholder(
        self,
        surf : 'pygame.Surface',
        cfg  : 'HUDConfig',
    ) -> None:
        """Draw 'NO CAMERA' placeholder when no image is available."""
        msg = self._font_alert.render("NO CAMERA", True, cfg.colour_text_muted)
        x   = self._width  // 2 - msg.get_width()  // 2
        y   = self._height // 2 - msg.get_height() // 2
        surf.blit(msg, (x, y))

    def _draw_sector_arcs(
        self,
        surf      : 'pygame.Surface',
        obstacles : List[FusedObstacle],
        cfg       : 'HUDConfig',
    ) -> None:
        """
        Draw coloured line segments at camera image edges indicating threat sectors.

        Layout (8 sectors -> 8 border positions):
          FRONT:        top edge, centre third
          FRONT_LEFT:   top edge, left third
          FRONT_RIGHT:  top edge, right third
          LEFT:         left edge, centre half
          REAR_LEFT:    left edge, lower half
          RIGHT:        right edge, centre half
          REAR_RIGHT:   right edge, lower half
          REAR:         bottom edge, centre third
        """
        order_map = {'BRAKE': 2, 'WARN': 1, 'SAFE': 0}
        sector_worst: Dict[str, str] = {}

        for obs in obstacles:
            s   = obs.sector
            sev = obs.severity
            curr = sector_worst.get(s, 'SAFE')
            if order_map.get(sev, 0) > order_map.get(curr, 0):
                sector_worst[s] = sev

        sev_col = {'BRAKE': cfg.colour_brake, 'WARN': cfg.colour_warn,
                   'SAFE': cfg.colour_safe}

        W = self._width
        H = self._height
        T = cfg.sector_arc_thickness

        # (start_point, end_point) for each sector
        sector_positions: Dict[str, Tuple[Tuple, Tuple]] = {
            'FRONT'      : ((W // 3,     0),     (2 * W // 3, 0)),
            'FRONT_LEFT' : ((0,          0),     (W // 3,     0)),
            'FRONT_RIGHT': ((2 * W // 3, 0),     (W,          0)),
            'LEFT'       : ((0,          H // 4),(0,      3 * H // 4)),
            'REAR_LEFT'  : ((0,      3 * H // 4),(0,          H)),
            'RIGHT'      : ((W,          H // 4),(W,      3 * H // 4)),
            'REAR_RIGHT' : ((W,      3 * H // 4),(W,          H)),
            'REAR'       : ((W // 3,     H),     (2 * W // 3, H)),
        }

        for sector, severity in sector_worst.items():
            if sector not in sector_positions or severity == 'SAFE':
                continue
            colour     = sev_col.get(severity, cfg.colour_unknown)
            start, end = sector_positions[sector]
            pygame.draw.line(surf, colour, start, end, T)

    def _draw_obstacle_overlays(
        self,
        surf : 'pygame.Surface',
        data : 'DisplayData',
        cfg  : 'HUDConfig',
    ) -> None:
        """
        Draw bounding boxes and labels for all obstacles on the camera image.

        Transforms YOLO bbox coordinates (original image pixels) to the scaled
        display coordinates using self._img_scale and self._img_y_offset.

        Draws:
          - Green box   : FULL fusion (YOLO bbox)
          - Blue box    : LIDAR_ONLY (projected LiDAR bbox)
          - Yellow box  : CAMERA_ONLY (YOLO bbox)
          - Orange box  : Projected LiDAR bbox overlay (thin, 1 px)
          - Label       : "{fused_class} {dist}m" above the box
          - TTC warning : below the box when TTC < threshold
        """
        s      = self._img_scale
        y_off  = self._img_y_offset

        def transform(x: float, y: float) -> Tuple[int, int]:
            return int(x * s), int(y * s - y_off)

        for obs in data.fused_obstacles:
            # ── YOLO camera bbox ──────────────────────────────────────────────
            if obs.camera_bbox_xyxy is not None:
                x1, y1, x2, y2 = obs.camera_bbox_xyxy
                dx1, dy1 = transform(x1, y1)
                dx2, dy2 = transform(x2, y2)

                col = {
                    'FULL'       : cfg.colour_full_fusion,
                    'CAMERA_ONLY': cfg.colour_camera_only,
                }.get(obs.fusion_method, cfg.colour_lidar_only)

                if dy2 > 0 and dy1 < self._height and dx2 > 0 and dx1 < self._width:
                    pygame.draw.rect(surf, col,
                                     (dx1, dy1, dx2 - dx1, dy2 - dy1),
                                     cfg.overlay_box_thickness)

                    # Label
                    dist_str = f"{obs.min_distance:.1f}m" if obs.min_distance >= 0 else "?m"
                    label    = f"{obs.fused_class} {dist_str}"
                    lbl      = self._font_small.render(label, True, col)

                    lx = max(0, dx1)
                    ly = max(0, dy1 - lbl.get_height() - 2)
                    self._draw_label_with_bg(surf, lbl, lx, ly, cfg)

                    # TTC warning
                    if (obs.ttc_seconds != float('inf') and
                            obs.ttc_seconds < cfg.overlay_ttc_warn_threshold):
                        ttc_col = (cfg.colour_brake if obs.ttc_seconds < cfg.overlay_ttc_brake_threshold
                                   else cfg.colour_warn)
                        ttc_txt = f"TTC {obs.ttc_seconds:.1f}s"
                        ttc_lbl = self._font_small.render(ttc_txt, True, ttc_col)
                        surf.blit(ttc_lbl, (dx1, min(dy2 + 2, self._height - 15)))

            # ── Projected LiDAR bbox (thin orange overlay) ────────────────────
            if (obs.projected_bbox_xyxy is not None and
                    obs.fusion_method != 'CAMERA_ONLY'):
                px1, py1, px2, py2 = obs.projected_bbox_xyxy
                dpx1, dpy1 = transform(px1, py1)
                dpx2, dpy2 = transform(px2, py2)
                if dpy2 > 0 and dpy1 < self._height:
                    pygame.draw.rect(surf, (255, 140, 0),
                                     (dpx1, dpy1, dpx2 - dpx1, dpy2 - dpy1), 1)

    def _draw_label_with_bg(
        self,
        surf : 'pygame.Surface',
        lbl  : 'pygame.Surface',
        x    : int,
        y    : int,
        cfg  : 'HUDConfig',
    ) -> None:
        """Draw a text label with a semi-transparent dark background rectangle."""
        bg = pygame.Surface((lbl.get_width() + 4, lbl.get_height() + 2),
                             pygame.SRCALPHA)
        bg.fill((0, 0, 0, cfg.overlay_label_bg_alpha))
        surf.blit(bg, (x - 2, y - 1))
        surf.blit(lbl, (x, y))

    def _draw_alert_overlay(
        self,
        surf : 'pygame.Surface',
        data : 'DisplayData',
        cfg  : 'HUDConfig',
    ) -> None:
        """
        Draw alert overlays based on worst severity:

          BRAKE: pulsing red fullscreen overlay + "!! BRAKE !!" text.
                 Alpha oscillates: alpha = brake_alpha_max * |sin(pi * flash_hz * t)|
                 Flash frequency: alert_flash_hz (default 4 Hz, max 6 Hz safe).

          WARN:  pulsing amber border (10 px thick).
                 Alpha oscillates: alpha = 100 * |sin(pi * pulse_hz * t)|
        """
        t   = time.time()
        sev = data.worst_severity

        if sev == 'BRAKE':
            alpha = int(cfg.alert_brake_alpha_max *
                        abs(math.sin(math.pi * cfg.alert_flash_hz * t)))
            flash = pygame.Surface((self._width, self._height), pygame.SRCALPHA)
            flash.fill((*cfg.colour_brake, alpha))
            surf.blit(flash, (0, 0))

            # Text
            msg = self._font_alert.render("!! BRAKE !!", True, (255, 255, 255))
            mx  = self._width  // 2 - msg.get_width()  // 2
            my  = self._height // 4 - msg.get_height() // 2
            surf.blit(msg, (mx, my))

        elif sev == 'WARN':
            alpha = int(100 * abs(math.sin(math.pi * cfg.alert_pulse_hz * t)))
            border = pygame.Surface((self._width, self._height), pygame.SRCALPHA)
            pygame.draw.rect(border, (*cfg.colour_warn, alpha),
                             (0, 0, self._width, self._height), 10)
            surf.blit(border, (0, 0))

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Dashboard helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_severity_bar(
        self,
        surf : 'pygame.Surface',
        data : 'DisplayData',
        cfg  : 'HUDConfig',
        y    : int,
        W    : int,
    ) -> int:
        """
        Draw a horizontal severity summary bar: [BRAKE N] [WARN N] [SAFE N].

        The bar width is proportional to obstacle count.
        Returns the updated y position after the bar.
        """
        total = max(data.n_obstacles_total, 1)
        bar_w = W - 20   # usable bar width
        bar_h = 16
        x     = 10

        # Draw total count label
        cnt_lbl = self._font_small.render(
            f"OBS: {data.n_obstacles_total}  |  "
            f"BRAKE:{data.n_brake}  WARN:{data.n_warn}  SAFE:{data.n_safe}",
            True, cfg.colour_text_primary
        )
        surf.blit(cnt_lbl, (W // 2 - cnt_lbl.get_width() // 2, y))
        y += cnt_lbl.get_height() + 4

        # Proportional coloured segments
        segments = [
            (data.n_brake, cfg.colour_brake),
            (data.n_warn,  cfg.colour_warn),
            (data.n_safe,  cfg.colour_safe),
        ]
        cx = x
        for count, col in segments:
            seg_w = int(bar_w * count / total)
            if seg_w > 0:
                pygame.draw.rect(surf, col, (cx, y, seg_w, bar_h))
            cx += seg_w

        # Bar border
        pygame.draw.rect(surf, cfg.colour_text_muted, (x, y, bar_w, bar_h), 1)
        return y + bar_h + 4

    def _draw_nearest_obstacle(
        self,
        surf : 'pygame.Surface',
        data : 'DisplayData',
        cfg  : 'HUDConfig',
        y    : int,
        W    : int,
    ) -> int:
        """
        Draw nearest obstacle highlight with key metrics.

        Shows: type, distance, TTC, sector, severity colour coding.
        Returns updated y after drawing.
        """
        title = self._font_med.render("NEAREST OBSTACLE", True, cfg.colour_text_label)
        surf.blit(title, (W // 2 - title.get_width() // 2, y))
        y += title.get_height() + 4

        obs = data.nearest_obstacle
        if obs is None:
            no_lbl = self._font_small.render("none", True, cfg.colour_text_muted)
            surf.blit(no_lbl, (W // 2 - no_lbl.get_width() // 2, y))
            return y + no_lbl.get_height() + 4

        sev_col = {'BRAKE': cfg.colour_brake, 'WARN': cfg.colour_warn,
                   'SAFE': cfg.colour_safe}.get(obs.severity, cfg.colour_unknown)

        ttc_str = (f"{obs.ttc_seconds:.1f}s"
                   if obs.ttc_seconds != float('inf') else "inf")

        lines = [
            (f"TYPE: {obs.fused_class}", sev_col),
            (f"DIST: {obs.min_distance:.1f} m", sev_col),
            (f"TTC:  {ttc_str}", sev_col),
            (f"SECT: {obs.sector}", cfg.colour_text_muted),
            (f"SEV:  {obs.severity}", sev_col),
        ]
        for text, col in lines:
            lbl = self._font_small.render(text, True, col)
            surf.blit(lbl, (W // 2 - lbl.get_width() // 2, y))
            y += lbl.get_height() + 2

        return y + 2

    def _draw_obstacle_table(
        self,
        surf : 'pygame.Surface',
        data : 'DisplayData',
        cfg  : 'HUDConfig',
        y    : int,
        W    : int,
    ) -> int:
        """
        Draw top-5 nearest obstacles as a compact table.

        Columns: ID | TYPE | DIST | TTC | SEV
        Returns updated y after the table.
        """
        header = self._font_small.render(
            "  ID  TYPE      DIST    TTC   SEV",
            True, cfg.colour_text_label
        )
        surf.blit(header, (4, y))
        y += header.get_height() + 2

        # Table rows (top-5 by min_distance, known distance first)
        known    = [o for o in data.fused_obstacles if o.min_distance >= 0]
        top5     = known[:5]

        for obs in top5:
            sev_col = {'BRAKE': cfg.colour_brake, 'WARN': cfg.colour_warn,
                       'SAFE': cfg.colour_safe}.get(obs.severity, cfg.colour_unknown)
            ttc_str = (f"{obs.ttc_seconds:.1f}" if obs.ttc_seconds != float('inf') else "  --")
            row = (f"  {obs.track_id:3d}  {obs.fused_class:<9s} "
                   f"{obs.min_distance:5.1f}m {ttc_str:>5s}s {obs.severity}")
            row_lbl = self._font_small.render(row, True, sev_col)
            surf.blit(row_lbl, (4, y))
            y += row_lbl.get_height() + 1

        return y + 2

    def _draw_latency_breakdown(
        self,
        surf : 'pygame.Surface',
        data : 'DisplayData',
        cfg  : 'HUDConfig',
        y    : int,
        W    : int,
    ) -> int:
        """
        Draw per-stage pipeline latency with coloured bars.

        Stages: lidar_preproc, clustering, tracking, fusion, total.
        Each bar is scaled to fill the panel width, coloured by latency.
        Returns updated y after drawing.
        """
        ms      = data.pipeline_ms
        stages  = [
            ('PRE',  ms.get('lidar_preproc', 0.0)),
            ('CLU',  ms.get('clustering',    0.0)),
            ('TRK',  ms.get('tracking',      0.0)),
            ('FSN',  ms.get('fusion',        0.0)),
            ('TOT',  ms.get('total',         0.0)),
        ]

        title = self._font_small.render("PIPELINE LATENCY", True, cfg.colour_text_label)
        surf.blit(title, (W // 2 - title.get_width() // 2, y))
        y += title.get_height() + 2

        max_budget = 50.0   # ms — bar scale reference
        bar_w_max  = W - 80

        for name, ms_val in stages:
            col  = (cfg.colour_safe if ms_val < 15
                    else cfg.colour_warn if ms_val < 30
                    else cfg.colour_brake)
            b_px = int(bar_w_max * min(ms_val, max_budget) / max_budget)

            lbl  = self._font_small.render(f"{name}:", True, cfg.colour_text_muted)
            surf.blit(lbl, (4, y + 1))

            pygame.draw.rect(surf, col, (50, y + 2, max(b_px, 1), 10))

            val_lbl = self._font_small.render(f"{ms_val:.0f}ms", True, cfg.colour_text_muted)
            surf.blit(val_lbl, (55 + b_px + 2, y + 1))

            y += 14

        return y + 2
