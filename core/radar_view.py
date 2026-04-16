"""
360° Radar / BEV Rendering — Task 06
=======================================

Renders a circular bird's-eye-view radar panel showing all obstacles
around the ego vehicle in real vehicle-frame coordinates. Designed for
Pygame rendering at 30 fps on a 480 × 500 pixel surface.

Coordinate mapping (vehicle frame -> radar pixels):
    Ego centre = radar centre = (cx_px, cy_px)
    +X (forward) = UP    on radar (y decreases)
    +Y (left)    = LEFT  on radar (x decreases)

Formula:
    radar_x = cx_px - obs.center_xyz[1] * scale   # Y-left -> pixel-left
    radar_y = cy_px - obs.center_xyz[0] * scale   # X-fwd  -> pixel-up

WHY Y IS NEGATED:
    Image y increases downward; vehicle X increases forward (maps to UP).
    radar_y = cy - X*scale  and  radar_x = cx - Y*scale.
    This matches standard BEV conventions (nuScenes viewer, Waymo, etc.).

BEV frame export:
    Every rendered frame is saved as PNG when config.save_bev_frames=True.
    These PNGs are direct inputs to BEVDet / BEVFormer / TPVFormer.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    from core.lidar_fusion import FusedObstacle
except ImportError:
    from lidar_fusion import FusedObstacle  # type: ignore

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][HUD] %(message)s',
    datefmt='%H:%M:%S',
)


# ══════════════════════════════════════════════════════════════════════════════
#  RadarView
# ══════════════════════════════════════════════════════════════════════════════

class RadarView:
    """
    360° Bird's Eye View radar display rendered on a Pygame Surface.

    All obstacles are drawn in vehicle-frame coordinates:
        +X forward -> UP on radar
        +Y left    -> LEFT on radar

    Rendering order (back to front):
      1. Background fill
      2. Range rings + labels
      3. Cardinal direction lines
      4. Camera FOV cone
      5. Obstacle bboxes + velocity arrows + labels
      6. Ego vehicle symbol
      7. Severity sector arcs at radar edge
      8. Pipeline latency footer
    """

    SECTOR_NAMES = [
        'FRONT', 'FRONT_LEFT', 'LEFT', 'REAR_LEFT',
        'REAR',  'REAR_RIGHT', 'RIGHT', 'FRONT_RIGHT',
    ]

    # Bearing centre (degrees) of each sector
    _SECTOR_BEARING = {
        'FRONT'      :   0, 'FRONT_LEFT' :  45,
        'LEFT'       :  90, 'REAR_LEFT'  : 135,
        'REAR'       : 180, 'REAR_RIGHT' : 225,
        'RIGHT'      : 270, 'FRONT_RIGHT': 315,
    }

    def __init__(self, config: 'HUDConfig') -> None:
        self._cfg    = config
        self._width  = config.right_panel_width
        self._height = config.radar_height

        # Radar centre pixel within radar surface
        self._cx = self._width  // 2
        self._cy = self._height // 2

        # Pixel scale: pixels per metre (90% of half-height for margins)
        self._scale: float = (
            (min(self._cx, self._cy) * 0.88) / config.radar_max_range_m
        )

        # Initialised in init_pygame()
        self._surface    : Optional[pygame.Surface]    = None
        self._font_large : Optional[pygame.font.Font]  = None
        self._font_med   : Optional[pygame.font.Font]  = None
        self._font_small : Optional[pygame.font.Font]  = None

        logger.info(
            "RadarView initialised | size=%d x %d | range=%.0fm | scale=%.2fpx/m",
            self._width, self._height,
            config.radar_max_range_m, self._scale,
        )

    def init_pygame(self) -> None:
        """
        Initialise Pygame surfaces and fonts. Must be called AFTER pygame.init().
        """
        cfg = self._cfg
        self._surface = pygame.Surface(
            (self._width, self._height), pygame.SRCALPHA
        )
        self._font_large = pygame.font.SysFont('monospace', cfg.font_size_large, bold=True)
        self._font_med   = pygame.font.SysFont('monospace', cfg.font_size_medium)
        self._font_small = pygame.font.SysFont('monospace', cfg.font_size_small)

    def render(self, data: 'DisplayData') -> 'pygame.Surface':
        """
        Render the full 360° radar view for one frame.

        Args:
            data: DisplayData snapshot from sensor pipeline.

        Returns:
            pygame.Surface (right_panel_width x radar_height) ready to blit.
        """
        cfg  = self._cfg
        surf = self._surface
        surf.fill(cfg.colour_bg_radar)

        self._draw_range_rings(surf, cfg)
        self._draw_cardinal_lines(surf, cfg)
        self._draw_camera_fov(surf, cfg)
        self._draw_obstacles(surf, data.fused_obstacles, cfg)
        self._draw_ego(surf, cfg)
        self._draw_sector_severity(surf, data.fused_obstacles, cfg)
        self._draw_fps_counter(surf, data, cfg)

        return surf

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Background elements
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_range_rings(self, surf: 'pygame.Surface', cfg: 'HUDConfig') -> None:
        """
        Draw concentric range rings at configured distances.

        Each ring is a circle centred at ego (cx, cy).
        Label at 3 o'clock position (right side of ring).
        """
        for r_m in cfg.radar_range_rings:
            r_px = int(r_m * self._scale)
            pygame.draw.circle(surf, cfg.colour_radar_ring,
                               (self._cx, self._cy), r_px, 1)
            # Label at 3 o'clock
            lx = self._cx + r_px + 3
            ly = self._cy - 8
            if lx < self._width - 30:
                lbl = self._font_small.render(f"{int(r_m)}m", True,
                                               cfg.colour_text_label)
                surf.blit(lbl, (lx, ly))

    def _draw_cardinal_lines(self, surf: 'pygame.Surface', cfg: 'HUDConfig') -> None:
        """
        Draw cardinal direction grid lines (F/B/L/R) and edge labels.

        F (forward) = UP; B (rear) = DOWN; L (left) = LEFT; R (right) = RIGHT.
        """
        max_r = int(min(self._cx, self._cy) * 0.92)

        directions = [
            ( 0,    -max_r, 'F'),
            (-max_r,  0,   'L'),
            ( max_r,  0,   'R'),
            ( 0,     max_r, 'B'),
        ]
        for dx, dy, label in directions:
            pygame.draw.line(surf, cfg.colour_radar_grid,
                             (self._cx, self._cy),
                             (self._cx + dx, self._cy + dy), 1)
            lbl = self._font_small.render(label, True, cfg.colour_text_label)
            lx  = self._cx + dx - lbl.get_width()  // 2
            ly  = self._cy + dy - lbl.get_height() // 2
            surf.blit(lbl, (lx, ly))

    def _draw_camera_fov(self, surf: 'pygame.Surface', cfg: 'HUDConfig') -> None:
        """
        Draw a faint wedge representing the camera's 90-degree field of view.

        Forward arc from -45° to +45° in vehicle bearing.
        Drawn as a semi-transparent pie-wedge on a temp SRCALPHA surface.
        """
        fov_deg  = 90.0
        half_fov = fov_deg / 2.0
        max_r    = int(min(self._cx, self._cy) * 0.90)
        n_pts    = 20

        points = [(self._cx, self._cy)]
        for i in range(n_pts + 1):
            bearing       = -half_fov + (fov_deg * i / n_pts)
            screen_rad    = math.radians(-bearing - 90)   # vehicle bearing -> screen angle
            px = self._cx + max_r * math.cos(screen_rad)
            py = self._cy + max_r * math.sin(screen_rad)
            points.append((int(px), int(py)))

        fov_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(fov_surf, (60, 100, 160, 18), points)
        surf.blit(fov_surf, (0, 0))

        # FOV edge lines
        for side in [-1, 1]:
            screen_rad = math.radians(-(-side * half_fov) - 90)
            ex = self._cx + max_r * math.cos(screen_rad)
            ey = self._cy + max_r * math.sin(screen_rad)
            pygame.draw.line(surf, (50, 80, 130),
                             (self._cx, self._cy), (int(ex), int(ey)), 1)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Obstacle rendering
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_obstacles(
        self,
        surf      : 'pygame.Surface',
        obstacles : List[FusedObstacle],
        cfg       : 'HUDConfig',
    ) -> None:
        """
        Draw all obstacles on the radar surface, far to near (correct z-order).

        CAMERA_ONLY obstacles are drawn at the radar edge in their bearing.
        LiDAR-backed obstacles are drawn at their 3D positions.
        """
        draw_order = sorted(
            obstacles,
            key=lambda o: -(o.min_distance if o.min_distance >= 0 else 999.0),
        )
        for obs in draw_order:
            self._draw_single_obstacle(surf, obs, cfg)

    def _draw_single_obstacle(
        self,
        surf : 'pygame.Surface',
        obs  : FusedObstacle,
        cfg  : 'HUDConfig',
    ) -> None:
        """
        Draw one obstacle on the radar surface.

        Coordinate transform:
            radar_x = cx - vehicle_y * scale    (left -> left on screen)
            radar_y = cy - vehicle_x * scale    (forward -> up on screen)
        """
        if obs.fusion_method == 'CAMERA_ONLY':
            self._draw_camera_only_indicator(surf, obs, cfg)
            return

        cx_v = float(obs.center_xyz[0])
        cy_v = float(obs.center_xyz[1])
        dist_2d = math.hypot(cx_v, cy_v)

        if dist_2d > cfg.radar_max_range_m:
            return

        # Vehicle frame -> radar pixels
        rx = int(self._cx - cy_v * self._scale)
        ry = int(self._cy - cx_v * self._scale)

        # Severity colour
        sev_col = {
            'BRAKE': cfg.colour_brake,
            'WARN' : cfg.colour_warn,
            'SAFE' : cfg.colour_safe,
        }.get(obs.severity, cfg.colour_unknown)

        alpha = cfg.radar_lost_alpha if obs.lost_frames > 0 else 255

        # ── Bounding box ──────────────────────────────────────────────────────
        # extent_lwh[0]=length (X), extent_lwh[1]=width (Y)
        # On radar: length -> vertical, width -> horizontal
        half_l = max(cfg.radar_min_box_px // 2,
                     int(obs.extent_lwh[0] * self._scale / 2))
        half_w = max(cfg.radar_min_box_px // 2,
                     int(obs.extent_lwh[1] * self._scale / 2))

        box_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        box_rect = pygame.Rect(rx - half_w, ry - half_l, half_w * 2, half_l * 2)
        pygame.draw.rect(box_surf, (*sev_col, alpha), box_rect, 2)
        surf.blit(box_surf, (0, 0))

        # ── Heading line ──────────────────────────────────────────────────────
        heading_rad = math.radians(-obs.heading_deg - 90)
        head_len    = max(half_l, 8)
        hx = rx + int(head_len * math.cos(heading_rad))
        hy = ry + int(head_len * math.sin(heading_rad))
        pygame.draw.line(surf, (*cfg.colour_text_muted, alpha), (rx, ry), (hx, hy), 1)

        # ── Velocity arrow (only for approaching obstacles) ───────────────────
        if obs.velocity_ms > 0.3 and dist_2d > 0.1:
            vx_screen = -cy_v / dist_2d
            vy_screen = -cx_v / dist_2d
            arrow_len = int(obs.velocity_ms * cfg.radar_velocity_arrow_scale)
            ax = rx + int(vx_screen * arrow_len)
            ay = ry + int(vy_screen * arrow_len)

            pygame.draw.line(surf, (*cfg.colour_brake, 200), (rx, ry), (ax, ay), 2)
            # Arrowhead
            perp_x = -vy_screen * 3
            perp_y =  vx_screen * 3
            tip_pts = [
                (ax, ay),
                (ax - int(vx_screen * 6 - perp_x), ay - int(vy_screen * 6 - perp_y)),
                (ax - int(vx_screen * 6 + perp_x), ay - int(vy_screen * 6 + perp_y)),
            ]
            pygame.draw.polygon(surf, (*cfg.colour_brake, 200), tip_pts)

        # ── Distance label ────────────────────────────────────────────────────
        lbl = self._font_small.render(f"{obs.min_distance:.0f}m", True, sev_col)
        surf.blit(lbl, (rx + half_w + 2, ry - 6))

        # ── Object type initial ───────────────────────────────────────────────
        char = {'vehicle': 'V', 'pedestrian': 'P',
                'cyclist': 'C', 'structure': 'S'}.get(obs.fused_class, '?')
        t_lbl = self._font_small.render(char, True, (*cfg.colour_text_muted, alpha))
        surf.blit(t_lbl, (rx - half_w - 12, ry - 6))

        # ── Fusion method dot ─────────────────────────────────────────────────
        dot_col = {'FULL': cfg.colour_full_fusion,
                   'LIDAR_ONLY': cfg.colour_lidar_only}.get(
                       obs.fusion_method, cfg.colour_camera_only)
        pygame.draw.circle(surf, dot_col, (rx, ry), 3)

    def _draw_camera_only_indicator(
        self,
        surf : 'pygame.Surface',
        obs  : FusedObstacle,
        cfg  : 'HUDConfig',
    ) -> None:
        """
        Draw a CAMERA_ONLY obstacle as a hollow diamond at the radar edge.

        Placed at 90% of max range in the obstacle's bearing direction.
        Hollow diamond = "detected but distance unknown".
        """
        bearing_rad = math.radians(-obs.bearing_deg - 90)
        edge_r      = int(cfg.radar_max_range_m * self._scale * 0.90)
        ex = self._cx + int(edge_r * math.cos(bearing_rad))
        ey = self._cy + int(edge_r * math.sin(bearing_rad))

        s = 6
        diamond = [(ex, ey - s), (ex + s, ey), (ex, ey + s), (ex - s, ey)]
        pygame.draw.polygon(surf, cfg.colour_camera_only, diamond, 2)

        lbl = self._font_small.render(
            obs.fused_class[0].upper() if obs.fused_class else '?',
            True, cfg.colour_camera_only
        )
        surf.blit(lbl, (ex + s + 2, ey - 6))

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Ego vehicle
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_ego(self, surf: 'pygame.Surface', cfg: 'HUDConfig') -> None:
        """
        Draw ego vehicle symbol at radar centre as a white upward-pointing triangle.
        """
        cx, cy = self._cx, self._cy
        tip   = (cx,     cy - 12)
        left  = (cx - 6, cy + 6)
        right = (cx + 6, cy + 6)
        pygame.draw.polygon(surf, cfg.colour_ego, [tip, left, right])
        pygame.draw.polygon(surf, cfg.colour_bg_radar, [tip, left, right], 1)
        pygame.draw.circle(surf, cfg.colour_bg_radar, (cx, cy), 3)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Severity sector arcs
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_sector_severity(
        self,
        surf      : 'pygame.Surface',
        obstacles : List[FusedObstacle],
        cfg       : 'HUDConfig',
    ) -> None:
        """
        Draw coloured arc segments at the radar edge for worst severity per sector.

        Provides instant 360-degree threat awareness at a glance.
        Only BRAKE and WARN arcs are drawn (SAFE is omitted to reduce clutter).

        Arc formula:
          Vehicle bearing b -> screen angle = 90 - b  (degrees)
          Pygame draw.arc uses CCW radians from +x (right of screen).
        """
        order_map = {'BRAKE': 2, 'WARN': 1, 'SAFE': 0}
        sector_worst: Dict[str, str] = {}

        for obs in obstacles:
            if obs.min_distance < 0:
                continue
            s    = obs.sector
            curr = sector_worst.get(s, 'SAFE')
            if order_map.get(obs.severity, 0) > order_map.get(curr, 0):
                sector_worst[s] = obs.severity

        if not sector_worst:
            return

        r_px = int(min(self._cx, self._cy) * 0.95)
        rect  = pygame.Rect(
            self._cx - r_px, self._cy - r_px, r_px * 2, r_px * 2
        )

        sev_to_col = {
            'BRAKE': cfg.colour_brake,
            'WARN' : cfg.colour_warn,
        }

        for sector, severity in sector_worst.items():
            if severity not in sev_to_col:
                continue
            col      = sev_to_col[severity]
            bear_ctr = self._SECTOR_BEARING.get(sector, 0)
            # Screen angle (CCW from right), converted to radians
            screen_ctr = 90 - bear_ctr
            start_a    = math.radians(screen_ctr - 22.5)
            stop_a     = math.radians(screen_ctr + 22.5)
            # pygame.draw.arc needs start_angle < stop_angle
            if start_a > stop_a:
                start_a, stop_a = stop_a, start_a
            try:
                pygame.draw.arc(surf, col, rect, start_a, stop_a, 5)
            except Exception:
                pass  # degenerate angle — skip silently

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Footer
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_fps_counter(
        self,
        surf : 'pygame.Surface',
        data : 'DisplayData',
        cfg  : 'HUDConfig',
    ) -> None:
        """
        Draw pipeline latency summary at the bottom of the radar surface.
        Format: "PRE:8 CLU:12 TRK:4 FSN:6 TOT:30ms"
        """
        ms  = data.pipeline_ms
        txt = (f"PRE:{ms.get('lidar_preproc', 0):.0f} "
               f"CLU:{ms.get('clustering', 0):.0f} "
               f"TRK:{ms.get('tracking', 0):.0f} "
               f"FSN:{ms.get('fusion', 0):.0f} "
               f"TOT:{ms.get('total', 0):.0f}ms")
        lbl = self._font_small.render(txt, True, cfg.colour_text_muted)
        surf.blit(lbl, (4, self._height - 18))

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: BEV frame export
    # ──────────────────────────────────────────────────────────────────────────

    def save_bev_frame(
        self,
        surface  : 'pygame.Surface',
        frame_id : int,
        save_dir : str,
    ) -> None:
        """
        Save the radar BEV surface as a lossless PNG for NN training.

        Filename: {save_dir}/frame_{frame_id:06d}.png
        Format  : RGB PNG (alpha stripped — BEV nets expect 3-channel input).

        Compatible with: BEVDet, BEVFormer, TPVFormer, BEV-occupancy networks.

        WHY PNG NOT JPEG:
          PNG is lossless.  BEV frames have sharp colour-coded obstacle edges.
          JPEG compression artefacts corrupt these edges and degrade NN accuracy.
        """
        os.makedirs(save_dir, exist_ok=True)
        rgb_surf = pygame.Surface(surface.get_size())
        rgb_surf.blit(surface, (0, 0))
        path = os.path.join(save_dir, f'frame_{frame_id:06d}.png')
        pygame.image.save(rgb_surf, path)

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Pure-Python coordinate helpers (no pygame needed)
    # ──────────────────────────────────────────────────────────────────────────

    def vehicle_to_radar_px(
        self,
        vehicle_x : float,
        vehicle_y : float,
    ) -> Tuple[int, int]:
        """
        Convert vehicle-frame position to radar pixel coordinates.

        Args:
            vehicle_x: Forward distance (m). Positive = ahead.
            vehicle_y: Lateral offset (m). Positive = left.

        Returns:
            (px_x, px_y) radar surface pixel coordinates.
        """
        rx = int(self._cx - vehicle_y * self._scale)
        ry = int(self._cy - vehicle_x * self._scale)
        return rx, ry

    @property
    def scale(self) -> float:
        """Pixels per metre on radar surface."""
        return self._scale

    @property
    def centre(self) -> Tuple[int, int]:
        """Radar centre pixel (cx, cy)."""
        return self._cx, self._cy
