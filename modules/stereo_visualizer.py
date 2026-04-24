"""
stereo_visualizer.py
────────────────────
Combines the annotated camera view and the stereo disparity map
into a single stacked display window for the FYP demo.

Output layout (1280 × 1080 pixels total):
  TOP    (1280 × 720) — annotated camera with YOLO boxes and distances
  BOTTOM (1280 × 360) — colour-coded disparity map (red=near, blue=far)
"""

import cv2
import numpy as np
from typing import Optional


class StereoVisualizer:
    """
    Builds a two-panel demo display that combines the driving camera view
    (top) with the stereo depth map (bottom) into one 1280×1080 window.

    Designed to be called once per frame from the main loop after both
    the driving agent and stereo estimator have been updated.
    """

    # Fixed output dimensions — chosen so the top panel is the full camera
    # resolution and the bottom panel is exactly half height.
    OUT_W      = 1280
    TOP_H      = 720
    BOTTOM_H   = 360
    OUT_H      = TOP_H + BOTTOM_H   # 1080 total

    # Width of the colour legend bar on the right edge of the bottom panel.
    BAR_W      = 50

    def render(self,
               left_annotated:   np.ndarray,
               disparity_color:  Optional[np.ndarray],
               nearest_distance: Optional[float] = None,
               stereo_active:    bool             = False,
               fps:              float            = 0.0) -> np.ndarray:
        """
        Builds the stacked demo display.

        Args:
            left_annotated:   Camera frame with YOLO boxes already drawn.
                              Expected shape (720, 1280, 3) uint8 BGR.
            disparity_color:  Jet-colourmap depth image from
                              StereoDepthEstimator.get_disparity_colormap().
                              Pass None if stereo is not computing yet.
            nearest_distance: Distance to nearest in-lane obstacle in metres,
                              or None if no obstacle is detected.
            stereo_active:    True when stereo depth is being computed this frame.
            fps:              Current frame rate for the HUD (set 0 to hide).

        Returns:
            BGR image shape (1080, 1280, 3) uint8 — pass to cv2.imshow().
        """
        top_panel    = self._build_top_panel(left_annotated, fps)
        bottom_panel = self._build_bottom_panel(
            disparity_color, nearest_distance, stereo_active
        )
        # Stack vertically — both panels must be OUT_W wide.
        return np.vstack([top_panel, bottom_panel])

    # ────────────────────────────────────────────────────────────────────────
    # PRIVATE PANEL BUILDERS
    # ────────────────────────────────────────────────────────────────────────

    def _build_top_panel(self,
                         left_annotated: np.ndarray,
                         fps: float) -> np.ndarray:
        """
        Returns the TOP panel: the driving camera frame at full resolution.
        Adds an FPS counter in the top-right corner.
        """
        # Guard: resize if the frame is not exactly (TOP_H × OUT_W).
        if left_annotated.shape[:2] != (self.TOP_H, self.OUT_W):
            panel = cv2.resize(left_annotated, (self.OUT_W, self.TOP_H))
        else:
            panel = left_annotated.copy()

        # FPS counter — top-right corner, small and unobtrusive.
        if fps > 0.0:
            label = f"FPS: {fps:.1f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            x = self.OUT_W - tw - 10
            cv2.putText(panel, label, (x, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return panel

    def _build_bottom_panel(self,
                             disparity_color:  Optional[np.ndarray],
                             nearest_distance: Optional[float],
                             stereo_active:    bool) -> np.ndarray:
        """
        Returns the BOTTOM panel (1280 × 360):
          - Colour-coded depth map, or a placeholder if stereo is inactive.
          - Vertical colour legend bar on the right edge.
          - Text overlays for distance and stereo status.
        """
        panel = self._make_depth_background(disparity_color)
        self._draw_colorbar(panel)
        self._draw_panel_labels(panel, nearest_distance, stereo_active)
        return panel

    def _make_depth_background(self,
                                disparity_color: Optional[np.ndarray]
                                ) -> np.ndarray:
        """
        Creates the 1280×360 depth background.
        If disparity_color is provided: resize it.
        Otherwise: dark grey placeholder with centred text.
        """
        if disparity_color is not None:
            # Resize the depth image to fit the bottom panel.
            bg = cv2.resize(disparity_color, (self.OUT_W, self.BOTTOM_H))
        else:
            # Right camera not active — informative placeholder.
            bg = np.full((self.BOTTOM_H, self.OUT_W, 3), 40, dtype=np.uint8)
            msg = "STEREO NOT COMPUTING  —  right camera inactive"
            (tw, th), _ = cv2.getTextSize(
                msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cx = (self.OUT_W  - tw) // 2
            cy = (self.BOTTOM_H + th) // 2
            cv2.putText(bg, msg, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        return bg

    def _draw_colorbar(self, panel: np.ndarray) -> None:
        """
        Draws a vertical colour legend bar (BAR_W px wide) on the right edge
        of the panel.

        The bar runs from blue (top = far = 50 m) to red (bottom = near = 0 m),
        matching the JET colourmap used in get_disparity_colormap().
        Tick labels are drawn at every 10 m.
        """
        bar_x = self.OUT_W - self.BAR_W   # left edge of the bar

        # Draw a dark background strip so labels are readable.
        cv2.rectangle(panel,
                      (bar_x - 5, 0),
                      (self.OUT_W, self.BOTTOM_H),
                      (20, 20, 20), -1)

        # Build the colour gradient: pixel row 0 (top) = far = blue;
        # pixel row BOTTOM_H (bottom) = near = red.
        # In JET: value 0 → dark blue, value 255 → red.
        # We want top=blue → value 0, bottom=red → value 255.
        bar_values = np.arange(self.BOTTOM_H, dtype=np.uint8)
        bar_values = (bar_values / (self.BOTTOM_H - 1) * 255).astype(np.uint8)
        bar_strip  = bar_values.reshape(-1, 1)                  # (H, 1)
        bar_strip  = np.repeat(bar_strip, self.BAR_W - 10, axis=1)  # (H, W)
        bar_colour = cv2.applyColorMap(bar_strip, cv2.COLORMAP_JET)  # (H, W, 3)

        panel[0:self.BOTTOM_H, bar_x:bar_x + self.BAR_W - 10] = bar_colour

        # Draw tick labels every 10 m.
        # At top (y=0) the label is 50 m; at bottom (y=BOTTOM_H) it is 0 m.
        max_depth_m = 50
        for depth_m in range(0, max_depth_m + 1, 10):
            # Fraction from bottom (0 m) to top (50 m).
            frac  = depth_m / max_depth_m        # 0 → bottom, 1 → top
            y_pos = int((1.0 - frac) * (self.BOTTOM_H - 1))
            label = f"{depth_m}m"
            cv2.putText(panel, label,
                        (bar_x - 38, y_pos + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
            # Small tick line
            cv2.line(panel,
                     (bar_x - 5, y_pos), (bar_x, y_pos),
                     (240, 240, 240), 1)

    def _draw_panel_labels(self,
                            panel:            np.ndarray,
                            nearest_distance: Optional[float],
                            stereo_active:    bool) -> None:
        """
        Draws three text overlays on the bottom panel:
          - Top-left:    panel title / range
          - Bottom-left: nearest obstacle distance (colour = urgency)
          - Bottom-right: STEREO ACTIVE / INACTIVE status
        """
        # ── Top-left title ───────────────────────────────────────────────
        cv2.putText(panel, "DISPARITY MAP  |  0 - 50 m",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # ── Bottom-left: nearest obstacle ────────────────────────────────
        if nearest_distance is not None:
            # Colour encodes urgency: red=close, yellow=medium, green=safe.
            if nearest_distance < 10.0:
                dist_colour = (0, 0, 255)     # red — imminent danger
            elif nearest_distance < 25.0:
                dist_colour = (0, 255, 255)   # yellow — caution
            else:
                dist_colour = (0, 255, 0)     # green — safe distance

            dist_text = f"NEAREST: {nearest_distance:.1f} m"
            cv2.putText(panel, dist_text,
                        (10, self.BOTTOM_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_colour, 2)

        # ── Bottom-right: stereo status ──────────────────────────────────
        if stereo_active:
            st_text   = "STEREO: ACTIVE"
            st_colour = (0, 220, 0)      # green
        else:
            st_text   = "STEREO: INACTIVE"
            st_colour = (0, 0, 220)      # red

        (tw, _), _ = cv2.getTextSize(st_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        x_pos = self.OUT_W - self.BAR_W - tw - 15
        cv2.putText(panel, st_text,
                    (x_pos, self.BOTTOM_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, st_colour, 2)
