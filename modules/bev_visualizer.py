"""
BEV Visualizer Module
Renders a real-time Bird's Eye View map from LiDAR point cloud data.

Canvas layout (700×700 px, 60 m range by default):
    - Ego vehicle at canvas centre, forward = UP
    - Grid lines every 10 m
    - Distance rings at 10 / 20 / 30 / 50 m
    - Raw point cloud (gray dots)
    - Per-cluster bounding boxes colour-coded by distance
    - Sector dividers and labels (FRONT / REAR / L / R)
    - HUD overlay with nearest obstacle per sector
    - Scale bar

CARLA sensor frame: x = forward, y = right, z = up
BEV canvas mapping : forward → canvas up   (canvas_y decreases)
                     right   → canvas right (canvas_x increases)
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Optional, Tuple


# Distance thresholds → BGR colour
_DIST_COLORS: List[Tuple[float, Tuple[int, int, int]]] = [
    (5.0,  (0,   0,   255)),   # < 5 m  : red
    (10.0, (0,   100, 255)),   # < 10 m : orange
    (20.0, (0,   255, 255)),   # < 20 m : yellow
    (float('inf'), (0, 200, 0)),   # ≥ 20 m : green
]


class BEVVisualizer:
    """Renders a top-down LiDAR obstacle map as an OpenCV BGR image."""

    def __init__(self, range_m: float = 60.0, canvas_size: int = 700):
        self.range_m = range_m
        self.canvas_size = canvas_size
        self.center = canvas_size // 2
        self.scale = canvas_size / (2.0 * range_m)   # pixels per metre

        # Pre-compute ego vehicle rectangle half-sizes (Tesla Model 3: 4.7 × 2.0 m)
        self._ego_half_l = max(2, int(4.7 / 2.0 * self.scale))
        self._ego_half_w = max(1, int(2.0 / 2.0 * self.scale))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        points_xyz: Optional[np.ndarray],
        obstacles: List[Dict],
        fusion: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Build and return a 700×700 BGR BEV image.

        Args:
            points_xyz: (N, 3) ground-removed point cloud in sensor frame,
                        or None when no data is available.
            obstacles:  List of obstacle dicts from LidarProcessor.process().
            fusion:     Optional dict from DrivingAgent.run_lidar_camera_fusion().
                        When provided, draws camera FOV cones and adds YOLO class
                        labels to clusters that were detected by a camera.
        """
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        self._draw_grid(canvas)

        # Camera FOV cones (drawn before points so they are in background)
        if fusion is not None:
            self._draw_camera_fov_cones(canvas, fusion)

        if points_xyz is not None and len(points_xyz) > 0:
            self._draw_points(canvas, points_xyz)

        # Build cluster → camera class mapping from fusion
        cluster_class_map = self._build_cluster_class_map(obstacles, fusion)

        self._draw_clusters(canvas, obstacles, cluster_class_map)
        self._draw_ego(canvas)
        self._draw_hud(canvas, obstacles)
        self._draw_scale_bar(canvas)
        return canvas

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _to_canvas(self, x: float, y: float) -> Tuple[int, int]:
        """Sensor-frame (x=forward, y=right) → canvas pixel (col, row)."""
        col = int(self.center + y * self.scale)
        row = int(self.center - x * self.scale)
        return col, row

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.canvas_size and 0 <= row < self.canvas_size

    # ------------------------------------------------------------------
    # Drawing layers
    # ------------------------------------------------------------------

    def _draw_grid(self, canvas: np.ndarray) -> None:
        c = self.center
        sz = self.canvas_size
        grid_color = (30, 30, 30)

        for d in range(10, int(self.range_m) + 1, 10):
            off = int(d * self.scale)
            cv2.line(canvas, (c + off, 0), (c + off, sz), grid_color, 1)
            cv2.line(canvas, (c - off, 0), (c - off, sz), grid_color, 1)
            cv2.line(canvas, (0, c + off), (sz, c + off), grid_color, 1)
            cv2.line(canvas, (0, c - off), (sz, c - off), grid_color, 1)

        # Axes
        cv2.line(canvas, (c, 0), (c, sz), (40, 40, 40), 1)
        cv2.line(canvas, (0, c), (sz, c), (40, 40, 40), 1)

        # Distance rings with labels
        for d, col in [(10, (0, 70, 0)), (20, (0, 80, 0)), (30, (0, 60, 0)), (50, (0, 50, 0))]:
            r = int(d * self.scale)
            if r < sz // 2:
                cv2.circle(canvas, (c, c), r, col, 1)
                cv2.putText(canvas, f"{d}m",
                            (c + r + 3, c - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (50, 90, 50), 1)

        # Sector dividers at ±45° and ±135° from forward
        for angle_deg in (45, -45, 135, -135):
            rad = math.radians(angle_deg)
            # forward=up → dx=sin(a), dy=-cos(a) in canvas coords
            ex = int(c + math.sin(rad) * sz // 2)
            ey = int(c - math.cos(rad) * sz // 2)
            self._dashed_line(canvas, (c, c), (ex, ey), (45, 45, 45), 1)

        # Sector text
        label_color = (75, 75, 75)
        cv2.putText(canvas, "FRONT",  (c - 22, 16),          cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)
        cv2.putText(canvas, "REAR",   (c - 18, sz - 6),      cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)
        cv2.putText(canvas, "L",      (6,  c + 5),            cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)
        cv2.putText(canvas, "R",      (sz - 16, c + 5),       cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)

    def _dashed_line(
        self,
        canvas: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        dash: int = 8,
        gap: int = 8,
    ) -> None:
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        pos = 0.0
        while pos < length:
            s = (int(pt1[0] + ux * pos), int(pt1[1] + uy * pos))
            e_pos = min(pos + dash, length)
            e = (int(pt1[0] + ux * e_pos), int(pt1[1] + uy * e_pos))
            cv2.line(canvas, s, e, color, thickness)
            pos += dash + gap

    def _draw_points(self, canvas: np.ndarray, points_xyz: np.ndarray) -> None:
        """Draw ground-removed point cloud as dim gray pixels (every 4th point)."""
        for pt in points_xyz[::4]:
            col, row = self._to_canvas(pt[0], pt[1])
            if self._in_bounds(col, row):
                canvas[row, col] = (55, 55, 55)

    def _dist_color(self, distance: float) -> Tuple[int, int, int]:
        for thresh, color in _DIST_COLORS:
            if distance < thresh:
                return color
        return _DIST_COLORS[-1][1]

    def _build_cluster_class_map(
        self,
        obstacles: List[Dict],
        fusion: Optional[Dict],
    ) -> Dict[int, str]:
        """
        Match each LiDAR cluster to a YOLO class label from fusion data.

        Strategy: for each camera detection that has a valid lidar_distance,
        find the cluster whose distance is within 3 m of that lidar_distance.
        Returns {cluster_id: class_label}.
        """
        if fusion is None or not obstacles:
            return {}

        class_map: Dict[int, str] = {}
        tolerance = 3.0  # metres

        for cam_key in ('front', 'rear'):
            cam_data = fusion.get(cam_key, {})
            for det in cam_data.get('detections', []):
                lid_d = det.get('lidar_distance')
                cls   = det.get('class', '')
                if lid_d is None or not cls:
                    continue
                for obs in obstacles:
                    if abs(obs['distance'] - lid_d) < tolerance:
                        if obs['id'] not in class_map:
                            class_map[obs['id']] = cls
                        break

        return class_map

    def _draw_camera_fov_cones(
        self,
        canvas: np.ndarray,
        fusion: Dict,
        front_fov_deg: float = 90.0,
        rear_fov_deg: float  = 120.0,
        cone_range_m: float  = 50.0,
        alpha: float         = 0.15,
    ) -> None:
        """
        Draw semi-transparent FOV cones for front (90°) and rear (120°) cameras.

        Front cone: centred on +x axis (forward), faint blue.
        Rear  cone: centred on -x axis (rearward), faint cyan.
        """
        overlay = canvas.copy()
        c       = self.center
        r_px    = int(cone_range_m * self.scale)

        def _cone_points(centre_angle_deg: float, half_fov_deg: float) -> np.ndarray:
            """Return polygon vertices for a cone on the BEV canvas."""
            pts = [(c, c)]   # apex at ego
            n_steps = 30
            for i in range(n_steps + 1):
                theta_sensor = (
                    centre_angle_deg - half_fov_deg
                    + 2 * half_fov_deg * i / n_steps
                )
                # sensor frame: x=forward, y=right
                # canvas: forward→up (row decreases), right→col increases
                sx = math.cos(math.radians(theta_sensor))  # forward component
                sy = math.sin(math.radians(theta_sensor))  # right component
                col = int(c + sy * r_px)
                row = int(c - sx * r_px)
                pts.append((col, row))
            return np.array(pts, dtype=np.int32)

        # Front camera: looking forward (0°), half-fov = front_fov_deg / 2
        front_pts = _cone_points(0.0, front_fov_deg / 2.0)
        cv2.fillPoly(overlay, [front_pts], (180, 80, 0))   # faint blue-ish

        # Rear camera: looking rearward (180°), half-fov = rear_fov_deg / 2
        rear_pts = _cone_points(180.0, rear_fov_deg / 2.0)
        cv2.fillPoly(overlay, [rear_pts], (140, 120, 0))   # faint cyan-ish

        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

        # Label the cones
        lc = (100, 140, 200)
        cv2.putText(canvas, f"FrontCam {int(front_fov_deg)}deg",
                    (c - 44, c - r_px + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, lc, 1)
        cv2.putText(canvas, f"RearCam {int(rear_fov_deg)}deg",
                    (c - 41, c + r_px - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, lc, 1)

    def _draw_clusters(
        self,
        canvas: np.ndarray,
        obstacles: List[Dict],
        cluster_class_map: Optional[Dict[int, str]] = None,
    ) -> None:
        if cluster_class_map is None:
            cluster_class_map = {}
        for rank, obs in enumerate(obstacles):
            color = self._dist_color(obs['distance'])
            min_x, min_y, max_x, max_y = obs['bbox_3d']

            # Four corners of the 2D bounding box
            corners = [
                self._to_canvas(max_x, min_y),  # far-left
                self._to_canvas(max_x, max_y),  # far-right
                self._to_canvas(min_x, max_y),  # near-right
                self._to_canvas(min_x, min_y),  # near-left
            ]
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

            # Centroid dot
            cc = self._to_canvas(obs['centroid'][0], obs['centroid'][1])
            if self._in_bounds(*cc):
                cv2.circle(canvas, cc, 4, color, -1)

            # Label: "#rank class dist"
            cls_label = cluster_class_map.get(obs['id'], '')
            if cls_label:
                label = f"#{rank} {cls_label} {obs['distance']:.1f}m"
            else:
                label = f"#{rank} {obs['distance']:.1f}m"

            lx = max(4, min(cc[0] + 6, self.canvas_size - 80))
            ly = max(10, min(cc[1] - 5, self.canvas_size - 4))
            cv2.putText(canvas, label,
                        (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (230, 230, 230), 1)

    def _draw_ego(self, canvas: np.ndarray) -> None:
        c = self.center
        hl, hw = self._ego_half_l, self._ego_half_w
        # Filled blue rectangle
        cv2.rectangle(canvas, (c - hw, c - hl), (c + hw, c + hl), (160, 70, 0), -1)
        cv2.rectangle(canvas, (c - hw, c - hl), (c + hw, c + hl), (220, 110, 0), 2)
        # Forward heading arrow
        cv2.arrowedLine(canvas, (c, c), (c, c - hl - 8), (255, 255, 255), 2, tipLength=0.35)

    def _draw_hud(self, canvas: np.ndarray, obstacles: List[Dict]) -> None:
        sector_min: Dict[str, Optional[float]] = {s: None for s in ('front', 'rear', 'left', 'right')}
        for obs in obstacles:
            s = obs['sector']
            if s in sector_min and (sector_min[s] is None or obs['distance'] < sector_min[s]):
                sector_min[s] = obs['distance']

        def fmt(v):
            return f"{v:.1f}m" if v is not None else "---"

        lines = [
            ("LiDAR BEV", (0, 220, 220)),
            (f"Clusters : {len(obstacles)}", (200, 200, 200)),
            (f"Front : {fmt(sector_min['front'])}", (200, 200, 200)),
            (f"Rear  : {fmt(sector_min['rear'])}", (200, 200, 200)),
            (f"Left  : {fmt(sector_min['left'])}", (200, 200, 200)),
            (f"Right : {fmt(sector_min['right'])}", (200, 200, 200)),
        ]

        # Semi-transparent background panel
        panel_h = len(lines) * 18 + 10
        panel_w = 130
        overlay = canvas.copy()
        cv2.rectangle(overlay, (4, 4), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        for i, (text, color) in enumerate(lines):
            cv2.putText(canvas, text, (8, 18 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    def _draw_scale_bar(self, canvas: np.ndarray) -> None:
        bar_m = 10
        bar_px = int(bar_m * self.scale)
        x0 = self.canvas_size - bar_px - 14
        y0 = self.canvas_size - 18
        cv2.line(canvas, (x0, y0), (x0 + bar_px, y0), (170, 170, 170), 2)
        cv2.line(canvas, (x0, y0 - 4), (x0, y0 + 4), (170, 170, 170), 1)
        cv2.line(canvas, (x0 + bar_px, y0 - 4), (x0 + bar_px, y0 + 4), (170, 170, 170), 1)
        cv2.putText(canvas, f"{bar_m}m",
                    (x0 + bar_px // 2 - 10, y0 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)
