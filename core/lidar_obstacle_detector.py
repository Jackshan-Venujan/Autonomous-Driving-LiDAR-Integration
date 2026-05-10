"""
LiDAR Obstacle Detector — DBSCAN clustering on filtered point cloud,
sector-based danger classification, and BEV visualisation canvas.
"""

import math
import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict

try:
    from sklearn.cluster import DBSCAN
    _DBSCAN_AVAILABLE = True
except ImportError:
    _DBSCAN_AVAILABLE = False
    print("⚠️  sklearn not available — LiDAR clustering disabled")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LidarObstacle:
    centroid_x: float        # forward distance (m)  +X = forward
    centroid_y: float        # lateral distance (m)  +Y = right
    centroid_z: float
    distance: float          # horizontal distance sqrt(x²+y²)
    angle_deg: float         # angle from forward axis (+right, -left)
    sector: str              # 'front' | 'side_left' | 'side_right'
    point_count: int
    danger_level: str        # 'drive'|'cautious'|'slow'|'stop'|'emergency_stop'
    bbox_min_x: float = 0.0
    bbox_max_x: float = 0.0
    bbox_min_y: float = 0.0
    bbox_max_y: float = 0.0
    bbox_min_z: float = 0.0  # lowest Z across accumulated point history (obstacle bottom)
    bbox_max_z: float = 0.0  # highest Z across accumulated point history (obstacle top)
    track_id: str = ""       # persistent track ID assigned by LidarObstacleDetector


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

ACTION_PRIORITY = {'drive': 0, 'cautious': 1, 'slow': 2, 'stop': 3, 'emergency_stop': 4}


class LidarObstacleDetector:
    """Clusters a preprocessed point cloud and classifies each cluster by danger level."""

    FRONT_ANGLE_DEG = 30.0   # ±60° from forward = front sector
    SIDE_ANGLE_DEG = 150.0   # 60°–120° = side sectors; >120° = rear (ignored)

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 3,
        base_emergency_dist: float = 3,         # 7.5
        base_stop_dist: float = 5.0,            # 15.0
        base_slow_dist: float = 10.0,           # 20.0
        base_cautious_dist: float = 15.0,       # 15.0
        speed_factor: float = 0.5,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.base_emergency_dist = base_emergency_dist
        self.base_stop_dist = base_stop_dist
        self.base_slow_dist = base_slow_dist
        self.base_cautious_dist = base_cautious_dist
        self.speed_factor = speed_factor

        self._last_obstacles: List[LidarObstacle] = []
        self._last_raw_points: Optional[np.ndarray] = None

        # Temporal filtering state
        self._EMA_ALPHA: float = 0.3          # matches lateral_error_alpha in driving_agent
        self._ANGLE_WINDOW: float = 10.0      # deg tolerance for cross-frame obstacle matching
        self._DEESC_REQUIRED: int = 3         # frames at lower level before de-escalating (150ms at 20Hz)
        self._MAX_STALE: int = 3              # frames before dropping an unmatched track
        self._POINT_HISTORY_LEN: int = 5      # frames of raw cluster points to accumulate per track

        # Track dict keyed by string ID. Each entry holds:
        #   sector, angle_deg, smooth_dist, committed_action,
        #   deesc_candidate, deesc_count, stale,
        #   point_history (deque of np.ndarray — raw XYZ points from last N frames)
        self._tracks: Dict[str, dict] = {}
        self._next_track_id: int = 0

        # Hysteresis state for the overall front action (no-obstacle → drive)
        self._committed_front_action: str = 'drive'
        self._no_front_obs_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_thresholds(self, speed_kmh: float) -> dict:
        extra = max(0.0, speed_kmh) * self.speed_factor
        return {
            'emergency': self.base_emergency_dist + extra * 0.3,
            'stop':      self.base_stop_dist      + extra * 0.5,
            'slow':      self.base_slow_dist      + extra * 0.5,
            'cautious':  self.base_cautious_dist  + extra * 0.5,
        }

    def _classify_danger(self, distance: float, sector: str, thresholds: dict) -> str:
        if sector in ('side_left', 'side_right'):
            return 'cautious'   # sides never trigger stop
        # front sector
        if distance <= thresholds['emergency']:
            return 'emergency_stop'
        if distance <= thresholds['stop']:
            return 'stop'
        if distance <= thresholds['slow']:
            return 'slow'
        if distance <= thresholds['cautious']:
            return 'cautious'
        return 'drive'

    def _make_track_id(self) -> str:
        tid = f"t{self._next_track_id}"
        self._next_track_id += 1
        return tid

    def _match_to_track(self, sector: str, angle_deg: float) -> Optional[str]:
        """Return the ID of the closest track in the same sector within ANGLE_WINDOW, or None.

        Uses angle only for matching — distance is the noisy variable being corrected
        so it must not be used as a match criterion.
        """
        best_id = None
        best_diff = float('inf')
        for tid, track in self._tracks.items():
            if track['sector'] != sector:
                continue
            diff = abs(track['angle_deg'] - angle_deg)
            if diff <= self._ANGLE_WINDOW and diff < best_diff:
                best_diff = diff
                best_id = tid
        return best_id

    def _apply_asymmetric_ema(self, raw_dist: float, prev_smooth: float) -> float:
        """Asymmetric EMA: instant escalation (closer), damped de-escalation (farther).

        If the raw distance is closer than history → accept raw immediately (safety-first).
        If farther → apply EMA damping so a single noisy far reading is absorbed gradually.
        """
        ema = self._EMA_ALPHA * raw_dist + (1.0 - self._EMA_ALPHA) * prev_smooth
        return min(raw_dist, ema)

    def _apply_action_hysteresis(self, track: dict, new_action: str) -> str:
        """Commit danger level changes asymmetrically.

        Escalation (higher danger): apply immediately.
        De-escalation (lower danger): require DEESC_REQUIRED consecutive frames first.
        Mutates track in place. Returns the committed action.
        """
        committed = track['committed_action']

        if ACTION_PRIORITY[new_action] >= ACTION_PRIORITY[committed]:
            track['committed_action'] = new_action
            track['deesc_candidate'] = None
            track['deesc_count'] = 0
        else:
            if track['deesc_candidate'] == new_action:
                track['deesc_count'] += 1
            else:
                track['deesc_candidate'] = new_action
                track['deesc_count'] = 1

            if track['deesc_count'] >= self._DEESC_REQUIRED:
                track['committed_action'] = new_action
                track['deesc_candidate'] = None
                track['deesc_count'] = 0

        return track['committed_action']

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, filtered_points: np.ndarray, vehicle_speed_kmh: float = 0.0) -> List[LidarObstacle]:
        """Run DBSCAN and classify clusters.

        Args:
            filtered_points: (M, 3) numpy array [x, y, z] from LidarProcessor.
            vehicle_speed_kmh: current ego speed for adaptive thresholds.

        Returns:
            List of LidarObstacle (front + side sectors; rear dropped).
        """
        self._last_obstacles = []
        self._last_raw_points = filtered_points

        if not _DBSCAN_AVAILABLE or len(filtered_points) < self.min_samples:
            return self._last_obstacles

        thresholds = self._get_thresholds(vehicle_speed_kmh)

        # Cluster on XY plane only (ignore height for grouping)
        xy = filtered_points[:, :2]
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(xy)

        seen_track_ids: set = set()

        for lbl in set(labels):
            if lbl == -1:
                continue  # noise

            mask = labels == lbl
            cluster = filtered_points[mask]

            cx = float(np.mean(cluster[:, 0]))
            cy = float(np.mean(cluster[:, 1]))
            dist = math.sqrt(cx ** 2 + cy ** 2)
            angle_deg = math.degrees(math.atan2(cy, cx))  # +right / -left

            abs_angle = abs(angle_deg)
            if abs_angle <= self.FRONT_ANGLE_DEG:
                sector = 'front'
            elif abs_angle <= self.SIDE_ANGLE_DEG:
                sector = 'side_right' if angle_deg > 0 else 'side_left'
            else:
                continue  # rear — skip

            # Match cluster to existing track (by sector + angle proximity)
            tid = self._match_to_track(sector, angle_deg)
            if tid is None:
                tid = self._make_track_id()
                self._tracks[tid] = {
                    'sector': sector,
                    'angle_deg': angle_deg,
                    'smooth_dist': dist,
                    'committed_action': 'drive',
                    'deesc_candidate': None,
                    'deesc_count': 0,
                    'stale': 0,
                    'point_history': deque(maxlen=self._POINT_HISTORY_LEN),
                }
            else:
                track = self._tracks[tid]
                track['angle_deg'] = angle_deg
                track['stale'] = 0

            seen_track_ids.add(tid)
            track = self._tracks[tid]

            # Point cloud accumulation: append this frame's points to the sliding window.
            # Stacking accumulated frames gives 3–5× more points → stable centroid geometry
            # and accurate Z extents for future 3D bounding box rendering.
            track['point_history'].append(cluster)
            accumulated = np.vstack(track['point_history'])

            acc_cx = float(np.mean(accumulated[:, 0]))
            acc_cy = float(np.mean(accumulated[:, 1]))
            acc_cz = float(np.mean(accumulated[:, 2]))
            acc_dist = math.sqrt(acc_cx ** 2 + acc_cy ** 2)

            track['smooth_dist'] = self._apply_asymmetric_ema(acc_dist, track['smooth_dist'])
            smoothed_dist = track['smooth_dist']

            acc_min_z = float(np.min(accumulated[:, 2]))
            acc_max_z = float(np.max(accumulated[:, 2]))

            danger_raw = self._classify_danger(smoothed_dist, sector, thresholds)
            danger = self._apply_action_hysteresis(track, danger_raw)

            self._last_obstacles.append(LidarObstacle(
                centroid_x=acc_cx,
                centroid_y=acc_cy,
                centroid_z=acc_cz,
                distance=smoothed_dist,
                angle_deg=angle_deg,
                sector=sector,
                point_count=len(accumulated),
                danger_level=danger,
                bbox_min_x=float(np.min(accumulated[:, 0])),
                bbox_max_x=float(np.max(accumulated[:, 0])),
                bbox_min_y=float(np.min(accumulated[:, 1])),
                bbox_max_y=float(np.max(accumulated[:, 1])),
                bbox_min_z=acc_min_z,
                bbox_max_z=acc_max_z,
                track_id=tid,
            ))

        # Evict stale tracks (point_history is freed automatically on deletion)
        for tid in list(self._tracks.keys()):
            if tid not in seen_track_ids:
                self._tracks[tid]['stale'] += 1
                if self._tracks[tid]['stale'] >= self._MAX_STALE:
                    del self._tracks[tid]

        # Update committed front action with hysteresis for the no-obstacle case
        front_obs = [o for o in self._last_obstacles if o.sector == 'front']
        if front_obs:
            self._no_front_obs_count = 0
            self._committed_front_action = max(
                front_obs, key=lambda o: ACTION_PRIORITY[o.danger_level]
            ).danger_level
        else:
            self._no_front_obs_count += 1
            if self._no_front_obs_count >= self._DEESC_REQUIRED:
                self._committed_front_action = 'drive'

        return self._last_obstacles

    def get_front_action(self) -> str:
        """Worst danger level among front-sector obstacles, with temporal hysteresis."""
        return self._committed_front_action

    def get_side_action(self) -> str:
        """'cautious' if any side obstacle exists, else 'drive'."""
        sides = [o for o in self._last_obstacles if o.sector in ('side_left', 'side_right')]
        return 'cautious' if sides else 'drive'

    def get_nearest_front(self) -> Optional[LidarObstacle]:
        front = [o for o in self._last_obstacles if o.sector == 'front']
        return min(front, key=lambda o: o.distance) if front else None

    # ------------------------------------------------------------------
    # BEV visualisation canvas
    # ------------------------------------------------------------------

    def render_bev(
        self,
        canvas_size: int = 400,
        meters_per_pixel: float = 0.25,
    ) -> np.ndarray:
        """Build a top-down bird's-eye-view image of the point cloud and clusters.

        Origin = ego vehicle at bottom-centre of canvas.
        +X (forward) points upward on the canvas.
        +Y (right) points to the right.

        Returns:
            (canvas_size, canvas_size, 3) uint8 BGR image.
        """
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        cx_px = canvas_size // 2        # pixel column for y=0 (centre)
        ego_row = canvas_size - 20      # pixel row for x=0 (ego vehicle)
        center = (cx_px, ego_row)

        def world_to_px(x_m, y_m):
            """Convert sensor-frame (x=forward, y=right) to canvas pixel."""
            col = int(cx_px + y_m / meters_per_pixel)
            row = int(ego_row - x_m / meters_per_pixel)
            return col, row

        def w2cv(a_world_deg: float) -> float:
            """World angle (0=fwd, +right) → cv2 ellipse angle (0=right, +clockwise-in-image).

            World direction (cos θ, sin θ) maps to image (col=sin θ, row=−cos θ).
            cv2 uses (cos α, sin α) in image coords, so α = θ − 90°.
            """
            return a_world_deg - 90.0

        def arc_poly(start_deg: float, end_deg: float, r_px: int) -> np.ndarray:
            """Filled sector wedge polygon — world-angle convention, 2° step."""
            pts = [center]
            for a in range(int(start_deg), int(end_deg) + 1, 2):
                a_rad = math.radians(a)
                pts.append((
                    max(0, min(canvas_size - 1, cx_px + int(math.sin(a_rad) * r_px))),
                    max(0, min(canvas_size - 1, ego_row - int(math.cos(a_rad) * r_px))),
                ))
            return np.array(pts, np.int32)

        max_r_px = int(50.0 / meters_per_pixel)

        # ── Sector fill overlay (drawn first — lowest layer) ────────────────
        overlay = np.zeros_like(canvas)
        cv2.fillPoly(overlay,
                     [arc_poly(-self.FRONT_ANGLE_DEG, self.FRONT_ANGLE_DEG, max_r_px)],
                     (30, 30, 80))   # dark red tint   — front
        cv2.fillPoly(overlay,
                     [arc_poly(self.FRONT_ANGLE_DEG, self.SIDE_ANGLE_DEG, max_r_px)],
                     (0, 35, 55))    # dark orange tint — side-right
        cv2.fillPoly(overlay,
                     [arc_poly(-self.SIDE_ANGLE_DEG, -self.FRONT_ANGLE_DEG, max_r_px)],
                     (0, 45, 45))    # dark yellow tint — side-left
        cv2.addWeighted(canvas, 1.0, overlay, 0.5, 0, canvas)

        # ── Sector boundary lines at ±FRONT_ANGLE and ±SIDE_ANGLE ───────────
        for a_deg, line_color in [
            (-self.SIDE_ANGLE_DEG,  (40, 70, 70)),    # rear-left  boundary
            (-self.FRONT_ANGLE_DEG, (60, 60, 140)),   # front-left boundary
            ( self.FRONT_ANGLE_DEG, (60, 60, 140)),   # front-right boundary
            ( self.SIDE_ANGLE_DEG,  (40, 70, 70)),    # rear-right boundary
        ]:
            a_rad = math.radians(a_deg)
            end_c = max(0, min(canvas_size - 1, cx_px + int(math.sin(a_rad) * max_r_px)))
            end_r = max(0, min(canvas_size - 1, ego_row - int(math.cos(a_rad) * max_r_px)))
            cv2.line(canvas, center, (end_c, end_r), line_color, 1, cv2.LINE_AA)

        # ── Distance rings — sector-coloured arcs (replace plain circles) ───
        for r_m in (10, 25, 50):
            r_px = int(r_m / meters_per_pixel)
            # Front arc  (world −FRONT → +FRONT, upper portion)
            cv2.ellipse(canvas, center, (r_px, r_px), 0,
                        w2cv(-self.FRONT_ANGLE_DEG), w2cv(self.FRONT_ANGLE_DEG),
                        (80, 80, 200), 1)
            # Side-right arc  (world +FRONT → +SIDE, right side)
            cv2.ellipse(canvas, center, (r_px, r_px), 0,
                        w2cv(self.FRONT_ANGLE_DEG), w2cv(self.SIDE_ANGLE_DEG),
                        (0, 150, 180), 1)
            # Side-left arc  (world −SIDE → −FRONT, left side)
            cv2.ellipse(canvas, center, (r_px, r_px), 0,
                        w2cv(-self.SIDE_ANGLE_DEG), w2cv(-self.FRONT_ANGLE_DEG),
                        (0, 150, 180), 1)
            # Rear arc  (world +SIDE → +240°, lower/ignored sector — very dim)
            cv2.ellipse(canvas, center, (r_px, r_px), 0,
                        w2cv(self.SIDE_ANGLE_DEG), w2cv(360.0 - self.SIDE_ANGLE_DEG),
                        (35, 35, 35), 1)
            # Distance label at horizon (ego_row)
            label_col = cx_px + r_px + 2
            if 0 <= label_col < canvas_size:
                cv2.putText(canvas, f"{r_m}m", (label_col, ego_row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

        # ── Angle annotations on boundary lines ─────────────────────────────
        # ±FRONT_ANGLE° at 15 m along the line (clearly visible in canvas)
        r_ann_front = int(15.0 / meters_per_pixel)
        for a_deg in (-self.FRONT_ANGLE_DEG, self.FRONT_ANGLE_DEG):
            a_rad = math.radians(a_deg)
            lc = max(5, min(canvas_size - 25, cx_px + int(math.sin(a_rad) * r_ann_front)))
            lr = max(8, min(canvas_size - 5,  ego_row - int(math.cos(a_rad) * r_ann_front)))
            cv2.putText(canvas, f"{int(self.FRONT_ANGLE_DEG)}", (lc, lr),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 160), 1)

        # ±SIDE_ANGLE° at 5 m — kept close to ego because the line exits canvas quickly
        r_ann_side = int(5.0 / meters_per_pixel)
        for a_deg in (-self.SIDE_ANGLE_DEG, self.SIDE_ANGLE_DEG):
            a_rad = math.radians(a_deg)
            lc = max(5, min(canvas_size - 25, cx_px + int(math.sin(a_rad) * r_ann_side)))
            lr = max(8, min(canvas_size - 5,  ego_row - int(math.cos(a_rad) * r_ann_side)))
            cv2.putText(canvas, f"{int(self.SIDE_ANGLE_DEG)}", (lc, lr),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (70, 100, 100), 1)

        # Sector name labels at mid-angle of each sector
        r_name_px = int(35.0 / meters_per_pixel)
        for a_deg, name, name_color in [
            (  0, "FRONT",  (100, 100, 200)),
            ( 90, "SIDE R", (0, 150, 180)),
            (-90, "SIDE L", (0, 150, 180)),
        ]:
            a_rad = math.radians(a_deg)
            lc = cx_px + int(math.sin(a_rad) * r_name_px)
            lr = ego_row - int(math.cos(a_rad) * r_name_px)
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            lc = max(2, min(canvas_size - tw - 2, lc - tw // 2))
            lr = max(th + 2, min(canvas_size - 2, lr + th // 2))
            cv2.putText(canvas, name, (lc, lr),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, name_color, 1)

        # --- Raw points (white dots) ---
        if self._last_raw_points is not None and len(self._last_raw_points) > 0:
            for pt in self._last_raw_points[::3]:   # every 3rd point for speed
                col, row = world_to_px(pt[0], pt[1])
                if 0 <= col < canvas_size and 0 <= row < canvas_size:
                    canvas[row, col] = (200, 200, 200)

        # --- Cluster bounding boxes ---
        sector_colors = {
            'front':      (0, 0, 255),    # red
            'side_left':  (0, 215, 255),  # yellow
            'side_right': (0, 140, 255),  # orange
        }
        for obs in self._last_obstacles:
            color = sector_colors.get(obs.sector, (200, 200, 200))
            c1 = world_to_px(obs.bbox_max_x, obs.bbox_min_y)
            c2 = world_to_px(obs.bbox_min_x, obs.bbox_max_y)
            # Clamp to canvas
            x1, y1 = max(0, min(canvas_size - 1, c1[0])), max(0, min(canvas_size - 1, c1[1]))
            x2, y2 = max(0, min(canvas_size - 1, c2[0])), max(0, min(canvas_size - 1, c2[1]))
            if x1 != x2 and y1 != y2:
                cv2.rectangle(canvas, (min(x1, x2), min(y1, y2)),
                              (max(x1, x2), max(y1, y2)), color, 2)
            # Label: distance + danger + track_id
            lx, ly = world_to_px(obs.centroid_x, obs.centroid_y)
            label = f"{obs.distance:.1f}m {obs.danger_level.upper()[:4]}"
            cv2.putText(canvas, label, (max(0, lx - 30), max(10, ly - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
            if obs.track_id:
                cv2.putText(canvas, obs.track_id, (max(0, lx - 30), max(10, ly + 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # --- Ego vehicle rectangle ---
        ev_w, ev_h = 8, 14   # pixels
        ev_c, ev_r = cx_px, ego_row
        cv2.rectangle(canvas,
                      (ev_c - ev_w // 2, ev_r - ev_h),
                      (ev_c + ev_w // 2, ev_r + 4),
                      (255, 120, 0), -1)
        cv2.putText(canvas, "EGO", (ev_c - 10, ev_r + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 120, 0), 1)

        # --- Forward direction arrow ---
        cv2.arrowedLine(canvas, (cx_px, ego_row - 16),
                        (cx_px, ego_row - 36), (180, 180, 180), 1, tipLength=0.4)

        # --- Legend ---
        legend_y = 10
        for sector, color in sector_colors.items():
            cv2.putText(canvas, sector, (5, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            legend_y += 14

        return canvas
