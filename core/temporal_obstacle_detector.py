"""
temporal_obstacle_detector.py — 7-stage temporal LiDAR pipeline (0.5–50 m range).

Stages:
  1. Per-frame preprocessing: ROI cone, ego-box, RANSAC ground removal, stat outlier
  2. Ego-motion compensation: vehicle transform stored per frame
  3. Sliding window buffer: 10 frames maintained in EgoMotionBuffer
  4A. Pipeline A (static map): merge + occupancy filter + voxel downsample
  4B. Pipeline B (live frame): current preprocessed frame, no downsampling
  5. DBSCAN clustering on both pipelines independently (Open3D)
  6. Shape classification: wall / noise / obstacle
  7. Static vs dynamic cross-reference: Pipeline B vs Pipeline A centroids

Output: List[TemporalObstacle] — inherits from LidarObstacle for fusion compatibility.
"""

from __future__ import annotations

import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False
    print("⚠️  open3d not available — temporal LiDAR pipeline disabled")

from core.lidar_obstacle_detector import LidarObstacle, ACTION_PRIORITY
from core.ego_motion_buffer import EgoMotionBuffer
import core.lidar_config as cfg


# ---------------------------------------------------------------------------
# Extended data model
# ---------------------------------------------------------------------------

@dataclass
class TemporalObstacle(LidarObstacle):
    """LidarObstacle with temporal-pipeline classification fields.

    label:       'wall' | 'obstacle'  (noise clusters are discarded, never returned)
    motion_type: 'wall' | 'static_obstacle' | 'dynamic'

    Inherits centroid_x/y/z, distance, angle_deg, sector, point_count,
    danger_level, bbox fields from LidarObstacle — fully compatible with
    LidarFusion.fuse() which duck-types on those fields.
    """
    label: str = "obstacle"
    motion_type: str = "static_obstacle"


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class TemporalObstacleDetector:
    """Detects and classifies obstacles using a 10-frame temporal accumulation pipeline."""

    FRONT_ANGLE_DEG = 30.0    # ±30° from forward = front sector
    SIDE_ANGLE_DEG = 150.0    # 30°–150° = side sectors; >150° = rear (dropped)

    def __init__(
        self,
        buffer_size: int = cfg.TEMPORAL_BUFFER_SIZE,
        base_emergency_dist: float = cfg.BASE_EMERGENCY_DIST,
        base_stop_dist: float = cfg.BASE_STOP_DIST,
        base_slow_dist: float = cfg.BASE_SLOW_DIST,
        base_cautious_dist: float = cfg.BASE_CAUTIOUS_DIST,
        speed_factor: float = cfg.SPEED_FACTOR,
    ) -> None:
        self._buffer = EgoMotionBuffer(buffer_size=buffer_size)
        self.base_emergency_dist = base_emergency_dist
        self.base_stop_dist = base_stop_dist
        self.base_slow_dist = base_slow_dist
        self.base_cautious_dist = base_cautious_dist
        self.speed_factor = speed_factor

        self._last_obstacles: List[TemporalObstacle] = []
        self._last_raw_pts: Optional[np.ndarray] = None  # for BEV raw-point overlay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_temporal(
        self,
        raw_xyzI: np.ndarray,
        T_world: np.ndarray,
        timestamp: float,
        vehicle_speed_kmh: float = 0.0,
    ) -> List[TemporalObstacle]:
        """Run the full 7-stage temporal pipeline.

        Args:
            raw_xyzI:          (N, 4) float32 from LidarSensor.get_latest()
            T_world:           (4, 4) float64 = np.array(vehicle.get_transform().get_matrix())
            timestamp:         CARLA data.timestamp for this frame (seconds)
            vehicle_speed_kmh: ego speed for adaptive danger thresholds

        Returns:
            List of TemporalObstacle. Noise clusters are discarded. Rear-sector
            clusters are dropped. Empty if Open3D is unavailable or point count
            is below the DBSCAN minimum.
        """
        self._last_obstacles = []

        if not _O3D_AVAILABLE:
            return self._last_obstacles

        # ── Stage 1: Per-frame preprocessing ──────────────────────────
        pcd = self._preprocess_frame(raw_xyzI)
        # Keep a numpy copy for BEV raw-point overlay
        self._last_raw_pts = (
            np.asarray(pcd.points, dtype=np.float32) if len(pcd.points) > 0 else None
        )
        if len(pcd.points) < cfg.PIPELINE_B_DBSCAN_MIN_POINTS:
            return self._last_obstacles

        # ── Stages 2 & 3: Push to ego-motion buffer ───────────────────
        self._buffer.push(timestamp, pcd, T_world)

        # ── Stage 4A: Build static map (Pipeline A) ───────────────────
        compensated = self._buffer.get_compensated_frames()
        pcd_A = self._build_static_map(compensated)

        # ── Stage 4B: Pipeline B = current preprocessed frame ─────────
        pcd_B = pcd

        # ── Stage 5: DBSCAN on both pipelines independently ───────────
        # Pipeline A (dense, accumulated): strict params for clean static map
        clusters_A = self._cluster_pcd(
            pcd_A,
            eps=cfg.PIPELINE_A_DBSCAN_EPS,
            min_points=cfg.PIPELINE_A_DBSCAN_MIN_POINTS,
        )
        # Pipeline B (sparse, single frame): lenient params so distant objects
        # with 5-15 returns still form clusters
        clusters_B = self._cluster_pcd(
            pcd_B,
            eps=cfg.PIPELINE_B_DBSCAN_EPS,
            min_points=cfg.PIPELINE_B_DBSCAN_MIN_POINTS,
        )

        if not clusters_B:
            return self._last_obstacles

        # ── Stages 6 & 7: Classify and assign motion type ─────────────
        a_non_noise = [
            c for c in clusters_A if self._classify_shape(c) != "noise"
        ]
        a_centroids = [c["centroid"] for c in a_non_noise]

        thresholds = self._get_thresholds(vehicle_speed_kmh)

        for cl in clusters_B:
            shape = self._classify_shape(cl)
            if shape == "noise":
                continue

            motion = self._assign_motion_type(cl["centroid"], a_centroids, shape)
            obs = self._make_obstacle(cl, shape, motion, thresholds)
            if obs is not None:
                self._last_obstacles.append(obs)

        return self._last_obstacles

    # ------------------------------------------------------------------
    # Stage 1 — Per-frame preprocessing
    # ------------------------------------------------------------------

    def _preprocess_frame(self, raw_xyzI: np.ndarray) -> "o3d.geometry.PointCloud":
        """ROI → ego-box → RANSAC ground → statistical outlier → point budget."""
        empty = o3d.geometry.PointCloud()

        if raw_xyzI is None or len(raw_xyzI) == 0:
            return empty

        xyz = raw_xyzI[:, :3].astype(np.float64)

        # 1a. ROI cone filter: range [min, max] AND |angle from forward| ≤ half_angle
        dist_xy = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        angle_deg = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
        roi_mask = (
            (dist_xy >= cfg.TEMPORAL_ROI_MIN_DIST_M) &
            (dist_xy <= cfg.TEMPORAL_ROI_MAX_DIST_M) &
            (np.abs(angle_deg) <= cfg.TEMPORAL_ROI_HALF_ANGLE_DEG)
        )
        xyz = xyz[roi_mask]
        if len(xyz) == 0:
            return empty

        # 1b. Rectangular ego-body filter (removes vehicle self-returns)
        ego_mask = ~(
            (xyz[:, 0] >= cfg.EGO_BOX_X_MIN) & (xyz[:, 0] <= cfg.EGO_BOX_X_MAX) &
            (xyz[:, 1] >= cfg.EGO_BOX_Y_MIN) & (xyz[:, 1] <= cfg.EGO_BOX_Y_MAX) &
            (xyz[:, 2] >= cfg.EGO_BOX_Z_MIN) & (xyz[:, 2] <= cfg.EGO_BOX_Z_MAX)
        )
        xyz = xyz[ego_mask]
        if len(xyz) == 0:
            return empty

        # Build Open3D cloud for the remaining stages
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # 1c. RANSAC ground removal
        _, ground_inliers = pcd.segment_plane(
            distance_threshold=cfg.RANSAC_DISTANCE_THRESHOLD,
            ransac_n=cfg.RANSAC_N,
            num_iterations=cfg.RANSAC_ITERATIONS,
        )
        pcd = pcd.select_by_index(ground_inliers, invert=True)
        if len(pcd.points) == 0:
            return empty

        # 1d. Statistical outlier removal — only when cloud is dense enough for
        # neighbor statistics to be meaningful (skip on sparse frames)
        if len(pcd.points) >= cfg.STAT_OUTLIER_MIN_POINTS:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=cfg.STAT_OUTLIER_NB_NEIGHBORS,
                std_ratio=cfg.STAT_OUTLIER_STD_RATIO,
            )
            if len(pcd.points) == 0:
                return empty

        # 1e. Point budget guard: random subsample if over limit
        n = len(pcd.points)
        if n > cfg.MAX_POINTS_PER_FRAME:
            idx = np.random.choice(n, cfg.MAX_POINTS_PER_FRAME, replace=False)
            pcd = pcd.select_by_index(idx.tolist())

        return pcd

    # ------------------------------------------------------------------
    # Stage 4A — Static map construction
    # ------------------------------------------------------------------

    def _build_static_map(
        self,
        compensated_frames: List[Tuple["o3d.geometry.PointCloud", int]],
    ) -> "o3d.geometry.PointCloud":
        """Merge compensated frames → occupancy filter → voxel downsample."""
        empty = o3d.geometry.PointCloud()

        if not compensated_frames:
            return empty

        all_pts_list: List[np.ndarray] = []
        frame_id_list: List[np.ndarray] = []

        for pcd, frame_idx in compensated_frames:
            pts = np.asarray(pcd.points)
            if len(pts) == 0:
                continue
            all_pts_list.append(pts)
            frame_id_list.append(np.full(len(pts), frame_idx, dtype=np.int32))

        if not all_pts_list:
            return empty

        all_pts = np.vstack(all_pts_list)           # (N_total, 3) float64
        frame_ids = np.concatenate(frame_id_list)   # (N_total,) int32

        persistent_mask = self._occupancy_filter(all_pts, frame_ids)
        persistent_pts = all_pts[persistent_mask]

        if len(persistent_pts) == 0:
            return empty

        pcd_merged = o3d.geometry.PointCloud()
        pcd_merged.points = o3d.utility.Vector3dVector(persistent_pts)
        pcd_merged = pcd_merged.voxel_down_sample(voxel_size=cfg.OCCUPANCY_VOXEL_SIZE_M)
        return pcd_merged

    # ------------------------------------------------------------------
    # Occupancy filter — NumPy void-view hashing (zero Python loops over points)
    # ------------------------------------------------------------------

    @staticmethod
    def _occupancy_filter(
        all_pts: np.ndarray,
        frame_ids: np.ndarray,
    ) -> np.ndarray:
        """Return boolean mask keeping points in voxels present in ≥ min_frame_count frames.

        Algorithm (no Python loops over points — all NumPy C-level):
          1. Quantize each point to a voxel index (ix, iy, iz) via floor division.
          2. Build (voxel, frame_id) pairs as (N, 4) int32.
          3. Unique rows via void-view trick → one row per (voxel, frame) combo.
             Dense clusters in the same voxel+frame collapse to a single row, so
             each voxel is counted at most once per frame regardless of point density.
          4. Strip frame column → count frames per unique voxel.
          5. Build persistent voxel set (frame_count ≥ threshold).
          6. Mask original points via np.isin on voxel void-keys.
        """
        vs = cfg.OCCUPANCY_VOXEL_SIZE_M
        min_fc = cfg.OCCUPANCY_MIN_FRAME_COUNT

        # Step 1: quantize to integer voxel indices
        vox_idx = np.floor(all_pts / vs).astype(np.int32)   # (N, 3)

        # Step 2: (voxel, frame_id) pairs → (N, 4) contiguous int32
        pairs = np.ascontiguousarray(
            np.concatenate([vox_idx, frame_ids[:, np.newaxis]], axis=1)
        )  # (N, 4) int32

        # Step 3: unique (voxel, frame) via void-view (4 × 4 bytes = 16 bytes per row)
        pairs_void = pairs.view(np.dtype((np.void, 16)))
        unique_pairs_void = np.unique(pairs_void)                   # (M,)
        unique_pairs = unique_pairs_void.view(np.int32).reshape(-1, 4)  # (M, 4)

        # Step 4: strip frame column, count frames per unique voxel
        unique_vox = np.ascontiguousarray(unique_pairs[:, :3])      # (M, 3)
        vox_void = unique_vox.view(np.dtype((np.void, 12)))         # 3 × 4 bytes
        _, inv_idx, frame_counts = np.unique(
            vox_void, return_inverse=True, return_counts=True
        )

        # Step 5: build persistent voxel key set
        persistent_mask_vox = frame_counts >= min_fc                # (K,) bool
        persistent_vox_void = np.unique(vox_void[persistent_mask_vox[inv_idx]])

        # Step 6: mask original points
        all_vox_c = np.ascontiguousarray(vox_idx)
        all_vox_void = all_vox_c.view(np.dtype((np.void, 12)))     # (N,)
        return np.isin(all_vox_void.ravel(), persistent_vox_void.ravel())  # (N,) bool

    # ------------------------------------------------------------------
    # Stage 5 — DBSCAN clustering
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_pcd(
        pcd: "o3d.geometry.PointCloud",
        eps: float = cfg.PIPELINE_A_DBSCAN_EPS,
        min_points: int = cfg.PIPELINE_A_DBSCAN_MIN_POINTS,
    ) -> List[Dict]:
        """Run Open3D DBSCAN and return list of cluster dicts.

        Each dict keys: pts (M,3), centroid (3,), dims (3,),
                        bbox_min (3,), bbox_max (3,), point_count int.
        """
        if len(pcd.points) < min_points:
            return []

        pts = np.asarray(pcd.points, dtype=np.float64)
        raw_labels = pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False,
        )
        labels = np.asarray(raw_labels, dtype=np.int32)

        max_lbl = int(labels.max()) if len(labels) > 0 else -1
        clusters: List[Dict] = []

        for lbl in range(max_lbl + 1):   # label -1 = noise, skipped
            mask = labels == lbl
            if not mask.any():
                continue
            cl_pts = pts[mask]
            centroid = cl_pts.mean(axis=0)                     # (3,)
            bbox_min = cl_pts.min(axis=0)                      # (3,)
            bbox_max = cl_pts.max(axis=0)                      # (3,)
            clusters.append({
                "pts": cl_pts,
                "centroid": centroid,
                "dims": bbox_max - bbox_min,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "point_count": int(mask.sum()),
            })

        return clusters

    # ------------------------------------------------------------------
    # Stage 6 — Shape-based classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_shape(cluster: Dict) -> str:
        """Return 'wall', 'noise', or 'obstacle'."""
        dims = cluster["dims"]                          # (dx, dy, dz) float64
        length = float(max(dims[0], dims[1]))           # larger horizontal dim
        width  = float(min(dims[0], dims[1]))           # smaller horizontal dim
        volume = float(dims[0] * dims[1] * dims[2])

        # Wall: very large in any direction, OR long-and-narrow (e.g. jersey barrier)
        if (float(dims.max()) > cfg.WALL_ANY_DIM_M or
                (length > cfg.WALL_LENGTH_M and width < cfg.WALL_WIDTH_M)):
            return "wall"

        # Noise: too few points or vanishingly small volume
        if cluster["point_count"] < cfg.NOISE_MIN_POINTS or volume < cfg.NOISE_MIN_VOLUME_M3:
            return "noise"

        return "obstacle"

    # ------------------------------------------------------------------
    # Stage 7 — Static / dynamic assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_motion_type(
        b_centroid: np.ndarray,
        a_centroids: List[np.ndarray],
        shape_label: str,
    ) -> str:
        """Return 'wall', 'static_obstacle', or 'dynamic'.

        Walls always stay labeled as walls regardless of motion cross-reference.
        If the Pipeline A centroid pool is empty (buffer still warming up), every
        non-wall obstacle is conservatively labeled dynamic.
        """
        if shape_label == "wall":
            return "wall"
        if not a_centroids:
            return "dynamic"

        a_arr = np.array(a_centroids)                       # (K, 3)
        dists = np.linalg.norm(a_arr - b_centroid[np.newaxis, :], axis=1)
        return "static_obstacle" if dists.min() <= cfg.STATIC_MATCH_DIST_M else "dynamic"

    # ------------------------------------------------------------------
    # Danger thresholds and sector assignment
    # ------------------------------------------------------------------

    def _get_thresholds(self, speed_kmh: float) -> Dict:
        extra = max(0.0, speed_kmh) * self.speed_factor
        return {
            "emergency": self.base_emergency_dist + extra * 0.3,
            "stop":      self.base_stop_dist      + extra * 0.5,
            "slow":      self.base_slow_dist      + extra * 0.5,
            "cautious":  self.base_cautious_dist  + extra * 0.5,
        }

    @staticmethod
    def _classify_danger(distance: float, sector: str, thresholds: Dict) -> str:
        if sector in ("side_left", "side_right"):
            return "cautious"
        if distance <= thresholds["emergency"]:
            return "emergency_stop"
        if distance <= thresholds["stop"]:
            return "stop"
        if distance <= thresholds["slow"]:
            return "slow"
        if distance <= thresholds["cautious"]:
            return "cautious"
        return "drive"

    def _make_obstacle(
        self,
        cluster: Dict,
        shape_label: str,
        motion_type: str,
        thresholds: Dict,
    ) -> Optional[TemporalObstacle]:
        """Build a TemporalObstacle from a cluster dict. Returns None for rear-sector clusters."""
        cx, cy, cz = float(cluster["centroid"][0]), float(cluster["centroid"][1]), float(cluster["centroid"][2])
        dist = math.sqrt(cx ** 2 + cy ** 2)
        angle_deg = math.degrees(math.atan2(cy, cx))
        abs_angle = abs(angle_deg)

        if abs_angle <= self.FRONT_ANGLE_DEG:
            sector = "front"
        elif abs_angle <= self.SIDE_ANGLE_DEG:
            sector = "side_right" if angle_deg > 0 else "side_left"
        else:
            return None   # rear sector — drop

        danger = self._classify_danger(dist, sector, thresholds)

        return TemporalObstacle(
            centroid_x=cx,
            centroid_y=cy,
            centroid_z=cz,
            distance=dist,
            angle_deg=angle_deg,
            sector=sector,
            point_count=cluster["point_count"],
            danger_level=danger,
            bbox_min_x=float(cluster["bbox_min"][0]),
            bbox_max_x=float(cluster["bbox_max"][0]),
            bbox_min_y=float(cluster["bbox_min"][1]),
            bbox_max_y=float(cluster["bbox_max"][1]),
            bbox_min_z=float(cluster["bbox_min"][2]),
            bbox_max_z=float(cluster["bbox_max"][2]),
            label=shape_label,
            motion_type=motion_type,
        )

    # ------------------------------------------------------------------
    # Convenience accessors (mirror LidarObstacleDetector interface)
    # ------------------------------------------------------------------

    def get_front_action(self) -> str:
        """Worst danger level among front-sector obstacles."""
        front = [o for o in self._last_obstacles if o.sector == "front"]
        if not front:
            return "drive"
        return max(front, key=lambda o: ACTION_PRIORITY[o.danger_level]).danger_level

    def get_nearest_front(self) -> Optional[TemporalObstacle]:
        front = [o for o in self._last_obstacles if o.sector == "front"]
        return min(front, key=lambda o: o.distance) if front else None

    # ------------------------------------------------------------------
    # BEV visualisation
    # ------------------------------------------------------------------

    def render_bev_temporal(
        self,
        canvas_size: int = 400,
        meters_per_pixel: float = 0.25,
    ) -> np.ndarray:
        """Top-down bird's-eye-view canvas for temporal pipeline obstacles.

        Color legend (BGR):
            wall            → yellow  (0, 215, 255)
            static_obstacle → red     (0,   0, 255)
            dynamic         → cyan    (255, 215,   0)
        """
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        cx_px = canvas_size // 2
        ego_row = canvas_size - 20

        _MOTION_COLORS = {
            "wall":            (0, 215, 255),   # yellow
            "static_obstacle": (0,   0, 255),   # red
            "dynamic":         (255, 215,   0), # cyan
        }

        def world_to_px(x_m: float, y_m: float) -> Tuple[int, int]:
            col = int(cx_px + y_m / meters_per_pixel)
            row = int(ego_row - x_m / meters_per_pixel)
            return col, row

        # Distance rings
        for r_m in (10, 25, 50):
            r_px = int(r_m / meters_per_pixel)
            cv2.circle(canvas, (cx_px, ego_row), r_px, (45, 45, 45), 1)
            lc = cx_px + r_px + 2
            if 0 <= lc < canvas_size:
                cv2.putText(canvas, f"{r_m}m", (lc, ego_row),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1)

        # Sector boundary lines (±30° front cone)
        for a_deg in (-self.FRONT_ANGLE_DEG, self.FRONT_ANGLE_DEG):
            a_rad = math.radians(a_deg)
            max_r_px = int(cfg.TEMPORAL_ROI_MAX_DIST_M / meters_per_pixel)
            end_c = max(0, min(canvas_size - 1, cx_px + int(math.sin(a_rad) * max_r_px)))
            end_r = max(0, min(canvas_size - 1, ego_row - int(math.cos(a_rad) * max_r_px)))
            cv2.line(canvas, (cx_px, ego_row), (end_c, end_r), (60, 60, 140), 1, cv2.LINE_AA)

        # Raw point cloud overlay (sampled for speed)
        if self._last_raw_pts is not None and len(self._last_raw_pts) > 0:
            for pt in self._last_raw_pts[::4]:
                col, row = world_to_px(float(pt[0]), float(pt[1]))
                if 0 <= col < canvas_size and 0 <= row < canvas_size:
                    canvas[row, col] = (80, 80, 80)

        # Obstacle bounding boxes
        for obs in self._last_obstacles:
            color = _MOTION_COLORS.get(obs.motion_type, (200, 200, 200))
            c1 = world_to_px(obs.bbox_max_x, obs.bbox_min_y)
            c2 = world_to_px(obs.bbox_min_x, obs.bbox_max_y)
            x1 = max(0, min(canvas_size - 1, c1[0]))
            y1 = max(0, min(canvas_size - 1, c1[1]))
            x2 = max(0, min(canvas_size - 1, c2[0]))
            y2 = max(0, min(canvas_size - 1, c2[1]))
            if x1 != x2 and y1 != y2:
                cv2.rectangle(canvas,
                              (min(x1, x2), min(y1, y2)),
                              (max(x1, x2), max(y1, y2)),
                              color, 2)
            lx, ly = world_to_px(obs.centroid_x, obs.centroid_y)
            tag = f"{obs.distance:.1f}m {obs.motion_type[:4].upper()}"
            cv2.putText(canvas, tag, (max(0, lx - 30), max(10, ly - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # Ego vehicle marker
        cv2.rectangle(canvas,
                      (cx_px - 4, ego_row - 7),
                      (cx_px + 4, ego_row + 2),
                      (255, 120, 0), -1)
        cv2.arrowedLine(canvas, (cx_px, ego_row - 9),
                        (cx_px, ego_row - 26), (180, 180, 180), 1, tipLength=0.4)
        cv2.putText(canvas, "EGO", (cx_px - 10, ego_row + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 120, 0), 1)

        # Legend
        legend_y = 10
        for mt, col in _MOTION_COLORS.items():
            cv2.putText(canvas, mt, (5, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)
            legend_y += 13

        # Title
        cv2.putText(canvas, "TEMPORAL BEV",
                    (canvas_size // 2 - 48, canvas_size - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 140, 140), 1)

        return canvas
