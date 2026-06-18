"""
LiDAR-Camera Fusion Module

Projects CARLA ray_cast LiDAR points into front and rear camera images,
then fuses the projected cloud with YOLO detections to obtain accurate
LiDAR-derived distance for each bounding box.

Coordinate convention (same as lidar_processor.py / bev_visualizer.py):
    Sensor/vehicle frame: x = forward, y = right, z = up

Projection formula (camera local frame: x = forward, y = right, z = up):
    u = fx * (y_cam / x_cam) + cx
    v = cy - fy * (z_cam / x_cam)   <- negated because image v increases downward
    Valid only when x_cam > 0 (point in front of the camera)
"""

import math
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple


class LidarCameraFusion:
    """
    Projects LiDAR points into front / rear camera views and fuses the
    resulting depth measurements with YOLO bounding-box detections.

    Sensor layout (CARLA, relative to vehicle frame):
        LiDAR   : pos (0.0, 0.0, 2.0),  rot  (0°, 0°, 0°)
        FrontCam: pos (2.0, 0.0, 1.4),  pitch=-15°, yaw=  0°, fov=90°
        RearCam : pos (-2.5, 0.0, 1.4), pitch=-10°, yaw=180°, fov=120°
    """

    # LiDAR position offset in vehicle frame (sensor is 2 m above ground)
    LIDAR_POS = np.array([0.0, 0.0, 2.0], dtype=np.float32)

    FRONT_CAM_POS = np.array([2.0,  0.0, 1.4], dtype=np.float32)
    REAR_CAM_POS  = np.array([-2.5, 0.0, 1.4], dtype=np.float32)

    def __init__(
        self,
        img_w: int = 1280,
        img_h: int = 720,
        front_fov: float = 90.0,
        rear_fov: float  = 120.0,
        min_pts_for_distance: int = 3,
    ):
        """
        Args:
            img_w, img_h        : camera resolution (same for both cameras)
            front_fov           : horizontal FOV of the front camera in degrees
            rear_fov            : horizontal FOV of the rear  camera in degrees
            min_pts_for_distance: minimum LiDAR points inside a bbox to compute
                                  a valid LiDAR distance (robust against noise)
        """
        self.img_w = img_w
        self.img_h = img_h
        self.min_pts = min_pts_for_distance

        # Camera intrinsics
        self.K_front, self.fx_f, self.fy_f, self.cx_f, self.cy_f = \
            self._build_intrinsics(img_w, img_h, front_fov)
        self.K_rear,  self.fx_r, self.fy_r, self.cx_r, self.cy_r = \
            self._build_intrinsics(img_w, img_h, rear_fov)

        # Rotation matrices: vehicle frame → camera sensor frame
        # R_cam = (R_sensor_in_vehicle)^T
        self.R_front = self._build_front_rotation()
        self.R_rear  = self._build_rear_rotation()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project_to_camera(
        self,
        lidar_pts_xyz: Optional[np.ndarray],
        camera: str = 'front',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project LiDAR points (ground-removed, sensor frame) into a camera.

        Args:
            lidar_pts_xyz : (N, 3) array in LiDAR sensor frame, or None
            camera        : 'front' | 'rear'

        Returns:
            proj_uv   : (N, 2) float32 — pixel coordinates (u, v)
            depths    : (N,)   float32 — forward depth in camera frame (metres)
            valid     : (N,)   bool    — True for points that land in the image
        """
        empty_uv    = np.empty((0, 2), dtype=np.float32)
        empty_depth = np.empty(0,      dtype=np.float32)
        empty_mask  = np.zeros(0,      dtype=bool)
        empty_z     = np.empty(0,      dtype=np.float32)

        if lidar_pts_xyz is None or len(lidar_pts_xyz) == 0:
            return empty_uv, empty_depth, empty_mask, empty_z

        pts = lidar_pts_xyz.astype(np.float32)

        # 1. LiDAR sensor frame → vehicle frame
        pts_v = pts + self.LIDAR_POS  # broadcast (N,3) + (3,)
        z_vehicle = pts_v[:, 2].copy()   # vehicle-frame height (needed for ground coloring)

        # 2. Vehicle frame → camera sensor frame
        if camera == 'front':
            cam_pos = self.FRONT_CAM_POS
            R       = self.R_front
            fx, fy, cx, cy = self.fx_f, self.fy_f, self.cx_f, self.cy_f
        else:  # rear
            cam_pos = self.REAR_CAM_POS
            R       = self.R_rear
            fx, fy, cx, cy = self.fx_r, self.fy_r, self.cx_r, self.cy_r

        pts_c = (R @ (pts_v - cam_pos).T).T  # (N, 3)

        # 3. Perspective projection
        x_cam = pts_c[:, 0]   # forward depth
        y_cam = pts_c[:, 1]   # rightward
        z_cam = pts_c[:, 2]   # upward

        # Only consider points in front of the camera
        in_front = x_cam > 0.1
        # Avoid divide-by-zero for masked-out points
        safe_x = np.where(in_front, x_cam, 1.0)

        u = fx * (y_cam / safe_x) + cx
        v = cy - fy * (z_cam / safe_x)

        # Build validity mask: in front AND within image bounds
        valid = (
            in_front
            & (u >= 0) & (u < self.img_w)
            & (v >= 0) & (v < self.img_h)
        )

        proj_uv = np.stack([u, v], axis=1).astype(np.float32)
        depths  = x_cam.astype(np.float32)
        return proj_uv, depths, valid, z_vehicle

    def fuse_with_detections(
        self,
        proj_uv:    np.ndarray,
        depths:     np.ndarray,
        valid_mask: np.ndarray,
        yolo_dets:  List[Dict],
    ) -> List[Dict]:
        """
        For each YOLO detection, find projected LiDAR points inside its
        bounding box and compute a robust (median) LiDAR distance.

        Adds two keys to each detection dict (non-destructive copy):
            'lidar_distance'    : float in metres, or None
            'lidar_point_count' : int (0 if no match)

        Args:
            proj_uv    : (N, 2) pixel coordinates from project_to_camera()
            depths     : (N,)   forward depths from project_to_camera()
            valid_mask : (N,)   boolean validity mask
            yolo_dets  : list of YOLO detection dicts (must have 'bbox' key)

        Returns:
            List of enriched detection dicts (same order, shallow copies)
        """
        if not yolo_dets:
            return []

        # Only work with valid projected points
        if valid_mask.sum() == 0:
            return [dict(d, lidar_distance=None, lidar_point_count=0)
                    for d in yolo_dets]

        valid_uv = proj_uv[valid_mask]    # (M, 2)
        valid_d  = depths[valid_mask]     # (M,)
        ux, uy   = valid_uv[:, 0], valid_uv[:, 1]

        enriched = []
        for det in yolo_dets:
            x1, y1, x2, y2 = det['bbox']
            inside = (ux >= x1) & (ux <= x2) & (uy >= y1) & (uy <= y2)
            pts_inside = valid_d[inside]

            d = dict(det)  # shallow copy
            n = int(inside.sum())
            if n >= self.min_pts:
                d['lidar_distance']    = float(np.median(pts_inside))
                d['lidar_point_count'] = n
            else:
                d['lidar_distance']    = None
                d['lidar_point_count'] = n
            enriched.append(d)

        return enriched

    def draw_projected_points(
        self,
        image:           np.ndarray,
        proj_uv:         np.ndarray,
        depths:          np.ndarray,
        valid_mask:      np.ndarray,
        z_vehicle:       Optional[np.ndarray] = None,
        ground_z_thresh: float = 0.8,
        max_range:       float = 50.0,
    ) -> np.ndarray:
        """
        Draw all valid projected LiDAR points as 2 px circles.

        Color coding:
          • z_vehicle < ground_z_thresh  → blue  (road plane / near-ground)
          • z_vehicle ≥ ground_z_thresh  → depth-coded red (near) → green (far)

        Args:
            image           : BGR image to draw on (will be copied)
            proj_uv         : (N, 2) pixel coordinates
            depths          : (N,)   forward depths in metres
            valid_mask      : (N,)   bool validity mask
            z_vehicle       : (N,)   vehicle-frame z for each point (from project_to_camera)
            ground_z_thresh : points below this height (m) are coloured blue
            max_range       : depth (m) mapped to fully green for obstacle points

        Returns:
            BGR image with LiDAR overlay
        """
        vis = image.copy()
        if valid_mask.sum() == 0:
            return vis

        uv_valid = proj_uv[valid_mask].astype(np.int32)
        d_valid  = depths[valid_mask]
        zv_valid = z_vehicle[valid_mask] if z_vehicle is not None else None

        for idx, ((u, v), depth) in enumerate(zip(uv_valid, d_valid)):
            if zv_valid is not None and zv_valid[idx] < ground_z_thresh:
                color = (180, 60, 0)   # blue — road/ground plane
            else:
                ratio = float(min(depth / max_range, 1.0))
                r = int(255 * (1.0 - ratio))
                g = int(200 * ratio)
                color = (0, g, r)      # red (near) → green (far) — obstacles
            cv2.circle(vis, (u, v), 2, color, -1)

        return vis

    def draw_fused_detections(
        self,
        image:       np.ndarray,
        fused_dets:  List[Dict],
        show_cam_dist: bool = True,
    ) -> np.ndarray:
        """
        Draw YOLO bounding boxes enriched with LiDAR distance labels.

        Label format above each box:
            <class> #<id>   cam:<X.X>m  LiDAR:<X.X>m

        Args:
            image        : BGR image
            fused_dets   : enriched detection list from fuse_with_detections()
            show_cam_dist: whether to also show the monocular camera distance

        Returns:
            annotated BGR image
        """
        vis = image.copy()
        # Sort by monocular distance so IDs are distance-ordered
        sorted_dets = sorted(
            enumerate(fused_dets),
            key=lambda x: (x[1].get('distance') or 999)
        )

        for rank, (orig_idx, det) in enumerate(sorted_dets):
            x1, y1, x2, y2 = det['bbox']
            cls   = det.get('class', '?')
            conf  = det.get('confidence', 0.0)
            cam_d = det.get('distance')
            lid_d = det.get('lidar_distance')
            npts  = det.get('lidar_point_count', 0)

            # Box color: orange if LiDAR confirmed, yellow otherwise
            box_color = (0, 140, 255) if lid_d is not None else (0, 220, 220)
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

            # Build label lines
            line1 = f"{cls} #{rank}"
            parts = []
            if show_cam_dist and cam_d is not None:
                parts.append(f"cam:{cam_d:.1f}m")
            if lid_d is not None:
                parts.append(f"LiDAR:{lid_d:.1f}m")
            elif npts > 0:
                parts.append(f"pts:{npts}")
            line2 = "  ".join(parts) if parts else ""

            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness  = 1
            text_color = (255, 255, 255)

            # Background pill for readability
            for i, line in enumerate([line1, line2]):
                if not line:
                    continue
                (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
                tx = x1
                ty = max(y1 - 4 - (len([l for l in [line1, line2] if l]) - i) * (th + 4), th + 2)
                cv2.rectangle(vis, (tx - 1, ty - th - 2), (tx + tw + 1, ty + 2),
                              (0, 0, 0), -1)
                cv2.putText(vis, line, (tx, ty), font, font_scale, text_color, thickness)

        return vis

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_intrinsics(
        img_w: int, img_h: int, fov_deg: float
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """Build K matrix and (fx, fy, cx, cy) from horizontal FOV."""
        fx = fy = (img_w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        cx = img_w / 2.0
        cy = img_h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K, float(fx), float(fy), float(cx), float(cy)

    @staticmethod
    def _rot_y(theta: float) -> np.ndarray:
        """Rotation matrix around Y-axis (pitch). theta in radians."""
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    @staticmethod
    def _rot_z(theta: float) -> np.ndarray:
        """Rotation matrix around Z-axis (yaw). theta in radians."""
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    def _build_front_rotation(self) -> np.ndarray:
        """
        Front camera: CARLA pitch=-15° (nose DOWN), yaw=0°.

        CARLA pitch sign convention: negative pitch = nose DOWN.
        Our Ry(θ) maps +X to (cosθ, 0, -sinθ), so +θ gives -z (nose DOWN).
        Therefore CARLA pitch=-15° requires Ry(+15°) in our math.

        R_sensor_in_vehicle = Ry(+15°)   [columns = camera local axes in vehicle frame]
        R (vehicle→camera)  = R_sensor^T
        """
        R_sensor = self._rot_y(math.radians(+15.0))   # CARLA pitch=-15° → use +15° here
        return R_sensor.T

    def _build_rear_rotation(self) -> np.ndarray:
        """
        Rear camera: CARLA yaw=180°, pitch=-10° (nose DOWN).

        Same pitch sign fix: CARLA pitch=-10° → use Ry(+10°) in our math.
        Rotation composition: pitch applied after yaw in intrinsic order
        R_sensor_in_vehicle = Ry(+10°) @ Rz(180°)
        R (vehicle→camera)  = R_sensor^T
        """
        Rz180 = self._rot_z(math.radians(180.0))
        Ry10p = self._rot_y(math.radians(+10.0))      # CARLA pitch=-10° → use +10° here
        R_sensor = Ry10p @ Rz180
        return R_sensor.T
