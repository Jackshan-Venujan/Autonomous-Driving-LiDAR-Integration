"""
LiDAR-to-Camera projector.

Projects LiDAR points (sensor frame: X=fwd, Y=right, Z=up) into the camera
image using the known extrinsic offset between the two sensors and the camera
intrinsics derived from the FOV.

Sensor placement on the ego vehicle:
  Camera : x=2.0, z=1.4, pitch=-15 deg  (nose-down)
  LiDAR  : x=2.0, z=1.8, no rotation

Both sensors share the same X/Y offset from the vehicle body, so the only
extrinsic difference is a +0.4 m Z shift (LiDAR is above camera) followed by
the inverse of the camera pitch rotation.
"""

import math
import numpy as np
from typing import Optional, Tuple


class LidarCameraProjector:

    def __init__(
        self,
        image_width: int = 1280,
        image_height: int = 720,
        fov_deg: float = 90.0,
        cam_pitch_deg: float = -15.0,
        lidar_z_above_cam: float = 0.4,
    ):
        self.image_width = image_width
        self.image_height = image_height

        # Intrinsics
        half_fov = math.radians(fov_deg / 2.0)
        self.fx = (image_width / 2.0) / math.tan(half_fov)
        self.fy = self.fx
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0

        # Extrinsics — Z offset from LiDAR frame to camera frame
        self.dz = lidar_z_above_cam  # camera is dz metres below LiDAR

        # Camera pitch rotation: pitch=-15 deg means nose is tilted down.
        # To express a world/lidar point in camera sensor frame we apply
        # R_cam^T  where R_cam = R_y(pitch).
        # R_y^T(pitch) = R_y(-pitch), so we rotate by +|pitch|.
        pitch_inv = math.radians(-cam_pitch_deg)   # +15 deg
        self._cos_p = math.cos(pitch_inv)
        self._sin_p = math.sin(pitch_inv)

    # ------------------------------------------------------------------
    # Core projection
    # ------------------------------------------------------------------

    def project_points(
        self, lidar_xyz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Project LiDAR points to image coordinates.

        Args:
            lidar_xyz: (N, 3) float32 array in LiDAR sensor frame
                       (X=forward, Y=right, Z=up).

        Returns:
            u      : (M,) pixel column for valid points
            v      : (M,) pixel row for valid points
            depth  : (M,) forward distance (metres) for valid points
            valid  : (N,) bool mask — True for points that project onto the image
        """
        if lidar_xyz is None or len(lidar_xyz) == 0:
            empty = np.array([], dtype=np.float32)
            return empty, empty, empty, np.zeros(0, dtype=bool)

        x_l = lidar_xyz[:, 0]
        y_l = lidar_xyz[:, 1]
        z_l = lidar_xyz[:, 2]

        # Step 1: translate Z.
        # Camera is dz metres BELOW the LiDAR.  In camera-frame coordinates a
        # given point sits dz metres HIGHER than in LiDAR-frame coordinates,
        # so we add dz (not subtract).
        z_shifted = z_l + self.dz

        # Step 2: apply inverse camera pitch rotation (R_y^T(pitch))
        #   x_c =  cos_p * x_l + sin_p * z_shifted
        #   y_c =  y_l                               (lateral is unaffected by pitch)
        #   z_c = -sin_p * x_l + cos_p * z_shifted
        x_c = self._cos_p * x_l + self._sin_p * z_shifted
        y_c = y_l
        z_c = -self._sin_p * x_l + self._cos_p * z_shifted

        # Step 3: keep only points in front of camera
        forward_mask = x_c > 0.1

        xf = x_c[forward_mask]
        yf = y_c[forward_mask]
        zf = z_c[forward_mask]

        # Step 4: pinhole projection
        # CARLA camera frame: X=fwd, Y=right, Z=up → image: u=right, v=down
        u = self.fx * (yf / xf) + self.cx
        v = self.fy * (-zf / xf) + self.cy

        # Step 5: keep only points within image bounds
        in_image = (
            (u >= 0) & (u < self.image_width) &
            (v >= 0) & (v < self.image_height)
        )

        # Build full valid mask (same length as input)
        valid = np.zeros(len(lidar_xyz), dtype=bool)
        idx = np.where(forward_mask)[0]
        valid[idx[in_image]] = True

        return u[in_image], v[in_image], xf[in_image], valid

    # ------------------------------------------------------------------
    # BBox filtering
    # ------------------------------------------------------------------

    def filter_by_bbox(
        self,
        lidar_xyz: np.ndarray,
        bbox: Tuple[float, float, float, float],
        min_points: int = 2,
        padding_frac: float = 0.15,
    ) -> Tuple[Optional[float], int]:
        """Return the median forward depth of LiDAR points inside a 2D bbox.

        Args:
            lidar_xyz    : (N, 3) LiDAR points in sensor frame.
            bbox         : (x1, y1, x2, y2) pixel bounding box.
            min_points   : Minimum points required to trust the estimate.
            padding_frac : Fractional expansion applied to each side of the
                           bbox to absorb minor calibration drift.

        Returns:
            (median_depth_m, point_count)
            median_depth_m is None when fewer than min_points fall inside bbox.
        """
        if lidar_xyz is None or len(lidar_xyz) == 0:
            return None, 0

        u, v, depth, _ = self.project_points(lidar_xyz)
        if len(u) == 0:
            return None, 0

        x1, y1, x2, y2 = bbox
        pw = (x2 - x1) * padding_frac
        ph = (y2 - y1) * padding_frac
        in_bbox = (
            (u >= x1 - pw) & (u <= x2 + pw) &
            (v >= y1 - ph) & (v <= y2 + ph)
        )
        bbox_depths = depth[in_bbox]

        count = int(len(bbox_depths))
        if count < min_points:
            return None, count

        return float(np.median(bbox_depths)), count
