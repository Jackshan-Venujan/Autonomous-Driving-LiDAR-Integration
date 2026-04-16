"""
Camera Calibration Module — Task 05
=====================================

Provides two frozen dataclasses for Camera–LiDAR fusion calibration:

  CameraCalibration  — camera intrinsics (K matrix, FOV, image dimensions)
  FusionExtrinsics   — LiDAR-to-camera extrinsic rigid body transform

These are consumed by SensorFusion (lidar_fusion.py) to project 3D LiDAR
cluster points onto the 2D camera image plane for IoU matching with YOLO
bounding boxes.

Coordinate systems
------------------
  LiDAR / vehicle frame (ISO 8855 right-hand, after Task 01 Y-flip):
    +X = forward,  +Y = left,  +Z = up.

  Camera frame (OpenCV pinhole convention):
    +X = image right,  +Y = image down,  +Z = optical axis (depth/forward).

  CARLA uses left-hand coordinates (+Y = right).  The R_fix rotation in
  FusionExtrinsics corrects CARLA camera axes to OpenCV axes.

Usage
-----
    # From a live CARLA session:
    cal  = CameraCalibration.from_carla_blueprint(cam_bp, 1280, 720)
    extr = FusionExtrinsics.from_carla_transforms(lidar_t, camera_t)

    # For unit tests / offline processing (no CARLA needed):
    cal  = CameraCalibration.from_params(1280, 720, fov_deg=90.0)
    extr = FusionExtrinsics.from_offsets(
               lidar_xyz=(0.0, 0.0, 2.4),
               camera_xyz=(1.5, 0.0, 2.1)
           )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

import numpy as np

try:
    import carla  # type: ignore
except ImportError:
    carla = None   # CARLA not installed — from_carla_* class methods unavailable

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][Fusion] %(message)s',
    datefmt='%H:%M:%S',
)


# ══════════════════════════════════════════════════════════════════════════════
#  CameraCalibration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CameraCalibration:
    """
    Camera intrinsic parameters derived from CARLA blueprint attributes.

    The intrinsic matrix K maps 3D camera-frame points to 2D pixel coordinates:

      K = [[f,  0,  cx],
           [0,  f,  cy],
           [0,  0,   1]]

    Where:
      f  = focal length in pixels
      cx = principal point X (horizontal image centre)
      cy = principal point Y (vertical image centre)

    CARLA assumes a pinhole camera model with zero skew and square pixels.
    Therefore Kx = Ky = f (single focal length for both axes).

    Projection formula (for a point P in camera frame):
      u = f * (Px / Pz) + cx
      v = f * (Py / Pz) + cy

    Valid only for Pz > 0 (point in front of camera).
    """

    image_width  : int      # pixels, e.g. 1280
    image_height : int      # pixels, e.g. 720
    fov_deg      : float    # horizontal field of view in degrees, e.g. 90.0
    focal_length : float    # f in pixels, derived from fov and image_width
    cx           : float    # principal point X = image_width / 2
    cy           : float    # principal point Y = image_height / 2

    @property
    def K(self) -> np.ndarray:
        """
        3×3 intrinsic matrix K. float64 for numerical precision in projection.

        Returns:
            np.ndarray shape (3, 3) dtype float64.
        """
        return np.array([
            [self.focal_length, 0.0,              self.cx],
            [0.0,               self.focal_length, self.cy],
            [0.0,               0.0,               1.0   ]
        ], dtype=np.float64)

    @property
    def K_inv(self) -> np.ndarray:
        """
        Inverse of K — maps pixel coordinates back to normalised camera rays.
        Used for back-projection (pixel → ray direction).
        """
        return np.linalg.inv(self.K)

    def is_in_image(self, u: float, v: float, margin: int = 0) -> bool:
        """
        Check if pixel (u, v) falls within the image bounds.

        Args:
            u, v  : Pixel coordinates (can be float from projection).
            margin: Optional inset margin in pixels.  Positive margin shrinks
                    the valid region (useful to exclude boundary projections
                    which have unreliable IoU matching near edges).

        Returns:
            True if pixel is within image bounds with given margin.
        """
        return (margin <= u <= self.image_width  - margin and
                margin <= v <= self.image_height - margin)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_carla_blueprint(
        cls,
        camera_bp    : 'carla.ActorBlueprint',
        image_width  : int,
        image_height : int
    ) -> 'CameraCalibration':
        """
        Derive calibration from a CARLA camera blueprint.

        Focal length derivation:
          The horizontal FOV spans image_width pixels.
          For a pinhole camera: tan(FOV/2) = (image_width/2) / f
          Therefore: f = (image_width/2) / tan(FOV/2)

          Example: FOV=90°, width=1280
            f = 640 / tan(45°) = 640 / 1.0 = 640 pixels

        Args:
            camera_bp   : CARLA camera blueprint (after setting image_size_x/y).
            image_width : Image width in pixels.
            image_height: Image height in pixels.

        Returns:
            CameraCalibration instance.
        """
        fov_deg = float(camera_bp.get_attribute('fov').as_float())
        f = image_width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
        return cls(
            image_width  = image_width,
            image_height = image_height,
            fov_deg      = fov_deg,
            focal_length = f,
            cx           = image_width  / 2.0,
            cy           = image_height / 2.0,
        )

    @classmethod
    def from_params(
        cls,
        image_width  : int,
        image_height : int,
        fov_deg      : float
    ) -> 'CameraCalibration':
        """
        Construct from explicit parameters without a CARLA blueprint.
        Use this in unit tests and offline processing.

        Focal length: f = (image_width/2) / tan(fov_deg/2)
        """
        f = image_width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
        return cls(
            image_width  = image_width,
            image_height = image_height,
            fov_deg      = fov_deg,
            focal_length = f,
            cx           = image_width  / 2.0,
            cy           = image_height / 2.0,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  FusionExtrinsics
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FusionExtrinsics:
    """
    Rigid body transform from LiDAR frame to Camera frame.

    Notation:
      T_lidar_cam transforms a point FROM LiDAR frame TO Camera frame.
      p_cam = T_lidar_cam @ p_lidar_homogeneous

    CARLA coordinate systems:

      LiDAR frame (ISO 8855, after Task 01 Y-flip):
        +X = forward, +Y = left, +Z = up

      Camera frame (standard OpenCV pinhole convention):
        +X = image right, +Y = image down, +Z = optical axis (forward)

    TRANSFORMATION PIPELINE:
      T_lidar_cam = T_world_cam_inv @ T_world_lidar
      then apply CARLA camera → OpenCV axis fix.

    CARLA→OpenCV axis convention fix (R_fix):
      Camera X (right)    =  CARLA camera Y
      Camera Y (down)     = -CARLA camera Z
      Camera Z (forward)  =  CARLA camera X

      R_fix = [[0, 1,  0],
               [0, 0, -1],
               [1, 0,  0]]
    """

    # 4×4 homogeneous transform: LiDAR frame → Camera frame (OpenCV convention)
    # Stored as tuple of 16 floats (row-major) for frozen dataclass compatibility.
    T_lidar_cam_data : Tuple

    @property
    def T_lidar_cam(self) -> np.ndarray:
        """4×4 float64 homogeneous transform matrix."""
        return np.array(self.T_lidar_cam_data, dtype=np.float64).reshape(4, 4)

    @property
    def R(self) -> np.ndarray:
        """3×3 rotation sub-matrix."""
        return self.T_lidar_cam[:3, :3]

    @property
    def t(self) -> np.ndarray:
        """3×1 translation vector."""
        return self.T_lidar_cam[:3, 3:4]

    def transform_points(self, points_lidar: np.ndarray) -> np.ndarray:
        """
        Transform 3D points from LiDAR frame to Camera frame.

        Args:
            points_lidar: shape (N, 3) or (3,) float32/float64

        Returns:
            points_cam: shape (N, 3) float64 in OpenCV camera frame.
                        (+X=right, +Y=down, +Z=forward/depth)
        """
        pts = np.atleast_2d(points_lidar).astype(np.float64)   # (N, 3)
        N   = len(pts)

        # Convert to homogeneous: (N, 4)
        ones   = np.ones((N, 1), dtype=np.float64)
        pts_h  = np.hstack([pts, ones])                          # (N, 4)

        # Apply transform: T (4,4) @ pts_h.T (4,N) → (4,N) → transpose → (N,4)
        pts_cam_h = (self.T_lidar_cam @ pts_h.T).T              # (N, 4)

        return pts_cam_h[:, :3]                                  # (N, 3)

    def to_dict(self) -> dict:
        return {'T_lidar_cam': list(self.T_lidar_cam_data)}

    @classmethod
    def from_carla_transforms(
        cls,
        lidar_transform  : 'carla.Transform',
        camera_transform : 'carla.Transform'
    ) -> 'FusionExtrinsics':
        """
        Compute T_lidar_cam from CARLA sensor transforms.

        Both transforms are in CARLA world frame (left-hand coordinates).
        We compute the RELATIVE transform from LiDAR to camera, then apply
        the CARLA camera → OpenCV axis convention correction.

        Algorithm:
          1. T_world_lidar  = lidar_transform.get_matrix()   (4×4)
          2. T_world_cam    = camera_transform.get_matrix()  (4×4)
          3. T_lidar_cam_carla = inv(T_world_cam) @ T_world_lidar
          4. Apply R_fix to convert CARLA camera axes → OpenCV camera axes

        CARLA→OpenCV R_fix (4×4 homogeneous):
          [[0, 1,  0, 0],
           [0, 0, -1, 0],
           [1, 0,  0, 0],
           [0, 0,  0, 1]]

        Usage in CARLA:
            lidar_t  = lidar_sensor.get_transform()
            camera_t = camera_sensor.get_transform()
            extr = FusionExtrinsics.from_carla_transforms(lidar_t, camera_t)
        """
        T_world_lidar = np.array(lidar_transform.get_matrix(),  dtype=np.float64)
        T_world_cam   = np.array(camera_transform.get_matrix(), dtype=np.float64)

        # Relative transform: LiDAR frame → camera frame (CARLA axes)
        T_lidar_cam_carla = np.linalg.inv(T_world_cam) @ T_world_lidar

        # Axis convention: CARLA camera → OpenCV pinhole camera
        R_fix = np.array([
            [0, 1,  0, 0],
            [0, 0, -1, 0],
            [1, 0,  0, 0],
            [0, 0,  0, 1]
        ], dtype=np.float64)

        T_lidar_cam_opencv = R_fix @ T_lidar_cam_carla

        return cls(T_lidar_cam_data=tuple(T_lidar_cam_opencv.flatten().tolist()))

    @classmethod
    def from_offsets(
        cls,
        lidar_xyz  : Tuple[float, float, float],
        camera_xyz : Tuple[float, float, float]
    ) -> 'FusionExtrinsics':
        """
        Compute approximate extrinsics from sensor mounting positions only.
        Use when CARLA transforms are not yet available (unit tests, offline).

        Assumes both sensors have zero rotation relative to vehicle frame.
        Only the translation offset between sensors is modelled.

        Typical CARLA mounts:
          LiDAR:  (x=0.0, y=0.0, z=2.4) — roof centre
          Camera: (x=1.5, y=0.0, z=2.1) — front windshield

        The relative translation in CARLA vehicle frame:
          t_lidar_cam_vehicle = camera_xyz - lidar_xyz = [1.5, 0.0, -0.3]

        Applying CARLA→OpenCV axis fix:
          OpenCV t.x =  CARLA t.y =  0.0  (lateral)
          OpenCV t.y = -CARLA t.z =  0.3  (vertical, flipped)
          OpenCV t.z =  CARLA t.x =  1.5  (forward)

        Args:
            lidar_xyz : LiDAR mounting position [x, y, z] in vehicle frame.
            camera_xyz: Camera mounting position [x, y, z] in vehicle frame.

        Returns:
            FusionExtrinsics with pure translation (rotation = axis fix only).
        """
        lx, ly, lz = lidar_xyz
        cx, cy, cz = camera_xyz

        # Translation: LiDAR to camera in CARLA vehicle frame
        dt_carla = np.array([cx - lx, cy - ly, cz - lz], dtype=np.float64)

        # CARLA→OpenCV rotation matrix (3×3)
        R_fix = np.array([
            [0, 1,  0],
            [0, 0, -1],
            [1, 0,  0]
        ], dtype=np.float64)

        dt_opencv = R_fix @ dt_carla

        # Build 4×4 homogeneous transform
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_fix
        T[:3,  3] = dt_opencv

        return cls(T_lidar_cam_data=tuple(T.flatten().tolist()))
