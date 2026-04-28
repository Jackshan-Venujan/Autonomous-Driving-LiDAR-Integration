import threading
import math
import numpy as np
import cv2
import carla
from typing import Optional, Tuple


class StereoCameraError(RuntimeError):
    pass


class StereoPair:
    """
    Spawns two CARLA RGB cameras separated by baseline_m on the Y axis and
    produces synchronised rectified stereo frame pairs via SGBM disparity.

    Left camera position matches the monocular camera in main.py so that
    coordinate systems stay consistent between Experiment 1 and 2.

    Threading model mirrors LidarSensor: callback-based push into pending
    slots, _try_pair() promotes to latest when both frames share the same
    simulation timestamp (guaranteed identical in synchronous mode).
    """

    def __init__(
        self,
        world: carla.World,
        vehicle: carla.Vehicle,
        baseline_m: float = 0.12,
        image_width: int = 1280,
        image_height: int = 720,
        fov_degrees: float = 90.0,
        cam_x: float = 2.0,
        cam_z: float = 1.4,
        cam_pitch: float = -15.0,
        sgbm_num_disparities: int = 128,
        sgbm_block_size: int = 11,
        sgbm_p1: int = 2904,
        sgbm_p2: int = 11616,
        sgbm_disp12_max_diff: int = 1,
        sgbm_uniqueness_ratio: int = 10,
        sgbm_speckle_window_size: int = 100,
        sgbm_speckle_range: int = 32,
        sgbm_pre_filter_cap: int = 63,
        sgbm_mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    ):
        self._world = world
        self._vehicle = vehicle
        self._baseline_m = baseline_m
        self._image_width = image_width
        self._image_height = image_height
        self._fov_degrees = fov_degrees

        # focal length from pinhole model
        self._focal_length_px = (image_width / 2.0) / math.tan(
            math.radians(fov_degrees / 2.0)
        )

        self._lock = threading.Lock()
        self._pending_left: Optional[Tuple[np.ndarray, float]] = None
        self._pending_right: Optional[Tuple[np.ndarray, float]] = None
        self._latest_left: Optional[np.ndarray] = None
        self._latest_right: Optional[np.ndarray] = None
        self._latest_ts: Optional[float] = None
        self._pair_count = 0

        # SGBM matcher — created once, thread-safe for compute
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=sgbm_num_disparities,
            blockSize=sgbm_block_size,
            P1=sgbm_p1,
            P2=sgbm_p2,
            disp12MaxDiff=sgbm_disp12_max_diff,
            uniquenessRatio=sgbm_uniqueness_ratio,
            speckleWindowSize=sgbm_speckle_window_size,
            speckleRange=sgbm_speckle_range,
            preFilterCap=sgbm_pre_filter_cap,
            mode=sgbm_mode,
        )

        bp_lib = world.get_blueprint_library()

        left_transform = carla.Transform(
            carla.Location(x=cam_x, y=0.0, z=cam_z),
            carla.Rotation(pitch=cam_pitch),
        )
        right_transform = carla.Transform(
            carla.Location(x=cam_x, y=baseline_m, z=cam_z),
            carla.Rotation(pitch=cam_pitch),
        )

        self._left_cam = self._spawn_camera(bp_lib, left_transform, image_width, image_height, fov_degrees)
        self._right_cam = self._spawn_camera(bp_lib, right_transform, image_width, image_height, fov_degrees)

        self._left_cam.listen(self._left_callback)
        self._right_cam.listen(self._right_callback)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def focal_length_px(self) -> float:
        return self._focal_length_px

    @property
    def baseline(self) -> float:
        return self._baseline_m

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """
        Returns (left_bgr, right_bgr, timestamp) or (None, None, None).
        Both frames are guaranteed to share the same CARLA tick timestamp.
        Thread-safe; non-blocking.
        """
        with self._lock:
            if self._latest_left is None:
                return None, None, None
            return self._latest_left.copy(), self._latest_right.copy(), self._latest_ts

    def compute_disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """
        Run SGBM on a rectified pair and return a float32 disparity map (pixels).

        SGBM returns fixed-point int16 scaled by 16; we convert to float32.
        Pixels with disparity <= 0 are invalid.
        """
        left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        disp_int16 = self._matcher.compute(left_gray, right_gray)
        disp_float = disp_int16.astype(np.float32) / 16.0
        return disp_float

    def is_ready(self) -> bool:
        with self._lock:
            return self._pair_count > 0

    def destroy(self):
        try:
            self._left_cam.stop()
            self._left_cam.destroy()
        except Exception:
            pass
        try:
            self._right_cam.stop()
            self._right_cam.destroy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spawn_camera(self, bp_lib, transform: carla.Transform,
                      width: int, height: int, fov: float) -> carla.Actor:
        bp = bp_lib.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(width))
        bp.set_attribute('image_size_y', str(height))
        bp.set_attribute('fov', str(fov))
        actor = self._world.spawn_actor(bp, transform, attach_to=self._vehicle)
        return actor

    def _left_callback(self, image: carla.Image):
        frame = self._carla_image_to_bgr(image)
        with self._lock:
            self._pending_left = (frame, image.timestamp)
            self._try_pair()

    def _right_callback(self, image: carla.Image):
        frame = self._carla_image_to_bgr(image)
        with self._lock:
            self._pending_right = (frame, image.timestamp)
            self._try_pair()

    def _try_pair(self):
        """
        Called inside each callback (already under lock).
        When both pending slots hold frames from the same CARLA tick
        (identical simulation timestamp), promote them to latest.
        """
        if self._pending_left is None or self._pending_right is None:
            return
        left_frame, left_ts = self._pending_left
        right_frame, right_ts = self._pending_right
        if left_ts == right_ts:
            self._latest_left = left_frame
            self._latest_right = right_frame
            self._latest_ts = left_ts
            self._pair_count += 1
            self._pending_left = None
            self._pending_right = None

    @staticmethod
    def _carla_image_to_bgr(image: carla.Image) -> np.ndarray:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        return array[:, :, :3]  # drop alpha, keep BGR
