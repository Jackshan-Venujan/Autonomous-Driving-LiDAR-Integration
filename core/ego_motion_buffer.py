"""
ego_motion_buffer.py — Pose-stamped point cloud deque with ego-motion compensation.

Stores the last N (timestamp, pcd, T_world_4x4) frames and returns all of them
transformed into the current sensor frame, ready for temporal accumulation.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Deque, List, Optional, Tuple

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False

from core.lidar_config import TEMPORAL_BUFFER_SIZE


class EgoMotionBuffer:
    """Maintains a sliding window of pose-stamped point clouds.

    Each frame is stored as (timestamp, pcd_in_sensor_local_frame, T_world_4x4).
    T_world_4x4 is np.array(vehicle.get_transform().get_matrix()) — the 4x4 matrix
    that maps sensor-local coordinates to world coordinates at capture time.

    get_compensated_frames() returns all buffered clouds transformed into the
    coordinate system of the most recent frame, enabling ego-motion-corrected
    temporal accumulation without point smearing.
    """

    def __init__(self, buffer_size: int = TEMPORAL_BUFFER_SIZE) -> None:
        # Each entry: (timestamp, o3d.PointCloud, T_world_4x4 as float64 ndarray)
        self._buffer: Deque[Tuple[float, object, np.ndarray]] = deque(maxlen=buffer_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        timestamp: float,
        pcd: "o3d.geometry.PointCloud",
        T_world: np.ndarray,
    ) -> None:
        """Append a new frame to the sliding window.

        Args:
            timestamp: CARLA data.timestamp for this frame (seconds).
            pcd: Open3D PointCloud in sensor-local coordinates.
            T_world: (4, 4) float64 ndarray =
                     np.array(vehicle.get_transform().get_matrix()).
                     Maps sensor-local → world at the moment of capture.
        """
        self._buffer.append((timestamp, pcd, np.array(T_world, dtype=np.float64)))

    def get_compensated_frames(
        self,
    ) -> List[Tuple["o3d.geometry.PointCloud", int]]:
        """Return all buffered frames aligned into the current sensor frame.

        Returns:
            List of (transformed_pcd, frame_index) tuples.
            frame_index 0 = oldest, len-1 = current (identity transform applied).
            Returns [] if buffer is empty or Open3D is unavailable.
        """
        if not _O3D_AVAILABLE or len(self._buffer) == 0:
            return []

        # Current frame's world transform is the reference coordinate system
        _, _, T_world_current = self._buffer[-1]
        T_current_from_world = np.linalg.inv(T_world_current)  # (4,4)

        result: List[Tuple["o3d.geometry.PointCloud", int]] = []
        for frame_idx, (ts, pcd, T_world_old) in enumerate(self._buffer):
            # Copy so transform() does not mutate the stored cloud
            pcd_copy = _copy_pcd(pcd)

            if frame_idx < len(self._buffer) - 1:
                # Relative transform: old sensor frame → current sensor frame
                T_curr_from_old = T_current_from_world @ T_world_old
                pcd_copy.transform(T_curr_from_old)
            # Current frame: identity — no transform needed

            result.append((pcd_copy, frame_idx))

        return result

    def is_full(self) -> bool:
        """True when the buffer holds exactly buffer_size frames."""
        return len(self._buffer) == self._buffer.maxlen

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()


# ------------------------------------------------------------------
# Private helper
# ------------------------------------------------------------------

def _copy_pcd(pcd: "o3d.geometry.PointCloud") -> "o3d.geometry.PointCloud":
    """Return a new PointCloud with a copy of pcd's points array.

    Open3D's copy constructor shares the internal data buffer in some
    versions, so we explicitly copy the numpy array to guarantee
    independence before calling transform() on the copy.
    """
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(
        np.asarray(pcd.points).copy()
    )
    return new_pcd
