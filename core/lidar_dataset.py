"""
PyTorch Dataset & TensorFlow Dataset Builder — Task 07
=======================================================

Provides zero-copy, memory-mapped loaders for the ADAS dataset format
recorded by DataExporter. Compatible with PointPillars, CenterPoint,
BEVDet, BEVFormer, LSTM trajectory prediction, and occupancy networks.

Usage:
    # PyTorch
    from lidar_dataset import ADASDriveDataset
    ds = ADASDriveDataset('recordings/session_20240115_143022')
    loader = DataLoader(ds, batch_size=4, num_workers=4, pin_memory=True)

    # Frame access
    sample = ds[0]
    lidar_pts = sample['lidar_filtered']     # (M, 3) float32 tensor
    obstacles  = sample['obstacles']         # list of dicts
    ego_speed  = sample['ego_speed_ms']      # float tensor

    # TensorFlow
    from lidar_dataset import build_tf_dataset
    tf_ds = build_tf_dataset('recordings/session_20240115_143022',
                              batch_size=8, shuffle=True)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Dataset = object   # type: ignore

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  Frame index helpers
# ══════════════════════════════════════════════════════════════════════════════

def discover_frames(session_dir: Path) -> List[Path]:
    """
    Discover all frame directories in a session, sorted by frame_id.

    A valid frame directory must contain at minimum obstacles.json.

    Args:
        session_dir: Path to session directory
                     (the one containing metadata.json and frames/).

    Returns:
        List of frame directory Paths, sorted by integer frame_id.
    """
    frames_dir = Path(session_dir) / 'frames'
    if not frames_dir.exists():
        raise FileNotFoundError(f"No 'frames' subdirectory in {session_dir}")

    valid = []
    for d in frames_dir.iterdir():
        if d.is_dir() and (d / 'obstacles.json').exists():
            valid.append(d)

    # Sort by numeric frame_id (directory name is zero-padded integer)
    valid.sort(key=lambda p: int(p.name))
    return valid


def load_metadata(session_dir: Path) -> dict:
    """Load and return metadata.json from a session directory."""
    meta_path = Path(session_dir) / 'metadata.json'
    if not meta_path.exists():
        return {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  ADASDriveDataset — PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

class ADASDriveDataset(Dataset):
    """
    PyTorch Dataset for ADAS LiDAR pipeline recordings.

    Supports lazy loading: arrays and images are loaded on-demand per sample.
    LiDAR arrays use np.load with mmap_mode='r' for zero-copy random access.

    Args:
        session_dir     : Path to session directory.
        load_lidar_raw  : Load lidar_raw.npy (N, 4) float32.
        load_lidar_filt : Load lidar_filtered.npy (M, 3) float32.
        load_camera     : Load camera_rgb.jpg as (H, W, 3) uint8 tensor.
        load_radar_bev  : Load radar_bev.png as (H, W, 3) uint8 tensor.
        max_obstacles   : Pad/truncate obstacle list to this length.
                          None = variable-length (cannot batch without collate_fn).
        transform       : Optional callable applied to each sample dict.
        min_frame_id    : If set, skip frames with frame_id < this value.
        max_frame_id    : If set, skip frames with frame_id > this value.
    """

    def __init__(
        self,
        session_dir     : str,
        load_lidar_raw  : bool = False,
        load_lidar_filt : bool = True,
        load_camera     : bool = True,
        load_radar_bev  : bool = True,
        max_obstacles   : Optional[int] = 20,
        transform       : Optional[Callable] = None,
        min_frame_id    : Optional[int] = None,
        max_frame_id    : Optional[int] = None,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required: pip install torch")

        self._session_dir    = Path(session_dir)
        self._load_raw       = load_lidar_raw
        self._load_filt      = load_lidar_filt
        self._load_camera    = load_camera
        self._load_radar     = load_radar_bev
        self._max_obs        = max_obstacles
        self._transform      = transform

        self._metadata = load_metadata(self._session_dir)
        all_frames     = discover_frames(self._session_dir)

        # Apply frame_id range filter
        self._frames: List[Path] = []
        for f in all_frames:
            fid = int(f.name)
            if min_frame_id is not None and fid < min_frame_id:
                continue
            if max_frame_id is not None and fid > max_frame_id:
                continue
            self._frames.append(f)

        logger.info(
            "ADASDriveDataset: session=%s | frames=%d",
            self._session_dir.name, len(self._frames),
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict:
        frame_dir = self._frames[idx]
        sample: dict = {'frame_id': int(frame_dir.name)}

        # ── LiDAR ─────────────────────────────────────────────────────────
        if self._load_raw:
            raw_path = frame_dir / 'lidar_raw.npy'
            if raw_path.exists():
                arr = np.load(str(raw_path), mmap_mode='r').copy()
                sample['lidar_raw'] = torch.from_numpy(arr)  # (N, 4) float32

        if self._load_filt:
            filt_path = frame_dir / 'lidar_filtered.npy'
            if filt_path.exists():
                arr = np.load(str(filt_path), mmap_mode='r').copy()
                sample['lidar_filtered'] = torch.from_numpy(arr)  # (M, 3)
            else:
                sample['lidar_filtered'] = torch.zeros((0, 3), dtype=torch.float32)

        # ── Camera ────────────────────────────────────────────────────────
        if self._load_camera and _PIL_AVAILABLE:
            cam_path = frame_dir / 'camera_rgb.jpg'
            if cam_path.exists():
                img = np.array(Image.open(str(cam_path)), dtype=np.uint8)
                sample['camera_rgb'] = torch.from_numpy(img)  # (H, W, 3)

        # ── Radar BEV ─────────────────────────────────────────────────────
        if self._load_radar and _PIL_AVAILABLE:
            bev_path = frame_dir / 'radar_bev.png'
            if bev_path.exists():
                img = np.array(Image.open(str(bev_path)), dtype=np.uint8)
                sample['radar_bev'] = torch.from_numpy(img)  # (H, W, 3)

        # ── Obstacles ─────────────────────────────────────────────────────
        obs_path = frame_dir / 'obstacles.json'
        obstacles_raw = []
        if obs_path.exists():
            with open(obs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sample['timestamp']         = float(data.get('timestamp', 0.0))
            sample['sensor_latency_ms'] = float(data.get('sensor_latency_ms', 0.0))
            obstacles_raw               = data.get('obstacles', [])

        # NN detection labels: (max_obstacles, 8) — [cx,cy,cz,l,w,h,heading_rad,class_id]
        if self._max_obs is not None:
            det_labels = np.zeros((self._max_obs, 8), dtype=np.float32)
            risk_labels = np.zeros((self._max_obs, 3), dtype=np.float32)
            valid_mask  = np.zeros(self._max_obs, dtype=bool)

            for i, obs in enumerate(obstacles_raw[:self._max_obs]):
                nn = obs.get('nn_label', {})
                d3 = nn.get('detection_3d', {})
                rk = nn.get('risk_label', {})
                c  = d3.get('center', [0, 0, 0])
                sz = d3.get('size',   [0, 0, 0])
                det_labels[i] = [
                    c[0], c[1], c[2],
                    sz[0], sz[1], sz[2],
                    float(d3.get('heading_rad', 0.0)),
                    float(d3.get('class_id', 5)),
                ]
                ttc = rk.get('ttc_s')
                risk_labels[i] = [
                    float(ttc) if ttc is not None else 999.0,
                    float(rk.get('severity_id', 0)),
                    float(rk.get('is_threat', False)),
                ]
                valid_mask[i] = True

            sample['detection_labels'] = torch.from_numpy(det_labels)
            sample['risk_labels']      = torch.from_numpy(risk_labels)
            sample['obstacle_mask']    = torch.from_numpy(valid_mask)

        sample['obstacles_raw'] = obstacles_raw

        # ── Ego state ─────────────────────────────────────────────────────
        ego_path = frame_dir / 'ego_state.json'
        if ego_path.exists():
            with open(ego_path, 'r', encoding='utf-8') as f:
                ego = json.load(f)
            sample['ego_speed_ms']  = torch.tensor(float(ego.get('speed_ms', 0.0)))
            sample['ego_speed_kmh'] = torch.tensor(float(ego.get('speed_kmh', 0.0)))
            ctrl = ego.get('control', {})
            sample['ego_control']   = torch.tensor([
                float(ctrl.get('throttle', 0.0)),
                float(ctrl.get('brake',    0.0)),
                float(ctrl.get('steer',    0.0)),
            ])
            pos = ego.get('position_xyz', [0, 0, 0])
            sample['ego_position']  = torch.tensor(pos, dtype=torch.float32)
            sample['ego_heading_rad'] = torch.tensor(
                float(ego.get('heading_rad', 0.0))
            )

        if self._transform is not None:
            sample = self._transform(sample)

        return sample

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def session_dir(self) -> Path:
        return self._session_dir


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-session dataset (concatenates multiple sessions)
# ══════════════════════════════════════════════════════════════════════════════

class MultiSessionDataset(Dataset):
    """
    Concatenate multiple ADASDriveDataset sessions into one dataset.

    Useful for training across several recording runs.

    Args:
        session_dirs: List of session directory paths.
        **kwargs    : Passed to ADASDriveDataset constructor.
    """

    def __init__(self, session_dirs: List[str], **kwargs):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required: pip install torch")
        self._datasets = [
            ADASDriveDataset(d, **kwargs) for d in session_dirs
        ]
        self._lengths  = [len(d) for d in self._datasets]
        self._total    = sum(self._lengths)
        logger.info(
            "MultiSessionDataset: %d sessions | %d total frames",
            len(self._datasets), self._total,
        )

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict:
        offset = 0
        for ds, length in zip(self._datasets, self._lengths):
            if idx < offset + length:
                return ds[idx - offset]
            offset += length
        raise IndexError(f"Index {idx} out of range for {self._total} frames")


# ══════════════════════════════════════════════════════════════════════════════
#  TensorFlow dataset builder
# ══════════════════════════════════════════════════════════════════════════════

def build_tf_dataset(
    session_dir     : str,
    batch_size      : int  = 8,
    shuffle         : bool = True,
    shuffle_buffer  : int  = 500,
    load_lidar_filt : bool = True,
    load_camera     : bool = False,
    prefetch        : int  = 2,
) -> 'tf.data.Dataset':
    """
    Build a tf.data.Dataset from an ADAS session recording.

    Yields batches of:
        lidar_filtered : (batch, M, 3) float32 — may be ragged if M varies
        detection_labels: (batch, 20, 8) float32
        ego_speed_ms   : (batch,) float32

    Args:
        session_dir    : Path to session directory.
        batch_size     : Training batch size.
        shuffle        : Shuffle frame order.
        shuffle_buffer : Shuffle buffer size (frames).
        load_lidar_filt: Load filtered LiDAR point clouds.
        load_camera    : Load camera images.
        prefetch       : Number of batches to prefetch.

    Returns:
        tf.data.Dataset

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required: pip install tensorflow")

    frames = discover_frames(Path(session_dir))
    frame_paths = [str(f) for f in frames]

    def _load_frame(frame_dir_bytes: bytes) -> Tuple:
        frame_dir = Path(frame_dir_bytes.decode('utf-8'))

        # LiDAR
        lidar = np.zeros((0, 3), dtype=np.float32)
        if load_lidar_filt:
            p = frame_dir / 'lidar_filtered.npy'
            if p.exists():
                lidar = np.load(str(p)).astype(np.float32)

        # Detection labels (max 20 obstacles)
        det_labels = np.zeros((20, 8), dtype=np.float32)
        ego_speed  = np.float32(0.0)

        obs_path = frame_dir / 'obstacles.json'
        if obs_path.exists():
            with open(obs_path) as f:
                data = json.load(f)
            for i, obs in enumerate(data.get('obstacles', [])[:20]):
                nn = obs.get('nn_label', {}).get('detection_3d', {})
                c  = nn.get('center', [0, 0, 0])
                sz = nn.get('size',   [0, 0, 0])
                det_labels[i] = [c[0], c[1], c[2], sz[0], sz[1], sz[2],
                                  float(nn.get('heading_rad', 0.0)),
                                  float(nn.get('class_id', 5))]

        ego_path = frame_dir / 'ego_state.json'
        if ego_path.exists():
            with open(ego_path) as f:
                ego = json.load(f)
            ego_speed = np.float32(ego.get('speed_ms', 0.0))

        return lidar, det_labels, ego_speed

    def tf_load(frame_path: tf.Tensor) -> dict:
        lidar, det_labels, ego_speed = tf.py_function(
            func=_load_frame,
            inp=[frame_path],
            Tout=[tf.float32, tf.float32, tf.float32],
        )
        # Set static shapes where possible
        det_labels.set_shape([20, 8])
        return {
            'lidar_filtered'  : lidar,
            'detection_labels': det_labels,
            'ego_speed_ms'    : ego_speed,
        }

    ds = tf.data.Dataset.from_tensor_slices(frame_paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(prefetch)
    return ds


# ══════════════════════════════════════════════════════════════════════════════
#  Collate functions for PyTorch DataLoader
# ══════════════════════════════════════════════════════════════════════════════

def collate_variable_lidar(batch: List[dict]) -> dict:
    """
    Collate fn for variable-length LiDAR point clouds.

    Pads all clouds in the batch to the same length (max N in batch).
    Adds 'lidar_num_points' to track real point count before padding.

    Use with:
        DataLoader(ds, collate_fn=collate_variable_lidar)
    """
    import torch

    result: dict = {}

    # Detect keys from first sample
    for key in batch[0].keys():
        if key == 'lidar_filtered':
            clouds = [s['lidar_filtered'] for s in batch]
            max_n  = max(c.shape[0] for c in clouds)
            padded = torch.zeros(len(clouds), max_n, 3, dtype=torch.float32)
            counts = torch.zeros(len(clouds), dtype=torch.long)
            for i, c in enumerate(clouds):
                n = c.shape[0]
                padded[i, :n] = c
                counts[i]     = n
            result['lidar_filtered']    = padded
            result['lidar_num_points']  = counts
        elif key == 'lidar_raw':
            clouds = [s['lidar_raw'] for s in batch]
            max_n  = max(c.shape[0] for c in clouds)
            padded = torch.zeros(len(clouds), max_n, 4, dtype=torch.float32)
            counts = torch.zeros(len(clouds), dtype=torch.long)
            for i, c in enumerate(clouds):
                n = c.shape[0]
                padded[i, :n] = c
                counts[i]     = n
            result['lidar_raw']          = padded
            result['lidar_raw_num_pts']  = counts
        elif key == 'obstacles_raw':
            result[key] = [s[key] for s in batch]   # keep as list of lists
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([s[key] for s in batch])
        else:
            result[key] = [s[key] for s in batch]

    return result
