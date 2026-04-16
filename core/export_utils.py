"""
Export Utility Functions — Task 07
====================================

Pure serialisation helpers used by DataExporter and verify_export.
No classes — only module-level functions.

All functions are stateless and thread-safe.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

if TYPE_CHECKING:
    from lidar_fusion import FusedObstacle


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS_ID_MAP — nuScenes compatible class ids
# ══════════════════════════════════════════════════════════════════════════════

CLASS_ID_MAP = {
    'car'        : 0,
    'vehicle'    : 0,
    'truck'      : 1,
    'bus'        : 1,
    'pedestrian' : 2,
    'person'     : 2,
    'cyclist'    : 3,
    'bicycle'    : 3,
    'motorcycle' : 4,
    'motorbike'  : 4,
    'unknown'    : 5,
    'structure'  : 5,
}


# ══════════════════════════════════════════════════════════════════════════════
#  JSON serialisation
# ══════════════════════════════════════════════════════════════════════════════

def json_serialise_safe(obj: Any) -> Any:
    """
    Recursively convert an object to a JSON-serialisable form.

    Handles: np.ndarray -> list, np.floating -> float, np.integer -> int,
             float('inf') -> None, float('nan') -> None,
             Path -> str, dataclass -> dict.

    Used as the 'default' parameter to json.dumps().
    Never raises TypeError — always returns something serialisable.

    WHY NONE FOR INF:
        JSON does not support Infinity. None (null in JSON) signals
        "no value" which is correct for TTC=inf (not approaching).
        NN training code must handle null TTC as a special case (not a threat).
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, '__dataclass_fields__'):
        import dataclasses
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def write_json_atomic(data: dict, path: Path) -> None:
    """
    Write a JSON file atomically using a temp file + rename.

    WHY ATOMIC WRITES:
        If the process crashes mid-write, a partial JSON file is worse than
        no file (it corrupts the dataset and is hard to detect).
        Writing to a temp file then renaming is atomic on POSIX filesystems.
        The rename either completes fully or does not happen — no partial files.

    Algorithm:
        1. Write to path.with_suffix('.tmp')
        2. Flush + fsync (ensure OS buffers are committed to disk)
        3. os.replace(tmp_path, path) — atomic rename on POSIX/Windows

    Args:
        data: JSON-serialisable dict.
        path: Target .json file path.
    """
    tmp_path = path.with_suffix('.tmp')
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, default=json_serialise_safe, indent=2,
                      ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Atomic JSON write failed for {path}: {e}") from e


# ══════════════════════════════════════════════════════════════════════════════
#  NumPy saving
# ══════════════════════════════════════════════════════════════════════════════

def save_numpy_compressed(array: np.ndarray, path: Path) -> None:
    """
    Save a numpy array using np.save (uncompressed binary .npy format).

    WHY UNCOMPRESSED .npy (not .npz):
        np.load('.npy') returns the array immediately — zero decompression time.
        PointPillars and SECOND expect .npy format directly.
        .npz compression saves ~30% disk but adds 5-20 ms load time per file.
        At 10 Hz x 60 minutes = 36,000 files, fast loading matters.
        The header magic 0x9304 allows instant dtype/shape inspection
        without loading the full array.

    Verifies:
        Array dtype is float32 (critical — NN inputs must not be float64).
        Array is C-contiguous (required by PointPillars voxelisation).

    Args:
        array: numpy array to save.
        path : .npy file path.
    """
    if array.dtype != np.float32:
        raise TypeError(
            f"Array dtype must be float32 for NN compatibility, "
            f"got {array.dtype}. Cast with array.astype(np.float32) before saving."
        )
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    np.save(str(path), array)


# ══════════════════════════════════════════════════════════════════════════════
#  Image saving
# ══════════════════════════════════════════════════════════════════════════════

def save_image_rgb(
    image_rgb : np.ndarray,
    path      : Path,
    quality   : int = 90
) -> None:
    """
    Save an RGB image using Pillow.

    Supports .jpg/.jpeg (JPEG) and .png (PNG) based on file extension.
    Always verifies that the input is uint8 and 3-channel RGB.

    JPEG settings:
        quality=90, subsampling=0 (4:4:4 — no chroma downsampling).
        Subsampling=0 preserves fine colour detail needed for depth
        estimation and semantic segmentation training.

    PNG settings:
        compress_level=3 (fast, moderate compression).
        Higher levels (6-9) are too slow for real-time recording.

    Args:
        image_rgb: np.ndarray (H, W, 3) uint8 RGB.
        path     : Output file path (.jpg or .png).
        quality  : JPEG quality (1-95). Ignored for PNG.
    """
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required for image saving: pip install Pillow")
    if image_rgb.dtype != np.uint8:
        raise TypeError(f"Image must be uint8, got {image_rgb.dtype}")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Image must be (H,W,3), got {image_rgb.shape}")

    pil_img = Image.fromarray(image_rgb, mode='RGB')
    ext = path.suffix.lower()

    if ext in ('.jpg', '.jpeg'):
        pil_img.save(str(path), format='JPEG',
                     quality=quality, subsampling=0,
                     optimize=False)  # optimize=True is 3x slower for marginal gain
    elif ext == '.png':
        pil_img.save(str(path), format='PNG', compress_level=3)
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


# ══════════════════════════════════════════════════════════════════════════════
#  NN label builders
# ══════════════════════════════════════════════════════════════════════════════

def build_nn_label_detection_3d(obs: 'FusedObstacle') -> dict:
    """
    Build a nuScenes-compatible 3D detection label from a FusedObstacle.

    nuScenes format (used by PointPillars, SECOND, CenterPoint, BEVFusion):
      - center: [x, y, z] in metres, ego frame
      - size:   [length, width, height] in metres
      - heading: yaw in radians [-pi, pi]
      - velocity: [vx, vy] in m/s (2D BEV, no z velocity)
      - class_name: string
      - score: detection confidence [0, 1]

    WHY RADIANS FOR HEADING:
        All major 3D detection networks output and expect radians.
        Converting at training time risks subtle sign errors.
        Store in radians now, avoid conversion later.

    Returns:
        dict compatible with nuScenes detection annotation format.
    """
    return {
        'class_id'   : CLASS_ID_MAP.get(obs.fused_class, 5),
        'class_name' : obs.fused_class,
        'center'     : [round(float(v), 4) for v in obs.center_xyz],
        'size'       : [round(float(v), 4) for v in obs.extent_lwh],
        'heading_rad': round(float(np.radians(obs.heading_deg)), 6),
        'velocity'   : [round(float(obs.velocity_xyz[0]), 4),
                        round(float(obs.velocity_xyz[1]), 4)],
        'score'      : round(float(obs.fused_confidence), 4),
    }


def build_nn_label_risk(obs: 'FusedObstacle') -> dict:
    """
    Build a risk assessment label for safety prediction NNs.

    severity_id: 0=SAFE, 1=WARN, 2=BRAKE
    is_threat: True if severity is WARN or BRAKE.
    """
    SEVERITY_ID = {'SAFE': 0, 'WARN': 1, 'BRAKE': 2}
    return {
        'ttc_s'        : (round(float(obs.ttc_seconds), 3)
                          if obs.ttc_seconds != float('inf') else None),
        'severity_id'  : SEVERITY_ID.get(obs.severity, 0),
        'severity_name': obs.severity,
        'is_threat'    : obs.severity in ('WARN', 'BRAKE'),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Integrity / checksum helpers
# ══════════════════════════════════════════════════════════════════════════════

def md5_of_file(path: Path, chunk_size: int = 65536) -> str:
    """Compute MD5 hex digest of a file without loading it fully into memory."""
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_npy_shape(path: Path, expected_ndim: int, expected_dtype: np.dtype) -> bool:
    """
    Quick shape/dtype check for a .npy file without loading the full array.

    Uses np.lib.format.open_memmap for zero-copy header reading.
    Returns True if shape and dtype match, False otherwise.
    """
    try:
        arr = np.load(str(path), mmap_mode='r')
        return arr.ndim == expected_ndim and arr.dtype == expected_dtype
    except Exception:
        return False


def load_json_safe(path: Path) -> dict:
    """Load a JSON file, returning an empty dict on any error."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}
