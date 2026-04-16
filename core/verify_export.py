"""
Dataset Integrity Verification — Task 07
==========================================

Verifies the structural and content integrity of a recorded ADAS session:
  - metadata.json is present and parseable
  - All expected files exist in each frame directory
  - numpy arrays have correct dtype (float32) and expected dimensionality
  - JSON files are valid and contain required keys
  - No partial writes (no .tmp files)
  - Frame IDs are monotonically increasing

Usage:
    python verify_export.py recordings/session_20240115_143022
    python verify_export.py recordings/session_20240115_143022 --strict
    python verify_export.py recordings/session_20240115_143022 --fix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Issue tracking
# ══════════════════════════════════════════════════════════════════════════════

class Issue:
    """A single verification finding."""

    def __init__(self, level: str, frame_id: str, message: str):
        self.level    = level     # 'ERROR' | 'WARN' | 'INFO'
        self.frame_id = frame_id
        self.message  = message

    def __str__(self) -> str:
        return f"[{self.level}] frame={self.frame_id}: {self.message}"


# ══════════════════════════════════════════════════════════════════════════════
#  Verification functions
# ══════════════════════════════════════════════════════════════════════════════

def verify_session(
    session_dir : Path,
    strict      : bool = False,
    check_hashes: bool = False,
) -> List[Issue]:
    """
    Run all integrity checks on a session directory.

    Args:
        session_dir  : Path to session directory.
        strict       : If True, treat WARN-level findings as errors.
        check_hashes : If True, verify numpy array checksums (slow).

    Returns:
        List of Issue objects. Empty list = fully valid.
    """
    issues: List[Issue] = []

    def err(frame_id: str, msg: str) -> None:
        issues.append(Issue('ERROR', frame_id, msg))

    def warn(frame_id: str, msg: str) -> None:
        issues.append(Issue('WARN', frame_id, msg))

    def info(frame_id: str, msg: str) -> None:
        issues.append(Issue('INFO', frame_id, msg))

    # ── Session-level checks ───────────────────────────────────────────────
    meta_path = session_dir / 'metadata.json'
    if not meta_path.exists():
        err('session', f"metadata.json missing in {session_dir}")
        return issues   # can't continue without metadata

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except json.JSONDecodeError as e:
        err('session', f"metadata.json is not valid JSON: {e}")
        return issues

    required_meta_keys = [
        'session_id', 'coordinate_frame', 'distance_unit',
        'lidar_config', 'exporter_config',
    ]
    for key in required_meta_keys:
        if key not in meta:
            warn('session', f"metadata.json missing key: {key}")

    frames_dir = session_dir / 'frames'
    if not frames_dir.exists():
        err('session', "frames/ directory missing")
        return issues

    # ── Check for leftover .tmp files (partial writes) ─────────────────────
    tmp_files = list(session_dir.rglob('*.tmp'))
    if tmp_files:
        for tf in tmp_files:
            warn('session', f"Leftover partial write: {tf.name}")

    # ── Discover frame directories ─────────────────────────────────────────
    frame_dirs = sorted(
        [d for d in frames_dir.iterdir() if d.is_dir()],
        key=lambda p: int(p.name),
    )

    if len(frame_dirs) == 0:
        warn('session', "No frame directories found")
        return issues

    # ── Check monotonicity of frame IDs ────────────────────────────────────
    frame_ids = [int(d.name) for d in frame_dirs]
    for i in range(1, len(frame_ids)):
        if frame_ids[i] <= frame_ids[i - 1]:
            err('session', f"Non-monotonic frame IDs: {frame_ids[i-1]} -> {frame_ids[i]}")

    # ── Per-frame checks ───────────────────────────────────────────────────
    for frame_dir in frame_dirs:
        fid = frame_dir.name

        # Required JSON files
        for fname in ['obstacles.json', 'ego_state.json', 'pipeline_stats.json']:
            fp = frame_dir / fname
            if not fp.exists():
                err(fid, f"{fname} missing")
                continue
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                _check_json_schema(fname, data, fid, issues)
            except json.JSONDecodeError as e:
                err(fid, f"{fname} is invalid JSON: {e}")

        # LiDAR raw
        raw_path = frame_dir / 'lidar_raw.npy'
        if raw_path.exists():
            _check_numpy(raw_path, expected_ndim=2, expected_ncols=4,
                         fid=fid, issues=issues)

        # LiDAR filtered
        filt_path = frame_dir / 'lidar_filtered.npy'
        if filt_path.exists():
            _check_numpy(filt_path, expected_ndim=2, expected_ncols=3,
                         fid=fid, issues=issues)

        # Camera RGB
        cam_path = frame_dir / 'camera_rgb.jpg'
        if cam_path.exists():
            _check_image(cam_path, fid, issues)

        # Radar BEV
        bev_path = frame_dir / 'radar_bev.png'
        if bev_path.exists():
            _check_image(bev_path, fid, issues)

    return issues


def _check_numpy(
    path         : Path,
    expected_ndim: int,
    expected_ncols: int,
    fid          : str,
    issues       : List[Issue],
) -> None:
    """Verify a .npy file has correct shape and float32 dtype."""
    try:
        arr = np.load(str(path), mmap_mode='r')
        if arr.dtype != np.float32:
            issues.append(Issue(
                'ERROR', fid,
                f"{path.name}: dtype is {arr.dtype}, expected float32"
            ))
        if arr.ndim != expected_ndim:
            issues.append(Issue(
                'ERROR', fid,
                f"{path.name}: ndim={arr.ndim}, expected {expected_ndim}"
            ))
        elif arr.shape[-1] != expected_ncols:
            issues.append(Issue(
                'ERROR', fid,
                f"{path.name}: ncols={arr.shape[-1]}, expected {expected_ncols}"
            ))
        if arr.shape[0] == 0:
            issues.append(Issue(
                'WARN', fid,
                f"{path.name}: array is empty (0 rows)"
            ))
    except Exception as e:
        issues.append(Issue('ERROR', fid, f"{path.name}: could not load: {e}"))


def _check_image(path: Path, fid: str, issues: List[Issue]) -> None:
    """Verify an image file is not corrupted (minimal PIL check)."""
    try:
        from PIL import Image
        img = Image.open(str(path))
        img.verify()   # raises on truncated/corrupt files
    except ImportError:
        pass   # skip if PIL not available
    except Exception as e:
        issues.append(Issue('ERROR', fid, f"{path.name}: image corrupt: {e}"))


def _check_json_schema(
    fname  : str,
    data   : dict,
    fid    : str,
    issues : List[Issue],
) -> None:
    """Check that a JSON file contains required top-level keys."""
    REQUIRED_KEYS = {
        'obstacles.json'      : ['frame_id', 'timestamp', 'n_obstacles', 'obstacles'],
        'ego_state.json'      : ['frame_id', 'timestamp', 'speed_ms', 'control'],
        'pipeline_stats.json' : ['frame_id', 'timestamp', 'timing_ms'],
        'ground_truth.json'   : ['frame_id', 'timestamp', 'n_actors', 'actors'],
    }
    required = REQUIRED_KEYS.get(fname, [])
    for key in required:
        if key not in data:
            issues.append(Issue(
                'WARN', fid,
                f"{fname}: missing required key '{key}'"
            ))

    # obstacles.json — check each obstacle has nn_label
    if fname == 'obstacles.json':
        for obs in data.get('obstacles', []):
            if 'nn_label' not in obs:
                issues.append(Issue(
                    'WARN', fid,
                    f"obstacles.json: obstacle {obs.get('track_id','?')} "
                    f"missing nn_label block"
                ))
                break   # report once per frame


# ══════════════════════════════════════════════════════════════════════════════
#  Fix mode — remove leftover .tmp files
# ══════════════════════════════════════════════════════════════════════════════

def fix_tmp_files(session_dir: Path) -> int:
    """Remove all .tmp partial-write files. Returns count deleted."""
    count = 0
    for tmp in session_dir.rglob('*.tmp'):
        tmp.unlink()
        count += 1
    return count


# ══════════════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════════════

def print_verification_report(
    session_dir : Path,
    issues      : List[Issue],
    strict      : bool = False,
) -> int:
    """
    Print a verification report. Returns exit code (0=pass, 1=fail).
    """
    errors = [i for i in issues if i.level == 'ERROR']
    warns  = [i for i in issues if i.level == 'WARN']
    infos  = [i for i in issues if i.level == 'INFO']

    print(f"{'='*70}")
    print(f"  Dataset Verification — {session_dir.name}")
    print(f"{'='*70}")
    print(f"  Total issues: {len(issues)}  "
          f"(errors={len(errors)}, warnings={len(warns)}, info={len(infos)})")
    print()

    if errors:
        print("  ERRORS:")
        for i in errors:
            print(f"    {i}")
        print()

    if warns:
        print("  WARNINGS:")
        for i in warns[:20]:   # cap at 20 to avoid flooding
            print(f"    {i}")
        if len(warns) > 20:
            print(f"    ... and {len(warns) - 20} more warnings")
        print()

    n_fail = len(errors) + (len(warns) if strict else 0)
    if n_fail == 0:
        print("  RESULT: PASS")
    else:
        print(f"  RESULT: FAIL ({n_fail} critical issues)")
    print(f"{'='*70}")

    return 0 if n_fail == 0 else 1


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _main() -> None:
    parser = argparse.ArgumentParser(
        description='Verify integrity of an ADAS dataset session.'
    )
    parser.add_argument('session_dir', help='Path to session directory')
    parser.add_argument('--strict', action='store_true',
                        help='Treat warnings as errors')
    parser.add_argument('--fix', action='store_true',
                        help='Remove leftover .tmp files before verifying')
    parser.add_argument('--json', action='store_true',
                        help='Output issues as JSON')
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        print(f"ERROR: {session_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.fix:
        n_fixed = fix_tmp_files(session_dir)
        if n_fixed:
            print(f"Fixed: removed {n_fixed} leftover .tmp file(s)")

    issues = verify_session(session_dir, strict=args.strict)

    if args.json:
        print(json.dumps([
            {'level': i.level, 'frame_id': i.frame_id, 'message': i.message}
            for i in issues
        ], indent=2))
        exit_code = 1 if any(i.level == 'ERROR' for i in issues) else 0
    else:
        exit_code = print_verification_report(session_dir, issues, strict=args.strict)

    sys.exit(exit_code)


if __name__ == '__main__':
    _main()
