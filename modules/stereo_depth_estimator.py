"""
stereo_depth_estimator.py
─────────────────────────
Computes obstacle distance using a stereo camera pair.

HOW STEREO DEPTH WORKS (plain English):
  When you look at an object with both eyes, your left eye and right
  eye see it at slightly different positions. The brain uses that
  difference (called disparity) to judge how far away the object is.
  A stereo camera pair works the same way.

  The depth formula is:
      Z = (focal_length × baseline) / disparity

  Where:
    Z           = depth to the object in metres
    focal_length= camera property in pixels (derived from FOV)
    baseline    = physical distance between the two cameras (0.54 m)
    disparity   = how many pixels the object shifted between left
                  and right images (bigger shift = closer object)

WHAT THIS MODULE DOES:
  1. Takes a LEFT frame and a RIGHT frame from CARLA
  2. Computes a dense disparity map using OpenCV SGBM algorithm
  3. Converts disparity to metric depth in metres
  4. Answers "how far is the object inside this bounding box?"
     for use by ObstacleDetector

USED BY:
  - modules/driving_agent.py  (calls compute() each frame)
  - modules/obstacle_detector.py (calls get_depth_at_bbox() per bbox)
  - evaluate_all_methods.py   (calls both for the FYP experiment)
"""

import cv2
import numpy as np
import time
from typing import Optional


class StereoDepthEstimator:
    """
    Computes per-pixel metric depth from a calibrated stereo camera pair
    using OpenCV's Semi-Global Block Matching (SGBM) algorithm.

    Usage pattern each frame:
        estimator.compute(left_frame, right_frame)   # build depth map
        dist = estimator.get_depth_at_bbox(x1,y1,x2,y2)  # query per box
    """

    def __init__(self,
                 image_width:  int   = 1280,
                 image_height: int   = 720,
                 fov_degrees:  float = 90.0,
                 baseline_m:   float = 0.54):
        """
        Prepares the stereo depth estimator.

        In CARLA, cameras have no lens distortion and we know the exact FOV,
        so we can compute all camera parameters mathematically — no physical
        calibration checkerboard is needed.

        Args:
            image_width:  Camera frame width in pixels (default 1280)
            image_height: Camera frame height in pixels (default 720)
            fov_degrees:  Horizontal field of view of each camera (default 90°)
            baseline_m:   Physical distance between the two camera lenses in
                          metres (default 0.54 m). This matches the y-offset
                          used when spawning the right camera in main.py.
        """

        # ── STEP A: Camera intrinsic matrix ─────────────────────────────────
        # The intrinsic matrix K describes the camera's internal geometry.
        # For a CARLA camera with known FOV and resolution, we compute it
        # analytically. fx and fy are the focal lengths in pixels — they
        # convert real-world distances to pixel distances on the image.
        fov_rad = np.radians(fov_degrees)
        fx = fy = (image_width / 2.0) / np.tan(fov_rad / 2.0)
        cx = image_width  / 2.0   # optical centre = middle of image
        cy = image_height / 2.0
        K  = np.array([[fx, 0,  cx],
                       [0,  fy, cy],
                       [0,  0,  1 ]], dtype=np.float64)
        D  = np.zeros(5, dtype=np.float64)   # no lens distortion in CARLA

        # Store the values we need later for the depth formula
        self._fx       = fx
        self._baseline = baseline_m
        self._K        = K
        self._img_w    = image_width
        self._img_h    = image_height

        # ── STEP B: Right camera position relative to left ───────────────────
        # The right camera is directly to the right of the left camera —
        # same height, same angle, just shifted sideways by baseline_m.
        # R = identity means no rotation difference between the cameras.
        # In OpenCV stereo convention, T is the translation from the LEFT
        # camera to the RIGHT camera expressed in the LEFT camera's frame.
        # Since the right camera is offset along the camera x-axis (left→right),
        # T = [baseline_m, 0, 0].
        R = np.eye(3, dtype=np.float64)
        T = np.array([baseline_m, 0.0, 0.0], dtype=np.float64)

        # ── STEP C: Stereo rectification ────────────────────────────────────
        # Rectification mathematically rotates both images so that matching
        # pixels in the left and right frames always appear on the same
        # horizontal row. This is required by SGBM which searches for
        # pixel matches only along horizontal lines (epipolar lines).
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K, D, K, D,
            (image_width, image_height),
            R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0          # alpha=0 crops to valid pixels only (no black borders)
        )
        # Q is the 4×4 reprojection matrix. It converts (x, y, disparity)
        # to real-world (X, Y, Z) coordinates in metres.
        self._Q = Q

        # Build remap lookup tables for fast per-frame rectification.
        # Instead of computing the rectification transform every frame,
        # we precompute a pixel mapping table here and just apply it.
        self._map1_L, self._map2_L = cv2.initUndistortRectifyMap(
            K, D, R1, P1, (image_width, image_height), cv2.CV_32FC1)
        self._map1_R, self._map2_R = cv2.initUndistortRectifyMap(
            K, D, R2, P2, (image_width, image_height), cv2.CV_32FC1)

        # ── STEP D: SGBM stereo matcher ──────────────────────────────────────
        # Semi-Global Block Matching (SGBM) is OpenCV's best classical
        # stereo algorithm. It compares small patches of pixels between
        # left and right images to find how far each pixel has shifted.
        # Parameters are tuned for CARLA's clean synthetic textures.
        block = 11   # size of the patch compared between frames (must be odd)
        self._matcher = cv2.StereoSGBM_create(
            minDisparity      = 0,
            numDisparities    = 128,              # max pixel shift to search (must be div by 16)
            blockSize         = block,
            P1                = 8  * 3 * block ** 2,   # smoothness: penalty for 1-px disparity change
            P2                = 32 * 3 * block ** 2,   # smoothness: penalty for larger changes
            disp12MaxDiff     = 1,                # left-right consistency check tolerance (pixels)
            uniquenessRatio   = 10,               # reject matches within 10% of next-best
            speckleWindowSize = 100,              # remove isolated speckle noise regions
            speckleRange      = 32,               # max disparity variation within a speckle
            mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY  # best quality mode
        )

        # ── STEP E: Internal state ───────────────────────────────────────────
        self._depth_map       = None   # float32 array, metres, shape (H, W)
        self._disparity_map   = None   # float32 raw disparity, shape (H, W)
        self._computed        = False  # True after first successful compute()
        self._last_compute_ms = 0.0    # ms taken by last compute() call
        self._valid_pixel_pct = 0.0    # % of pixels with valid depth (0-100)

        print(f"✓ StereoDepthEstimator initialized")
        print(f"  Focal length: {fx:.1f} px | Baseline: {baseline_m} m")
        print(f"  Theoretical range: 0.5 m – {fx * baseline_m / 1.0:.0f} m")

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ────────────────────────────────────────────────────────────────────────

    def compute(self,
                left_frame:  np.ndarray,
                right_frame: np.ndarray) -> None:
        """
        Computes a per-pixel depth map from a stereo frame pair.

        Call this ONCE PER FRAME before calling get_depth_at_bbox().
        The result is stored internally and overwritten each call.

        Args:
            left_frame:  BGR image from the LEFT camera  (H×W×3 uint8)
            right_frame: BGR image from the RIGHT camera (H×W×3 uint8)
                         Must be captured at the same moment as left_frame.
        """
        t0 = time.perf_counter()

        # Step 1 — Grayscale conversion
        # SGBM works on intensity values, not colour. Converting to
        # grayscale reduces computation and avoids colour channel noise.
        left_gray  = cv2.cvtColor(left_frame,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Step 2 — Rectification
        # Apply the precomputed remap tables to align both images so
        # matching pixels lie on the same horizontal row (epipolar constraint).
        left_rect  = cv2.remap(left_gray,  self._map1_L, self._map2_L,
                               cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_gray, self._map1_R, self._map2_R,
                               cv2.INTER_LINEAR)

        # Step 3 — Disparity computation
        # SGBM searches for each pixel in left_rect within right_rect,
        # sliding horizontally. The horizontal shift found is the disparity.
        raw_disp = self._matcher.compute(left_rect, right_rect)

        # Step 4 — Convert to float32 pixels
        # SGBM returns disparity scaled by 16 (for sub-pixel precision).
        # Dividing by 16.0 gives the actual disparity in pixels.
        disp = raw_disp.astype(np.float32) / 16.0

        # Step 5 — Convert disparity to metric depth
        # Z = (focal_length × baseline) / disparity
        # A larger disparity (bigger pixel shift) means the object is
        # physically closer. We guard against division by zero by only
        # computing depth where disparity > 0.1 pixels.
        depth = np.full(disp.shape, np.nan, dtype=np.float32)
        valid_mask = disp > 0.1
        depth[valid_mask] = (self._fx * self._baseline) / disp[valid_mask]

        # Clamp to physically meaningful range: 0.5 m to 200 m.
        # Values outside this range are noise, not real obstacles.
        depth[(depth < 0.5) | (depth > 200.0)] = np.nan

        # Step 6 — Store results and timing
        self._depth_map       = depth
        self._disparity_map   = disp
        self._valid_pixel_pct = 100.0 * float(valid_mask.sum()) / float(disp.size)
        self._last_compute_ms = (time.perf_counter() - t0) * 1000.0
        self._computed        = True

    def get_depth_at_bbox(self,
                          x1: int, y1: int,
                          x2: int, y2: int,
                          min_valid: int = 20) -> Optional[float]:
        """
        Returns the estimated distance in metres to the object inside a
        YOLO bounding box.

        WHY CENTRAL 60%?
          The edges of a bounding box often contain background pixels
          (e.g. sky or road behind a car). Using only the central 60%
          of the box avoids mixing foreground and background depths.

        WHY 25th PERCENTILE?
          We want the distance to the nearest surface of the object.
          The 25th percentile picks a value near the minimum while
          ignoring noisy pixels that may read too close.

        Args:
            x1, y1: Top-left corner of bounding box (pixels)
            x2, y2: Bottom-right corner of bounding box (pixels)
            min_valid: Minimum valid pixels needed. Returns None if
                       fewer are found (not enough stereo texture data).

        Returns:
            Distance in metres (float), or None if unreliable.
        """
        if not self._computed or self._depth_map is None:
            return None

        # Shrink the bounding box by 20% on each edge → central 60% remains.
        margin_x = int((x2 - x1) * 0.20)
        margin_y = int((y2 - y1) * 0.20)

        # Clamp to image bounds so we never read outside the array.
        rx1 = max(0, x1 + margin_x)
        rx2 = min(self._depth_map.shape[1], x2 - margin_x)
        ry1 = max(0, y1 + margin_y)
        ry2 = min(self._depth_map.shape[0], y2 - margin_y)

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        region = self._depth_map[ry1:ry2, rx1:rx2]
        valid  = region[~np.isnan(region)]

        # Not enough valid stereo pixels in this region — fall back to pinhole.
        if len(valid) < min_valid:
            return None

        # 25th percentile ≈ distance to the nearest surface of the object.
        return float(np.percentile(valid, 25))

    def get_depth_map(self) -> Optional[np.ndarray]:
        """
        Returns the full per-pixel depth map (float32, metres).
        Returns None if compute() has not been called yet.
        Used by the visualiser to colour-code the depth image.
        """
        return self._depth_map if self._computed else None

    def get_disparity_colormap(self,
                               max_depth_m: float = 50.0
                               ) -> Optional[np.ndarray]:
        """
        Returns a colour-coded BGR image of the depth map for display.

        Colour meaning:
          RED   = very close (near 0 m)
          GREEN = medium distance
          BLUE  = far away (near max_depth_m)
        NaN pixels (no valid depth) are shown as black.

        Args:
            max_depth_m: Depths beyond this value are shown as blue.

        Returns:
            BGR image uint8 same size as input frame, or None.
        """
        if not self._computed or self._depth_map is None:
            return None

        depth = self._depth_map.copy()

        # Invert the normalisation so that 0 m → 255 (red in JET) and
        # max_depth_m → 0 (blue in JET). This makes close objects red.
        depth_clipped = np.clip(depth, 0.0, max_depth_m)
        norm = ((max_depth_m - depth_clipped) / max_depth_m * 255.0).astype(np.uint8)

        # Apply JET colormap: 0=blue(far), 128=green(mid), 255=red(near)
        color_img = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Restore black for pixels with no valid stereo depth (NaN)
        color_img[np.isnan(depth)] = 0

        return color_img

    def get_status(self) -> dict:
        """
        Returns a summary dict for logging and HUD display.

        Keys:
          'active'          (bool)  — True if compute() called this session
          'last_compute_ms' (float) — time taken by last compute() in ms
          'valid_pixel_pct' (float) — % of pixels with valid depth (0–100)
        """
        return {
            'active':          self._computed,
            'last_compute_ms': self._last_compute_ms,
            'valid_pixel_pct': self._valid_pixel_pct,
        }


# ────────────────────────────────────────────────────────────────────────────
# SELF-TEST — run directly: python modules/stereo_depth_estimator.py
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    Quick self-test with synthetic frames.
    A white rectangle on the left appears 20 px to the right on the
    right frame — simulating disparity of 20 pixels.

    Expected depth ≈ (640 × 0.54) / 20 = 17.28 m

    Run with:  python modules/stereo_depth_estimator.py
    Expected output: PASS if computed depth is within ±3 m of 17.28 m
    """
    print("=== StereoDepthEstimator self-test ===")

    W, H = 1280, 720
    # Left frame: white rectangle at columns 580-680, rows 300-420
    left_frame = np.zeros((H, W, 3), dtype=np.uint8)
    left_frame[300:420, 580:680] = 200   # bright grey (not pure white avoids clipping)

    # Right frame: same rectangle shifted RIGHT by 20 px (simulates disparity = 20 px)
    # In stereo, if the right image has an object shifted right by D pixels,
    # the disparity is D → depth = fx * baseline / D = 640 * 0.54 / 20 ≈ 17.28 m
    right_frame = np.zeros((H, W, 3), dtype=np.uint8)
    right_frame[300:420, 600:700] = 200

    expected_m = (640.0 * 0.54) / 20.0   # ≈ 17.28 m

    est = StereoDepthEstimator()
    est.compute(left_frame, right_frame)

    status = est.get_status()
    print(f"  Compute time  : {status['last_compute_ms']:.1f} ms")
    print(f"  Valid pixels  : {status['valid_pixel_pct']:.1f}%")

    depth = est.get_depth_at_bbox(580, 300, 680, 420)
    if depth is not None and abs(depth - expected_m) < 3.0:
        print(f"  PASS  — measured {depth:.2f} m, expected {expected_m:.2f} m")
    else:
        print(f"  FAIL  — measured {depth}, expected {expected_m:.2f} m")
        print("  Note: synthetic frames have minimal texture; SGBM may")
        print("  find fewer matches than real-world CARLA frames.")

    # Verify colormap does not crash
    cmap = est.get_disparity_colormap()
    assert cmap is not None and cmap.shape == (H, W, 3), "Colormap shape mismatch"
    print(f"  Colormap OK   — shape {cmap.shape}")
    print("=== self-test complete ===")
