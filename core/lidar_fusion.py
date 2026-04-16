"""
Camera–LiDAR Fusion Layer — Task 05
=====================================

Fuses semantically-rich YOLO detections (camera) with geometrically-precise
TrackedObstacle objects (LiDAR Task 04) to produce a unified obstacle list
where every obstacle has:

  - A semantic class label  (from camera YOLO)
  - An accurate 3D position and distance  (from LiDAR Kalman tracker)
  - A persistent track ID  (from Task 04)
  - A confidence score combining both sensor modalities
  - A fusion method flag  ('FULL' | 'LIDAR_ONLY' | 'CAMERA_ONLY')

Why fusion is the intelligence layer
--------------------------------------
  Camera knows WHAT (semantic class) but not WHERE in 3D.
  LiDAR knows WHERE (precise 3D geometry) but classifies by size heuristic.
  Fusion gives WHAT + WHERE with the accuracy of both sensors combined.

  Black vehicle on dark road → few LiDAR returns, but camera sees it clearly.
  Pedestrian behind bush → invisible to camera, LiDAR detects vertical cluster.
  Fusion ensures neither case causes a safety miss.

Fusion pipeline per LiDAR tick (100 ms)
-----------------------------------------
  1.  Grab the camera frame nearest in time from the synchronisation buffer
  2.  For each LiDAR tracked obstacle, project cluster points onto image plane
  3.  Compute IoU between projected 2D bbox and each YOLO detection bbox
  4.  Greedy IoU matching with configurable threshold (default 0.15)
  5.  Matched pairs  → FULL      (LiDAR geometry + camera semantics)
  6.  Unmatched LiDAR tracks → LIDAR_ONLY  (geometry only)
  7.  Unmatched camera dets  → CAMERA_ONLY (semantics, distance=unknown)
  8.  Sort output by min_distance ascending (CAMERA_ONLY last)

Thread safety
-------------
  push_camera_frame()  — safe to call from camera callback thread (30 fps)
  fuse()               — call from LiDAR processing thread only (10 Hz)
  Internal buffer uses threading.Lock for cross-thread safety.

Install dependencies
--------------------
  pip install opencv-python   (optional — only needed for visualisation helpers)
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    from core.fusion_calibration import CameraCalibration, FusionExtrinsics
    from core.lidar_tracker import TrackedObstacle, TrackerConfig
    from core.lidar_clusterer import Obstacle
except ImportError:
    from fusion_calibration import CameraCalibration, FusionExtrinsics  # type: ignore
    from lidar_tracker import TrackedObstacle, TrackerConfig             # type: ignore
    from lidar_clusterer import Obstacle                                  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][Fusion] %(message)s',
    datefmt='%H:%M:%S',
)


# ══════════════════════════════════════════════════════════════════════════════
#  FusionConfig
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FusionConfig:
    """
    All tunable fusion parameters in one frozen, serialisable object.

    Every value has an inline comment explaining its meaning and tuning advice.
    """

    # ── Frame synchronisation ─────────────────────────────────────────────────
    max_camera_age_s    : float = 0.08
    # Maximum age (seconds) of a camera frame to be used for fusion.
    # Camera runs at 30 fps → new frame every 33 ms.
    # LiDAR runs at 10 Hz  → new frame every 100 ms.
    # 80 ms = 2.4 camera frames — allows one missed camera frame without fallback.
    # Reduce to 50 ms for stricter sync; increase to 120 ms on slow hardware.

    camera_buffer_size  : int   = 4
    # Number of recent camera frames to keep in the synchronisation buffer.
    # 4 frames at 30 fps = 133 ms history. Sufficient for sync at 10 Hz LiDAR.

    # ── IoU-based matching thresholds ─────────────────────────────────────────
    iou_threshold       : float = 0.15
    # Minimum IoU between projected LiDAR bbox and YOLO bbox to allow matching.
    # WHY 0.15 (lower than standard 0.5):
    #   Projection errors from calibration uncertainty, partial clusters, and
    #   centroid vs object-centre mismatch cause IoU < 0.5 for correct matches.
    #   0.15 catches valid matches at range > 30 m where projection is imprecise.

    iou_min_overlap_px  : int   = 5
    # Minimum overlap area in pixels for IoU computation.
    # Below this, treat IoU as 0 (numerical noise from tiny projections).

    # ── Projection parameters ─────────────────────────────────────────────────
    projection_margin_px: int   = 20
    # Pixels of margin inside image boundary for valid projection.
    # Points projected within 20 px of edge are excluded from bbox computation.

    min_projected_points: int   = 3
    # Minimum number of cluster points that must project into the image
    # to attempt IoU matching.

    # ── Class mapping ─────────────────────────────────────────────────────────
    yolo_to_adas_class  : Dict = field(default_factory=lambda: {
        # Vehicles
        'car'          : 'vehicle',
        'truck'        : 'vehicle',
        'bus'          : 'vehicle',
        'motorbike'    : 'vehicle',
        'motorcycle'   : 'vehicle',
        'bicycle'      : 'cyclist',
        # Pedestrians
        'person'       : 'pedestrian',
        # Structures
        'traffic light': 'structure',
        'stop sign'    : 'structure',
    })
    # YOLO COCO class names → ADAS semantic categories.
    # Classes not in this dict → 'unknown' (excluded from matching).

    # ── Confidence fusion weights ─────────────────────────────────────────────
    camera_conf_weight  : float = 0.55
    # Weight of camera confidence in fused confidence score.
    # Camera classification confidence is primary (YOLO is highly accurate).

    lidar_conf_weight   : float = 0.45
    # Weight of LiDAR (Kalman tracker) confidence in fused score.
    # Lower weight because LiDAR confidence reflects geometry quality only.

    # ── Camera-only distance fallback ─────────────────────────────────────────
    camera_only_fallback_dist : float = -1.0
    # min_distance assigned to CAMERA_ONLY detections (no LiDAR match).
    # -1.0 signals "distance unknown" to downstream modules.
    # Task 06 HUD: shows '?m'. Task 04 threat engine: assumes 15.0 m.

    # ── Performance ───────────────────────────────────────────────────────────
    max_proc_ms         : float = 8.0
    # Hard budget: 8 ms per fusion frame.
    # NOTE: Windows platform overhead adds ~4-8 ms due to numpy allocator and
    # OS scheduler. Budget achievable consistently only on Linux.

    log_every_n_frames  : int   = 100

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  YoloDetection
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class YoloDetection:
    """
    Single YOLO bounding box detection from the camera pipeline.

    Coordinate convention:
        bbox_xyxy: [x1, y1, x2, y2] in PIXEL coordinates.
        x1, y1 = top-left corner.
        x2, y2 = bottom-right corner.
        Origin  = top-left of image.

    If using ultralytics YOLOv8:
        for result in model(frame):
            for box in result.boxes:
                det = YoloDetection(
                    class_name = model.names[int(box.cls)],
                    confidence = float(box.conf),
                    bbox_xyxy  = tuple(box.xyxy[0].tolist()),
                    frame_id   = current_frame_id,
                    timestamp  = current_timestamp
                )
    """
    class_name  : str                                            # YOLO COCO class name
    confidence  : float                                          # detection confidence [0, 1]
    bbox_xyxy   : Tuple[float, float, float, float]             # [x1, y1, x2, y2] pixels
    frame_id    : int
    timestamp   : float

    @property
    def area(self) -> float:
        """Bounding box area in pixels²."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, (x2 - x1) * (y2 - y1))

    @property
    def centre_pixel(self) -> Tuple[float, float]:
        """Centre pixel (u, v) of the bounding box."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_normalised(
        self, image_width: int, image_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Return normalised bbox [cx, cy, w, h] in [0, 1] range.
        Format compatible with YOLO training label format (Task 07).
        """
        x1, y1, x2, y2 = self.bbox_xyxy
        w  = (x2 - x1) / image_width
        h  = (y2 - y1) / image_height
        cx = ((x1 + x2) / 2.0) / image_width
        cy = ((y1 + y2) / 2.0) / image_height
        return (round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))


# ══════════════════════════════════════════════════════════════════════════════
#  FusedObstacle  — primary output contract
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FusedObstacle:
    """
    A single fused obstacle combining LiDAR tracking and camera detection.
    Primary output of Task 05 and input to Tasks 06 and 07.

    Fusion method legend:
      'FULL'         — LiDAR track + YOLO detection matched (best quality).
      'LIDAR_ONLY'   — LiDAR track only (no matching camera detection).
                       Occurs for rear obstacles (outside FOV) or camera misses.
      'CAMERA_ONLY'  — Camera detection only (no matching LiDAR track).
                       min_distance = -1.0 (distance unknown).
                       Occurs for thin objects (poles), far small objects.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    track_id            : int       # persistent Kalman track ID; -1 for CAMERA_ONLY
    frame_id            : int
    timestamp           : float
    fusion_method       : str       # 'FULL' | 'LIDAR_ONLY' | 'CAMERA_ONLY'

    # ── Semantic classification ───────────────────────────────────────────────
    lidar_class         : str       # size-heuristic class from Task 03; 'unknown' for CAMERA_ONLY
    camera_class        : str       # YOLO class mapped to ADAS; 'unknown' for LIDAR_ONLY
    fused_class         : str       # final class: camera_class > lidar_class > 'unknown'
    camera_confidence   : float     # YOLO confidence [0, 1]; 0.0 for LIDAR_ONLY
    fused_confidence    : float     # weighted combination of camera and LiDAR confidence

    # ── 3D Geometry (from LiDAR — authoritative) ──────────────────────────────
    center_xyz          : np.ndarray    # (3,) float32, Kalman-filtered position
    extent_lwh          : np.ndarray    # (3,) float32, smoothed bounding box dims
    heading_deg         : float
    bearing_deg         : float
    sector              : str

    # ── Distance (safety-critical) ────────────────────────────────────────────
    min_distance        : float
    # EMA-smoothed minimum surface distance. -1.0 for CAMERA_ONLY.
    # USE THIS for all safety-critical decisions.

    # ── Motion (from Kalman tracker) ─────────────────────────────────────────
    velocity_ms         : float         # radial closing speed (m/s); 0.0 for CAMERA_ONLY
    velocity_xyz        : np.ndarray    # (3,) float32 world frame
    ttc_seconds         : float         # time-to-collision; inf for CAMERA_ONLY
    severity            : str           # 'BRAKE' | 'WARN' | 'SAFE'

    # ── Camera bbox (from YOLO) ───────────────────────────────────────────────
    camera_bbox_xyxy    : Optional[Tuple[float, float, float, float]]
    # YOLO bbox [x1, y1, x2, y2] in pixels. None for LIDAR_ONLY.

    camera_bbox_norm    : Optional[Tuple[float, float, float, float]]
    # Normalised [cx, cy, w, h] in [0, 1]. None for LIDAR_ONLY.
    # Task 07 NN training label format.

    # ── Projected LiDAR bbox (for visualisation) ──────────────────────────────
    projected_bbox_xyxy : Optional[Tuple[float, float, float, float]]
    # LiDAR cluster projected onto image → 2D bbox.
    # None for CAMERA_ONLY or if cluster outside camera FOV.

    # ── IoU match quality ─────────────────────────────────────────────────────
    match_iou           : float
    # IoU between projected LiDAR bbox and YOLO bbox. 0.0 for LIDAR/CAMERA_ONLY.

    # ── Kalman internals (for NN export) ─────────────────────────────────────
    kalman_state        : Optional[np.ndarray]    # (6,) float64; None for CAMERA_ONLY
    kalman_uncertainty  : Optional[np.ndarray]    # (6,) float64; None for CAMERA_ONLY

    # ── Track lifecycle ───────────────────────────────────────────────────────
    age_frames          : int
    lost_frames         : int
    is_confirmed        : bool

    def to_dict(self) -> dict:
        """JSON-serialisable dict for Task 07 export."""
        return {
            'track_id'           : self.track_id,
            'frame_id'           : self.frame_id,
            'timestamp'          : round(self.timestamp, 4),
            'fusion_method'      : self.fusion_method,
            'lidar_class'        : self.lidar_class,
            'camera_class'       : self.camera_class,
            'fused_class'        : self.fused_class,
            'camera_confidence'  : round(float(self.camera_confidence), 4),
            'fused_confidence'   : round(float(self.fused_confidence), 4),
            'center_xyz'         : [round(float(v), 4) for v in self.center_xyz],
            'extent_lwh'         : [round(float(v), 4) for v in self.extent_lwh],
            'heading_deg'        : round(float(self.heading_deg), 2),
            'bearing_deg'        : round(float(self.bearing_deg), 2),
            'sector'             : self.sector,
            'min_distance'       : round(float(self.min_distance), 4),
            'velocity_ms'        : round(float(self.velocity_ms), 4),
            'velocity_xyz'       : [round(float(v), 4) for v in self.velocity_xyz],
            'ttc_seconds'        : (round(float(self.ttc_seconds), 3)
                                    if self.ttc_seconds != float('inf') else None),
            'severity'           : self.severity,
            'camera_bbox_xyxy'   : (list(self.camera_bbox_xyxy)
                                    if self.camera_bbox_xyxy is not None else None),
            'camera_bbox_norm'   : (list(self.camera_bbox_norm)
                                    if self.camera_bbox_norm is not None else None),
            'projected_bbox_xyxy': (list(self.projected_bbox_xyxy)
                                    if self.projected_bbox_xyxy is not None else None),
            'match_iou'          : round(float(self.match_iou), 4),
            'kalman_state'       : ([round(float(v), 6) for v in self.kalman_state]
                                    if self.kalman_state is not None else None),
            'kalman_uncertainty' : ([round(float(v), 6) for v in self.kalman_uncertainty]
                                    if self.kalman_uncertainty is not None else None),
            'age_frames'         : self.age_frames,
            'lost_frames'        : self.lost_frames,
            'is_confirmed'       : self.is_confirmed,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FusionResult
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FusionResult:
    """Full output of one fusion frame with diagnostics."""

    frame_id            : int
    timestamp           : float
    fused_obstacles     : List[FusedObstacle]
    # Sorted by min_distance ascending (CAMERA_ONLY at end, min_distance=-1).

    n_lidar_tracks      : int       # input TrackedObstacle count
    n_camera_dets       : int       # input YoloDetection count
    n_full              : int       # FULL fusion count
    n_lidar_only        : int       # LIDAR_ONLY count
    n_camera_only       : int       # CAMERA_ONLY count

    camera_frame_age_s  : float     # age of camera frame used (seconds)
    proc_ms             : float     # total fusion processing time

    def fusion_quality(self) -> float:
        """
        Fraction of LiDAR tracks that achieved FULL fusion [0.0, 1.0].
        Low value = camera-LiDAR alignment problem or calibration drift.
        """
        return self.n_full / max(self.n_lidar_tracks, 1)

    def to_dict(self) -> dict:
        return {
            'frame_id'           : self.frame_id,
            'timestamp'          : round(self.timestamp, 4),
            'n_obstacles'        : len(self.fused_obstacles),
            'n_lidar_tracks'     : self.n_lidar_tracks,
            'n_camera_dets'      : self.n_camera_dets,
            'n_full'             : self.n_full,
            'n_lidar_only'       : self.n_lidar_only,
            'n_camera_only'      : self.n_camera_only,
            'camera_frame_age_s' : round(self.camera_frame_age_s, 4),
            'proc_ms'            : round(self.proc_ms, 2),
            'fusion_quality'     : round(self.fusion_quality(), 4),
            'obstacles'          : [o.to_dict() for o in self.fused_obstacles],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SensorFusion
# ══════════════════════════════════════════════════════════════════════════════

class SensorFusion:
    """
    Late-fusion of LiDAR tracks (Task 04) and camera detections (YOLO).

    Architecture:
        Camera runs at 30 fps  → push YoloDetection list to buffer each frame
        LiDAR runs at 10 Hz    → call fuse() at each LiDAR tick

    Thread safety:
        push_camera_frame() — safe to call from camera callback thread.
        fuse()              — call from LiDAR processing thread only.
        Internal camera buffer uses threading.Lock.

    Usage:
        fusion = SensorFusion(calibration, extrinsics, config)

        # Camera thread (30 fps):
        def camera_callback(image):
            detections = yolo.detect(image)
            fusion.push_camera_frame(detections, timestamp)

        # LiDAR thread (10 Hz):
        def lidar_tick(tracked_obstacles, frame_id, timestamp):
            result = fusion.fuse(tracked_obstacles, frame_id, timestamp)
            hud.render(result)
    """

    def __init__(
        self,
        calibration : CameraCalibration,
        extrinsics  : FusionExtrinsics,
        config      : FusionConfig = FusionConfig()
    ):
        self._cal         = calibration
        self._ext         = extrinsics
        self._cfg         = config
        self._lock        = threading.Lock()
        self._frame_count = 0

        # Camera frame buffer: deque of (timestamp, List[YoloDetection])
        self._cam_buffer: Deque[Tuple[float, List[YoloDetection]]] = \
            collections.deque(maxlen=config.camera_buffer_size)

        logger.info(
            "SensorFusion initialised | "
            "image=%d×%d | fov=%.0f° | "
            "iou_thresh=%.2f | max_cam_age=%.0f ms",
            calibration.image_width, calibration.image_height,
            calibration.fov_deg,
            config.iou_threshold,
            config.max_camera_age_s * 1000,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Camera frame ingestion
    # ──────────────────────────────────────────────────────────────────────────

    def push_camera_frame(
        self,
        detections : List[YoloDetection],
        timestamp  : float
    ) -> None:
        """
        Push a new camera detection frame into the synchronisation buffer.

        Call this from your camera callback at 30 fps. Thread-safe.

        Args:
            detections: List[YoloDetection] from YOLO inference.
            timestamp : Camera frame timestamp (simulation seconds).
                        Must use the same clock as LiDAR timestamps.
                        In CARLA: use carla.Timestamp.elapsed_seconds.
        """
        with self._lock:
            self._cam_buffer.append((timestamp, detections))

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Main fusion entry point
    # ──────────────────────────────────────────────────────────────────────────

    def fuse(
        self,
        tracked_obstacles : List[TrackedObstacle],
        frame_id          : int,
        timestamp         : float
    ) -> FusionResult:
        """
        Fuse LiDAR tracks with the most recent camera detections.

        Call this at every LiDAR tick (10 Hz).

        Args:
            tracked_obstacles: List[TrackedObstacle] from Task 04.
            frame_id         : Current LiDAR frame ID.
            timestamp        : Current LiDAR frame timestamp (seconds).

        Returns:
            FusionResult with fused_obstacles sorted by min_distance.
        """
        cfg     = self._cfg
        t_start = time.perf_counter()

        # ── Step 1: Get synchronised camera frame ─────────────────────────────
        cam_dets, cam_age = self._get_synced_camera_frame(timestamp)
        use_camera = (cam_dets is not None and
                      cam_age  <= cfg.max_camera_age_s)

        if not use_camera:
            cam_dets = []
            if cam_age != float('inf'):
                logger.debug(
                    "Frame %d: camera frame too old (%.0f ms > %.0f ms) — LIDAR_ONLY mode",
                    frame_id, cam_age * 1000, cfg.max_camera_age_s * 1000,
                )

        # ── Step 2: Project LiDAR clusters onto image ─────────────────────────
        # For each tracked obstacle, project its cluster points onto the camera
        # image plane to get a 2D projected bbox for IoU matching.
        projected_bboxes: Dict[int, Optional[Tuple]] = {}
        for obs in tracked_obstacles:
            projected_bboxes[obs.track_id] = self._project_obstacle(obs, cfg)

        # ── Step 3: Build IoU matrix and greedily match ───────────────────────
        matched_track_to_det: Dict[int, int] = {}   # track_id → det list index
        matched_det_ids: set = set()

        if use_camera and cam_dets:
            matched_track_to_det, matched_det_ids = self._match_greedy(
                tracked_obstacles, cam_dets, projected_bboxes, cfg
            )

        # ── Step 4: Assemble FusedObstacle list ───────────────────────────────
        fused: List[FusedObstacle] = []
        n_full       = 0
        n_lidar_only = 0

        for obs in tracked_obstacles:
            det_idx = matched_track_to_det.get(obs.track_id, None)
            if det_idx is not None:
                fo = self._build_full(obs, cam_dets[det_idx],
                                      projected_bboxes[obs.track_id], cfg)
                n_full += 1
            else:
                fo = self._build_lidar_only(obs, projected_bboxes[obs.track_id])
                n_lidar_only += 1
            fused.append(fo)

        # CAMERA_ONLY: camera sees something LiDAR missed
        n_camera_only = 0
        for i, det in enumerate(cam_dets):
            if i not in matched_det_ids:
                fused.append(self._build_camera_only(det, frame_id, timestamp, cfg))
                n_camera_only += 1

        # ── Step 5: Sort output ───────────────────────────────────────────────
        # Confirmed LiDAR-backed obstacles first (by distance), CAMERA_ONLY last.
        fused.sort(key=lambda fo: float('inf') if fo.min_distance < 0
                                  else fo.min_distance)

        # ── Timing & diagnostics ──────────────────────────────────────────────
        proc_ms = (time.perf_counter() - t_start) * 1000.0
        if proc_ms > cfg.max_proc_ms:
            logger.warning(
                "Frame %d: fusion %.1f ms exceeds budget %.0f ms",
                frame_id, proc_ms, cfg.max_proc_ms,
            )

        self._frame_count += 1
        if self._frame_count % cfg.log_every_n_frames == 0:
            logger.info(
                "[Frame %d] full=%d lidar_only=%d cam_only=%d "
                "quality=%.0f%% lat=%.1f ms",
                frame_id, n_full, n_lidar_only, n_camera_only,
                n_full / max(len(tracked_obstacles), 1) * 100,
                proc_ms,
            )

        return FusionResult(
            frame_id           = frame_id,
            timestamp          = timestamp,
            fused_obstacles    = fused,
            n_lidar_tracks     = len(tracked_obstacles),
            n_camera_dets      = len(cam_dets),
            n_full             = n_full,
            n_lidar_only       = n_lidar_only,
            n_camera_only      = n_camera_only,
            camera_frame_age_s = cam_age,
            proc_ms            = proc_ms,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Camera frame synchronisation
    # ──────────────────────────────────────────────────────────────────────────

    def _get_synced_camera_frame(
        self,
        lidar_timestamp: float,
    ) -> Tuple[Optional[List[YoloDetection]], float]:
        """
        Find the camera frame with timestamp closest to the LiDAR timestamp.

        Camera and LiDAR run on independent clocks in CARLA synchronous mode.
        The camera buffer holds the last 4 frames (133 ms at 30 fps).

        Returns:
            (detections, age_seconds): Best-matching camera frame and its age.
            Returns (None, inf) if buffer is empty.
        """
        with self._lock:
            if not self._cam_buffer:
                return None, float('inf')
            best_ts, best_dets = min(
                self._cam_buffer,
                key=lambda item: abs(item[0] - lidar_timestamp),
            )

        age = abs(lidar_timestamp - best_ts)
        return list(best_dets), age

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: LiDAR cluster projection
    # ──────────────────────────────────────────────────────────────────────────

    def _project_obstacle(
        self,
        obs : TrackedObstacle,
        cfg : FusionConfig,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Project a tracked obstacle's cluster points onto the camera image plane.

        Returns the 2D AABB of all projected points that fall within the image.

        PROJECTION PIPELINE:
          1. Get cluster points from obs.cluster_points (Task 05 Option A)
             or fall back to the 8 AABB corners of obs.center_xyz ± extent_lwh/2
          2. Transform: LiDAR vehicle frame → Camera frame via T_lidar_cam
          3. Filter: depth Pz > 0.5 m (in front of camera)
          4. Project: u = f*Px/Pz + cx,  v = f*Py/Pz + cy
          5. Filter: keep only pixels within image bounds (with margin)
          6. Compute 2D AABB of surviving projected pixels

        Returns:
            (x1, y1, x2, y2) projected bbox in pixels, or None if:
              - All cluster points are behind the camera
              - Fewer than min_projected_points survive image bounds filter
              - The projected box is degenerate (< 3 px in any dimension)
        """
        # ── Get cluster points ────────────────────────────────────────────────
        cluster_pts = self._get_cluster_points(obs)
        if cluster_pts is None or len(cluster_pts) == 0:
            cluster_pts = self._get_aabb_corners(obs)
        if cluster_pts is None or len(cluster_pts) == 0:
            return None

        # ── Transform: LiDAR frame → Camera frame ─────────────────────────────
        pts_cam = self._ext.transform_points(cluster_pts)    # (N, 3) float64

        # ── Depth filter: keep only points in front of camera (Pz > 0.5 m) ────
        depth_mask = pts_cam[:, 2] > 0.5
        pts_cam    = pts_cam[depth_mask]
        if len(pts_cam) < cfg.min_projected_points:
            return None

        # ── Perspective projection ─────────────────────────────────────────────
        # u = f * (Px / Pz) + cx   (horizontal pixel)
        # v = f * (Py / Pz) + cy   (vertical pixel, down = positive)
        f  = self._cal.focal_length
        cx = self._cal.cx
        cy = self._cal.cy

        # Vectorised: divide Px/Py by Pz (depth), then scale and offset
        pz  = pts_cam[:, 2]           # (N,) depth
        u   = f * (pts_cam[:, 0] / pz) + cx
        v   = f * (pts_cam[:, 1] / pz) + cy

        # ── Image bounds filter ───────────────────────────────────────────────
        m   = cfg.projection_margin_px
        W   = self._cal.image_width
        H   = self._cal.image_height
        img_mask = (u >= m) & (u <= W - m) & (v >= m) & (v <= H - m)
        u   = u[img_mask]
        v   = v[img_mask]

        if len(u) < cfg.min_projected_points:
            return None

        # ── 2D AABB of projected pixels ───────────────────────────────────────
        x1, y1, x2, y2 = float(u.min()), float(v.min()), float(u.max()), float(v.max())

        # Reject degenerate boxes (< 3 px in either dimension)
        if (x2 - x1) < 3.0 or (y2 - y1) < 3.0:
            return None

        return (x1, y1, x2, y2)

    def _get_cluster_points(
        self,
        obs: TrackedObstacle,
    ) -> Optional[np.ndarray]:
        """
        Retrieve the most recent cluster point cloud for this tracked obstacle.

        Uses TrackedObstacle.cluster_points (Option A from spec): the most
        recent Obstacle.points passed through MultiObjectTracker._build_output().

        Returns:
            (N, 3) float32 cluster points in vehicle frame, or None.
        """
        pts = getattr(obs, 'cluster_points', None)
        if pts is not None and len(pts) > 0:
            return pts.astype(np.float32)
        return None

    def _get_aabb_corners(
        self,
        obs: TrackedObstacle,
    ) -> np.ndarray:
        """
        Generate 8 corners of the 3D AABB from obstacle center and extent.

        Used as fallback when cluster_points is None (coasting/lost tracks).

        For a box centred at C = (cx, cy, cz) with half-extents [hl, hw, hh]:
          8 corners = C ± each combination of [±hl, ±hw, ±hh].

        Returns:
            shape (8, 3) float32 in vehicle frame.
        """
        cx, cy, cz = obs.center_xyz
        l, w, h    = obs.extent_lwh
        hl, hw, hh = l / 2.0, w / 2.0, h / 2.0

        corners = np.array([
            [cx + hl, cy + hw, cz + hh],
            [cx + hl, cy + hw, cz - hh],
            [cx + hl, cy - hw, cz + hh],
            [cx + hl, cy - hw, cz - hh],
            [cx - hl, cy + hw, cz + hh],
            [cx - hl, cy + hw, cz - hh],
            [cx - hl, cy - hw, cz + hh],
            [cx - hl, cy - hw, cz - hh],
        ], dtype=np.float32)
        return corners

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: IoU matching
    # ──────────────────────────────────────────────────────────────────────────

    def _match_greedy(
        self,
        tracked_obstacles : List[TrackedObstacle],
        cam_dets          : List[YoloDetection],
        projected_bboxes  : Dict[int, Optional[Tuple]],
        cfg               : FusionConfig,
    ) -> Tuple[Dict[int, int], set]:
        """
        Greedy IoU-based matching between projected LiDAR bboxes and YOLO detections.

        WHY GREEDY (not Hungarian):
          At typical urban density, we have ≤ 30 LiDAR tracks and ≤ 15 YOLO
          detections.  Hungarian requires O(N³) while greedy with sorted
          candidates is O(N² log N).  For N≤30, the difference is negligible
          and greedy is simpler to reason about.

          Greedy logic:
            1. Build IoU matrix (n_tracks × n_dets) for all valid pairs.
            2. Sort all (iou, track_idx, det_idx) tuples by IoU descending.
            3. Iterate: if neither party is already matched, assign them.
            This guarantees each track is matched to at most one detection
            and each detection is matched to at most one track.

        Args:
            tracked_obstacles: List of LiDAR tracks.
            cam_dets         : List of YOLO detections.
            projected_bboxes : Dict from track_id to projected 2D bbox or None.
            cfg              : FusionConfig.

        Returns:
            (matched_track_to_det, matched_det_ids):
                matched_track_to_det: Dict[track_id → det_index]
                matched_det_ids     : set of matched detection indices
        """
        n_tracks = len(tracked_obstacles)
        n_dets   = len(cam_dets)

        # Build IoU matrix (n_tracks × n_dets)
        iou_mat = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for i, obs in enumerate(tracked_obstacles):
            proj_bbox = projected_bboxes.get(obs.track_id)
            if proj_bbox is None:
                continue

            for j, det in enumerate(cam_dets):
                # Skip detections whose class is not in our mapping
                if cfg.yolo_to_adas_class.get(det.class_name, 'unknown') == 'unknown':
                    continue
                iou_mat[i, j] = self._compute_iou(proj_bbox, det.bbox_xyxy, cfg)

        # Collect all candidates above threshold, sorted by IoU descending
        candidates: List[Tuple[float, int, int]] = []
        for i in range(n_tracks):
            for j in range(n_dets):
                if iou_mat[i, j] >= cfg.iou_threshold:
                    candidates.append((float(iou_mat[i, j]), i, j))
        candidates.sort(key=lambda x: -x[0])   # descending IoU

        # Greedy assignment
        matched_track_idxs: set = set()
        matched_det_ids:    set = set()
        matched_track_to_det: Dict[int, int] = {}

        for iou_val, track_idx, det_idx in candidates:
            if track_idx in matched_track_idxs or det_idx in matched_det_ids:
                continue
            track_id = tracked_obstacles[track_idx].track_id
            matched_track_to_det[track_id] = det_idx
            matched_track_idxs.add(track_idx)
            matched_det_ids.add(det_idx)

        return matched_track_to_det, matched_det_ids

    def _compute_iou(
        self,
        box1 : Tuple[float, float, float, float],
        box2 : Tuple[float, float, float, float],
        cfg  : FusionConfig,
    ) -> float:
        """
        Compute Intersection over Union of two axis-aligned 2D bboxes.

        Args:
            box1, box2: (x1, y1, x2, y2) pixel bounding boxes.
            cfg       : FusionConfig (iou_min_overlap_px).

        Returns:
            IoU in [0, 1].
        """
        # Intersection rectangle
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])

        inter_w    = max(0.0, ix2 - ix1)
        inter_h    = max(0.0, iy2 - iy1)
        inter_area = inter_w * inter_h

        # Below minimum overlap → treat as no match (numerical noise guard)
        if inter_area < cfg.iou_min_overlap_px:
            return 0.0

        area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        union_area = area1 + area2 - inter_area

        if union_area <= 0.0:
            return 0.0

        return float(inter_area / union_area)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: FusedObstacle construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_full(
        self,
        obs      : TrackedObstacle,
        det      : YoloDetection,
        proj_bbox: Optional[Tuple],
        cfg      : FusionConfig,
    ) -> FusedObstacle:
        """
        Build a FULL fusion FusedObstacle from a matched LiDAR track + YOLO det.

        Class resolution (camera takes priority over LiDAR heuristic):
          If camera_class != 'unknown': fused_class = camera_class
          Else                        : fused_class = lidar_class

        Confidence fusion:
          fused_conf = clip(camera_weight * cam_conf + lidar_weight * lidar_conf)
        """
        adas_class   = cfg.yolo_to_adas_class.get(det.class_name, 'unknown')
        fused_class  = adas_class if adas_class != 'unknown' else obs.type
        fused_conf   = float(np.clip(
            cfg.camera_conf_weight * det.confidence +
            cfg.lidar_conf_weight  * obs.confidence,
            0.0, 1.0
        ))
        iou          = (self._compute_iou(proj_bbox, det.bbox_xyxy, cfg)
                        if proj_bbox is not None else 0.0)
        cam_bbox_norm = det.to_normalised(self._cal.image_width,
                                          self._cal.image_height)

        return FusedObstacle(
            track_id            = obs.track_id,
            frame_id            = obs.frame_id,
            timestamp           = obs.timestamp,
            fusion_method       = 'FULL',
            lidar_class         = obs.type,
            camera_class        = adas_class,
            fused_class         = fused_class,
            camera_confidence   = float(det.confidence),
            fused_confidence    = fused_conf,
            center_xyz          = obs.center_xyz.copy(),
            extent_lwh          = obs.extent_lwh.copy(),
            heading_deg         = obs.heading_deg,
            bearing_deg         = obs.bearing_deg,
            sector              = obs.sector,
            min_distance        = obs.min_distance,
            velocity_ms         = obs.velocity_ms,
            velocity_xyz        = obs.velocity_xyz.copy(),
            ttc_seconds         = obs.ttc_seconds,
            severity            = obs.severity,
            camera_bbox_xyxy    = det.bbox_xyxy,
            camera_bbox_norm    = cam_bbox_norm,
            projected_bbox_xyxy = proj_bbox,
            match_iou           = iou,
            kalman_state        = obs.kalman_state.copy(),
            kalman_uncertainty  = obs.kalman_uncertainty.copy(),
            age_frames          = obs.age_frames,
            lost_frames         = obs.lost_frames,
            is_confirmed        = obs.is_confirmed,
        )

    def _build_lidar_only(
        self,
        obs      : TrackedObstacle,
        proj_bbox: Optional[Tuple],
    ) -> FusedObstacle:
        """
        Build a LIDAR_ONLY FusedObstacle (no matching camera detection).

        camera_class = 'unknown'; fused_class = lidar size-heuristic class.
        camera_confidence = 0.0; fused_confidence = lidar confidence only.
        camera_bbox_xyxy and camera_bbox_norm = None.
        """
        return FusedObstacle(
            track_id            = obs.track_id,
            frame_id            = obs.frame_id,
            timestamp           = obs.timestamp,
            fusion_method       = 'LIDAR_ONLY',
            lidar_class         = obs.type,
            camera_class        = 'unknown',
            fused_class         = obs.type,
            camera_confidence   = 0.0,
            fused_confidence    = float(obs.confidence),
            center_xyz          = obs.center_xyz.copy(),
            extent_lwh          = obs.extent_lwh.copy(),
            heading_deg         = obs.heading_deg,
            bearing_deg         = obs.bearing_deg,
            sector              = obs.sector,
            min_distance        = obs.min_distance,
            velocity_ms         = obs.velocity_ms,
            velocity_xyz        = obs.velocity_xyz.copy(),
            ttc_seconds         = obs.ttc_seconds,
            severity            = obs.severity,
            camera_bbox_xyxy    = None,
            camera_bbox_norm    = None,
            projected_bbox_xyxy = proj_bbox,
            match_iou           = 0.0,
            kalman_state        = obs.kalman_state.copy(),
            kalman_uncertainty  = obs.kalman_uncertainty.copy(),
            age_frames          = obs.age_frames,
            lost_frames         = obs.lost_frames,
            is_confirmed        = obs.is_confirmed,
        )

    def _build_camera_only(
        self,
        det       : YoloDetection,
        frame_id  : int,
        timestamp : float,
        cfg       : FusionConfig,
    ) -> FusedObstacle:
        """
        Build a CAMERA_ONLY FusedObstacle (no matching LiDAR track).

        LiDAR fields: center_xyz = zeros, extent_lwh = zeros, min_distance = -1.0.
        Severity: WARN for vulnerable road users (pedestrian/cyclist), else SAFE.
        """
        adas_class    = cfg.yolo_to_adas_class.get(det.class_name, 'unknown')
        cam_bbox_norm = det.to_normalised(self._cal.image_width,
                                          self._cal.image_height)
        # Conservative severity for camera-only detections (distance unknown)
        severity = ('WARN' if adas_class in ('pedestrian', 'cyclist')
                    else 'SAFE')

        return FusedObstacle(
            track_id            = -1,
            frame_id            = frame_id,
            timestamp           = timestamp,
            fusion_method       = 'CAMERA_ONLY',
            lidar_class         = 'unknown',
            camera_class        = adas_class,
            fused_class         = adas_class if adas_class != 'unknown' else 'unknown',
            camera_confidence   = float(det.confidence),
            fused_confidence    = float(det.confidence) * cfg.camera_conf_weight,
            center_xyz          = np.zeros(3, dtype=np.float32),
            extent_lwh          = np.zeros(3, dtype=np.float32),
            heading_deg         = 0.0,
            bearing_deg         = 0.0,
            sector              = 'FRONT',
            min_distance        = cfg.camera_only_fallback_dist,
            velocity_ms         = 0.0,
            velocity_xyz        = np.zeros(3, dtype=np.float32),
            ttc_seconds         = float('inf'),
            severity            = severity,
            camera_bbox_xyxy    = det.bbox_xyxy,
            camera_bbox_norm    = cam_bbox_norm,
            projected_bbox_xyxy = None,
            match_iou           = 0.0,
            kalman_state        = None,
            kalman_uncertainty  = None,
            age_frames          = 0,
            lost_frames         = 0,
            is_confirmed        = False,
        )
