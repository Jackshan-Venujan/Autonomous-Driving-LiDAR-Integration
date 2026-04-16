"""
Real-Time Data Exporter — Task 07
===================================

Records every sensor frame — synchronised point clouds, camera images,
BEV radar frames, fused obstacle detections, ego state, and CARLA ground truth
— in a structured format directly loadable by PyTorch and TensorFlow for NN
training without any reprocessing.

DESIGN PHILOSOPHY:
  1. NEVER SLOW THE SAFETY PIPELINE.
     All disk I/O is asynchronous. record_frame() completes in < 0.5 ms.
  2. EVERY FRAME IS SELF-DESCRIBING.
     metadata.json + per-frame JSON schemas make every file standalone.
  3. RECORD NOW, TRAIN ANYWHERE.
     Output is natively compatible with PointPillars, CenterPoint, BEVDet,
     BEVFormer, LSTM trajectory prediction — no intermediate conversion.

Session lifecycle:
    exporter = DataExporter(ExporterConfig())
    exporter.open_session(lidar_config, ...)
    try:
        while recording:
            exporter.record_frame(lidar_frame, cloud, cluster_result,
                                  fusion_result, ...)
    finally:
        stats = exporter.close_session()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from lidar_sensor import LidarFrame                          # Task 01
except ImportError:
    LidarFrame = None  # type: ignore

try:
    from lidar_preprocessor import PreprocessedCloud, PreprocessConfig  # Task 02
except ImportError:
    PreprocessedCloud = None  # type: ignore
    PreprocessConfig  = None  # type: ignore

try:
    from lidar_clusterer import ClusterResult                    # Task 03
except ImportError:
    ClusterResult = None  # type: ignore

try:
    from lidar_tracker import TrackedObstacle                    # Task 04
except ImportError:
    TrackedObstacle = None  # type: ignore

try:
    from lidar_fusion import FusionResult, FusedObstacle         # Task 05
except ImportError:
    FusionResult   = None  # type: ignore
    FusedObstacle  = None  # type: ignore

from export_utils import (
    write_json_atomic, save_numpy_compressed, save_image_rgb,
    build_nn_label_detection_3d, build_nn_label_risk,
    json_serialise_safe,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][Exporter] %(message)s',
    datefmt='%H:%M:%S',
)


# ══════════════════════════════════════════════════════════════════════════════
#  LidarConfig — lightweight descriptor (no CARLA dependency)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LidarConfig:
    """
    Serialisable LiDAR sensor configuration.
    Used for session metadata.json. All fields match CARLA blueprint attributes.
    """
    channels           : int   = 64
    range_m            : float = 100.0
    upper_fov          : float = 15.0
    lower_fov          : float = -25.0
    rotation_frequency : float = 10.0
    points_per_second  : int   = 1_120_000
    mount_xyz          : Tuple = (0.0, 0.0, 2.4)
    noise_stddev       : float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
#  ExporterConfig
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExporterConfig:
    """
    Configuration for the real-time data exporter.
    All parameters are tunable. No magic numbers in logic code.
    """

    # ── Output directory ──────────────────────────────────────────────────────
    base_output_dir     : str   = 'recordings'
    session_name        : str   = ''

    # ── File format settings ──────────────────────────────────────────────────
    jpeg_quality        : int   = 90
    save_lidar_raw      : bool  = True
    save_lidar_filtered : bool  = True
    save_lidar_intensity: bool  = True
    save_camera_rgb     : bool  = True
    save_radar_bev      : bool  = True
    save_cluster_points : bool  = False
    save_ground_truth   : bool  = True

    # ── Async writer settings ─────────────────────────────────────────────────
    write_queue_maxsize : int   = 60
    # At 10 Hz: 60 frames = 6 seconds of buffer. If full, frames are DROPPED.
    n_writer_threads    : int   = 2

    # ── Frame selection ───────────────────────────────────────────────────────
    save_every_n_frames : int   = 1
    min_obstacles_to_save: int  = 0

    # ── Ground truth collection ───────────────────────────────────────────────
    gt_max_range_m      : float = 120.0
    gt_actor_types      : Tuple = ('vehicle', 'walker', 'static')

    # ── Metadata ─────────────────────────────────────────────────────────────
    carla_map           : str   = 'Town05'
    carla_version       : str   = '0.9.14'
    recorder_version    : str   = '1.0.0'

    # ── Performance ───────────────────────────────────────────────────────────
    max_write_ms        : float = 80.0
    log_every_n_frames  : int   = 100

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  FramePacket — internal write-queue payload
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FramePacket:
    """
    Complete data bundle for one sensor frame, ready for async disk writing.
    Created synchronously in record_frame(). Written asynchronously by workers.

    All numpy arrays are already copied (not views) to ensure thread safety.
    All CARLA objects are already converted to plain Python types.
    The writer thread must not make any CARLA API calls.
    """

    frame_id            : int
    timestamp           : float
    session_dir         : Path
    created_at_wall     : float

    lidar_raw_xyz       : Optional[np.ndarray]          # (N, 3) float32
    lidar_raw_intensity : Optional[np.ndarray]          # (N,)   float32
    lidar_filtered_xyz  : Optional[np.ndarray]          # (M, 3) float32

    camera_rgb          : Optional[np.ndarray]          # (H, W, 3) uint8
    radar_bev           : Optional[np.ndarray]          # (H, W, 3) uint8

    obstacles_dict      : dict
    ego_state_dict      : dict
    ground_truth_dict   : dict
    pipeline_stats_dict : dict

    obstacle_points     : Optional[Dict[int, np.ndarray]]


# ══════════════════════════════════════════════════════════════════════════════
#  DataExporter — main class
# ══════════════════════════════════════════════════════════════════════════════

class DataExporter:
    """
    Real-time asynchronous data exporter for ADAS LiDAR pipeline.

    Receives complete sensor frame data at 10 Hz from the pipeline thread
    and writes it to disk in the background without blocking the pipeline.

    Thread model:
        Pipeline thread: calls record_frame() — fast (<0.5 ms), queues data.
        Writer threads (N=2): consume queue, write to disk (~20-80 ms/frame).
    """

    def __init__(self, config: ExporterConfig = ExporterConfig()):
        self._cfg           = config
        self._session_dir   : Optional[Path]  = None
        self._frames_dir    : Optional[Path]  = None
        self._write_queue   : queue.Queue     = queue.Queue(maxsize=config.write_queue_maxsize)
        self._workers       : List[threading.Thread] = []
        self._running       : bool = False
        self._frame_count   : int  = 0
        self._saved_count   : int  = 0
        self._dropped_count : int  = 0
        self._error_count   : int  = 0
        self._lock          : threading.Lock = threading.Lock()
        self._session_open  : bool = False
        self._t_session_start: float = 0.0

        logger.info(
            "DataExporter initialised | queue=%d | writers=%d | every_n=%d",
            config.write_queue_maxsize,
            config.n_writer_threads,
            config.save_every_n_frames,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Session management
    # ──────────────────────────────────────────────────────────────────────────

    def open_session(
        self,
        lidar_config   : 'LidarConfig',
        camera_config  : Optional[dict] = None,
        preprocess_cfg : Optional[dict] = None,
        cluster_cfg    : Optional[dict] = None,
        tracker_cfg    : Optional[dict] = None,
        extra_metadata : Optional[dict] = None,
    ) -> Path:
        """
        Open a new recording session. Creates directory structure and writes metadata.json.

        Returns:
            Path to session directory.
        Raises:
            RuntimeError: If a session is already open.
        """
        if self._session_open:
            raise RuntimeError(
                "Cannot open session: previous session still open. Call close_session() first."
            )

        from datetime import datetime
        ts         = datetime.now().strftime('%Y%m%d_%H%M%S')
        name_sfx   = f"_{self._cfg.session_name}" if self._cfg.session_name else ""
        session_id = f"session_{ts}{name_sfx}"
        session_dir = Path(self._cfg.base_output_dir) / session_id
        frames_dir  = session_dir / 'frames'

        session_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(exist_ok=True)

        self._session_dir = session_dir
        self._frames_dir  = frames_dir

        metadata = {
            'session_id'        : session_id,
            'created_at'        : datetime.now().isoformat(),
            'carla_version'     : self._cfg.carla_version,
            'carla_map'         : self._cfg.carla_map,
            'recorder_version'  : self._cfg.recorder_version,

            'coordinate_frame'  : 'ISO8855_vehicle_RH',
            'coordinate_note'   : '+X=forward +Y=left +Z=up. Sensor at origin.',
            'distance_unit'     : 'metres',
            'angle_unit'        : 'degrees',
            'velocity_unit'     : 'metres_per_second',
            'time_unit'         : 'seconds_since_simulation_start',

            'lidar_config'      : lidar_config.to_dict(),
            'camera_config'     : camera_config  or {},
            'preprocess_config' : preprocess_cfg or {},
            'cluster_config'    : cluster_cfg    or {},
            'tracker_config'    : tracker_cfg    or {},
            'exporter_config'   : self._cfg.to_dict(),

            'nn_compatibility'  : {
                'lidar_filtered_npy': ['PointPillars','SECOND','CenterPoint','PointNet++'],
                'lidar_raw_npy'     : ['PointNet','VoxelNet','custom'],
                'camera_rgb_jpg'    : ['BEVFusion','FUTR3D','depth_estimation'],
                'radar_bev_png'     : ['BEVDet','BEVFormer','occupancy_prediction'],
                'obstacles_json'    : ['3D_detection','velocity_regression','risk_prediction'],
                'ego_state_json'    : ['trajectory_prediction','imitation_learning'],
                'ground_truth_json' : ['supervised_detection','evaluation'],
            },
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        write_json_atomic(metadata, session_dir / 'metadata.json')

        # ── Start writer threads ──────────────────────────────────────────────
        self._running        = True
        self._workers        = []
        self._frame_count    = 0
        self._saved_count    = 0
        self._dropped_count  = 0
        self._error_count    = 0
        self._t_session_start = time.time()
        self._session_open   = True

        for i in range(self._cfg.n_writer_threads):
            t = threading.Thread(
                target=self._writer_worker,
                name=f'ExporterWriter-{i}',
                daemon=True,
            )
            t.start()
            self._workers.append(t)

        logger.info(
            "Session opened: %s | %d writer threads started",
            session_dir, self._cfg.n_writer_threads,
        )
        return session_dir

    def close_session(self) -> dict:
        """
        Gracefully close the recording session.

        Waits for all queued frames to be written before returning.
        Writes session_stats.json. Should ALWAYS be called in a finally block.

        Returns:
            dict with session statistics.
        """
        if not self._session_open:
            logger.warning("close_session() called but no session is open.")
            return {}

        logger.info(
            "Closing session — draining %d queued frames ...",
            self._write_queue.qsize(),
        )

        for _ in self._workers:
            self._write_queue.put(None)   # sentinel — tells each worker to exit

        for worker in self._workers:
            worker.join(timeout=60.0)
            if worker.is_alive():
                logger.error("Writer thread %s did not stop in 60 s", worker.name)

        self._running      = False
        self._session_open = False

        duration_s = time.time() - self._t_session_start
        stats = {
            'session_dir'      : str(self._session_dir),
            'duration_s'       : round(duration_s, 2),
            'total_frames_seen': self._frame_count,
            'frames_saved'     : self._saved_count,
            'frames_dropped'   : self._dropped_count,
            'write_errors'     : self._error_count,
            'effective_hz'     : round(self._saved_count / max(duration_s, 1), 2),
            'drop_rate_pct'    : round(
                self._dropped_count / max(self._frame_count, 1) * 100, 2
            ),
        }

        if self._session_dir:
            write_json_atomic(stats, self._session_dir / 'session_stats.json')

        logger.info(
            "Session closed | saved=%d | dropped=%d | errors=%d | duration=%.1fs",
            self._saved_count, self._dropped_count,
            self._error_count, duration_s,
        )
        return stats

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Main record entry point
    # ──────────────────────────────────────────────────────────────────────────

    def record_frame(
        self,
        lidar_frame    : 'LidarFrame',
        cloud          : 'PreprocessedCloud',
        cluster_result : 'ClusterResult',
        fusion_result  : 'FusionResult',
        ego_vehicle    : Optional[object]     = None,
        carla_world    : Optional[object]     = None,
        camera_image   : Optional[np.ndarray] = None,
        radar_bev      : Optional[np.ndarray] = None,
        pipeline_ms    : Optional[dict]       = None,
        tracked_obs    : Optional[List]       = None,
    ) -> bool:
        """
        Record one sensor frame. Returns True if queued, False if dropped.

        MUST complete in < 0.5 ms — no disk I/O. Only data preparation +
        queue insertion. All slow work happens in _writer_worker() threads.

        Frame selection logic:
            - Only save every save_every_n_frames frames.
            - Only save if confirmed obstacle count >= min_obstacles_to_save.
            - Drop if write queue is full (pipeline takes priority).
        """
        if not self._session_open:
            return False

        with self._lock:
            self._frame_count += 1
            frame_count = self._frame_count

        # ── Frame selection ────────────────────────────────────────────────
        if frame_count % self._cfg.save_every_n_frames != 0:
            return False

        n_confirmed = sum(
            1 for o in fusion_result.fused_obstacles
            if o.is_confirmed and o.min_distance >= 0
        )
        if n_confirmed < self._cfg.min_obstacles_to_save:
            return False

        # ── Build FramePacket (fast, no I/O) ──────────────────────────────
        t_build = time.perf_counter()
        packet  = self._build_packet(
            lidar_frame, cloud, cluster_result, fusion_result,
            ego_vehicle, carla_world, camera_image, radar_bev,
            pipeline_ms, tracked_obs,
        )
        build_ms = (time.perf_counter() - t_build) * 1000.0

        if build_ms > 1.0:
            logger.warning(
                "Frame %d: packet build took %.1f ms (target < 0.5 ms).",
                lidar_frame.frame_id, build_ms,
            )

        # ── Enqueue (non-blocking) ─────────────────────────────────────────
        try:
            self._write_queue.put_nowait(packet)
            with self._lock:
                self._saved_count += 1
            if frame_count % self._cfg.log_every_n_frames == 0:
                logger.info(
                    "Frame %d queued | saved=%d | dropped=%d | q=%d/%d",
                    lidar_frame.frame_id, self._saved_count,
                    self._dropped_count, self._write_queue.qsize(),
                    self._cfg.write_queue_maxsize,
                )
            return True
        except queue.Full:
            with self._lock:
                self._dropped_count += 1
            logger.warning(
                "Frame %d: write queue full (%d/%d). Frame dropped.",
                lidar_frame.frame_id,
                self._write_queue.qsize(), self._cfg.write_queue_maxsize,
            )
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Packet builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_packet(
        self,
        lidar_frame    : 'LidarFrame',
        cloud          : 'PreprocessedCloud',
        cluster_result : 'ClusterResult',
        fusion_result  : 'FusionResult',
        ego_vehicle    : Optional[object],
        carla_world    : Optional[object],
        camera_image   : Optional[np.ndarray],
        radar_bev      : Optional[np.ndarray],
        pipeline_ms    : Optional[dict],
        tracked_obs    : Optional[List],
    ) -> FramePacket:
        """
        Build a FramePacket from pipeline outputs.

        CRITICAL: All numpy arrays MUST be .copy()-ed here.
        The pipeline may reuse arrays after this call returns.
        All CARLA API calls must complete here (on pipeline thread).
        Writer threads cannot call CARLA API.
        """
        cfg = self._cfg
        fid = lidar_frame.frame_id
        ts  = lidar_frame.timestamp

        # ── Arrays (copy to own memory) ────────────────────────────────────
        lidar_raw_xyz = lidar_raw_intensity = lidar_filtered_xyz = None

        if cfg.save_lidar_raw and lidar_frame.points is not None:
            lidar_raw_xyz       = lidar_frame.points.copy()    # (N, 3) float32
        if cfg.save_lidar_intensity and lidar_frame.intensity is not None:
            lidar_raw_intensity = lidar_frame.intensity.copy() # (N,)   float32
        if cfg.save_lidar_filtered and cloud.points is not None:
            lidar_filtered_xyz  = cloud.points.copy()          # (M, 3) float32

        cam_copy = camera_image.copy() if camera_image is not None else None
        bev_copy = radar_bev.copy()    if radar_bev    is not None else None

        # ── Per-obstacle cluster points (optional) ─────────────────────────
        obstacle_points = None
        if cfg.save_cluster_points:
            obstacle_points = {}
            for obs in fusion_result.fused_obstacles:
                pts = getattr(obs, 'cluster_points', None)
                if pts is not None and obs.track_id >= 0:
                    obstacle_points[obs.track_id] = pts.copy()

        # ── Structured dicts (all CARLA API calls happen here) ─────────────
        obstacles_dict = self._build_obstacles_dict(
            fusion_result, fid, ts, pipeline_ms,
        )
        ego_dict       = self._build_ego_dict(ego_vehicle, fid, ts)
        gt_dict        = self._build_ground_truth_dict(
            carla_world, ego_vehicle, fusion_result, fid, ts,
        )
        stats_dict     = self._build_pipeline_stats_dict(
            lidar_frame, cloud, cluster_result, fusion_result,
            tracked_obs, pipeline_ms, fid, ts,
        )

        return FramePacket(
            frame_id            = fid,
            timestamp           = ts,
            session_dir         = self._session_dir,
            created_at_wall     = time.time(),
            lidar_raw_xyz       = lidar_raw_xyz,
            lidar_raw_intensity = lidar_raw_intensity,
            lidar_filtered_xyz  = lidar_filtered_xyz,
            camera_rgb          = cam_copy,
            radar_bev           = bev_copy,
            obstacles_dict      = obstacles_dict,
            ego_state_dict      = ego_dict,
            ground_truth_dict   = gt_dict,
            pipeline_stats_dict = stats_dict,
            obstacle_points     = obstacle_points,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Dict builders (called on pipeline thread)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_obstacles_dict(
        self,
        fusion_result : 'FusionResult',
        frame_id      : int,
        timestamp     : float,
        pipeline_ms   : Optional[dict],
    ) -> dict:
        total_ms = (pipeline_ms.get('total_pipeline', 0.0)
                    if pipeline_ms else 0.0)

        obs_list = []
        for idx, obs in enumerate(fusion_result.fused_obstacles):
            obs_d = obs.to_dict()
            obs_d['obstacle_idx'] = idx
            obs_d['nn_label'] = {
                'detection_3d': build_nn_label_detection_3d(obs),
                'risk_label'  : build_nn_label_risk(obs),
            }
            obs_list.append(obs_d)

        return {
            'frame_id'          : frame_id,
            'timestamp'         : timestamp,
            'sensor_latency_ms' : round(float(total_ms), 2),
            'n_obstacles'       : len(obs_list),
            'obstacles'         : obs_list,
        }

    def _build_ego_dict(
        self,
        ego_vehicle : Optional[object],
        frame_id    : int,
        timestamp   : float,
    ) -> dict:
        """
        Build ego_state.json dict from CARLA vehicle.
        Handles None ego_vehicle gracefully (returns zeros).
        """
        base: dict = {
            'frame_id'           : frame_id,
            'timestamp'          : timestamp,
            'position_xyz'       : [0.0, 0.0, 0.0],
            'heading_deg'        : 0.0,
            'heading_rad'        : 0.0,
            'velocity_xyz_world' : [0.0, 0.0, 0.0],
            'speed_ms'           : 0.0,
            'speed_kmh'          : 0.0,
            'acceleration_xyz'   : [0.0, 0.0, 0.0],
            'angular_velocity'   : [0.0, 0.0, 0.0],
            'control'            : {
                'throttle'  : 0.0,
                'brake'     : 0.0,
                'steer'     : 0.0,
                'gear'      : 0,
                'hand_brake': False,
                'reverse'   : False,
            },
            'map_name'           : self._cfg.carla_map,
            'weather'            : 'Unknown',
            'nn_label'           : {
                'future_waypoints_10hz': None,
                'nn_label_note'        : (
                    'future_waypoints filled offline by trajectory labeller'
                ),
            },
        }

        if ego_vehicle is None:
            return base

        try:
            t    = ego_vehicle.get_transform()
            v    = ego_vehicle.get_velocity()
            acc  = ego_vehicle.get_acceleration()
            av   = ego_vehicle.get_angular_velocity()
            ctrl = ego_vehicle.get_control()

            speed_ms  = float(np.sqrt(v.x**2 + v.y**2 + v.z**2))
            heading_d = float(t.rotation.yaw)

            base.update({
                'position_xyz'       : [
                    round(t.location.x, 4),
                    round(t.location.y, 4),
                    round(t.location.z, 4),
                ],
                'heading_deg'        : round(heading_d, 4),
                'heading_rad'        : round(float(np.radians(heading_d)), 6),
                'velocity_xyz_world' : [
                    round(float(v.x), 4),
                    round(float(v.y), 4),
                    round(float(v.z), 4),
                ],
                'speed_ms'           : round(speed_ms, 4),
                'speed_kmh'          : round(speed_ms * 3.6, 4),
                'acceleration_xyz'   : [
                    round(float(acc.x), 4),
                    round(float(acc.y), 4),
                    round(float(acc.z), 4),
                ],
                'angular_velocity'   : [
                    round(float(av.x), 6),
                    round(float(av.y), 6),
                    round(float(av.z), 6),
                ],
                'control'            : {
                    'throttle'  : round(float(ctrl.throttle), 4),
                    'brake'     : round(float(ctrl.brake), 4),
                    'steer'     : round(float(ctrl.steer), 4),
                    'gear'      : int(ctrl.gear),
                    'hand_brake': bool(ctrl.hand_brake),
                    'reverse'   : bool(ctrl.reverse),
                },
            })

            # Weather (best-effort)
            try:
                world = ego_vehicle.get_world()
                weather = world.get_weather()
                base['weather'] = str(weather.sun_altitude_angle)
            except Exception:
                pass

        except Exception as e:
            logger.debug("ego state collection failed: %s", e)

        return base

    def _build_ground_truth_dict(
        self,
        carla_world   : Optional[object],
        ego_vehicle   : Optional[object],
        fusion_result : 'FusionResult',
        frame_id      : int,
        timestamp     : float,
    ) -> dict:
        """
        Build ground_truth.json from CARLA actor list.
        Matches GT actors to tracker outputs by 3D proximity.
        """
        ego_pos = [0.0, 0.0, 0.0]
        if ego_vehicle is not None:
            try:
                t = ego_vehicle.get_transform()
                ego_pos = [t.location.x, t.location.y, t.location.z]
            except Exception:
                pass

        actors = []

        if carla_world is not None and ego_vehicle is not None:
            try:
                actor_list = carla_world.get_actors()
                ego_id     = ego_vehicle.id

                for actor in actor_list:
                    if actor.id == ego_id:
                        continue

                    type_id = actor.type_id
                    # Filter by configured actor type prefixes
                    if not any(type_id.startswith(t)
                               for t in self._cfg.gt_actor_types):
                        continue

                    try:
                        at = actor.get_transform()
                        av = actor.get_velocity()

                        # Position relative to ego
                        dx = at.location.x - ego_pos[0]
                        dy = at.location.y - ego_pos[1]
                        dz = at.location.z - ego_pos[2]
                        dist = float(np.sqrt(dx*dx + dy*dy + dz*dz))

                        if dist > self._cfg.gt_max_range_m:
                            continue

                        # ADAS class
                        adas_class = 'unknown'
                        if type_id.startswith('vehicle'):
                            adas_class = 'vehicle'
                        elif type_id.startswith('walker'):
                            adas_class = 'pedestrian'
                        elif type_id.startswith('static'):
                            adas_class = 'structure'

                        from export_utils import CLASS_ID_MAP
                        class_id = CLASS_ID_MAP.get(adas_class, 5)

                        # Match to tracked obstacles by proximity
                        matched_track_id = -1
                        detection_iou    = 0.0
                        best_dist        = float('inf')
                        for obs in fusion_result.fused_obstacles:
                            if obs.min_distance < 0:
                                continue
                            cx = float(obs.center_xyz[0])
                            cy = float(obs.center_xyz[1])
                            d_to_track = float(np.sqrt(
                                (cx - dx)**2 + (cy - dy)**2
                            ))
                            if d_to_track < best_dist and d_to_track < 3.0:
                                best_dist = d_to_track
                                matched_track_id = obs.track_id
                                detection_iou    = max(0.0, 1.0 - d_to_track / 3.0)

                        try:
                            bb     = actor.bounding_box
                            bbox_l = round(bb.extent.x * 2, 3)
                            bbox_w = round(bb.extent.y * 2, 3)
                            bbox_h = round(bb.extent.z * 2, 3)
                        except Exception:
                            bbox_l = bbox_w = bbox_h = 0.0

                        speed = float(np.sqrt(av.x**2 + av.y**2 + av.z**2))
                        heading_world = float(at.rotation.yaw)
                        heading_ego   = heading_world  # simplified

                        actors.append({
                            'actor_id'         : actor.id,
                            'type_id'          : type_id,
                            'adas_class'       : adas_class,
                            'class_id'         : class_id,
                            'position_world'   : [
                                round(at.location.x, 4),
                                round(at.location.y, 4),
                                round(at.location.z, 4),
                            ],
                            'position_ego'     : [round(dx, 4), round(dy, 4), round(dz, 4)],
                            'heading_world_deg': round(heading_world, 2),
                            'heading_ego_deg'  : round(heading_ego, 2),
                            'velocity_world'   : [
                                round(float(av.x), 4),
                                round(float(av.y), 4),
                                round(float(av.z), 4),
                            ],
                            'speed_ms'         : round(speed, 4),
                            'bbox_lwh'         : [bbox_l, bbox_w, bbox_h],
                            'distance_to_ego'  : round(dist, 4),
                            'matched_track_id' : matched_track_id,
                            'detection_iou'    : round(detection_iou, 4),
                            'nn_label'         : {
                                'class_id'   : class_id,
                                'center_ego' : [round(dx, 4), round(dy, 4), round(dz, 4)],
                                'size'       : [bbox_l, bbox_w, bbox_h],
                                'heading_rad': round(float(np.radians(heading_world)), 6),
                                'velocity_2d': [round(float(av.x), 4), round(float(av.y), 4)],
                                'is_occluded': False,
                                'visibility' : 1.0,
                            },
                        })
                    except Exception as ae:
                        logger.debug("GT actor %d error: %s", actor.id, ae)

            except Exception as e:
                logger.debug("ground truth collection failed: %s", e)

        return {
            'frame_id'   : frame_id,
            'timestamp'  : timestamp,
            'ego_position': ego_pos,
            'n_actors'   : len(actors),
            'actors'     : actors,
        }

    def _build_pipeline_stats_dict(
        self,
        lidar_frame    : 'LidarFrame',
        cloud          : 'PreprocessedCloud',
        cluster_result : 'ClusterResult',
        fusion_result  : 'FusionResult',
        tracked_obs    : Optional[List],
        pipeline_ms    : Optional[dict],
        frame_id       : int,
        timestamp      : float,
    ) -> dict:
        pm      = pipeline_ms or {}
        n_raw   = lidar_frame.num_points
        n_filt  = int(cloud.points.shape[0]) if cloud.points is not None else 0
        n_raw_cl = len(cluster_result.obstacles)

        n_confirmed = sum(1 for t in (tracked_obs or [])
                          if getattr(t, 'is_confirmed', False))
        n_active    = len(tracked_obs) if tracked_obs else 0

        return {
            'frame_id'  : frame_id,
            'timestamp' : timestamp,
            'wall_time' : time.time(),

            'lidar_frame' : {
                'n_raw_points'  : n_raw,
                'n_filtered_pts': n_filt,
                'filter_ratio'  : round(n_filt / max(n_raw, 1), 4),
            },

            'timing_ms' : {
                'lidar_preproc'  : round(float(pm.get('lidar_preproc',  0.0)), 2),
                'clustering'     : round(float(pm.get('clustering',     0.0)), 2),
                'tracking'       : round(float(pm.get('tracking',       0.0)), 2),
                'fusion'         : round(float(pm.get('fusion',         0.0)), 2),
                'export_queue'   : round(float(pm.get('export_queue',   0.0)), 2),
                'total_pipeline' : round(float(pm.get('total_pipeline', 0.0)), 2),
            },

            'cluster_stats' : {
                'n_raw_clusters' : n_raw_cl,
                'n_kept_clusters': fusion_result.n_lidar_tracks,
                'n_noise_points' : 0,     # not tracked in clusterer output
                'dbscan_ms'      : round(float(pm.get('clustering', 0.0)), 2),
            },

            'tracker_stats' : {
                'n_active_tracks' : n_active,
                'n_confirmed'     : n_confirmed,
                'n_new_tracks'    : 0,    # not exposed by tracker
                'n_deleted_tracks': 0,
            },

            'fusion_stats' : {
                'n_full'         : fusion_result.n_full,
                'n_lidar_only'   : fusion_result.n_lidar_only,
                'n_camera_only'  : fusion_result.n_camera_only,
                'fusion_quality' : round(float(fusion_result.fusion_quality()), 4),
                'camera_age_ms'  : round(float(fusion_result.camera_frame_age_s * 1000), 2),
            },
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: Background writer worker
    # ──────────────────────────────────────────────────────────────────────────

    def _writer_worker(self) -> None:
        """
        Background thread: drain the write queue and write each FramePacket to disk.
        Exits when it receives a None sentinel from close_session().
        """
        thread_name = threading.current_thread().name
        logger.debug("%s started", thread_name)

        while True:
            try:
                packet = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                if not self._running:
                    break
                continue

            if packet is None:   # sentinel
                self._write_queue.task_done()
                break

            t_start = time.perf_counter()
            try:
                self._write_packet(packet)
            except Exception as e:
                with self._lock:
                    self._error_count += 1
                logger.error(
                    "Write error for frame %d: %s",
                    packet.frame_id, e, exc_info=True,
                )
            finally:
                self._write_queue.task_done()

            write_ms = (time.perf_counter() - t_start) * 1000.0
            if write_ms > self._cfg.max_write_ms:
                logger.warning(
                    "%s: frame %d took %.1f ms to write (budget %.0f ms)",
                    thread_name, packet.frame_id, write_ms, self._cfg.max_write_ms,
                )

        logger.debug("%s exiting", thread_name)

    def _write_packet(self, packet: FramePacket) -> None:
        """Write all files for one FramePacket to the frames/ directory."""
        cfg      = self._cfg
        frame_dir = packet.session_dir / 'frames' / f'{packet.frame_id:06d}'
        frame_dir.mkdir(parents=True, exist_ok=True)

        # ── LiDAR arrays ──────────────────────────────────────────────────
        if packet.lidar_raw_xyz is not None and cfg.save_lidar_raw:
            # Pack (N,3) xyz + intensity as (N,4) per schema
            if packet.lidar_raw_intensity is not None:
                raw4 = np.concatenate(
                    [packet.lidar_raw_xyz,
                     packet.lidar_raw_intensity[:, np.newaxis]],
                    axis=1,
                ).astype(np.float32)
            else:
                raw4 = np.zeros(
                    (len(packet.lidar_raw_xyz), 4), dtype=np.float32
                )
                raw4[:, :3] = packet.lidar_raw_xyz
            save_numpy_compressed(raw4, frame_dir / 'lidar_raw.npy')

        if packet.lidar_filtered_xyz is not None and cfg.save_lidar_filtered:
            save_numpy_compressed(
                packet.lidar_filtered_xyz.astype(np.float32),
                frame_dir / 'lidar_filtered.npy',
            )

        # ── Images ────────────────────────────────────────────────────────
        if packet.camera_rgb is not None and cfg.save_camera_rgb and _PIL_AVAILABLE:
            save_image_rgb(
                packet.camera_rgb,
                frame_dir / 'camera_rgb.jpg',
                quality=cfg.jpeg_quality,
            )

        if packet.radar_bev is not None and cfg.save_radar_bev and _PIL_AVAILABLE:
            save_image_rgb(
                packet.radar_bev,
                frame_dir / 'radar_bev.png',
            )

        # ── JSON files ─────────────────────────────────────────────────────
        write_json_atomic(packet.obstacles_dict,
                          frame_dir / 'obstacles.json')
        write_json_atomic(packet.ego_state_dict,
                          frame_dir / 'ego_state.json')
        if cfg.save_ground_truth:
            write_json_atomic(packet.ground_truth_dict,
                              frame_dir / 'ground_truth.json')
        write_json_atomic(packet.pipeline_stats_dict,
                          frame_dir / 'pipeline_stats.json')

        # ── Per-obstacle cluster points (optional) ─────────────────────────
        if packet.obstacle_points is not None and cfg.save_cluster_points:
            for track_id, pts in packet.obstacle_points.items():
                fname = frame_dir / f'obstacle_{track_id:04d}_pts.npy'
                save_numpy_compressed(pts.astype(np.float32), fname)

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC: Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def session_dir(self) -> Optional[Path]:
        return self._session_dir

    @property
    def queue_size(self) -> int:
        return self._write_queue.qsize()

    def get_stats(self) -> dict:
        """Return a snapshot of current recording statistics."""
        with self._lock:
            return {
                'frame_count'  : self._frame_count,
                'saved_count'  : self._saved_count,
                'dropped_count': self._dropped_count,
                'error_count'  : self._error_count,
                'queue_size'   : self._write_queue.qsize(),
            }
