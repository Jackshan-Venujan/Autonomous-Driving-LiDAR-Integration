import math
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import carla


@dataclass
class GTActor:
    """Ground truth state of a single actor at one frame."""
    actor_id: int
    class_name: str        # 'vehicle', 'pedestrian', 'cyclist'
    world_x: float
    world_y: float
    world_z: float
    ego_local_x: float     # forward in ego frame
    ego_local_y: float     # lateral in ego frame
    distance_m: float      # sqrt(local_x² + local_y²)
    azimuth_deg: float     # atan2(local_y, local_x) in degrees
    bbox_extent_x: float   # CARLA half-extents (metres)
    bbox_extent_y: float
    bbox_extent_z: float


@dataclass
class FrameGT:
    """Ground truth snapshot for one simulation frame."""
    frame_id: int
    timestamp: float       # simulation time (seconds)
    ego_speed_kmh: float
    weather_tag: str
    actors: List[GTActor]


# Blueprint prefix → cyclist class
_CYCLIST_PREFIXES = (
    'vehicle.bh.crossbike',
    'vehicle.gazelle.omafiets',
    'vehicle.diamondback.century',
)


class GroundTruthLogger:
    """
    Captures CARLA actor ground truth per frame and matches them to
    detector outputs for metric computation.

    Ground truth extraction:
        CARLA provides world-space actor positions via actor.get_transform().
        We convert to ego-local coordinates using the inverse of the ego
        vehicle's yaw rotation, matching the convention in lidar_obstacle_detector.py
        (azimuth = atan2(lateral, forward)).
    """

    def __init__(
        self,
        world: carla.World,
        ego_vehicle: carla.Vehicle,
        log_path: str = 'output/gt_log.jsonl',
        max_range_m: float = 80.0,
    ):
        self._world = world
        self._ego = ego_vehicle
        self._log_path = log_path
        self._max_range_m = max_range_m
        self._buffer: List[FrameGT] = []
        self._frame_index: Dict[int, FrameGT] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_frame(
        self,
        frame_id: int,
        weather_tag: str = 'unknown',
    ) -> FrameGT:
        """
        Snapshot current world state. Reads all actors, filters to non-ego
        vehicles and walkers within max_range_m, computes ego-local coords.
        """
        ego_tf = self._ego.get_transform()
        ego_vel = self._ego.get_velocity()
        ego_speed = 3.6 * math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)

        actors: List[GTActor] = []
        for actor in self._world.get_actors():
            if actor.id == self._ego.id:
                continue
            type_id = actor.type_id
            if not (type_id.startswith('vehicle.') or type_id.startswith('walker.')):
                continue

            world_loc = actor.get_transform().location
            local_x, local_y, local_z = self._world_to_ego_local(ego_tf, world_loc)
            distance = math.sqrt(local_x**2 + local_y**2)
            if distance > self._max_range_m:
                continue

            azimuth = math.degrees(math.atan2(local_y, local_x))
            class_name = self._classify_actor(type_id)

            bb = actor.bounding_box
            gt = GTActor(
                actor_id=actor.id,
                class_name=class_name,
                world_x=world_loc.x,
                world_y=world_loc.y,
                world_z=world_loc.z,
                ego_local_x=local_x,
                ego_local_y=local_y,
                distance_m=distance,
                azimuth_deg=azimuth,
                bbox_extent_x=bb.extent.x,
                bbox_extent_y=bb.extent.y,
                bbox_extent_z=bb.extent.z,
            )
            actors.append(gt)

        frame_gt = FrameGT(
            frame_id=frame_id,
            timestamp=self._world.get_snapshot().timestamp.elapsed_seconds,
            ego_speed_kmh=ego_speed,
            weather_tag=weather_tag,
            actors=actors,
        )
        self._buffer.append(frame_gt)
        self._frame_index[frame_id] = frame_gt
        return frame_gt

    def flush(self):
        """Write all buffered FrameGTs to log_path as JSONL."""
        os.makedirs(os.path.dirname(self._log_path) or '.', exist_ok=True)
        with open(self._log_path, 'a') as f:
            for fg in self._buffer:
                f.write(json.dumps(asdict(fg)) + '\n')
        self._buffer.clear()

    def get_frame(self, frame_id: int) -> Optional[FrameGT]:
        return self._frame_index.get(frame_id)

    def match_detection_to_gt(
        self,
        detection: Dict,
        frame_gt: FrameGT,
        focal_length_px: float,
        img_width: int,
        azimuth_tolerance_deg: float = 15.0,
        distance_tolerance_m: float = 8.0,
    ) -> Optional[GTActor]:
        """
        Find the closest GT actor to a detection dict using greedy
        angular + distance proximity matching.

        det_azimuth is computed from bbox centre using the pinhole model.
        """
        det_dist = detection.get('stereo_distance') or detection.get('distance')
        bbox = detection.get('bbox')
        if det_dist is None or bbox is None:
            return None

        cx = (bbox[0] + bbox[2]) / 2.0
        # Azimuth from image centre: positive = right
        det_azimuth = math.degrees(math.atan2(cx - img_width / 2.0, focal_length_px))

        best: Optional[GTActor] = None
        best_score = float('inf')

        for actor in frame_gt.actors:
            az_diff = abs(det_azimuth - actor.azimuth_deg)
            dist_diff = abs(det_dist - actor.distance_m)
            if az_diff <= azimuth_tolerance_deg and dist_diff <= distance_tolerance_m:
                score = az_diff + dist_diff
                if score < best_score:
                    best_score = score
                    best = actor

        return best

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_actor(type_id: str) -> str:
        if type_id.startswith('walker.'):
            return 'pedestrian'
        for prefix in _CYCLIST_PREFIXES:
            if type_id.startswith(prefix):
                return 'cyclist'
        return 'vehicle'

    @staticmethod
    def _world_to_ego_local(
        ego_transform: carla.Transform,
        world_location: carla.Location,
    ) -> Tuple[float, float, float]:
        """
        Convert world_location to ego-local coordinates (x=forward, y=lateral).
        Uses inverse yaw rotation (CARLA is left-handed, Y points right in world).
        """
        yaw_rad = math.radians(-ego_transform.rotation.yaw)
        dx = world_location.x - ego_transform.location.x
        dy = world_location.y - ego_transform.location.y
        dz = world_location.z - ego_transform.location.z
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        local_x = dx * cos_y - dy * sin_y
        local_y = dx * sin_y + dy * cos_y
        return local_x, local_y, dz
