"""
CARLA Ground Truth Extractor — Extract vehicle/pedestrian positions from CARLA simulation API.

Automatically captures all actors' 3D positions relative to ego vehicle each frame.
"""

import json
import carla
import numpy as np
from typing import List, Dict, Optional
import os


class CarlaGTExtractor:
    """Extracts ground truth from CARLA actors."""

    # CARLA actor type → class name mapping
    CLASS_MAPPING = {
        'vehicle.tesla.model3': 'car',
        'vehicle.audi.a2': 'car',
        'vehicle.bmw.grandtourer': 'car',
        'vehicle.mercedes.coupe': 'car',
        'vehicle.toyota.prius': 'car',
        'vehicle.citroen.c3': 'car',
        'vehicle.jeep.wrangler': 'car',
        'vehicle.lincoln.mkz': 'car',
        'vehicle.nissan.patrol': 'suv',
        'vehicle.dodge.charger': 'car',
        'vehicle.carla.dolly': 'truck',
        'vehicle.carla.firetruck': 'truck',
        'vehicle.ford.mustang': 'car',
        'vehicle.micro.microlino': 'car',
        'vehicle.kawasaki.ninja': 'motorcycle',
        'vehicle.vespa.zj300': 'motorcycle',
        'vehicle.harley.davidson': 'motorcycle',
        'vehicle.diamondback.century': 'bicycle',
        'walker.pedestrian.0001': 'person',
        'walker.pedestrian.0002': 'person',
    }

    def __init__(self, world: carla.World, ego_vehicle: carla.Vehicle, output_dir: str = 'output/ground_truth'):
        """
        Args:
            world: CARLA world object
            ego_vehicle: ego vehicle actor
            output_dir: where to save GT JSONL
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # File to append GT records
        self.gt_file = os.path.join(output_dir, 'carla_ground_truth.jsonl')
        self.frame_count = 0

    def extract_frame_gt(self, frame_id: str) -> List[Dict]:
        """
        Extract all visible actors as GT for current frame.

        Args:
            frame_id: unique frame identifier (e.g., 'frame_0001')

        Returns:
            List of GT object dicts with keys: frame_id, gt_id, class, centroid, distance, bbox
        """
        self.frame_count += 1
        records = []

        #Get ego position and rotation
        ego_loc = self.ego_vehicle.get_location()
        ego_rot = self.ego_vehicle.get_transform().rotation

        # Convert ego rotation to forward vector (+X)
        ego_forward_rad = np.radians(ego_rot.yaw)
        ego_forward = np.array([np.cos(ego_forward_rad), np.sin(ego_forward_rad)])
        ego_right = np.array([-np.sin(ego_forward_rad), np.cos(ego_forward_rad)])

        # Iterate all actors
        actor_id = 0
        for actor in self.world.get_actors():
            # Skip ego vehicle itself
            if actor.id == self.ego_vehicle.id:
                continue

            # Filter for vehicles and pedestrians only
            is_vehicle = actor.type_id.startswith('vehicle.')
            is_pedestrian = actor.type_id.startswith('walker.pedestrian.')

            if not (is_vehicle or is_pedestrian):
                continue

            # Get actor location
            actor_loc = actor.get_location()

            # Convert to ego-relative coordinates
            # X = forward (along ego heading), Y = right (perpendicular)
            rel_pos = np.array([
                actor_loc.x - ego_loc.x,
                actor_loc.y - ego_loc.y
            ])

            # Rotate to ego frame
            x_ego = np.dot(rel_pos, ego_forward)
            y_ego = np.dot(rel_pos, ego_right)
            z_ego = actor_loc.z - ego_loc.z

            # Skip if actor is behind ego or too far away
            if x_ego < 0 or x_ego > 100:
                continue

            # Compute distance
            distance = np.sqrt(x_ego**2 + y_ego**2)
            if distance > 100:
                continue

            # Determine class
            class_name = self._get_actor_class(actor.type_id)

            # Get bounding box (optional, approximate for now)
            bbox = self._get_approximate_bbox(x_ego, y_ego, actor.type_id)

            # Create GT record
            gt = {
                'frame_id': str(frame_id),
                'gt_id': f"{actor.type_id.replace('.', '-')}-{actor.id}",
                'class': class_name,
                'centroid': [float(x_ego), float(y_ego), float(z_ego)],
                'distance': float(distance),
                'bbox': bbox,  # [x1, y1, x2, y2] in image coords (approximate)
                'actor_id': actor.id,
                'actor_type': actor.type_id,
            }
            records.append(gt)
            actor_id += 1

        return records

    def save_frame_gt(self, records: List[Dict]):
        """Append frame GT records to JSONL file."""
        with open(self.gt_file, 'a') as fh:
            for rec in records:
                fh.write(json.dumps(rec) + '\n')

    def _get_actor_class(self, actor_type_id: str) -> str:
        """Map CARLA actor type to class name."""
        # Direct lookup
        if actor_type_id in self.CLASS_MAPPING:
            return self.CLASS_MAPPING[actor_type_id]

        # Fuzzy matching
        for prefix, class_name in [
            ('vehicle.', 'car'),
            ('walker.pedestrian.', 'person'),
        ]:
            if actor_type_id.startswith(prefix):
                return class_name

        return 'unknown'

    def _get_approximate_bbox(self, x_ego: float, y_ego: float, actor_type_id: str) -> List[int]:
        """
        Approximate 3D position → 2D image bounding box.
        This is a heuristic; requires camera calibration for accuracy.

        Args:
            x_ego: forward distance (m)
            y_ego: lateral distance (m)
            actor_type_id: CARLA actor type

        Returns:
            [x1, y1, x2, y2] approximate bbox in image (1280x720 assumed)
        """
        # Assume camera: focal_length ≈ 640px (90° FOV on 1280px width)
        # Principal point: (640, 360)
        focal_length = 640
        cx, cy = 640, 360
        img_w, img_h = 1280, 720

        # Assume actor heights (meters)
        heights = {
            'vehicle.': 1.5,
            'walker.pedestrian.': 1.7,
            'vehicle.carla.firetruck': 3.5,
            'vehicle.carla.dolly': 2.0,
        }

        height = 1.5
        for prefix, h in heights.items():
            if actor_type_id.startswith(prefix):
                height = h
                break

        # Project 3D → 2D
        if x_ego <= 0.1:  # Too close
            return [0, 0, 100, 100]

        # Image coordinates (in pixels)
        # Assuming pinhole camera model
        x_img = cx + (focal_length * y_ego / x_ego)
        y_img_top = cy - (focal_length * height / x_ego)
        y_img_bot = cy + (focal_length * 0.5 / x_ego)  # Assume 0.5m ground clearance

        # Width estimate (rough)
        width_px = height * focal_length / x_ego * 0.6  # 0.6 = width/height ratio

        x1 = int(max(0, x_img - width_px / 2))
        x2 = int(min(img_w - 1, x_img + width_px / 2))
        y1 = int(max(0, y_img_top))
        y2 = int(min(img_h - 1, y_img_bot))

        return [x1, y1, x2, y2]

    def get_summary(self) -> Dict:
        """Return summary of extracted GT."""
        # Count records in file
        count = 0
        classes = {}
        with open(self.gt_file, 'r') as f:
            for line in f:
                count += 1
                rec = json.loads(line)
                cls = rec.get('class', 'unknown')
                classes[cls] = classes.get(cls, 0) + 1

        return {
            'total_records': count,
            'frames_captured': self.frame_count,
            'class_distribution': classes,
            'output_file': self.gt_file,
        }

    def print_summary(self):
        """Print extraction summary."""
        summary = self.get_summary()
        print("\n" + "=" * 70)
        print("  CARLA GROUND TRUTH EXTRACTION SUMMARY")
        print("=" * 70)
        print(f"  Frames captured: {summary['frames_captured']}")
        print(f"  Total objects: {summary['total_records']}")
        print(f"  Avg objects/frame: {summary['total_records'] / max(1, summary['frames_captured']):.2f}")
        print(f"\n  Class distribution:")
        for cls, count in summary['class_distribution'].items():
            print(f"    {cls}: {count}")
        print(f"\n  GT file: {summary['output_file']}")
        print(f"  Ready for evaluation!")
        print("=" * 70)

    @staticmethod
    def load_gt_jsonl(path: str) -> List[Dict]:
        """Load GT JSONL file."""
        records = []
        with open(path, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        return records
