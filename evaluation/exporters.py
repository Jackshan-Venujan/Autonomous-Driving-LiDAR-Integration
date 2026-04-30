"""
Simple exporters to save camera and LiDAR detections in a unified JSONL schema.
"""
import os
import json
import uuid
from typing import List, Dict, Iterable


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def export_camera_detections_jsonl(detections: Iterable[Dict], out_dir: str, frame_id: str):
    """Append camera detections for a frame to out_dir/camera_detections.jsonl.

    Each line is a JSON object with keys:
      frame_id, detection_id, sensor='camera', class, class_id, bbox, bbox_center,
      distance, confidence, is_dangerous, danger_level, in_lane
    """
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "camera_detections.jsonl")
    with open(path, "a") as fh:
        for i, d in enumerate(detections):
            rec = {
                'frame_id': str(frame_id),
                'detection_id': d.get('id', f"cam-{frame_id}-{i}-{uuid.uuid4().hex[:6]}"),
                'sensor': 'camera',
                'class': d.get('class'),
                'class_id': d.get('class_id'),
                'bbox': list(d.get('bbox') if d.get('bbox') is not None else []),
                'bbox_center': list(d.get('bbox_center') if d.get('bbox_center') is not None else []),
                'distance': None if d.get('distance') is None else float(d.get('distance')),
                'confidence': float(d.get('confidence', 0.0)),
                'is_dangerous': bool(d.get('is_dangerous', False)),
                'danger_level': d.get('danger_level'),
                'in_lane': bool(d.get('in_lane', False)),
            }
            fh.write(json.dumps(rec) + "\n")
    return path


def export_lidar_obstacles_jsonl(obstacles: Iterable[object], out_dir: str, frame_id: str):
    """Append LiDAR obstacles for a frame to out_dir/lidar_obstacles.jsonl.

    Each line is a JSON object with keys:
      frame_id, detection_id, sensor='lidar', centroid (x,y,z), distance, angle_deg,
      sector, point_count, danger_level, bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y

    Accepts either dataclass-like objects with attributes or dicts.
    """
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "lidar_obstacles.jsonl")
    with open(path, "a") as fh:
        for i, o in enumerate(obstacles):
            if isinstance(o, dict):
                obj = o
                centroid = obj.get('centroid', None)
            else:
                centroid = [getattr(o, 'centroid_x', None), getattr(o, 'centroid_y', None), getattr(o, 'centroid_z', None)]
                obj = {
                    'centroid_x': getattr(o, 'centroid_x', None),
                    'centroid_y': getattr(o, 'centroid_y', None),
                    'centroid_z': getattr(o, 'centroid_z', None),
                    'distance': getattr(o, 'distance', None),
                    'angle_deg': getattr(o, 'angle_deg', None),
                    'sector': getattr(o, 'sector', None),
                    'point_count': getattr(o, 'point_count', None),
                    'danger_level': getattr(o, 'danger_level', None),
                    'bbox_min_x': getattr(o, 'bbox_min_x', None),
                    'bbox_max_x': getattr(o, 'bbox_max_x', None),
                    'bbox_min_y': getattr(o, 'bbox_min_y', None),
                    'bbox_max_y': getattr(o, 'bbox_max_y', None),
                }

            rec = {
                'frame_id': str(frame_id),
                'detection_id': obj.get('detection_id', f"lidar-{frame_id}-{i}-{uuid.uuid4().hex[:6]}"),
                'sensor': 'lidar',
                'centroid': centroid,
                'distance': None if obj.get('distance') is None else float(obj.get('distance')),
                'angle_deg': obj.get('angle_deg'),
                'sector': obj.get('sector'),
                'point_count': int(obj.get('point_count') or 0),
                'danger_level': obj.get('danger_level'),
                'bbox_min_x': obj.get('bbox_min_x'),
                'bbox_max_x': obj.get('bbox_max_x'),
                'bbox_min_y': obj.get('bbox_min_y'),
                'bbox_max_y': obj.get('bbox_max_y'),
            }
            fh.write(json.dumps(rec) + "\n")
    return path
