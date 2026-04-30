"""
Live evaluation reporter: tracks camera and LiDAR detections during runtime.

Integrates with DrivingAgent to export detections and generate comparison reports.
"""
import os
import json
from collections import defaultdict
from datetime import datetime


def _convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(v) for v in obj]
    
    # Convert numpy types
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.int_, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    
    return obj


class LiveEvaluationReporter:
    """Tracks detections during a run and produces live comparison reports."""

    def __init__(self, output_dir='output/eval_reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.frame_count = 0
        self.start_time = datetime.now()

        # Tracks by frame and sensor
        self.camera_dets_by_frame = defaultdict(list)  # frame_id -> list of dets
        self.lidar_dets_by_frame = defaultdict(list)   # frame_id -> list of dets

        # Aggregate stats
        self.camera_total_dets = 0
        self.lidar_total_dets = 0
        self.camera_dangerous = 0
        self.lidar_dangerous = 0

        # Latest frame info
        self.latest_frame_id = None
        self.latest_camera_count = 0
        self.latest_lidar_count = 0

    def log_frame_detections(self, frame_id, camera_dets, lidar_obstacles):
        """Log detections for a frame.

        Args:
            frame_id: unique frame identifier
            camera_dets: list of camera detection dicts from ObstacleDetector
            lidar_obstacles: list of LidarObstacle objects from LidarObstacleDetector
        """
        self.frame_count += 1
        self.latest_frame_id = frame_id

        # Store camera detections
        for i, d in enumerate(camera_dets):
            rec = {
                'frame_id': str(frame_id),
                'detection_id': f"cam-{frame_id}-{i}",
                'sensor': 'camera',
                'class': d.get('class'),
                'distance': d.get('distance'),
                'confidence': float(d.get('confidence', 0.0)),
                'danger_level': d.get('danger_level'),
                'in_lane': d.get('in_lane', False),
            }
            self.camera_dets_by_frame[str(frame_id)].append(rec)
            self.camera_total_dets += 1
            if d.get('is_dangerous'):
                self.camera_dangerous += 1

        # Store LiDAR obstacles
        for i, o in enumerate(lidar_obstacles):
            if isinstance(o, dict):
                obj = o
            else:
                obj = {
                    'centroid': [getattr(o, 'centroid_x', None),
                                getattr(o, 'centroid_y', None),
                                getattr(o, 'centroid_z', None)],
                    'distance': getattr(o, 'distance', None),
                    'sector': getattr(o, 'sector', None),
                    'point_count': getattr(o, 'point_count', None),
                    'danger_level': getattr(o, 'danger_level', None),
                }
            rec = {
                'frame_id': str(frame_id),
                'detection_id': f"lidar-{frame_id}-{i}",
                'sensor': 'lidar',
                'distance': obj.get('distance'),
                'sector': obj.get('sector'),
                'point_count': obj.get('point_count'),
                'danger_level': obj.get('danger_level'),
            }
            self.lidar_dets_by_frame[str(frame_id)].append(rec)
            self.lidar_total_dets += 1
            if obj.get('danger_level') in ('stop', 'emergency_stop'):
                self.lidar_dangerous += 1

        self.latest_camera_count = len(camera_dets)
        self.latest_lidar_count = len(lidar_obstacles)

    def get_live_stats(self):
        """Return current live statistics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            'frames_processed': self.frame_count,
            'elapsed_seconds': elapsed,
            'fps': self.frame_count / elapsed if elapsed > 0 else 0,
            'camera_total_detections': self.camera_total_dets,
            'camera_dangerous': self.camera_dangerous,
            'camera_avg_per_frame': (self.camera_total_dets / self.frame_count) if self.frame_count > 0 else 0,
            'lidar_total_detections': self.lidar_total_dets,
            'lidar_dangerous': self.lidar_dangerous,
            'lidar_avg_per_frame': (self.lidar_total_dets / self.frame_count) if self.frame_count > 0 else 0,
            'latest_frame': self.latest_frame_id,
            'latest_camera_count': self.latest_camera_count,
            'latest_lidar_count': self.latest_lidar_count,
        }

    def print_live_report(self):
        """Print a simple live report to console."""
        stats = self.get_live_stats()
        print("\n" + "=" * 70)
        print("  LIVE EVALUATION REPORT")
        print("=" * 70)
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Elapsed time: {stats['elapsed_seconds']:.1f}s ({stats['fps']:.1f} fps)")
        print()
        print(f"  CAMERA:")
        print(f"    Total detections: {stats['camera_total_detections']}")
        print(f"    Dangerous: {stats['camera_dangerous']}")
        print(f"    Avg per frame: {stats['camera_avg_per_frame']:.2f}")
        print()
        print(f"  LIDAR:")
        print(f"    Total detections: {stats['lidar_total_detections']}")
        print(f"    Dangerous: {stats['lidar_dangerous']}")
        print(f"    Avg per frame: {stats['lidar_avg_per_frame']:.2f}")
        print()
        print(f"  Latest Frame: {stats['latest_frame']}")
        print(f"    Camera detections: {stats['latest_camera_count']}")
        print(f"    LiDAR detections: {stats['latest_lidar_count']}")
        print("=" * 70)

    def save_jsonl_exports(self):
        """Save all logged detections to JSONL files."""
        cam_file = os.path.join(self.output_dir, 'camera_detections.jsonl')
        lidar_file = os.path.join(self.output_dir, 'lidar_obstacles.jsonl')

        # Camera detections
        with open(cam_file, 'w') as fh:
            for frame_id in sorted(self.camera_dets_by_frame.keys()):
                for det in self.camera_dets_by_frame[frame_id]:
                    fh.write(json.dumps(_convert_to_serializable(det)) + '\n')

        # LiDAR detections
        with open(lidar_file, 'w') as fh:
            for frame_id in sorted(self.lidar_dets_by_frame.keys()):
                for det in self.lidar_dets_by_frame[frame_id]:
                    fh.write(json.dumps(_convert_to_serializable(det)) + '\n')

        print(f"\n✓ Exported camera detections: {cam_file}")
        print(f"✓ Exported LiDAR detections: {lidar_file}")

    def save_summary_json(self):
        """Save aggregate statistics to JSON."""
        stats = self.get_live_stats()
        summary_file = os.path.join(self.output_dir, 'live_report.json')
        with open(summary_file, 'w') as fh:
            json.dump(_convert_to_serializable(stats), fh, indent=2)
        print(f"✓ Saved live report: {summary_file}")
        return summary_file

    def print_comparison_summary(self):
        """Print a sensor comparison summary."""
        stats = self.get_live_stats()
        print("\n" + "=" * 70)
        print("  SENSOR COMPARISON SUMMARY")
        print("=" * 70)

        # Detection rates
        cam_rate = stats['camera_avg_per_frame']
        lidar_rate = stats['lidar_avg_per_frame']
        print(f"  Detection rate (avg per frame):")
        print(f"    Camera: {cam_rate:.2f} objects/frame")
        print(f"    LiDAR:  {lidar_rate:.2f} objects/frame")
        if cam_rate > 0:
            print(f"    Difference: {lidar_rate - cam_rate:+.2f} ({(lidar_rate/cam_rate - 1)*100:+.1f}%)")

        # Danger detection
        cam_danger_pct = (stats['camera_dangerous'] / stats['camera_total_detections'] * 100) if stats['camera_total_detections'] > 0 else 0
        lidar_danger_pct = (stats['lidar_dangerous'] / stats['lidar_total_detections'] * 100) if stats['lidar_total_detections'] > 0 else 0
        print()
        print(f"  Dangerous objects detected (% of total):")
        print(f"    Camera: {stats['camera_dangerous']}/{stats['camera_total_detections']} ({cam_danger_pct:.1f}%)")
        print(f"    LiDAR:  {stats['lidar_dangerous']}/{stats['lidar_total_detections']} ({lidar_danger_pct:.1f}%)")

        print("=" * 70)
