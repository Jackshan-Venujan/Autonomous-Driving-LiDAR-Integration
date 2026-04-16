"""
Core utility modules for autonomous driving system
Separated from model utils to avoid confusion
"""

from .roi_selector import ROISelector
from .pid_controller import PIDController
from .carla_spawner import CarlaSpawner
from .lidar_sensor import LidarSensor, LidarFrame
from .lidar_preprocessor import LidarPreprocessor, PreprocessConfig, PreprocessedCloud
from .lidar_clusterer import LidarClusterer, ClusterConfig, Obstacle, ClusterResult
from .kalman_filter        import ObjectKalmanFilter
from .lidar_tracker        import MultiObjectTracker, TrackerConfig, TrackedObstacle
from .fusion_calibration   import CameraCalibration, FusionExtrinsics
from .lidar_fusion         import (
    SensorFusion, FusionConfig, FusedObstacle, FusionResult, YoloDetection,
)
from .lidar_hud            import HUDConfig, DisplayData, LiDARHUD
from .hud_renderer         import HUDRenderer
from .radar_view           import RadarView

__all__ = [
    'ROISelector', 'PIDController', 'CarlaSpawner',
    'LidarSensor', 'LidarFrame',
    'LidarPreprocessor', 'PreprocessConfig', 'PreprocessedCloud',
    'LidarClusterer', 'ClusterConfig', 'Obstacle', 'ClusterResult',
    'ObjectKalmanFilter',
    'MultiObjectTracker', 'TrackerConfig', 'TrackedObstacle',
    'CameraCalibration', 'FusionExtrinsics',
    'SensorFusion', 'FusionConfig', 'FusedObstacle', 'FusionResult', 'YoloDetection',
    'HUDConfig', 'DisplayData', 'LiDARHUD',
    'HUDRenderer', 'RadarView',
]
