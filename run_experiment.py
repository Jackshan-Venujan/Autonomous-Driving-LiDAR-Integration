"""
ADAS Level 2 Sensor Fusion Experiment — Entry Point

Usage:
    python run_experiment.py --experiment lidar     # Experiment 1 only
    python run_experiment.py --experiment stereo    # Experiment 2 only
    python run_experiment.py --experiment both      # Both sequentially (default)
    python run_experiment.py --report-only          # Generate report from saved CSVs

Options:
    --config PATH       Path to config_experiment.yaml  (default: ./config_experiment.yaml)
    --experiment        'lidar' | 'stereo' | 'both'
    --report-only       Skip simulation; generate report from existing CSV files
    --scenarios LIST    Comma-separated scenario names to run (default: all)
    --output-dir PATH   Override output directory from config

Experiment flow (--experiment both):
    Phase 1  →  LiDAR + monocular camera  →  exp1_metrics.csv
    Phase 2  →  Stereo camera pair        →  exp2_metrics.csv
    Phase 3  →  Comparison report         →  comparison_table.csv + radar_chart.png
"""

import argparse
import os
import csv
import sys
import time
import traceback
import threading
import yaml
from typing import Optional, List, Dict

import carla
import numpy as np

from modules.driving_agent import DrivingAgent
from core.lidar_sensor import LidarSensor
from core.stereo_camera import StereoPair
from core.stereo_depth import StereoDepthEstimator
from evaluation.ground_truth_logger import GroundTruthLogger
from evaluation.metrics_engine import MetricsEngine, ExperimentResults, CLASSES
from evaluation.scenario_runner import (
    ScenarioRunner, ScenarioDefinition, PREDEFINED_SCENARIOS, scenarios_from_config,
)
from evaluation.comparison_report import ComparisonReport


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def filter_scenarios(
    scenarios: List[ScenarioDefinition],
    names_csv: Optional[str],
) -> List[ScenarioDefinition]:
    if not names_csv:
        return scenarios
    names = set(n.strip() for n in names_csv.split(','))
    filtered = [s for s in scenarios if s.name in names]
    if not filtered:
        print(f"[Warning] No scenarios matched '{names_csv}'. Using all.")
        return scenarios
    return filtered


# ---------------------------------------------------------------------------
# CARLA setup helpers
# ---------------------------------------------------------------------------

def connect_carla(cfg: dict):
    carla_cfg = cfg.get('carla', {})
    client = carla.Client(carla_cfg.get('host', 'localhost'), carla_cfg.get('port', 2000))
    client.set_timeout(carla_cfg.get('timeout_s', 10.0))
    world = client.load_world(carla_cfg.get('world_map', 'Town04'))
    time.sleep(1.0)  # let world settle after loading
    return client, world


def spawn_ego_vehicle(world: carla.World, cfg: dict) -> carla.Vehicle:
    bp_lib = world.get_blueprint_library()
    blueprint = cfg.get('ego_vehicle', {}).get('blueprint', 'vehicle.tesla.model3')
    spawn_index = cfg.get('ego_vehicle', {}).get('spawn_index', 0)
    bp = bp_lib.find(blueprint)
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(bp, spawn_points[spawn_index])
    return vehicle


def spawn_monocular_camera(world: carla.World, vehicle: carla.Vehicle, cfg: dict):
    """Returns (actor, frame_ref) where frame_ref is a list holding the latest BGR frame."""
    cam_cfg = cfg.get('camera', {})
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(cam_cfg.get('image_width', 1280)))
    bp.set_attribute('image_size_y', str(cam_cfg.get('image_height', 720)))
    bp.set_attribute('fov', str(cam_cfg.get('fov_degrees', 90.0)))

    transform = carla.Transform(
        carla.Location(x=cam_cfg.get('position_x', 2.0), z=cam_cfg.get('position_z', 1.4)),
        carla.Rotation(pitch=cam_cfg.get('pitch_degrees', -15.0)),
    )
    actor = world.spawn_actor(bp, transform, attach_to=vehicle)

    frame_ref: List[Optional[np.ndarray]] = [None]
    lock = threading.Lock()

    def callback(image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        with lock:
            frame_ref[0] = arr

    actor.listen(callback)
    return actor, frame_ref, lock


def focal_length_from_config(cfg: dict) -> float:
    cam = cfg.get('camera', {})
    w = cam.get('image_width', 1280)
    fov = cam.get('fov_degrees', 90.0)
    import math
    return (w / 2.0) / math.tan(math.radians(fov / 2.0))


# ---------------------------------------------------------------------------
# Experiment 1: LiDAR + monocular camera
# ---------------------------------------------------------------------------

def run_experiment_lidar(
    client: carla.Client,
    world: carla.World,
    vehicle: carla.Vehicle,
    cfg: dict,
    scenarios: List[ScenarioDefinition],
    output_dir: str,
) -> ExperimentResults:
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: LiDAR + Monocular Camera Fusion")
    print("=" * 60)

    cam_cfg = cfg.get('camera', {})
    lidar_cfg = cfg.get('experiment_1_lidar_mono', {}).get('lidar', {})
    focal_px = focal_length_from_config(cfg)
    img_width = cam_cfg.get('image_width', 1280)

    # Spawn sensors
    cam_actor, frame_ref, frame_lock = spawn_monocular_camera(world, vehicle, cfg)

    # Build a configured carla.ActorBlueprint (LidarSensor expects the blueprint object)
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels',           str(lidar_cfg.get('channels', 32)))
    lidar_bp.set_attribute('points_per_second',  str(lidar_cfg.get('points_per_second', 56000)))
    lidar_bp.set_attribute('rotation_frequency', str(lidar_cfg.get('rotation_frequency', 20)))
    lidar_bp.set_attribute('range',              str(lidar_cfg.get('range_m', 50)))
    lidar_bp.set_attribute('upper_fov',          str(lidar_cfg.get('upper_fov', 10)))
    lidar_bp.set_attribute('lower_fov',          str(lidar_cfg.get('lower_fov', -10)))

    lidar_transform = carla.Transform(
        carla.Location(
            x=lidar_cfg.get('position_x', 2.0),
            z=lidar_cfg.get('position_z', 1.8),
        )
    )
    lidar = LidarSensor(world, vehicle, lidar_bp, lidar_transform)

    # Wait for first frames
    time.sleep(1.0)

    agent = DrivingAgent(world, vehicle, lidar_sensor=lidar)

    gt_logger = GroundTruthLogger(
        world, vehicle,
        log_path=os.path.join(output_dir, cfg.get('output', {}).get('gt_log_filename', 'gt_log_exp1.jsonl')),
        max_range_m=cfg.get('metrics', {}).get('max_range_m', 80.0),
    )
    engine = MetricsEngine(
        experiment_name='lidar_monocular_fusion',
        focal_length_px=focal_px,
        img_width=img_width,
        azimuth_tolerance_deg=cfg.get('metrics', {}).get('azimuth_tolerance_deg', 15.0),
        distance_tolerance_m=cfg.get('metrics', {}).get('distance_tolerance_m', 8.0),
    )

    def get_frame():
        with frame_lock:
            f = frame_ref[0]
        return f.copy() if f is not None else None

    def frame_callback(camera_frame: np.ndarray, frame_id: int) -> dict:
        return agent.process_frame(camera_frame)

    runner = ScenarioRunner(
        client=client,
        world=world,
        vehicle=vehicle,
        frame_callback=frame_callback,
        gt_logger=gt_logger,
        metrics_engine=engine,
        scenarios=scenarios,
        get_camera_frame=get_frame,
        use_synchronous_mode=cfg.get('carla', {}).get('synchronous_mode', True),
    )

    try:
        runner.run_all()
    finally:
        # agent.cleanup() already destroys the LiDAR sensor internally — do NOT call lidar.destroy() again
        try:
            agent.cleanup()
        except Exception as e:
            print(f"[Exp1] agent.cleanup error (non-fatal): {e}")
        try:
            cam_actor.stop()
            cam_actor.destroy()
        except Exception as e:
            print(f"[Exp1] camera cleanup error (non-fatal): {e}")
        try:
            gt_logger.flush()
        except Exception as e:
            print(f"[Exp1] gt_logger flush error (non-fatal): {e}")

    results = engine.compute()
    csv_path = os.path.join(output_dir, 'exp1_metrics.csv')
    engine.save_csv(csv_path)
    print(f"[Exp1] Results saved to {csv_path}")
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Stereo camera
# ---------------------------------------------------------------------------

def run_experiment_stereo(
    client: carla.Client,
    world: carla.World,
    vehicle: carla.Vehicle,
    cfg: dict,
    scenarios: List[ScenarioDefinition],
    output_dir: str,
) -> ExperimentResults:
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Stereo Camera Fusion")
    print("=" * 60)

    cam_cfg = cfg.get('camera', {})
    stereo_cfg = cfg.get('experiment_2_stereo', {}).get('stereo', {})
    metrics_cfg = cfg.get('metrics', {})
    focal_px = focal_length_from_config(cfg)
    img_width = cam_cfg.get('image_width', 1280)
    baseline_m = stereo_cfg.get('baseline_m', 0.12)

    stereo_pair = StereoPair(
        world=world,
        vehicle=vehicle,
        baseline_m=baseline_m,
        image_width=cam_cfg.get('image_width', 1280),
        image_height=cam_cfg.get('image_height', 720),
        fov_degrees=cam_cfg.get('fov_degrees', 90.0),
        cam_x=cam_cfg.get('position_x', 2.0),
        cam_z=cam_cfg.get('position_z', 1.4),
        cam_pitch=cam_cfg.get('pitch_degrees', -15.0),
        sgbm_num_disparities=stereo_cfg.get('sgbm_num_disparities', 128),
        sgbm_block_size=stereo_cfg.get('sgbm_block_size', 11),
        sgbm_p1=stereo_cfg.get('sgbm_p1', 2904),
        sgbm_p2=stereo_cfg.get('sgbm_p2', 11616),
        sgbm_disp12_max_diff=stereo_cfg.get('sgbm_disp12_max_diff', 1),
        sgbm_uniqueness_ratio=stereo_cfg.get('sgbm_uniqueness_ratio', 10),
        sgbm_speckle_window_size=stereo_cfg.get('sgbm_speckle_window_size', 100),
        sgbm_speckle_range=stereo_cfg.get('sgbm_speckle_range', 32),
        sgbm_pre_filter_cap=stereo_cfg.get('sgbm_pre_filter_cap', 63),
    )

    depth_estimator = StereoDepthEstimator(
        focal_length_px=focal_px,
        baseline_m=baseline_m,
        depth_percentile=metrics_cfg.get('depth_percentile', 20.0),
    )

    # Agent runs in camera-only mode (no LiDAR); stereo depth injected externally
    agent = DrivingAgent(world, vehicle, lidar_sensor=None)

    gt_logger = GroundTruthLogger(
        world, vehicle,
        log_path=os.path.join(output_dir, 'gt_log_exp2.jsonl'),
        max_range_m=metrics_cfg.get('max_range_m', 80.0),
    )
    engine = MetricsEngine(
        experiment_name='stereo_camera_fusion',
        focal_length_px=focal_px,
        img_width=img_width,
        azimuth_tolerance_deg=metrics_cfg.get('azimuth_tolerance_deg', 15.0),
        distance_tolerance_m=metrics_cfg.get('distance_tolerance_m', 8.0),
    )

    def get_frame():
        left, _, _ = stereo_pair.get_latest()
        return left.copy() if left is not None else None

    def frame_callback(camera_frame: np.ndarray, frame_id: int) -> dict:
        left, right, _ = stereo_pair.get_latest()
        if left is None or right is None:
            return agent.process_frame(camera_frame)
        disparity = stereo_pair.compute_disparity(left, right)
        result = agent.process_frame(left)
        obs_data = result.get('obstacle_data') or {}
        dets = obs_data.get('lane_detections') or []
        speed = 0.0
        try:
            vel = vehicle.get_velocity()
            import math
            speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        except Exception:
            pass
        fused_dets, fused_action, _ = depth_estimator.fuse(dets, disparity, 'drive', speed)
        if obs_data:
            obs_data['lane_detections'] = fused_dets
        return result

    runner = ScenarioRunner(
        client=client,
        world=world,
        vehicle=vehicle,
        frame_callback=frame_callback,
        gt_logger=gt_logger,
        metrics_engine=engine,
        scenarios=scenarios,
        get_camera_frame=get_frame,
        use_synchronous_mode=cfg.get('carla', {}).get('synchronous_mode', True),
    )

    try:
        runner.run_all()
    finally:
        try:
            agent.cleanup()
        except Exception as e:
            print(f"[Exp2] agent.cleanup error (non-fatal): {e}")
        try:
            stereo_pair.destroy()
        except Exception as e:
            print(f"[Exp2] stereo_pair cleanup error (non-fatal): {e}")
        try:
            gt_logger.flush()
        except Exception as e:
            print(f"[Exp2] gt_logger flush error (non-fatal): {e}")

    results = engine.compute()
    csv_path = os.path.join(output_dir, 'exp2_metrics.csv')
    engine.save_csv(csv_path)
    print(f"[Exp2] Results saved to {csv_path}")
    return results


# ---------------------------------------------------------------------------
# Report generation from saved CSVs
# ---------------------------------------------------------------------------

def results_from_csv(csv_path: str, experiment_name: str) -> ExperimentResults:
    """Reconstruct ExperimentResults from a previously saved flat CSV row."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)

    def _f(key: float) -> float:
        val = row.get(key)
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    per_cls_05 = {cls: _f(f'ap05_{cls}') for cls in CLASSES}
    per_cls_07 = {cls: _f(f'ap07_{cls}') for cls in CLASSES}
    fnr = {cls: _f(f'fnr_{cls}') for cls in CLASSES}
    fpr = {cls: _f(f'fpr_{cls}') for cls in CLASSES}
    recall = {cls: _f(f'recall_{cls}') for cls in CLASSES}

    weather_tags = ['clear', 'rain', 'night', 'cloudy_evening']
    map_weather = {tag: _f(f'map_{tag}') for tag in weather_tags if f'map_{tag}' in row}

    return ExperimentResults(
        experiment_name=experiment_name,
        map_iou05=_f('map_iou05'),
        map_iou07=_f('map_iou07'),
        per_class_ap_iou05=per_cls_05,
        per_class_ap_iou07=per_cls_07,
        depth_rmse=_f('depth_rmse'),
        depth_mae=_f('depth_mae'),
        depth_delta_1_25=_f('depth_delta_1_25'),
        fnr_per_class=fnr,
        fpr_per_class=fpr,
        recall_per_class=recall,
        mean_latency_ms=_f('mean_latency_ms'),
        p95_latency_ms=_f('p95_latency_ms'),
        mean_fps=_f('mean_fps'),
        map_by_weather=map_weather,
    )


def generate_report(
    result1: ExperimentResults,
    result2: ExperimentResults,
    comparison_dir: str,
):
    report = ComparisonReport(result1, result2, output_dir=comparison_dir)
    report.generate_all()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ADAS Level 2 Sensor Fusion Experiment')
    parser.add_argument('--config', default='config_experiment.yaml',
                        help='Path to config_experiment.yaml')
    parser.add_argument('--experiment', choices=['lidar', 'stereo', 'both'],
                        default='both', help='Which experiment to run')
    parser.add_argument('--report-only', action='store_true',
                        help='Generate comparison report from existing CSVs (no CARLA)')
    parser.add_argument('--scenarios', default=None,
                        help='Comma-separated scenario names to run (default: all)')
    parser.add_argument('--output-dir', default=None,
                        help='Override output directory')
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg.get('output', {}).get('base_dir', 'output/experiment_results')
    comparison_dir = cfg.get('output', {}).get('comparison_dir', 'output/comparison')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    # Build scenario list
    scenario_list_raw = cfg.get('scenarios', [])
    if scenario_list_raw:
        all_scenarios = scenarios_from_config(scenario_list_raw)
    else:
        all_scenarios = PREDEFINED_SCENARIOS
    scenarios = filter_scenarios(all_scenarios, args.scenarios)
    print(f"[main] Running {len(scenarios)} scenario(s): {[s.name for s in scenarios]}")

    # --report-only: skip CARLA and reconstruct from CSVs
    if args.report_only:
        csv1 = os.path.join(output_dir, 'exp1_metrics.csv')
        csv2 = os.path.join(output_dir, 'exp2_metrics.csv')
        for path in [csv1, csv2]:
            if not os.path.exists(path):
                print(f"[Error] CSV not found: {path}  — run the experiments first.")
                sys.exit(1)
        r1 = results_from_csv(csv1, 'lidar_monocular_fusion')
        r2 = results_from_csv(csv2, 'stereo_camera_fusion')
        generate_report(r1, r2, comparison_dir)
        return

    # Connect to CARLA — shared for both experiments
    client, world = connect_carla(cfg)
    vehicle = spawn_ego_vehicle(world, cfg)

    result1: Optional[ExperimentResults] = None
    result2: Optional[ExperimentResults] = None

    try:
        if args.experiment in ('lidar', 'both'):
            result1 = run_experiment_lidar(client, world, vehicle, cfg, scenarios, output_dir)

        if args.experiment in ('stereo', 'both'):
            result2 = run_experiment_stereo(client, world, vehicle, cfg, scenarios, output_dir)

        if result1 and result2:
            generate_report(result1, result2, comparison_dir)
        elif result1 or result2:
            r = result1 or result2
            name = r.experiment_name if r else 'unknown'
            print(f"\n[main] Single experiment '{name}' complete. Run both to generate comparison report.")

    except Exception:
        traceback.print_exc()
    finally:
        try:
            vehicle.destroy()
        except Exception:
            pass


if __name__ == '__main__':
    main()
