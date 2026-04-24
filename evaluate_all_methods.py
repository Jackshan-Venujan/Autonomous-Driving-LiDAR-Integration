"""
evaluate_all_methods.py
───────────────────────────────────────────────────────────────────────────────
FYP Experiment — Section 4
Measures and compares three depth-estimation methods simultaneously on
identical CARLA simulation frames.

RESEARCH QUESTION:
  "Which depth-sensing method — monocular pinhole, stereo camera, or LiDAR —
   is most accurate for vehicle-following distance estimation in an autonomous
   driving system?"

METHODS COMPARED:
  1. Monocular Pinhole  — Z = (focal_length × known_height) / bbox_height_px
                          The classic single-camera distance formula. Fast,
                          but assumes the object height is always 1.5 m.
  2. Stereo Camera      — Z = (focal_length × baseline) / disparity
                          Uses two cameras 0.54 m apart and SGBM matching.
                          More accurate than pinhole; affected by lighting.
  3. LiDAR              — Nearest point-cloud cluster in forward cone.
                          Highest accuracy; unaffected by lighting.
                          Cannot tell WHAT the obstacle is (YOLO does that).

EXPERIMENT DESIGN:
  - A static red target vehicle is placed at N known distances (ground truth).
  - All three methods are run on the SAME tick so timing is identical.
  - 15 frames are recorded at each distance → statistical stability.
  - Run repeated in day and night lighting → robustness comparison.

OUTPUT FILES (written to ./results/):
  results_day.csv       — per-frame rows for day lighting
  results_night.csv     — per-frame rows for night lighting
  summary.csv           — per-distance RMSE for each method × lighting
  comparison.png        — 2×2 matplotlib comparison chart

USAGE:
  python evaluate_all_methods.py
  python evaluate_all_methods.py --distances 5 10 20 30 50 --frames 20
  python evaluate_all_methods.py --no-night
"""

import argparse
import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import carla
import cv2
import numpy as np

from modules.stereo_depth_estimator import StereoDepthEstimator
from modules.lidar_distance_estimator import LidarDistanceEstimator


# ── CONFIGURATION CONSTANTS ──────────────────────────────────────────────────
# Camera parameters — must match the values used in main.py exactly.
IMG_W      = 1280
IMG_H      = 720
FOV_DEG    = 90.0
BASELINE_M = 0.54   # metres between left camera (y=0.0) and right (y=0.54)

# YOLO model path — same model used by YOLODistanceDetector in the live system.
YOLO_MODEL = 'yolo11n.pt'

# Known real-world height of a passenger vehicle for the pinhole formula.
# Tesla Model 3 is ~1.44 m tall; 1.5 m is a standard automotive assumption.
VEHICLE_HEIGHT_M = 1.5

# Focal length in pixels.
# At 90° horizontal FOV and 1280 px width: fx = (1280/2) / tan(45°) = 640 px.
FX = (IMG_W / 2.0) / np.tan(np.radians(FOV_DEG / 2.0))

# LiDAR → camera coordinate transform.
#
# CARLA LiDAR axes: x=forward, y=left,  z=up
# Camera axes:      x=right,   y=down,  z=forward
#
# The rotation matrix maps each LiDAR axis to its camera-frame equivalent:
#   camera_x = -lidar_y   (camera right  = opposite of LiDAR left)
#   camera_y = -lidar_z   (camera down   = opposite of LiDAR up)
#   camera_z =  lidar_x   (camera forward = LiDAR forward)
CAMERA_R = np.array([[0, -1,  0],
                     [0,  0, -1],
                     [1,  0,  0]], dtype=np.float64)

# LiDAR is mounted at ego (x=0, y=0, z=2.4).
# Left camera is at ego (x=2.0, y=0.0, z=1.4).
# Translation from LiDAR origin to camera origin in LiDAR space:
#   t_lidar_to_cam = cam_pos - lidar_pos = [2.0, 0.0, -1.0]
# Expressed in camera coordinates: camera_T = -(CAMERA_R @ t)
_t_lidar_to_cam = np.array([2.0, 0.0, -1.0], dtype=np.float64)
CAMERA_T = -(CAMERA_R @ _t_lidar_to_cam)   # = [0., -1., -2.] metres

# Test distances: covers urban stop-and-go (5–15 m) and highway following
# (20–50 m). Chosen to span the useful range of all three sensors.
DEFAULT_DISTANCES = [5, 10, 15, 20, 25, 30, 40, 50]

# Number of frames captured at each test distance.
# 15 frames gives a stable mean while keeping total runtime ~10 minutes.
DEFAULT_FRAMES_PER_DIST = 15

# Ticks between repositioning the target and starting to record.
# Gives SGBM and LiDAR time to produce their first clean output after
# the target vehicle appears at a new position.
WARMUP_TICKS = 20

# Results are written here (directory is created if it does not exist).
RESULTS_DIR = 'results'

# COCO class IDs that correspond to driveable vehicles.
# Used to filter YOLO detections to vehicles only.
VEHICLE_CLASS_IDS = {2, 5, 7}   # car, bus, truck


# ── SETUP ────────────────────────────────────────────────────────────────────

def setup_carla(town: str = 'Town04') -> Tuple:
    """
    Connects to CARLA, spawns all actors, and enables synchronous mode.

    Synchronous mode means world.tick() blocks until the simulator has
    completed the next physics step AND every sensor has produced exactly
    one new frame. This guarantees that left camera, right camera, and
    LiDAR all contain data from the identical simulation moment — essential
    for a fair comparison.

    Returns:
        (world, ego_vehicle, left_cam, right_cam, lidar_sensor,
         target_vehicle, sensor_data)

        sensor_data is a dict of mutable single-element lists so that
        sensor callbacks can update values that the main loop can read.
        Keys: 'left', 'left_ts', 'right', 'right_ts', 'lidar', 'lidar_ts'
    """
    print('-' * 60)
    print('Connecting to CARLA ...')
    client = carla.Client('localhost', 2000)
    client.set_timeout(15.0)
    world = client.load_world(town)

    # Enable synchronous mode.
    # Without this, sensors deliver data asynchronously and comparisons
    # between methods would be invalid (different simulation times).
    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05   # 20 Hz physics
    world.apply_settings(settings)
    print(f'Connected to {town} | synchronous mode ON (20 Hz)')

    bp_lib      = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # ── Ego vehicle ───────────────────────────────────────────────────────────
    vehicle_bp  = bp_lib.filter('vehicle.tesla.model3')[0]
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    ego_vehicle.set_simulate_physics(True)
    print(f'Ego vehicle spawned at {spawn_points[0].location}')

    # ── Camera blueprint (shared settings for both cameras) ───────────────────
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(IMG_W))
    cam_bp.set_attribute('image_size_y', str(IMG_H))
    cam_bp.set_attribute('fov',          str(FOV_DEG))

    # ── Left camera ───────────────────────────────────────────────────────────
    left_transform = carla.Transform(
        carla.Location(x=2.0, z=1.4),
        carla.Rotation(pitch=-15)
    )
    left_cam = world.spawn_actor(cam_bp, left_transform, attach_to=ego_vehicle)

    # ── Right camera (stereo partner, baseline = 0.54 m to the right) ─────────
    right_transform = carla.Transform(
        carla.Location(x=2.0, y=BASELINE_M, z=1.4),
        carla.Rotation(pitch=-15)
    )
    right_cam = world.spawn_actor(cam_bp, right_transform, attach_to=ego_vehicle)

    # ── LiDAR sensor ──────────────────────────────────────────────────────────
    # Parameters match main.py so the comparison is fair.
    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels',           '32')
    lidar_bp.set_attribute('range',              '50.0')
    lidar_bp.set_attribute('points_per_second',  '56000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('upper_fov',          '10.0')
    lidar_bp.set_attribute('lower_fov',          '-30.0')
    lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.4))
    lidar_sensor    = world.spawn_actor(lidar_bp, lidar_transform,
                                        attach_to=ego_vehicle)

    # ── Sensor data containers ─────────────────────────────────────────────────
    # Python closures cannot rebind an outer name with =, but they can mutate
    # a list. Each sensor callback writes into [0] of its own list; the main
    # loop reads from [0]. This is the standard CARLA callback pattern.
    sensor_data = {
        'left':    [None],
        'left_ts': [None],
        'right':   [None],
        'right_ts':[None],
        'lidar':   [None],
        'lidar_ts':[None],
    }

    def _left_cb(image,
                 d=sensor_data['left'],
                 t=sensor_data['left_ts']):
        arr  = np.frombuffer(image.raw_data, dtype=np.uint8)
        d[0] = arr.reshape((image.height, image.width, 4))[:, :, :3].copy()
        t[0] = image.timestamp

    def _right_cb(image,
                  d=sensor_data['right'],
                  t=sensor_data['right_ts']):
        arr  = np.frombuffer(image.raw_data, dtype=np.uint8)
        d[0] = arr.reshape((image.height, image.width, 4))[:, :, :3].copy()
        t[0] = image.timestamp

    def _lidar_cb(data,
                  d=sensor_data['lidar'],
                  t=sensor_data['lidar_ts']):
        pts  = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        d[0] = pts
        t[0] = data.timestamp

    left_cam.listen(_left_cb)
    right_cam.listen(_right_cb)
    lidar_sensor.listen(_lidar_cb)

    # ── Target vehicle (static red car placed ahead at each test distance) ─────
    target_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    target_bp.set_attribute('color', '255,0,0')   # red — visually distinct
    target_vehicle = world.spawn_actor(target_bp, spawn_points[1])
    target_vehicle.set_simulate_physics(False)     # static — will not move

    # Tick a few times to prime the sensor pipelines before recording.
    for _ in range(5):
        world.tick()

    print('Left cam, right cam, LiDAR, target vehicle ready')
    print('-' * 60)
    return (world, ego_vehicle, left_cam, right_cam, lidar_sensor,
            target_vehicle, sensor_data)


def setup_estimators() -> Tuple:
    """
    Creates StereoDepthEstimator, LidarDistanceEstimator, and loads YOLO.

    Parameters mirror main.py exactly so the experiment uses the same
    sensor models as the live autonomous driving system.

    Returns:
        (stereo_estimator, lidar_estimator, yolo_model)
    """
    from ultralytics import YOLO

    stereo = StereoDepthEstimator(
        image_width  = IMG_W,
        image_height = IMG_H,
        fov_degrees  = FOV_DEG,
        baseline_m   = BASELINE_M,
    )
    lidar = LidarDistanceEstimator(
        max_range_m        = 50.0,
        ground_z_threshold = -1.5,
        forward_cone_deg   = 30.0,
        cluster_eps_m      = 0.5,
        cluster_min_points = 5,
    )
    yolo = YOLO(YOLO_MODEL)
    print(f'StereoDepthEstimator, LidarDistanceEstimator, YOLO({YOLO_MODEL}) ready')
    return stereo, lidar, yolo


# ── FRAME MEASUREMENT ────────────────────────────────────────────────────────

def measure_one_frame(world,
                      sensor_data:       dict,
                      stereo_estimator:  StereoDepthEstimator,
                      lidar_estimator:   LidarDistanceEstimator,
                      yolo_model,
                      ground_truth_m:    float) -> dict:
    """
    Advances the simulation by one tick, then runs all three depth methods
    on the resulting sensor data.

    In synchronous mode, world.tick() guarantees that when it returns,
    every sensor has delivered exactly one new frame for this tick.
    All three methods therefore operate on data from the SAME moment —
    making the comparison scientifically valid.

    Args:
        ground_truth_m: The known distance we set the target at (metres).
                        Used only for error calculation; not fed to any method.

    Returns:
        A dict of measurements and errors (see column list below).
        Missing values (method failed / no detection) are stored as None.
    """
    world.tick()   # advance one simulation step; all callbacks fire here

    left_frame  = sensor_data['left'][0]
    right_frame = sensor_data['right'][0]
    lidar_pts   = sensor_data['lidar'][0]

    stereo_est  : Optional[float] = None
    lidar_est   : Optional[float] = None
    pinhole_est : Optional[float] = None
    yolo_detected = False
    best_box      = None   # (x1, y1, x2, y2) of the largest detected vehicle

    # ── YOLO detection ────────────────────────────────────────────────────────
    # Run on the left frame (same as the live system). We pick the largest
    # bounding box area, which corresponds to the closest vehicle.
    if left_frame is not None:
        results = yolo_model(left_frame, verbose=False)
        best_area = 0
        for det in results[0].boxes:
            if int(det.cls[0]) in VEHICLE_CLASS_IDS:
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box  = (x1, y1, x2, y2)

        if best_box is not None:
            yolo_detected = True
            x1, y1, x2, y2 = best_box

            # ── Method A: Stereo depth ─────────────────────────────────────────
            # Run SGBM on the left+right pair, then query depth inside the
            # detected bounding box. Only possible when right frame exists.
            if right_frame is not None:
                stereo_estimator.compute(left_frame, right_frame)
                stereo_est = stereo_estimator.get_depth_at_bbox(x1, y1, x2, y2)

            # ── Method B: Monocular pinhole ────────────────────────────────────
            # Pinhole formula: Z = (focal_length * real_height) / bbox_height_px
            # This is the baseline method — fast but assumes a fixed object size.
            bbox_h_px   = max(y2 - y1, 1)   # guard against zero-height box
            pinhole_est = (FX * VEHICLE_HEIGHT_M) / bbox_h_px

    # ── Method C: LiDAR ───────────────────────────────────────────────────────
    # If YOLO found a box, project LiDAR points onto the image and use only
    # the points inside the bounding box — this targets the specific vehicle.
    # If YOLO missed, fall back to the nearest forward-cone cluster — less
    # precise (could be any obstacle ahead) but still useful data.
    if lidar_pts is not None:
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            lidar_est = lidar_estimator.get_distance_in_camera_bbox(
                lidar_pts, x1, y1, x2, y2,
                camera_K=stereo_estimator._K,   # 3x3 intrinsic matrix
                camera_R=CAMERA_R,
                camera_T=CAMERA_T,
            )
        # Fallback (or primary if YOLO gave no box)
        if lidar_est is None:
            lidar_est = lidar_estimator.get_nearest_obstacle_distance(lidar_pts)

    # ── Error calculation ──────────────────────────────────────────────────────
    def _errors(est: Optional[float]):
        """Returns (abs_error, pct_error) or (None, None) if est is missing."""
        if est is None:
            return None, None
        abs_err = abs(est - ground_truth_m)
        pct_err = (abs_err / ground_truth_m) * 100.0
        return abs_err, pct_err

    s_err, s_pct = _errors(stereo_est)
    l_err, l_pct = _errors(lidar_est)
    p_err, p_pct = _errors(pinhole_est)

    return {
        'ground_truth_m':    ground_truth_m,
        'stereo_est_m':      stereo_est,
        'lidar_est_m':       lidar_est,
        'pinhole_est_m':     pinhole_est,
        'stereo_error_m':    s_err,
        'lidar_error_m':     l_err,
        'pinhole_error_m':   p_err,
        'stereo_pct_error':  s_pct,
        'lidar_pct_error':   l_pct,
        'pinhole_pct_error': p_pct,
        'stereo_valid':      stereo_est  is not None,
        'lidar_valid':       lidar_est   is not None,
        'yolo_detected':     yolo_detected,
    }


# ── EXPERIMENT RUNNER ────────────────────────────────────────────────────────

def run_experiment(world,
                   ego_vehicle,
                   target_vehicle,
                   sensor_data:      dict,
                   stereo_estimator: StereoDepthEstimator,
                   lidar_estimator:  LidarDistanceEstimator,
                   yolo_model,
                   test_distances:   List[float],
                   frames_per_dist:  int,
                   lighting:         str) -> List[dict]:
    """
    Runs the full experiment loop for ONE lighting condition.

    For each test distance:
      1. Apply lighting (day = ClearNoon, night = sun_altitude = -10)
      2. Move the static target vehicle to exactly that distance ahead
         of the ego vehicle, snapped to the nearest road waypoint
      3. Wait WARMUP_TICKS so sensors stabilise after the reposition
      4. Record frames_per_dist frames via measure_one_frame()

    Args:
        lighting: 'day' or 'night' — label attached to each output row.

    Returns:
        List of result dicts, each containing all measurement columns
        plus 'frame_id', 'test_distance_m', and 'lighting'.
    """
    # ── Apply lighting preset ─────────────────────────────────────────────────
    if lighting == 'night':
        weather = world.get_weather()
        weather.sun_altitude_angle = -10.0   # below horizon = darkness
        world.set_weather(weather)
        try:
            vls    = carla.VehicleLightState
            lights = vls.Position | vls.LowBeam
            ego_vehicle.set_light_state(carla.VehicleLightState(lights))
        except Exception:
            pass
        print('Night lighting applied')
    else:
        world.set_weather(carla.WeatherParameters.ClearNoon)
        try:
            ego_vehicle.set_light_state(carla.VehicleLightState(0))
        except Exception:
            pass
        print('Day lighting applied')

    spawn_map = world.get_map()
    all_rows  = []
    frame_id  = 0

    for dist_m in test_distances:
        print(f'\n  [{lighting}] Positioning target at {dist_m} m ...', end='', flush=True)

        # ── Position target vehicle ────────────────────────────────────────────
        # Compute a point directly ahead of the ego vehicle and snap it to
        # the nearest road waypoint so the target sits on the road surface.
        ego_tf  = ego_vehicle.get_transform()
        ego_fwd = ego_tf.get_forward_vector()
        raw_loc = carla.Location(
            x=ego_tf.location.x + ego_fwd.x * dist_m,
            y=ego_tf.location.y + ego_fwd.y * dist_m,
            z=ego_tf.location.z,
        )
        wp = spawn_map.get_waypoint(
            raw_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is not None:
            snapped      = wp.transform.location
            snapped.z   += 0.3   # slight lift so tyres sit on road, not below it
            target_tf    = carla.Transform(snapped, wp.transform.rotation)
        else:
            raw_loc.z   += 0.3
            target_tf    = carla.Transform(raw_loc, ego_tf.rotation)

        target_vehicle.set_transform(target_tf)

        # ── Warm-up ticks ──────────────────────────────────────────────────────
        # After repositioning, the target appears in the scene but sensors
        # need a few ticks to fill their pipelines with new data.
        # SGBM needs one frame; LiDAR needs ~100 ms (2 ticks at 20 Hz).
        # We use WARMUP_TICKS = 20 to be safe and let disparity stabilise.
        for _ in range(WARMUP_TICKS):
            world.tick()
        print(f' ready — recording {frames_per_dist} frames')

        # ── Record frames ──────────────────────────────────────────────────────
        for f in range(frames_per_dist):
            row = measure_one_frame(
                world, sensor_data,
                stereo_estimator, lidar_estimator, yolo_model,
                ground_truth_m=float(dist_m),
            )
            row['frame_id']        = frame_id
            row['test_distance_m'] = dist_m
            row['lighting']        = lighting
            all_rows.append(row)
            frame_id += 1

            # One-line progress so the user can watch in real time.
            s = f"{row['stereo_est_m']:.1f}"  if row['stereo_est_m']  is not None else ' --- '
            l = f"{row['lidar_est_m']:.1f}"   if row['lidar_est_m']   is not None else ' --- '
            p = f"{row['pinhole_est_m']:.1f}" if row['pinhole_est_m'] is not None else ' --- '
            print(f'    frame {f+1:02d}/{frames_per_dist}  '
                  f'GT={dist_m}m  stereo={s}  lidar={l}  pinhole={p}')

    return all_rows


# ── RESULTS ──────────────────────────────────────────────────────────────────

def _rmse(rows: List[dict], est_key: str) -> float:
    """
    Root Mean Square Error for one method over a set of rows.
    Rows where the estimate is None are excluded (not counted as zero error).
    Returns NaN if there are no valid rows.
    """
    errors_sq = []
    for r in rows:
        est = r[est_key]
        if est is not None:
            errors_sq.append((est - r['ground_truth_m']) ** 2)
    if not errors_sq:
        return float('nan')
    return float(np.sqrt(np.mean(errors_sq)))


def save_results(day_rows: List[dict], night_rows: List[dict]) -> None:
    """
    Writes all output files and prints the RMSE table to the terminal.

    Output:
      results_day.csv   — per-frame data, day lighting
      results_night.csv — per-frame data, night lighting
      summary.csv       — per-distance RMSE (all methods × both lightings)
      comparison.png    — 2x2 matplotlib figure
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    FIELDNAMES = [
        'frame_id', 'test_distance_m', 'ground_truth_m', 'lighting',
        'stereo_est_m', 'lidar_est_m', 'pinhole_est_m',
        'stereo_error_m', 'lidar_error_m', 'pinhole_error_m',
        'stereo_pct_error', 'lidar_pct_error', 'pinhole_pct_error',
        'stereo_valid', 'lidar_valid', 'yolo_detected',
    ]

    def _write_csv(rows, filename):
        path = os.path.join(RESULTS_DIR, filename)
        with open(path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES,
                                    extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f'  Saved: {path}')

    _write_csv(day_rows,   'results_day.csv')
    _write_csv(night_rows, 'results_night.csv')

    # ── RMSE table ────────────────────────────────────────────────────────────
    methods = [
        ('stereo_est_m',  'Stereo'),
        ('lidar_est_m',   'LiDAR'),
        ('pinhole_est_m', 'Pinhole'),
    ]
    distances = sorted(set(r['test_distance_m'] for r in day_rows))
    if night_rows:
        distances = sorted(set(distances)
                           | set(r['test_distance_m'] for r in night_rows))

    def _fmt(v):
        return f'{v:.3f}' if not (isinstance(v, float) and np.isnan(v)) else '  --- '

    print()
    print('=' * 76)
    print('  RMSE COMPARISON TABLE (metres)')
    print('=' * 76)
    header = (f"  {'Dist':>6}  "
              f"{'S-Day':>8}  {'L-Day':>8}  {'P-Day':>8}  "
              f"{'S-Night':>9}  {'L-Night':>9}  {'P-Night':>9}")
    print(header)
    print('  ' + '-' * 72)

    summary_rows = []
    for d in distances:
        row_d = [r for r in day_rows   if r['test_distance_m'] == d]
        row_n = [r for r in night_rows if r['test_distance_m'] == d]
        vals  = [_rmse(row_d, mk) for mk, _ in methods] + \
                [_rmse(row_n, mk) for mk, _ in methods]
        print(f"  {int(d):>5}m  "
              f"  {_fmt(vals[0]):>8}  {_fmt(vals[1]):>8}  {_fmt(vals[2]):>8}  "
              f"  {_fmt(vals[3]):>9}  {_fmt(vals[4]):>9}  {_fmt(vals[5]):>9}")

        summary_rows.append({
            'distance_m':         d,
            'stereo_rmse_day':    round(vals[0], 4) if not np.isnan(vals[0]) else '',
            'lidar_rmse_day':     round(vals[1], 4) if not np.isnan(vals[1]) else '',
            'pinhole_rmse_day':   round(vals[2], 4) if not np.isnan(vals[2]) else '',
            'stereo_rmse_night':  round(vals[3], 4) if not np.isnan(vals[3]) else '',
            'lidar_rmse_night':   round(vals[4], 4) if not np.isnan(vals[4]) else '',
            'pinhole_rmse_night': round(vals[5], 4) if not np.isnan(vals[5]) else '',
        })

    # Overall RMSE across all distances
    overall_d = [_rmse(day_rows,   mk) for mk, _ in methods]
    overall_n = [_rmse(night_rows, mk) for mk, _ in methods]
    print('  ' + '-' * 72)
    print(f"  {'OVERALL':>6}  "
          f"  {_fmt(overall_d[0]):>8}  {_fmt(overall_d[1]):>8}  {_fmt(overall_d[2]):>8}  "
          f"  {_fmt(overall_n[0]):>9}  {_fmt(overall_n[1]):>9}  {_fmt(overall_n[2]):>9}")
    print('=' * 76)
    print()

    # ── summary.csv ───────────────────────────────────────────────────────────
    summary_fields = ['distance_m',
                      'stereo_rmse_day', 'lidar_rmse_day', 'pinhole_rmse_day',
                      'stereo_rmse_night', 'lidar_rmse_night', 'pinhole_rmse_night']
    summary_path = os.path.join(RESULTS_DIR, 'summary.csv')
    with open(summary_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'  Saved: {summary_path}')

    # ── comparison.png ────────────────────────────────────────────────────────
    _build_chart(day_rows, night_rows, distances)


def _build_chart(day_rows:   List[dict],
                 night_rows: List[dict],
                 distances:  List[float]) -> None:
    """
    Saves a 2x2 matplotlib figure showing all four key comparisons.

    Panel layout:
      [0,0] Estimated vs true distance (day data, median per distance)
      [0,1] RMSE bar chart per distance per method (day)
      [1,0] Median percentage error by distance (day)
      [1,1] Overall RMSE — day vs night, per method
    """
    try:
        import matplotlib
        matplotlib.use('Agg')   # non-interactive backend; no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not installed — skipping comparison.png')
        return

    COLOURS = {'Stereo': '#2196F3', 'LiDAR': '#4CAF50', 'Pinhole': '#FF5722'}
    METHODS = [
        ('stereo_est_m',  'stereo_pct_error',  'Stereo'),
        ('lidar_est_m',   'lidar_pct_error',   'LiDAR'),
        ('pinhole_est_m', 'pinhole_pct_error', 'Pinhole'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Depth Estimation Method Comparison — FYP Experiment',
                 fontsize=14, fontweight='bold')

    def _median_est(rows, est_key):
        """Median estimate at each test distance (NaN if no valid rows)."""
        out = []
        for d in distances:
            vals = [r[est_key] for r in rows
                    if r['test_distance_m'] == d and r[est_key] is not None]
            out.append(float(np.median(vals)) if vals else float('nan'))
        return out

    def _rmse_per_dist(rows, est_key):
        """RMSE at each test distance."""
        out = []
        for d in distances:
            subset = [r for r in rows if r['test_distance_m'] == d]
            out.append(_rmse(subset, est_key))
        return out

    def _median_pct(rows, pct_key):
        """Median percentage error at each test distance."""
        out = []
        for d in distances:
            vals = [r[pct_key] for r in rows
                    if r['test_distance_m'] == d and r[pct_key] is not None]
            out.append(float(np.median(vals)) if vals else float('nan'))
        return out

    # ── [0,0] Estimated vs true distance (day) ────────────────────────────────
    ax = axes[0, 0]
    ax.plot(distances, distances, 'k--', label='Perfect', linewidth=1.5)
    for est_key, _, label in METHODS:
        ax.plot(distances, _median_est(day_rows, est_key),
                'o-', color=COLOURS[label], label=label,
                linewidth=2, markersize=6)
    ax.set_title('Estimated vs True Distance (Day, Median)')
    ax.set_xlabel('True distance (m)')
    ax.set_ylabel('Estimated distance (m)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── [0,1] RMSE bar chart per distance (day) ───────────────────────────────
    ax    = axes[0, 1]
    x     = np.arange(len(distances))
    width = 0.25
    for i, (est_key, _, label) in enumerate(METHODS):
        ax.bar(x + i * width, _rmse_per_dist(day_rows, est_key),
               width, label=label, color=COLOURS[label], alpha=0.85)
    ax.set_title('RMSE per Distance (Day)')
    ax.set_xlabel('True distance (m)')
    ax.set_ylabel('RMSE (m)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(int(d)) for d in distances])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ── [1,0] Percentage error by distance (day) ─────────────────────────────
    ax = axes[1, 0]
    for _, pct_key, label in METHODS:
        ax.plot(distances, _median_pct(day_rows, pct_key),
                'o-', color=COLOURS[label], label=label,
                linewidth=2, markersize=6)
    ax.axhline(5,  color='orange', linestyle='--', linewidth=1, label='5% line')
    ax.axhline(15, color='red',    linestyle='--', linewidth=1, label='15% line')
    ax.set_title('Median Percentage Error by Distance (Day)')
    ax.set_xlabel('True distance (m)')
    ax.set_ylabel('Median % error')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── [1,1] Overall RMSE — Day vs Night ─────────────────────────────────────
    ax            = axes[1, 1]
    method_labels = [label for _, _, label in METHODS]
    x             = np.arange(len(method_labels))
    width         = 0.35
    day_rmses   = [_rmse(day_rows,   ek) for ek, _, _ in METHODS]
    night_rmses = [_rmse(night_rows, ek) for ek, _, _ in METHODS]
    ax.bar(x - width / 2, day_rmses,   width, label='Day',   color='#FFC107', alpha=0.85)
    ax.bar(x + width / 2, night_rmses, width, label='Night', color='#37474F', alpha=0.85)
    ax.set_title('Overall RMSE — Day vs Night')
    ax.set_xlabel('Method')
    ax.set_ylabel('RMSE (m)')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='FYP: Compare stereo, LiDAR, and pinhole depth estimation in CARLA'
    )
    parser.add_argument(
        '--distances', type=float, nargs='+', default=DEFAULT_DISTANCES,
        help='Test distances in metres (default: 5 10 15 20 25 30 40 50)',
    )
    parser.add_argument(
        '--frames', type=int, default=DEFAULT_FRAMES_PER_DIST,
        help=f'Frames per distance (default: {DEFAULT_FRAMES_PER_DIST})',
    )
    parser.add_argument(
        '--no-night', action='store_true',
        help='Skip the night lighting experiment',
    )
    args = parser.parse_args()

    test_distances  = sorted(args.distances)
    frames_per_dist = args.frames
    run_night       = not args.no_night

    print()
    print('=' * 60)
    print('FYP Experiment — Depth Method Comparison')
    print(f'  Town       : Town04')
    print(f'  Distances  : {test_distances} m')
    print(f'  Frames/dist: {frames_per_dist}')
    print(f'  Night run  : {run_night}')
    print(f'  YOLO model : {YOLO_MODEL}')
    print(f'  Results dir: ./{RESULTS_DIR}/')
    print('=' * 60)

    # Declare all CARLA actors as None so the finally block can check safely.
    world = ego_vehicle = left_cam = right_cam = None
    lidar_sensor = target_vehicle = None

    try:
        # ── Setup ─────────────────────────────────────────────────────────────
        (world, ego_vehicle, left_cam, right_cam,
         lidar_sensor, target_vehicle, sensor_data) = setup_carla()

        stereo_estimator, lidar_estimator, yolo_model = setup_estimators()

        # ── Day experiment ─────────────────────────────────────────────────────
        print('\n>>> Running DAY experiment ...')
        day_rows = run_experiment(
            world, ego_vehicle, target_vehicle, sensor_data,
            stereo_estimator, lidar_estimator, yolo_model,
            test_distances, frames_per_dist, lighting='day',
        )

        # ── Night experiment ───────────────────────────────────────────────────
        if run_night:
            print('\n>>> Running NIGHT experiment ...')
            night_rows = run_experiment(
                world, ego_vehicle, target_vehicle, sensor_data,
                stereo_estimator, lidar_estimator, yolo_model,
                test_distances, frames_per_dist, lighting='night',
            )
        else:
            night_rows = []

        # ── Save results ───────────────────────────────────────────────────────
        print('\n>>> Saving results ...')
        save_results(day_rows, night_rows)

        print()
        print('=' * 60)
        print('Experiment complete.')
        print(f'Output files are in ./{RESULTS_DIR}/')
        print('=' * 60)

    except KeyboardInterrupt:
        print('\nInterrupted by user.')

    finally:
        # ── Restore CARLA and destroy actors ───────────────────────────────────
        # This block always runs — even after an exception — to ensure CARLA
        # is NOT left in synchronous mode, which would block future connections.
        print('\nCleaning up ...')
        if world is not None:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
                print('  Synchronous mode disabled')
            except Exception:
                pass

        for actor, name in [
            (left_cam,       'left camera'),
            (right_cam,      'right camera'),
            (lidar_sensor,   'LiDAR sensor'),
            (target_vehicle, 'target vehicle'),
            (ego_vehicle,    'ego vehicle'),
        ]:
            if actor is not None:
                try:
                    if hasattr(actor, 'stop'):
                        actor.stop()
                except Exception:
                    pass
                try:
                    actor.destroy()
                    print(f'  Destroyed {name}')
                except Exception:
                    pass

        print('Cleanup complete.')


if __name__ == '__main__':
    main()
