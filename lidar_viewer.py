#!/usr/bin/env python3
"""
lidar_viewer.py — Full LiDAR Perception Viewer (Stage 2)
=========================================================
Standalone script that connects to CARLA, spawns a vehicle (autopilot),
attaches standard LiDAR, runs obstacle detection, and shows a **live
Open3D 3D viewer** with toggleable display modes.

Includes:
    ✓ Live raw point cloud
    ✓ DBSCAN clustering with colour-coded obstacles
    ✓ 3D AABB bounding boxes per cluster
    ✓ Centroid spheres with distance labels
    ✓ Ground-point toggle
    ✓ Console HUD (frame, timestamp, pts, fps, watchdog)
    ✓ Watchdog timer
    ✓ Clean Ctrl+C shutdown

Usage:
    python lidar_viewer.py

Viewer controls:
    [1]  Raw points (intensity-coloured)
    [2]  Cluster colours (default)
    [3]  Clusters + 3D bounding boxes
    [4]  Clusters + centroid spheres
    [G]  Toggle ground visibility
    [R]  Reset camera (bird's-eye)
    [+/-]  Point size
    [Q / Esc]  Quit

Requirements:
    pip install numpy scikit-learn open3d
    CARLA server running on localhost:2000
"""

import carla
import numpy as np
import time
import sys
import os
import signal

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from modules.lidar_based_obstacle_detector import LidarManager
from core.lidar_processor import LidarProcessor
from core.lidar_visualizer import LidarVisualizer


# ─── Configuration ─────────────────────────────────────────────────────────────

CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0
CARLA_MAP = 'Town04'

AUTOPILOT = True
SPAWN_TRAFFIC = True
NUM_TRAFFIC_VEHICLES = 30
DURATION = 300  # seconds

# LiDAR config (passed to LidarManager)
LIDAR_CONFIG = {
    'channels': 32,
    'range': 100.0,
    'points_per_second': 600000,
    'rotation_frequency': 20.0,
    'upper_fov': 10.0,
    'lower_fov': -30.0,
    'verbose': False,          # reduce console spam — HUD handles logging
    'save_interval': 0,       # set >0 to save .npy files
    'watchdog_timeout': 5.0,
    'stats_interval': 30.0,   # LidarManager's own stats (we have HUD)
}

# Processing config (passed to LidarProcessor)
PROC_CONFIG = {
    'roi_x_min': -5.0,
    'roi_x_max': 60.0,
    'roi_y_min': -15.0,
    'roi_y_max': 15.0,
    'roi_z_min': -2.5,
    'roi_z_max': 2.5,
    'ground_z_threshold': -1.5,
    'cluster_eps': 1.2,
    'cluster_min_points': 8,
}

# Viewer update rate cap (Hz) — avoids burning CPU on rendering
VIEWER_MAX_FPS = 30


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    client = None
    vehicle = None
    lidar_mgr = None
    viewer = None
    traffic_actors = []

    def cleanup():
        nonlocal vehicle, lidar_mgr, viewer, traffic_actors
        print("\n[Viewer] Cleaning up...")

        # 1. Close the Open3D window first
        if viewer is not None:
            viewer.destroy()
            viewer = None

        # 2. Shutdown LiDAR (stop listener + destroy sensor)
        if lidar_mgr is not None:
            lidar_mgr.shutdown()
            lidar_mgr = None

        # 3. Small delay to let CARLA drain any in-flight callbacks
        time.sleep(0.5)

        # 4. Disable autopilot on traffic before destroying
        for a in traffic_actors:
            try:
                a.set_autopilot(False)
            except Exception:
                pass
        # 5. Destroy traffic actors
        if traffic_actors:
            try:
                # Batch destroy is safer than one-by-one
                client.apply_batch([carla.command.DestroyActor(a) for a in traffic_actors])
            except Exception:
                for a in traffic_actors:
                    try:
                        a.destroy()
                    except Exception:
                        pass
        traffic_actors.clear()

        # 6. Disable autopilot + destroy ego vehicle last
        if vehicle is not None:
            try:
                vehicle.set_autopilot(False)
            except Exception:
                pass
            try:
                vehicle.destroy()
                print("[Viewer]   ✓ Vehicle destroyed")
            except Exception:
                pass
            vehicle = None

        print("[Viewer] ✓ Cleanup complete")

    def signal_handler(sig, frame):
        print("\n[Viewer] ⏹ Ctrl+C")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # ── 1. Connect ──
        print(f"[Viewer] Connecting to CARLA at {CARLA_HOST}:{CARLA_PORT} ...")
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(CARLA_TIMEOUT)
        world = client.load_world(CARLA_MAP)
        print(f"[Viewer] ✓ Connected — Map: {CARLA_MAP}")

        # ── 2. Spawn ego vehicle ──
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        print(f"[Viewer] ✓ Ego vehicle spawned (id={vehicle.id})")
        if AUTOPILOT:
            vehicle.set_autopilot(True)
            print(f"[Viewer]   Autopilot: ON")

        # ── 3. Spawn traffic ──
        if SPAWN_TRAFFIC:
            print(f"[Viewer] Spawning {NUM_TRAFFIC_VEHICLES} traffic vehicles...")
            vehicle_bps = bp_lib.filter('vehicle.*')
            import random
            random.shuffle(spawn_points)
            for sp in spawn_points[1:NUM_TRAFFIC_VEHICLES + 1]:
                try:
                    vbp = random.choice(vehicle_bps)
                    if vbp.has_attribute('color'):
                        vbp.set_attribute('color', random.choice(
                            vbp.get_attribute('color').recommended_values))
                    npc = world.try_spawn_actor(vbp, sp)
                    if npc:
                        npc.set_autopilot(True)
                        traffic_actors.append(npc)
                except Exception:
                    pass
            print(f"[Viewer]   ✓ {len(traffic_actors)} traffic vehicles spawned")

        # ── 4. Attach LiDAR ──
        lidar_mgr = LidarManager(world, vehicle, config=LIDAR_CONFIG, auto_start=True)

        # ── 5. Wait for first LiDAR frame ──
        print("[Viewer] Waiting for first LiDAR frame...")
        t_wait = time.time()
        while lidar_mgr.get_latest() is None:
            if time.time() - t_wait > 10.0:
                print("[Viewer] ❌ Timeout — no LiDAR data received!")
                cleanup()
                return
            time.sleep(0.1)
        first = lidar_mgr.get_latest()
        print(f"[Viewer] ✓ First frame: {first['points'].shape}")

        # ── 6. Init processor + viewer ──
        processor = LidarProcessor(config=PROC_CONFIG)
        viewer = LidarVisualizer(window_name="CARLA LiDAR Viewer")
        viewer.create_window()

        # ── 7. Main loop ──
        print(f"\n[Viewer] Running for {DURATION}s — close viewer or Ctrl+C to stop")
        print("=" * 72)

        start_time = time.time()
        frame_interval = 1.0 / VIEWER_MAX_FPS
        last_frame_time = 0.0
        last_lidar_frame = -1

        while time.time() - start_time < DURATION:
            # Rate-limit viewer updates
            now = time.time()
            if now - last_frame_time < frame_interval:
                time.sleep(0.001)
                if not viewer.tick():
                    print("[Viewer] Window closed by user")
                    break
                continue
            last_frame_time = now

            # Get latest LiDAR data
            data = lidar_mgr.get_latest()
            if data is None:
                if not viewer.tick():
                    break
                continue

            # Skip if same frame
            if data['frame'] == last_lidar_frame:
                if not viewer.tick():
                    break
                continue
            last_lidar_frame = data['frame']

            # Compute watchdog gap
            stats = lidar_mgr.get_stats()
            with lidar_mgr._lock:
                last_recv = lidar_mgr._last_receive_time or now
            watchdog_gap = now - last_recv

            # Run detection pipeline
            detection = processor.process(data['points'])

            # Update viewer
            viewer.update(
                points=data['points'],
                detection=detection,
                frame=data['frame'],
                timestamp=data['timestamp'],
                watchdog_gap=watchdog_gap,
            )

            # Poll Open3D events
            if not viewer.tick():
                print("[Viewer] Window closed by user")
                break

    except Exception as e:
        print(f"\n[Viewer] ❌ Fatal: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cleanup()


if __name__ == '__main__':
    main()
