#!/usr/bin/env python3
"""
LiDAR Viewer Script (File-based IPC)
=====================================
Standalone Open3D LiDAR visualization that reads point cloud data
from a shared numpy file written by main.py.

This approach uses file-based IPC which is more reliable than
multiprocessing when dealing with GUI libraries like Open3D and OpenCV.

Usage:
    # From main.py, press [O] to launch this script
    # Or run directly:
    python core/lidar_viewer_script.py

Controls (in Open3D window):
    [1]  Intensity colouring
    [2]  Height colouring
    [G]  Toggle ground points
    [R]  Reset camera (bird's-eye)
    [+/-]  Point size
    [Q / Esc]  Quit
"""

import numpy as np
import time
import os
import sys
import signal

# Shared data file path
SHARED_DATA_FILE = '/tmp/lidar_viewer_data.npz'
LOCK_FILE = '/tmp/lidar_viewer_data.lock'

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d not installed. Run: pip install open3d")
    sys.exit(1)


def main():
    print("="*60)
    print("LiDAR 3D Viewer (File-based IPC)")
    print("="*60)
    print("Waiting for LiDAR data from main.py...")
    print("Controls: [1]intensity [2]height [G]ground [R]reset [+/-]size [Q]quit")
    print("="*60)

    # Config
    window_name = 'LiDAR Viewer (Ego Vehicle)'
    width, height = 1280, 720
    point_size = 2.0
    ground_z = -1.5
    hud_interval = 1.0

    # State
    display_mode = 1  # 1=intensity, 2=height
    show_ground = True
    running = True

    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name, width=width, height=height)

    ro = vis.get_render_option()
    ro.background_color = np.array([0.02, 0.02, 0.05])
    ro.point_size = point_size

    # Point clouds
    pcd = o3d.geometry.PointCloud()
    ground_pcd = o3d.geometry.PointCloud()

    # Init with dummy points
    dummy = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(dummy)
    pcd.colors = o3d.utility.Vector3dVector(np.full((3, 3), 0.5))
    vis.add_geometry(pcd)
    ground_added = False

    # Key callbacks
    def reset_camera(v):
        try:
            c = v.get_view_control()
            c.set_front([0.0, 0.0, 1.0])
            c.set_lookat([20.0, 0.0, 0.0])
            c.set_up([1.0, 0.0, 0.0])
            c.set_zoom(0.15)
        except:
            pass
        return False

    def set_mode_1(v):
        nonlocal display_mode
        display_mode = 1
        print("[Viewer] Mode: INTENSITY")
        return False

    def set_mode_2(v):
        nonlocal display_mode
        display_mode = 2
        print("[Viewer] Mode: HEIGHT")
        return False

    def toggle_ground(v):
        nonlocal show_ground
        show_ground = not show_ground
        print(f"[Viewer] Ground: {'ON' if show_ground else 'OFF'}")
        return False

    def inc_size(v):
        nonlocal point_size
        point_size = min(10.0, point_size + 1)
        ro.point_size = point_size
        print(f"[Viewer] Point size: {point_size}")
        return False

    def dec_size(v):
        nonlocal point_size
        point_size = max(1.0, point_size - 1)
        ro.point_size = point_size
        print(f"[Viewer] Point size: {point_size}")
        return False

    def on_quit(v):
        nonlocal running
        running = False
        return True

    vis.register_key_callback(ord('R'), reset_camera)
    vis.register_key_callback(ord('1'), set_mode_1)
    vis.register_key_callback(ord('2'), set_mode_2)
    vis.register_key_callback(ord('G'), toggle_ground)
    vis.register_key_callback(ord('='), inc_size)
    vis.register_key_callback(ord('-'), dec_size)
    vis.register_key_callback(ord('Q'), on_quit)

    reset_camera(vis)
    vis.poll_events()
    vis.update_renderer()

    print("[Viewer] Window ready. Waiting for data...")

    # Main loop
    last_mtime = 0
    last_hud = time.time()
    frame_count = 0
    start_time = time.time()
    last_frame_num = -1

    def sigint_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        while running:
            loop_start = time.time()

            # Check for new data file
            try:
                if os.path.exists(SHARED_DATA_FILE):
                    mtime = os.path.getmtime(SHARED_DATA_FILE)
                    if mtime > last_mtime:
                        last_mtime = mtime
                        # Load data
                        data = np.load(SHARED_DATA_FILE, allow_pickle=True)
                        points = data['points']
                        frame_num = int(data['frame'])
                        timestamp = float(data['timestamp'])

                        if len(points) > 0 and frame_num != last_frame_num:
                            last_frame_num = frame_num
                            frame_count += 1

                            # Separate ground and non-ground
                            if show_ground:
                                ground_mask = points[:, 2] < ground_z
                                ground_pts = points[ground_mask]
                                non_ground = points[~ground_mask]
                            else:
                                ground_pts = np.empty((0, 4))
                                non_ground = points

                            # Update non-ground points
                            if len(non_ground) > 0:
                                xyz = non_ground[:, :3].astype(np.float64)

                                if display_mode == 1:  # Intensity
                                    intensity = non_ground[:, 3].astype(np.float64)
                                    imin, imax = intensity.min(), intensity.max()
                                    if imax > imin:
                                        norm = (intensity - imin) / (imax - imin)
                                    else:
                                        norm = np.ones_like(intensity)
                                    colors = np.zeros((len(xyz), 3))
                                    colors[:, 0] = norm
                                    colors[:, 1] = 0.3 * norm
                                    colors[:, 2] = 1.0 - norm
                                else:  # Height
                                    z = non_ground[:, 2].astype(np.float64)
                                    zmin, zmax = z.min(), z.max()
                                    if zmax > zmin:
                                        norm = (z - zmin) / (zmax - zmin)
                                    else:
                                        norm = np.ones_like(z) * 0.5
                                    colors = np.zeros((len(xyz), 3))
                                    colors[:, 0] = norm
                                    colors[:, 1] = 1.0 - abs(norm - 0.5) * 2
                                    colors[:, 2] = 1.0 - norm

                                pcd.points = o3d.utility.Vector3dVector(xyz)
                                pcd.colors = o3d.utility.Vector3dVector(colors)
                            else:
                                pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                                pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

                            vis.update_geometry(pcd)

                            # Ground points
                            if len(ground_pts) > 0 and show_ground:
                                g_xyz = ground_pts[:, :3].astype(np.float64)
                                g_colors = np.full((len(g_xyz), 3), 0.3)
                                ground_pcd.points = o3d.utility.Vector3dVector(g_xyz)
                                ground_pcd.colors = o3d.utility.Vector3dVector(g_colors)
                                if not ground_added:
                                    vis.add_geometry(ground_pcd)
                                    ground_added = True
                                else:
                                    vis.update_geometry(ground_pcd)
                            elif ground_added:
                                ground_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                                vis.update_geometry(ground_pcd)

                            # HUD
                            now = time.time()
                            if now - last_hud >= hud_interval:
                                fps = frame_count / (now - start_time) if now > start_time else 0
                                n_ground = len(ground_pts) if show_ground else 0
                                n_obj = len(non_ground)
                                print(f"[HUD] frame={frame_num} | pts={len(points):,} "
                                      f"(ground:{n_ground:,}, obj:{n_obj:,}) | "
                                      f"t={timestamp:.2f}s | fps={fps:.1f}")
                                last_hud = now

            except Exception as e:
                pass  # File might be being written

            # Poll events
            if not vis.poll_events():
                print("[Viewer] Window closed")
                break
            vis.update_renderer()

            # Rate limit
            elapsed = time.time() - loop_start
            sleep_time = 0.033 - elapsed  # ~30 fps
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"[Viewer] Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        vis.destroy_window()
        print("[Viewer] Closed")


if __name__ == '__main__':
    main()
