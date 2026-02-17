#!/usr/bin/env python3
"""
lidar_viewer_stage1.py — Minimal LiDAR Viewer (Stage 1: Raw Points Only)
=========================================================================
Fast verification: raw point cloud in Open3D + console HUD.
No clustering, no detection, no bounding boxes.

Use this FIRST to confirm LiDAR is streaming correctly before
running the full lidar_viewer.py with detection overlays.

Usage:
    python lidar_viewer_stage1.py

Controls (in Open3D window):
    [R]  Reset camera to bird's-eye
    [+/-]  Point size
    [Q / Esc]  Quit

Console output:
    HUD line every second with frame, pts, fps, watchdog

Requirements:
    pip install numpy open3d
    CARLA server running on localhost:2000
"""

import carla
import numpy as np
import time
import sys
import os
import signal
import threading
import weakref
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d not installed.  pip install open3d")
    sys.exit(1)


# ─── Configuration ─────────────────────────────────────────────────────────────

CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_MAP = 'Town04'
AUTOPILOT = True
DURATION = 180  # seconds

# LiDAR attributes
LIDAR_CHANNELS = 32
LIDAR_RANGE = 100.0
LIDAR_PPS = 600000
LIDAR_FREQ = 20.0
LIDAR_UPPER_FOV = 10.0
LIDAR_LOWER_FOV = -30.0
LIDAR_MOUNT_Z = 2.4

# Viewer
VIEWER_FPS = 30
POINT_SIZE = 2.0
HUD_INTERVAL = 1.0
WATCHDOG_TIMEOUT = 5.0


# ─── Minimal inline LiDAR streamer (no external deps) ─────────────────────────

class MiniLidar:
    """Lightweight LiDAR streamer for Stage 1 — no LidarManager import needed."""

    def __init__(self):
        self.sensor = None
        self.vehicle = None
        self.client = None
        self.world = None

        self._lock = threading.Lock()
        self._cloud = None
        self._frame = -1
        self._timestamp = 0.0
        self._total_frames = 0
        self._total_points = 0
        self._last_recv = None
        self._start_time = time.time()
        self._shutdown = threading.Event()

    def setup(self):
        """Connect, spawn vehicle, attach LiDAR, start streaming."""
        print(f"[S1] Connecting to {CARLA_HOST}:{CARLA_PORT} ...")
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(CARLA_MAP)
        print(f"[S1] ✓ Map: {CARLA_MAP}")

        bp = self.world.get_blueprint_library()
        vbp = bp.filter('vehicle.tesla.model3')[0]
        sp = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vbp, sp[0])
        if AUTOPILOT:
            self.vehicle.set_autopilot(True)
        print(f"[S1] ✓ Vehicle spawned (id={self.vehicle.id}), autopilot={AUTOPILOT}")

        lidar_bp = bp.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(LIDAR_CHANNELS))
        lidar_bp.set_attribute('range', str(LIDAR_RANGE))
        lidar_bp.set_attribute('points_per_second', str(LIDAR_PPS))
        lidar_bp.set_attribute('rotation_frequency', str(LIDAR_FREQ))
        lidar_bp.set_attribute('upper_fov', str(LIDAR_UPPER_FOV))
        lidar_bp.set_attribute('lower_fov', str(LIDAR_LOWER_FOV))

        transform = carla.Transform(carla.Location(x=0, y=0, z=LIDAR_MOUNT_Z))
        self.sensor = self.world.spawn_actor(lidar_bp, transform, attach_to=self.vehicle)
        print(f"[S1] ✓ LiDAR attached (id={self.sensor.id})")

        weak = weakref.ref(self)
        self.sensor.listen(lambda m: MiniLidar._cb(weak, m))
        self._start_time = time.time()
        self._last_recv = time.time()
        print("[S1] ✓ Streaming started")

        # Watchdog thread
        threading.Thread(target=self._watchdog, daemon=True).start()

    @staticmethod
    def _cb(weak, m):
        s = weak()
        if s is None:
            return
        try:
            raw = np.frombuffer(m.raw_data, dtype=np.float32)
            if raw.size == 0 or raw.size % 4 != 0:
                return
            pts = raw.reshape((-1, 4))
            with s._lock:
                s._cloud = pts
                s._frame = m.frame
                s._timestamp = m.timestamp
                s._total_frames += 1
                s._total_points += len(pts)
                s._last_recv = time.time()
        except Exception as e:
            print(f"[S1] ❌ Callback: {e}")

    def get(self):
        with self._lock:
            if self._cloud is None:
                return None
            return self._cloud.copy(), self._frame, self._timestamp

    def stats(self):
        with self._lock:
            el = time.time() - self._start_time
            return self._total_frames, self._total_points, el, self._last_recv

    def _watchdog(self):
        while not self._shutdown.is_set():
            self._shutdown.wait(1.0)
            if self._shutdown.is_set():
                break
            with self._lock:
                lr = self._last_recv
            if lr and time.time() - lr > WATCHDOG_TIMEOUT:
                print(f"[S1] ⚠ WATCHDOG: no data for {time.time()-lr:.1f}s!")

    def destroy(self):
        self._shutdown.set()
        if self.sensor:
            try:
                self.sensor.stop()
                self.sensor.destroy()
            except Exception:
                pass
        if self.vehicle:
            try:
                self.vehicle.destroy()
            except Exception:
                pass
        print("[S1] ✓ Destroyed")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    lidar = MiniLidar()

    def sig(s, f):
        lidar.destroy()
        sys.exit(0)
    signal.signal(signal.SIGINT, sig)

    try:
        lidar.setup()

        # Wait for first frame
        print("[S1] Waiting for first frame...")
        t0 = time.time()
        while lidar.get() is None:
            if time.time() - t0 > 10:
                print("[S1] ❌ Timeout!")
                lidar.destroy()
                return
            time.sleep(0.1)
        pts, frm, ts = lidar.get()
        print(f"[S1] ✓ First frame: {pts.shape}")

        # Create Open3D viewer
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="LiDAR Stage 1 — Raw Points",
                          width=1280, height=720)
        ro = vis.get_render_option()
        ro.background_color = np.array([0.05, 0.05, 0.1])
        ro.point_size = POINT_SIZE
        vis.register_key_callback(ord('R'), lambda v: _reset_cam(v))

        pcd = o3d.geometry.PointCloud()
        added = False

        def _reset_cam(v):
            try:
                c = v.get_view_control()
                c.set_front([0, 0, 1])
                c.set_lookat([20, 0, 0])
                c.set_up([1, 0, 0])
                c.set_zoom(0.15)
            except Exception:
                pass
            return False

        # Main loop
        print(f"[S1] Running for {DURATION}s  |  Controls: [R]reset [+/-]size [Q]quit")
        print("=" * 60)

        start = time.time()
        last_hud = 0
        last_frame = -1
        frame_interval = 1.0 / VIEWER_FPS

        while time.time() - start < DURATION:
            now = time.time()
            data = lidar.get()
            if data is not None:
                pts, frm, ts = data
                if frm != last_frame:
                    last_frame = frm
                    xyz = pts[:, :3].astype(np.float64)

                    # Intensity colouring
                    intensity = pts[:, 3].astype(np.float64)
                    imin, imax = intensity.min(), intensity.max()
                    if imax > imin:
                        norm = (intensity - imin) / (imax - imin)
                    else:
                        norm = np.ones_like(intensity)
                    colors = np.zeros((len(xyz), 3))
                    colors[:, 0] = norm
                    colors[:, 1] = 0.3 * norm
                    colors[:, 2] = 1.0 - norm

                    pcd.points = o3d.utility.Vector3dVector(xyz)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    if not added:
                        vis.add_geometry(pcd)
                        added = True
                        _reset_cam(vis)
                    else:
                        vis.update_geometry(pcd)

            # Poll events
            if not (vis.poll_events() and vis.update_renderer()):
                print("[S1] Window closed")
                break

            # HUD
            if now - last_hud >= HUD_INTERVAL:
                tf, tp, el, lr = lidar.stats()
                fps = tf / el if el > 0 else 0
                pps = tp / el if el > 0 else 0
                gap = now - lr if lr else 0
                wd = f" ⚠WD:{gap:.1f}s" if gap > 2 else ""
                print(f"[HUD] frm={last_frame} | pts={len(pts) if data else 0:,} | "
                      f"fps={fps:.1f} | pps={pps:,.0f} | gap={gap:.2f}s{wd}")
                last_hud = now

            time.sleep(max(0, frame_interval - (time.time() - now)))

    except Exception as e:
        print(f"[S1] ❌ {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            vis.destroy_window()
        except Exception:
            pass
        lidar.destroy()


if __name__ == '__main__':
    main()
