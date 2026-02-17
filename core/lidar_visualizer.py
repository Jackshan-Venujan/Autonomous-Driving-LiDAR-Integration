"""
LiDAR Open3D Real-Time Visualizer
==================================
Live point-cloud viewer with overlay modes for CARLA LiDAR output.

Display modes (toggled by keyboard in the viewer window):
    [1]  Raw points only           — white/intensity-coloured
    [2]  Raw + cluster colours     — each DBSCAN cluster = unique colour
    [3]  Clusters + 3D AABB boxes  — axis-aligned bounding boxes per cluster
    [4]  Clusters + centroids + distance labels  (centroid spheres)

Other controls:
    [G]  Toggle ground-point visibility (grey)
    [R]  Reset camera viewpoint (BEV)
    [+]  Increase point size
    [-]  Decrease point size
    [Q / Esc]  Close viewer

Dependencies:
    open3d   (pip install open3d)
    numpy
"""

import numpy as np
import time
import threading
from typing import Optional, Dict, List
from collections import deque

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("[LidarVis] ⚠ open3d not installed — visualization disabled")
    print("              pip install open3d")


# ─── Colour palette for clusters ───────────────────────────────────────────────

def _cluster_palette(n: int) -> np.ndarray:
    """Generate n distinct RGB colours in [0,1]."""
    np.random.seed(42)
    colours = np.random.rand(max(n, 1), 3)
    # Make first few more vivid
    vivid = np.array([
        [1.0, 0.2, 0.2],   # red
        [0.2, 1.0, 0.2],   # green
        [0.3, 0.3, 1.0],   # blue
        [1.0, 1.0, 0.2],   # yellow
        [1.0, 0.5, 0.0],   # orange
        [0.8, 0.2, 1.0],   # purple
        [0.0, 1.0, 1.0],   # cyan
        [1.0, 0.0, 0.8],   # magenta
    ])
    colours[:min(n, len(vivid))] = vivid[:min(n, len(vivid))]
    return colours


class LidarVisualizer:
    """
    Non-blocking Open3D point-cloud viewer.

    Runs Open3D in the **main thread** (required on Windows/macOS).
    Data is fed from another thread via `update()`.

    Lifecycle:
        vis = LidarVisualizer()
        vis.create_window()       # must be called from main thread
        while running:
            vis.update(points, detection_result)
            if not vis.tick():    # poll events, returns False if window closed
                break
        vis.destroy()
    """

    # Display modes
    MODE_RAW       = 1
    MODE_CLUSTERS  = 2
    MODE_BOXES     = 3
    MODE_CENTROIDS = 4

    def __init__(self, window_name: str = "LiDAR Viewer", width: int = 1280, height: int = 720):
        if not OPEN3D_AVAILABLE:
            raise RuntimeError("open3d is required.  pip install open3d")

        self.window_name = window_name
        self.width = width
        self.height = height

        self._vis: Optional[o3d.visualization.VisualizerWithKeyCallback] = None
        self._pcd = o3d.geometry.PointCloud()             # main point cloud
        self._ground_pcd = o3d.geometry.PointCloud()      # ground points (toggleable)
        self._geometries_added = False
        self._ground_added = False

        # Bounding-box / centroid geometries
        self._box_lines: List[o3d.geometry.LineSet] = []
        self._centroid_spheres: List[o3d.geometry.TriangleMesh] = []

        # State
        self.display_mode = self.MODE_CLUSTERS
        self.show_ground = True
        self.point_size = 2.0

        # HUD / stats (printed to console since Open3D has no native 2D text overlay)
        self._fps_window: deque = deque(maxlen=60)
        self._last_hud_time = 0.0
        self._hud_interval = 1.0   # print HUD every N seconds

        # Palette
        self._palette = _cluster_palette(64)

    # ── Window lifecycle ───────────────────────────────────────────────────

    def create_window(self):
        """Create the Open3D window and register key callbacks. MUST call from main thread."""
        self._vis = o3d.visualization.VisualizerWithKeyCallback(self.window_name)
        self._vis.create_window(
            window_name=self.window_name,
            width=self.width,
            height=self.height,
        )

        # Key callbacks (GLFW key codes)
        self._vis.register_key_callback(ord('1'), lambda v: self._set_mode(self.MODE_RAW))
        self._vis.register_key_callback(ord('2'), lambda v: self._set_mode(self.MODE_CLUSTERS))
        self._vis.register_key_callback(ord('3'), lambda v: self._set_mode(self.MODE_BOXES))
        self._vis.register_key_callback(ord('4'), lambda v: self._set_mode(self.MODE_CENTROIDS))
        self._vis.register_key_callback(ord('G'), lambda v: self._toggle_ground())
        self._vis.register_key_callback(ord('R'), lambda v: self._reset_view())
        self._vis.register_key_callback(ord('='), lambda v: self._change_point_size(1))
        self._vis.register_key_callback(ord('-'), lambda v: self._change_point_size(-1))

        # Render options
        ro = self._vis.get_render_option()
        ro.background_color = np.array([0.05, 0.05, 0.1])  # dark background
        ro.point_size = self.point_size

        print(f"[LidarVis] ✓ Window created ({self.width}x{self.height})")
        print(f"[LidarVis]   Keys: [1]raw [2]clusters [3]boxes [4]centroids "
              f"[G]ground [R]reset [+/-]size [Q/Esc]close")

    def tick(self) -> bool:
        """
        Poll Open3D events. Returns False if window was closed.
        Must be called from the main thread at ~30-60 Hz.
        """
        if self._vis is None:
            return False
        try:
            return self._vis.poll_events() and self._vis.update_renderer()
        except Exception:
            return False

    def destroy(self):
        """Close the Open3D window."""
        if self._vis is not None:
            try:
                self._vis.destroy_window()
            except Exception:
                pass
            self._vis = None
            print("[LidarVis] ✓ Window destroyed")

    @property
    def is_open(self) -> bool:
        return self._vis is not None

    # ── Data update ────────────────────────────────────────────────────────

    def update(self, points: np.ndarray, detection: Optional[Dict] = None,
               frame: int = -1, timestamp: float = 0.0, watchdog_gap: float = 0.0):
        """
        Push new frame data into the visualizer.

        Args:
            points:       Nx4 float32 [x,y,z,intensity] raw cloud
            detection:    output dict from LidarProcessor.process()
            frame:        CARLA frame number
            timestamp:    CARLA sim time (s)
            watchdog_gap: seconds since last callback (for HUD)
        """
        if self._vis is None or points is None or len(points) == 0:
            return

        t0 = time.perf_counter()

        # ── Build main point cloud ──
        if self.display_mode == self.MODE_RAW or detection is None:
            self._update_raw(points)
        elif self.display_mode == self.MODE_CLUSTERS:
            self._update_clusters(detection)
        elif self.display_mode == self.MODE_BOXES:
            self._update_clusters(detection)
            self._update_boxes(detection)
        elif self.display_mode == self.MODE_CENTROIDS:
            self._update_clusters(detection)
            self._update_centroids(detection)

        # ── Ground points ──
        if self.show_ground and detection is not None and len(detection.get('ground_points', [])) > 0:
            gpts = detection['ground_points'][:, :3].astype(np.float64)
            self._ground_pcd.points = o3d.utility.Vector3dVector(gpts)
            grey = np.full((len(gpts), 3), 0.35)
            self._ground_pcd.colors = o3d.utility.Vector3dVector(grey)
            if not self._ground_added:
                self._vis.add_geometry(self._ground_pcd)
                self._ground_added = True
            else:
                self._vis.update_geometry(self._ground_pcd)
        elif self._ground_added:
            # Clear ground points if toggled off
            self._ground_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self._vis.update_geometry(self._ground_pcd)

        # ── FPS tracking ──
        dt = time.perf_counter() - t0
        self._fps_window.append(dt)

        # ── Console HUD ──
        now = time.time()
        if now - self._last_hud_time >= self._hud_interval:
            self._print_hud(frame, timestamp, len(points), detection, watchdog_gap)
            self._last_hud_time = now

    # ── Internal drawing helpers ───────────────────────────────────────────

    def _update_raw(self, points: np.ndarray):
        """Display raw points coloured by intensity."""
        xyz = points[:, :3].astype(np.float64)
        # CARLA standard LiDAR: intensity is in column 3
        intensity = points[:, 3].astype(np.float64)
        # Normalize intensity to [0,1]
        i_min, i_max = intensity.min(), intensity.max()
        if i_max > i_min:
            norm = (intensity - i_min) / (i_max - i_min)
        else:
            norm = np.ones_like(intensity)

        # Colour map: blue (far/low) → red (close/high)
        colors = np.zeros((len(xyz), 3))
        colors[:, 0] = norm        # R
        colors[:, 1] = 0.3 * norm  # G
        colors[:, 2] = 1.0 - norm  # B

        self._pcd.points = o3d.utility.Vector3dVector(xyz)
        self._pcd.colors = o3d.utility.Vector3dVector(colors)
        self._ensure_geometry_added()

    def _update_clusters(self, det: Dict):
        """Colour non-ground points by cluster ID."""
        non_ground = det.get('non_ground_points', np.empty((0, 4)))
        labels = det.get('cluster_labels', np.array([]))

        if len(non_ground) == 0:
            self._pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self._pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self._ensure_geometry_added()
            return

        xyz = non_ground[:, :3].astype(np.float64)
        colors = np.full((len(xyz), 3), 0.5)  # default grey

        if len(labels) == len(xyz):
            for label in np.unique(labels):
                if label == -1:
                    # noise = dark grey
                    colors[labels == label] = [0.25, 0.25, 0.25]
                else:
                    idx = int(label) % len(self._palette)
                    colors[labels == label] = self._palette[idx]

        self._pcd.points = o3d.utility.Vector3dVector(xyz)
        self._pcd.colors = o3d.utility.Vector3dVector(colors)
        self._ensure_geometry_added()

    def _update_boxes(self, det: Dict):
        """Draw AABB bounding boxes around each obstacle cluster."""
        from core.lidar_processor import LidarProcessor

        # Remove old boxes
        self._clear_boxes()

        obstacles = det.get('obstacles', [])
        for obs in obstacles:
            corners = LidarProcessor.compute_aabb_corners(obs['bbox_min'], obs['bbox_max'])
            edges = LidarProcessor.aabb_line_set_indices()

            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(corners)
            ls.lines = o3d.utility.Vector2iVector(edges)

            # Colour: nearest = red, others = green
            if obs is det.get('nearest_obstacle'):
                colour = [1.0, 0.0, 0.0]
            else:
                colour = [0.0, 1.0, 0.0]
            ls.colors = o3d.utility.Vector3dVector([colour] * len(edges))

            self._vis.add_geometry(ls)
            self._box_lines.append(ls)

    def _update_centroids(self, det: Dict):
        """Draw small spheres at obstacle centroids."""
        self._clear_centroids()

        obstacles = det.get('obstacles', [])
        for obs in obstacles:
            cx, cy, cz = obs['centroid']
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
            sphere.translate([cx, cy, cz])

            if obs is det.get('nearest_obstacle'):
                sphere.paint_uniform_color([1.0, 0.0, 0.0])
            else:
                sphere.paint_uniform_color([0.0, 0.8, 1.0])

            sphere.compute_vertex_normals()
            self._vis.add_geometry(sphere)
            self._centroid_spheres.append(sphere)

    def _ensure_geometry_added(self):
        """Add the main point cloud geometry on first call, update thereafter."""
        if not self._geometries_added:
            self._vis.add_geometry(self._pcd)
            self._geometries_added = True
            # Set initial viewpoint (bird's eye view looking forward)
            self._reset_view()
        else:
            self._vis.update_geometry(self._pcd)

    def _clear_boxes(self):
        for ls in self._box_lines:
            try:
                self._vis.remove_geometry(ls, reset_bounding_box=False)
            except Exception:
                pass
        self._box_lines.clear()

    def _clear_centroids(self):
        for s in self._centroid_spheres:
            try:
                self._vis.remove_geometry(s, reset_bounding_box=False)
            except Exception:
                pass
        self._centroid_spheres.clear()

    # ── Key callbacks ──────────────────────────────────────────────────────

    def _set_mode(self, mode: int):
        self._clear_boxes()
        self._clear_centroids()
        self.display_mode = mode
        names = {1: "RAW", 2: "CLUSTERS", 3: "BOXES", 4: "CENTROIDS"}
        print(f"[LidarVis] Display mode → {names.get(mode, mode)}")
        return False  # don't close window

    def _toggle_ground(self):
        self.show_ground = not self.show_ground
        print(f"[LidarVis] Ground points: {'ON' if self.show_ground else 'OFF'}")
        return False

    def _reset_view(self):
        """Reset camera to a bird's-eye view looking forward-down."""
        if self._vis is None:
            return False
        try:
            ctr = self._vis.get_view_control()
            ctr.set_front([0.0, 0.0, 1.0])       # looking down
            ctr.set_lookat([20.0, 0.0, 0.0])      # look at point 20m ahead
            ctr.set_up([1.0, 0.0, 0.0])            # forward is up in BEV
            ctr.set_zoom(0.15)
        except Exception:
            pass
        return False

    def _change_point_size(self, delta: int):
        self.point_size = max(1.0, self.point_size + delta)
        try:
            ro = self._vis.get_render_option()
            ro.point_size = self.point_size
        except Exception:
            pass
        print(f"[LidarVis] Point size: {self.point_size}")
        return False

    # ── Console HUD ────────────────────────────────────────────────────────

    def _print_hud(self, frame: int, timestamp: float, num_points: int,
                   detection: Optional[Dict], watchdog_gap: float):
        """Print a debug HUD line to the console."""
        mode_names = {1: "RAW", 2: "CLUSTER", 3: "BOXES", 4: "CENTROID"}
        mode_str = mode_names.get(self.display_mode, "?")

        # Vis FPS
        if self._fps_window:
            avg_dt = sum(self._fps_window) / len(self._fps_window)
            vis_fps = 1.0 / avg_dt if avg_dt > 0 else 0
        else:
            vis_fps = 0

        n_clusters = 0
        nearest_str = "—"
        proc_ms = 0.0
        if detection:
            n_clusters = detection.get('num_clusters', 0)
            proc_ms = detection.get('processing_time_ms', 0)
            nearest = detection.get('nearest_obstacle')
            if nearest:
                nearest_str = f"{nearest['distance']:.1f}m ({nearest['num_points']}pts)"

        watchdog_str = ""
        if watchdog_gap > 2.0:
            watchdog_str = f" ⚠ WD:{watchdog_gap:.1f}s"

        print(
            f"[HUD] frm={frame} | t={timestamp:.2f}s | pts={num_points:,} | "
            f"mode={mode_str} | clusters={n_clusters} | nearest={nearest_str} | "
            f"proc={proc_ms:.1f}ms | visFPS={vis_fps:.0f}{watchdog_str}"
        )
