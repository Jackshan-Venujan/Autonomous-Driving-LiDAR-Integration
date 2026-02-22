#!/usr/bin/env python3
"""
LiDAR Real-Time Viewer with Object Detection (High-Density)
============================================================
Standalone Open3D viewer optimized for dense 360° point clouds with:
- Multiple coloring modes (height, intensity, distance, clusters)
- 3D bounding boxes and object classification
- Top-down bird's eye view
- Distance rings for spatial reference

Usage:
    Launched automatically by main.py when pressing [O]
    Or run directly: python core/lidar_viewer_script.py

Controls:
    [1] Height coloring (blue=low, red=high)
    [2] Intensity coloring (from LiDAR return)
    [3] Distance coloring (green=near, red=far)
    [4] Cluster coloring (each object = unique color)
    [5] Full detection (clusters + boxes)
    [T] Toggle top-down / 3D perspective view
    [G] Toggle ground points
    [D] Toggle distance rings
    [R] Reset camera view
    [+/-] Point size
    [Q/Esc] Close
"""

import numpy as np
import time
import os
import sys
import signal

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("[LidarViewer] ERROR: open3d not installed")
    print("              pip install open3d")
    sys.exit(1)

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[LidarViewer] Warning: sklearn not installed, using basic clustering")

# ─── Configuration ─────────────────────────────────────────────────────────────
LIDAR_DATA_FILE = "/tmp/lidar_viewer_data.npz"
UPDATE_INTERVAL = 0.05  # ~20 FPS (reduced for dense clouds)

# Detection parameters (Y is forward in CARLA LiDAR)
ROI_X = (-60, 60)     # Lateral range (left/right) - extended for 360°
ROI_Y = (-60, 120)    # Forward range (behind to front)
ROI_Z = (-3, 15)      # Height range
GROUND_Z = -1.5       # Ground level threshold
CLUSTER_EPS = 1.0     # DBSCAN clustering radius (increased for dense clouds)
CLUSTER_MIN_PTS = 20  # Minimum points per cluster (increased)

# Visualization modes
MODE_HEIGHT = 1       # Color by Z height
MODE_INTENSITY = 2    # Color by LiDAR intensity
MODE_DISTANCE = 3     # Color by distance from ego
MODE_CLUSTERS = 4     # Color by cluster ID
MODE_FULL = 5         # Clusters + Boxes

# Distance rings configuration
DISTANCE_RINGS = [10, 20, 30, 50, 75, 100]  # meters

# Colors for different object types (RGB 0-1)
COLORS = {
    'ground': [0.3, 0.3, 0.3],       # Dark grey
    'unknown': [1.0, 1.0, 1.0],      # White
    'pedestrian': [1.0, 0.0, 0.0],   # Red
    'cyclist': [1.0, 0.5, 0.0],      # Orange
    'car': [0.0, 1.0, 0.0],          # Green
    'truck': [0.0, 0.5, 1.0],        # Light blue
    'building': [0.5, 0.5, 0.5],     # Grey
    'noise': [0.2, 0.2, 0.2],        # Dark grey
}

# Cluster color palette (for MODE_CLUSTERS)
def get_cluster_colors(n):
    """Generate n distinct colors."""
    np.random.seed(42)
    colors = np.random.rand(max(n, 20), 3)
    # Make first few vivid
    vivid = np.array([
        [1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.3, 0.3, 1.0],
        [1.0, 1.0, 0.2], [1.0, 0.5, 0.0], [0.8, 0.2, 1.0],
        [0.0, 1.0, 1.0], [1.0, 0.0, 0.8], [0.5, 1.0, 0.5],
        [1.0, 0.8, 0.6], [0.6, 0.8, 1.0], [0.8, 1.0, 0.6],
    ])
    colors[:len(vivid)] = vivid
    return colors

CLUSTER_PALETTE = get_cluster_colors(50)


class ObjectDetector:
    """Simple LiDAR-based object detector using DBSCAN clustering."""
    
    # Size ranges: (min_w, max_w, min_l, max_l, min_h, max_h)
    SIZE_TEMPLATES = {
        'pedestrian': (0.3, 1.0, 0.3, 1.0, 1.2, 2.2),
        'cyclist':    (0.4, 1.2, 1.2, 2.5, 1.0, 2.0),
        'car':        (1.4, 2.5, 3.0, 6.0, 1.2, 2.0),
        'truck':      (2.0, 3.5, 5.0, 15.0, 2.0, 4.5),
    }
    
    def __init__(self):
        self.frame_count = 0
    
    def process(self, points):
        """
        Process point cloud and detect objects.
        
        Args:
            points: Nx3 or Nx4 array
            
        Returns:
            dict with detection results
        """
        if points is None or len(points) < 10:
            return {'objects': [], 'ground': np.array([]), 'non_ground': np.array([]), 'labels': np.array([])}
        
        xyz = points[:, :3].astype(np.float64)
        
        # 1. ROI filtering
        mask = (
            (xyz[:, 0] >= ROI_X[0]) & (xyz[:, 0] <= ROI_X[1]) &
            (xyz[:, 1] >= ROI_Y[0]) & (xyz[:, 1] <= ROI_Y[1]) &
            (xyz[:, 2] >= ROI_Z[0]) & (xyz[:, 2] <= ROI_Z[1])
        )
        roi_points = xyz[mask]
        
        if len(roi_points) < 10:
            return {'objects': [], 'ground': np.array([]), 'non_ground': np.array([]), 'labels': np.array([])}
        
        # 2. Ground removal
        ground_mask = roi_points[:, 2] < GROUND_Z
        ground_points = roi_points[ground_mask]
        non_ground = roi_points[~ground_mask]
        
        if len(non_ground) < CLUSTER_MIN_PTS:
            return {'objects': [], 'ground': ground_points, 'non_ground': non_ground, 'labels': np.array([])}
        
        # 3. Clustering
        if SKLEARN_AVAILABLE:
            db = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_PTS)
            labels = db.fit_predict(non_ground)
        else:
            labels = self._simple_cluster(non_ground)
        
        # 4. Extract objects
        objects = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise
        
        for label_id in unique_labels:
            cluster_mask = labels == label_id
            cluster_pts = non_ground[cluster_mask]
            
            if len(cluster_pts) < CLUSTER_MIN_PTS:
                continue
            
            # Bounding box
            bbox_min = cluster_pts.min(axis=0)
            bbox_max = cluster_pts.max(axis=0)
            center = (bbox_min + bbox_max) / 2
            size = bbox_max - bbox_min
            
            # Distance from ego (Y is forward)
            distance = np.sqrt(center[0]**2 + center[1]**2)
            
            # Angle from ego (radians, 0 = forward/+Y)
            angle = np.arctan2(center[0], center[1])
            
            # Classification
            obj_type = self._classify(size)
            
            objects.append({
                'id': int(label_id),
                'points': cluster_pts,
                'center': center,
                'bbox_min': bbox_min,
                'bbox_max': bbox_max,
                'size': size,
                'distance': distance,
                'angle': angle,
                'angle_deg': np.degrees(angle),
                'type': obj_type,
                'color': COLORS.get(obj_type, COLORS['unknown']),
                'num_points': len(cluster_pts),
            })
        
        # Sort by distance
        objects.sort(key=lambda o: o['distance'])
        
        self.frame_count += 1
        
        return {
            'objects': objects,
            'ground': ground_points,
            'non_ground': non_ground,
            'labels': labels,
        }
    
    def _classify(self, size):
        """Classify object by size."""
        w, l, h = size
        
        for obj_type, (min_w, max_w, min_l, max_l, min_h, max_h) in self.SIZE_TEMPLATES.items():
            if (min_w <= w <= max_w and min_l <= l <= max_l and min_h <= h <= max_h):
                return obj_type
        
        # Check if it's a building/wall (very wide or long, tall)
        if (w > 5 or l > 10) and h > 2:
            return 'building'
        
        return 'unknown'
    
    def _simple_cluster(self, points):
        """Fallback grid-based clustering."""
        grid = (points[:, :2] / CLUSTER_EPS).astype(int)
        unique, inverse = np.unique(grid, axis=0, return_inverse=True)
        
        labels = np.full(len(points), -1)
        for i in range(len(unique)):
            mask = inverse == i
            if np.sum(mask) >= CLUSTER_MIN_PTS:
                labels[mask] = i
        return labels


class LidarViewer:
    """Real-time LiDAR visualization with object detection."""
    
    def __init__(self):
        self.vis = None
        self.pcd = o3d.geometry.PointCloud()
        self.ground_pcd = o3d.geometry.PointCloud()
        self.bbox_geometries = []
        
        self.detector = ObjectDetector()
        
        self.mode = MODE_FULL
        self.show_ground = True
        self.show_rings = True
        self.top_down_view = True
        self.point_size = 2.0  # Smaller for dense clouds
        self.running = True
        
        self._pcd_added = False
        self._ground_added = False
        self._last_file_time = 0
        self._rings_geometries = []
        self.frame_count = 0
        self.last_hud_time = 0
        self._last_points = None  # Cache for mode switching
    
    def create_window(self):
        """Create Open3D window."""
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("LiDAR 360° Viewer (High-Density)", width=1600, height=1000)
        
        # Register key callbacks
        self.vis.register_key_callback(ord('1'), lambda v: self._set_mode(MODE_HEIGHT))
        self.vis.register_key_callback(ord('2'), lambda v: self._set_mode(MODE_INTENSITY))
        self.vis.register_key_callback(ord('3'), lambda v: self._set_mode(MODE_DISTANCE))
        self.vis.register_key_callback(ord('4'), lambda v: self._set_mode(MODE_CLUSTERS))
        self.vis.register_key_callback(ord('5'), lambda v: self._set_mode(MODE_FULL))
        self.vis.register_key_callback(ord('G'), lambda v: self._toggle_ground())
        self.vis.register_key_callback(ord('T'), lambda v: self._toggle_view())
        self.vis.register_key_callback(ord('D'), lambda v: self._toggle_rings())
        self.vis.register_key_callback(ord('R'), lambda v: self._reset_view())
        self.vis.register_key_callback(ord('='), lambda v: self._change_point_size(0.5))
        self.vis.register_key_callback(ord('-'), lambda v: self._change_point_size(-0.5))
        self.vis.register_key_callback(ord('Q'), lambda v: self._quit())
        
        # Render options
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.1])  # Dark blue
        opt.point_size = self.point_size
        opt.show_coordinate_frame = True
        
        # Add coordinate frame at origin
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        self.vis.add_geometry(coord)
        
        # Add ego vehicle marker (small box at origin)
        ego_box = self._create_ego_marker()
        self.vis.add_geometry(ego_box)
        
        # Add distance rings
        self._create_distance_rings()
        
        # Initialize with dummy points
        dummy = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        self.pcd.points = o3d.utility.Vector3dVector(dummy)
        self.pcd.colors = o3d.utility.Vector3dVector(np.ones((3, 3)) * 0.5)
        self.vis.add_geometry(self.pcd)
        self._pcd_added = True
        
        self._reset_view()
        
        print("\n" + "=" * 70)
        print("  LiDAR 360° High-Density Viewer")
        print("=" * 70)
        print("  [1] Height  [2] Intensity  [3] Distance  [4] Clusters  [5] Full")
        print("  [T] Top-down/3D  [G] Ground  [D] Rings  [R] Reset  [+/-] Size")
        print("=" * 70 + "\n")
    
    def _create_ego_marker(self):
        """Create a marker for ego vehicle position."""
        # Create a small car-shaped box
        box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=4.5, depth=1.5)
        box.translate([-1.0, -2.25, -0.75])
        box.paint_uniform_color([0.9, 0.2, 0.2])
        box.compute_vertex_normals()
        return box
    
    def _create_distance_rings(self):
        """Create distance rings on the ground plane."""
        for r in DISTANCE_RINGS:
            # Create circle points
            theta = np.linspace(0, 2*np.pi, 100)
            circle = np.zeros((100, 3))
            circle[:, 0] = r * np.cos(theta)
            circle[:, 1] = r * np.sin(theta)
            circle[:, 2] = GROUND_Z
            
            # Create line set
            lines = [[i, (i+1) % 100] for i in range(100)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(circle)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            
            # Color based on distance (yellow to red)
            color_val = min(r / 100.0, 1.0)
            line_set.colors = o3d.utility.Vector3dVector([[1.0, 1.0 - color_val * 0.5, 0.0]] * 100)
            
            self.vis.add_geometry(line_set)
            self._rings_geometries.append(line_set)
    
    def _set_mode(self, mode):
        self.mode = mode
        names = {MODE_HEIGHT: "HEIGHT", MODE_INTENSITY: "INTENSITY", 
                 MODE_DISTANCE: "DISTANCE", MODE_CLUSTERS: "CLUSTERS", 
                 MODE_FULL: "FULL DETECTION"}
        print(f"[Mode] {names.get(mode, mode)}")
        return False
    
    def _toggle_view(self):
        self.top_down_view = not self.top_down_view
        print(f"[View] {'TOP-DOWN' if self.top_down_view else '3D PERSPECTIVE'}")
        self._reset_view()
        return False
    
    def _toggle_rings(self):
        self.show_rings = not self.show_rings
        for geom in self._rings_geometries:
            if self.show_rings:
                self.vis.add_geometry(geom, reset_bounding_box=False)
            else:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
        print(f"[Rings] {'ON' if self.show_rings else 'OFF'}")
        return False
    
    def _toggle_ground(self):
        self.show_ground = not self.show_ground
        print(f"[Ground] {'ON' if self.show_ground else 'OFF'}")
        return False
    
    def _reset_view(self):
        """Reset camera view (top-down or 3D perspective)."""
        try:
            ctr = self.vis.get_view_control()
            if self.top_down_view:
                # Bird's eye view (looking straight down)
                ctr.set_front([0, 0, 1])       # Look down from above
                ctr.set_lookat([0, 30, 0])     # Look at point ahead
                ctr.set_up([0, 1, 0])          # Y is forward
                ctr.set_zoom(0.02)             # Wide view for 360°
            else:
                # 3D perspective (behind and above vehicle)
                ctr.set_front([-0.3, -0.8, 0.5])  # Behind and above
                ctr.set_lookat([0, 20, 0])        # Look at point ahead
                ctr.set_up([0, 0, 1])             # Z is up
                ctr.set_zoom(0.05)
        except Exception as e:
            print(f"[View] Reset failed: {e}")
        return False
    
    def _change_point_size(self, delta):
        self.point_size = max(1.0, min(10.0, self.point_size + delta))
        try:
            self.vis.get_render_option().point_size = self.point_size
        except:
            pass
        print(f"[Point Size] {self.point_size}")
        return False
    
    def _quit(self):
        self.running = False
        return True
    
    def _clear_bboxes(self):
        """Remove all bounding box geometries."""
        for geom in self.bbox_geometries:
            try:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            except:
                pass
        self.bbox_geometries.clear()
    
    def update(self):
        """Update visualization with latest LiDAR data."""
        # Check for new data
        if not os.path.exists(LIDAR_DATA_FILE):
            return True
        
        try:
            mtime = os.path.getmtime(LIDAR_DATA_FILE)
            if mtime <= self._last_file_time:
                return True
            self._last_file_time = mtime
            
            data = np.load(LIDAR_DATA_FILE, allow_pickle=True)
            points = data['points']
            frame_num = int(data.get('frame', 0))
            timestamp = float(data.get('timestamp', 0))
        except:
            return True
        
        if len(points) < 10:
            return True
        
        self.frame_count += 1
        self._last_points = points  # Cache for mode switching
        
        # Debug output on first few frames
        if self.frame_count <= 3:
            print(f"[DEBUG] Loaded {len(points)} points")
            print(f"[DEBUG] X: {points[:,0].min():.1f} to {points[:,0].max():.1f}")
            print(f"[DEBUG] Y: {points[:,1].min():.1f} to {points[:,1].max():.1f}")
            print(f"[DEBUG] Z: {points[:,2].min():.1f} to {points[:,2].max():.1f}")
            if points.shape[1] >= 4:
                print(f"[DEBUG] Intensity: {points[:,3].min():.2f} to {points[:,3].max():.2f}")
        
        # Clear old bounding boxes
        self._clear_bboxes()
        
        # Update point clouds based on mode
        if self.mode in [MODE_HEIGHT, MODE_INTENSITY, MODE_DISTANCE]:
            self._update_colored_points(points, self.mode)
        elif self.mode in [MODE_CLUSTERS, MODE_FULL]:
            result = self.detector.process(points)
            self._update_detected_points(result)
            if self.mode == MODE_FULL:
                self._draw_bounding_boxes(result['objects'])
            # Print detection info
            now = time.time()
            if now - self.last_hud_time >= 1.0:
                self._print_detections(result['objects'], frame_num, timestamp, len(points))
                self.last_hud_time = now
        
        return True
    
    def _update_colored_points(self, points, color_mode):
        """Display points with various color schemes."""
        xyz = points[:, :3].astype(np.float64)
        
        # Get intensity if available
        intensity = points[:, 3] if points.shape[1] >= 4 else np.ones(len(points))
        
        # Filter ROI
        mask = (
            (xyz[:, 0] >= ROI_X[0]) & (xyz[:, 0] <= ROI_X[1]) &
            (xyz[:, 1] >= ROI_Y[0]) & (xyz[:, 1] <= ROI_Y[1]) &
            (xyz[:, 2] >= ROI_Z[0]) & (xyz[:, 2] <= ROI_Z[1])
        )
        
        # Remove ground if disabled
        if not self.show_ground:
            mask &= xyz[:, 2] > GROUND_Z
        
        xyz = xyz[mask]
        intensity = intensity[mask]
        
        if len(xyz) == 0:
            return
        
        # Calculate distance from ego (horizontal)
        distances = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        
        # Color based on mode
        colors = np.zeros((len(xyz), 3))
        
        if color_mode == MODE_HEIGHT:
            # Color by height: blue (low) → cyan → green → yellow → red (high)
            z_norm = (xyz[:, 2] - ROI_Z[0]) / (ROI_Z[1] - ROI_Z[0] + 0.001)
            z_norm = np.clip(z_norm, 0, 1)
            # More vivid color gradient
            colors[:, 0] = np.clip(2 * z_norm - 0.5, 0, 1)      # Red increases with height
            colors[:, 1] = np.clip(1 - 2 * abs(z_norm - 0.5), 0, 1)  # Green peaks in middle
            colors[:, 2] = np.clip(1 - 2 * z_norm, 0, 1)        # Blue decreases with height
            
        elif color_mode == MODE_INTENSITY:
            # Color by intensity: purple (low) → blue → cyan → green → yellow (high)
            i_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 0.001)
            i_norm = np.clip(i_norm, 0, 1)
            # HSV-like color mapping
            colors[:, 0] = np.where(i_norm > 0.5, 2 * i_norm - 1, 0.5 - i_norm)
            colors[:, 1] = np.clip(i_norm * 1.5, 0, 1)
            colors[:, 2] = np.clip(1 - i_norm, 0.2, 1)
            
        elif color_mode == MODE_DISTANCE:
            # Color by distance: green (near) → yellow → orange → red (far)
            d_norm = distances / 100.0  # Normalize to 100m
            d_norm = np.clip(d_norm, 0, 1)
            colors[:, 0] = np.clip(d_norm * 2, 0, 1)            # Red increases
            colors[:, 1] = np.clip(1 - d_norm, 0, 1)            # Green decreases
            colors[:, 2] = 0.1                                   # Minimal blue
        
        self.pcd.points = o3d.utility.Vector3dVector(xyz)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        
        if self.frame_count <= 3:
            print(f"[DEBUG] Colored points: {len(xyz)} pts, mode={color_mode}")
        
        self.vis.update_geometry(self.pcd)
        
        # Force view reset on first frame
        if self.frame_count == 1:
            self.vis.reset_view_point(True)
            self._reset_view()
    
    def _update_detected_points(self, result):
        """Display points with cluster/type coloring."""
        objects = result['objects']
        ground = result['ground']
        non_ground = result['non_ground']
        labels = result['labels']
        
        all_points = []
        all_colors = []
        
        # Ground points (grey)
        if self.show_ground and len(ground) > 0:
            all_points.append(ground)
            all_colors.append(np.tile(COLORS['ground'], (len(ground), 1)))
        
        # Object points (colored by cluster/type)
        if len(non_ground) > 0 and len(labels) == len(non_ground):
            if self.mode == MODE_CLUSTERS:
                # Color by cluster ID
                colors = np.zeros((len(non_ground), 3))
                for i, label in enumerate(labels):
                    if label == -1:
                        colors[i] = COLORS['noise']
                    else:
                        colors[i] = CLUSTER_PALETTE[label % len(CLUSTER_PALETTE)]
                all_points.append(non_ground)
                all_colors.append(colors)
            else:
                # Color by object type
                for obj in objects:
                    pts = obj['points']
                    all_points.append(pts)
                    all_colors.append(np.tile(obj['color'], (len(pts), 1)))
                
                # Add noise points (not belonging to any object)
                if len(labels) > 0:
                    noise_mask = labels == -1
                    noise_pts = non_ground[noise_mask]
                    if len(noise_pts) > 0:
                        all_points.append(noise_pts)
                        all_colors.append(np.tile(COLORS['noise'], (len(noise_pts), 1)))
        
        if all_points:
            xyz = np.vstack(all_points)
            colors = np.vstack(all_colors)
            
            if self.frame_count <= 3:
                print(f"[DEBUG] Detected points updated: {len(xyz)} pts, colors shape: {colors.shape}")
            
            self.pcd.points = o3d.utility.Vector3dVector(xyz)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.pcd)
            
            # Force view reset on first frame
            if self.frame_count == 1:
                self.vis.reset_view_point(True)
                self._reset_view()
    
    def _draw_bounding_boxes(self, objects):
        """Draw 3D bounding boxes around detected objects."""
        for obj in objects:
            # Create axis-aligned bounding box
            try:
                bbox = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=obj['bbox_min'],
                    max_bound=obj['bbox_max']
                )
                bbox.color = obj['color']
                
                self.vis.add_geometry(bbox, reset_bounding_box=False)
                self.bbox_geometries.append(bbox)
            except Exception as e:
                pass
            
            # Add vertical line from center to ground (distance indicator)
            try:
                center = obj['center']
                line_pts = np.array([
                    center,
                    [center[0], center[1], GROUND_Z]
                ], dtype=np.float64)
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector(line_pts)
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.colors = o3d.utility.Vector3dVector([obj['color']])
                
                self.vis.add_geometry(line, reset_bounding_box=False)
                self.bbox_geometries.append(line)
            except:
                pass
    
    def _print_detections(self, objects, frame_num, timestamp, total_pts):
        """Print detection summary to console."""
        # Count by type
        type_counts = {}
        for obj in objects:
            t = obj['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        type_str = " | ".join([f"{t}:{c}" for t, c in type_counts.items()]) or "none"
        
        print(f"[HUD] frame={frame_num} | pts={total_pts:,} | objects={len(objects)} | {type_str}")
        
        # Show closest objects
        if objects:
            # Closest in path (within 3.5m lateral X, and in front Y > 0)
            path_objects = [o for o in objects if abs(o['center'][0]) < 3.5 and o['center'][1] > 0]
            if path_objects:
                closest = path_objects[0]
                print(f"      ⚠️  IN PATH: {closest['type']} at {closest['distance']:.1f}m "
                      f"({closest['num_points']} pts, {closest['angle_deg']:.0f}°)")
    
    def run(self):
        """Main visualization loop."""
        while self.running:
            self.update()
            
            if not self.vis.poll_events():
                print("[Viewer] Window closed")
                break
            
            self.vis.update_renderer()
            time.sleep(UPDATE_INTERVAL)
        
        print("[Viewer] Closing...")
        try:
            self.vis.destroy_window()
        except:
            pass


def main():
    """Main entry point."""
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n[Viewer] Interrupted")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("  LiDAR Object Detection Viewer")
    print("  Waiting for data from main.py...")
    print("=" * 60)
    
    viewer = LidarViewer()
    viewer.create_window()
    viewer.run()


if __name__ == "__main__":
    main()
