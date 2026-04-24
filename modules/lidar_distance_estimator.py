"""
lidar_distance_estimator.py
───────────────────────────
Estimates obstacle distance using a LiDAR point cloud.

HOW LiDAR DISTANCE WORKS (plain English):
  A LiDAR fires laser pulses and measures how long each one takes
  to bounce back. This gives an exact 3D position (x, y, z) for
  each surface point it hits.

  To find the distance to an obstacle directly ahead:
    1. Take all the 3D points from one LiDAR scan
    2. Remove points that hit the ground (ground plane removal)
    3. Keep only points in a forward-facing cone in front of the car
    4. The distance to the nearest remaining point cluster is the
       distance to the nearest obstacle

  LiDAR distance accuracy: typically < 3% error at 5–50 m range.
  This is much better than stereo (~5–15%) or pinhole (~15–25%).
  However, LiDAR cannot tell you WHAT the obstacle is — it only
  gives geometry, not class labels. YOLO (camera) provides the class.

COORDINATE CONVENTION:
  CARLA LiDAR uses: x=forward, y=left, z=up
  This module works in CARLA LiDAR space (no conversion needed for
  distance measurement — we only need the forward distance which is x).

USED BY:
  - modules/driving_agent.py  (optional, future full integration)
  - evaluate_all_methods.py   (core FYP experiment — Section 4)
"""

import numpy as np
from typing import Optional, List


class LidarDistanceEstimator:
    """
    Estimates the distance to the nearest obstacle directly ahead of the
    vehicle using a LiDAR point cloud.

    Pipeline per frame:
      1. Remove ground points (height threshold)
      2. Keep points within a forward-facing cone
      3. Cluster remaining points with DBSCAN
      4. Return distance to the nearest cluster
    """

    def __init__(self,
                 max_range_m:        float = 50.0,
                 ground_z_threshold: float = -1.5,
                 forward_cone_deg:   float = 30.0,
                 cluster_eps_m:      float = 0.5,
                 cluster_min_points: int   = 5):
        """
        Prepares the LiDAR distance estimator.

        Args:
            max_range_m:        Ignore points beyond this distance (metres).
                                Matches the LiDAR sensor range in main.py.
            ground_z_threshold: Points with z below this value are ground.
                                In CARLA LiDAR space, z=up, so ground points
                                have large negative z. Default -1.5 m means
                                points more than 1.5 m below the LiDAR are
                                treated as ground and removed.
            forward_cone_deg:   Only consider points within this angle of the
                                vehicle's forward direction (x-axis in LiDAR
                                space). 30° means ±15° either side of forward.
            cluster_eps_m:      DBSCAN clustering radius in metres. Points
                                within this distance of each other form a
                                cluster (one obstacle).
            cluster_min_points: Minimum points to form a valid cluster.
                                Prevents noise spikes from being counted as objects.
        """
        self._max_range_m        = max_range_m
        self._ground_z_threshold = ground_z_threshold
        self._forward_cone_deg   = forward_cone_deg
        self._cluster_eps_m      = cluster_eps_m
        self._cluster_min_points = cluster_min_points

        # Internal state — populated by get_nearest_obstacle_distance()
        self._last_point_cloud = None   # filtered points after Step 3, for debugging
        self._last_clusters:  List     = []   # list of (centroid_x, cluster_size) tuples
        self._computed        = False

        print(f"✓ LidarDistanceEstimator initialized")
        print(f"  Max range: {max_range_m} m | Forward cone: {forward_cone_deg}°")
        print(f"  Ground threshold z < {ground_z_threshold} m")

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ────────────────────────────────────────────────────────────────────────

    def get_nearest_obstacle_distance(self,
                                      point_cloud: np.ndarray
                                      ) -> Optional[float]:
        """
        Takes a raw LiDAR point cloud and returns the distance to the
        nearest obstacle directly ahead of the vehicle.

        This is the main method used by the evaluation script to get
        lidar_est_m for each frame.

        Args:
            point_cloud: numpy array shape (N, 4) — columns are
                         [x, y, z, intensity]
                         x=forward, y=left, z=up (CARLA LiDAR convention).
                         N is typically 1000–5000 points per frame.

        Returns:
            Distance in metres to the nearest obstacle ahead, or None if
            no obstacle is detected in the forward cone.
        """
        # ── STEP 1: Input validation ─────────────────────────────────────
        # Guard against empty or malformed point clouds (e.g. sensor
        # not yet warmed up at the very start of the session).
        if point_cloud is None or len(point_cloud) < 10:
            return None

        # ── STEP 2: Ground plane removal ─────────────────────────────────
        # Points hitting the road surface are not obstacles. We remove them
        # by filtering out all points with z below our threshold.
        # In CARLA LiDAR space, z=up, so the ground is at large negative z.
        # Note: This is a simple height threshold — more robust methods
        # (RANSAC plane fitting) exist but are not needed for this FYP scope.
        pts = point_cloud[point_cloud[:, 2] > self._ground_z_threshold]

        if len(pts) < self._cluster_min_points:
            return None

        # ── STEP 3: Forward cone filter ──────────────────────────────────
        # We only care about obstacles directly ahead of the vehicle.
        # In CARLA LiDAR space, x=forward and y=left.
        # We keep points where:
        #   x > 0.5              (at least 0.5 m ahead — excludes car body)
        #   x < max_range_m      (within sensor range)
        #   |y/x| < tan(cone/2)  (within the forward half-cone angle)
        half_cone_rad = np.radians(self._forward_cone_deg / 2.0)
        in_front = pts[:, 0] > 0.5                      # must be ahead of vehicle
        in_range = pts[:, 0] < self._max_range_m        # within sensor range
        # Lateral angle filter: y/x is the tangent of the horizontal bearing.
        # We add 1e-6 to x to prevent division by zero on x≈0 points.
        in_cone  = (np.abs(pts[:, 1] / (pts[:, 0] + 1e-6))
                    < np.tan(half_cone_rad))

        pts = pts[in_front & in_range & in_cone]
        self._last_point_cloud = pts   # expose filtered cloud for debugging

        if len(pts) < self._cluster_min_points:
            return None

        # ── STEP 4: DBSCAN clustering ─────────────────────────────────────
        # DBSCAN groups nearby points into clusters. Each cluster represents
        # one physical object (car, pedestrian, wall). We then find which
        # cluster is closest to the ego vehicle.
        # We cluster on (x, y) — the 2D horizontal plane — ignoring z height.
        # This avoids tall objects (e.g. a truck) being split into two clusters
        # because of their vertical extent.
        from sklearn.cluster import DBSCAN
        coords = pts[:, :2]   # x (forward) and y (lateral) only
        labels = DBSCAN(
            eps        = self._cluster_eps_m,
            min_samples= self._cluster_min_points
        ).fit_predict(coords)

        # ── STEP 5: Find nearest cluster ──────────────────────────────────
        # For each valid cluster (label ≥ 0, not the DBSCAN noise label -1),
        # compute the median x-coordinate (forward distance).
        # We use median rather than minimum to reject noise spikes that may
        # read slightly too close.
        self._last_clusters = []
        min_distance: Optional[float] = None

        for label in set(labels):
            if label == -1:
                continue   # -1 is the DBSCAN noise label — not a real cluster
            cluster_pts  = pts[labels == label]
            cluster_dist = float(np.median(cluster_pts[:, 0]))
            self._last_clusters.append((cluster_dist, len(cluster_pts)))
            if min_distance is None or cluster_dist < min_distance:
                min_distance = cluster_dist

        self._computed = True
        return min_distance

    def get_distance_in_camera_bbox(self,
                                    point_cloud: np.ndarray,
                                    x1: int, y1: int,
                                    x2: int, y2: int,
                                    camera_K:    np.ndarray,
                                    camera_R:    np.ndarray,
                                    camera_T:    np.ndarray
                                    ) -> Optional[float]:
        """
        Projects LiDAR points onto the camera image and returns the depth
        of points that fall inside a YOLO bounding box.

        This gives LiDAR depth specifically for the detected obstacle,
        rather than just the nearest thing ahead. Used in the evaluation
        script when YOLO detects the target vehicle.

        WHY PROJECTION IS NEEDED:
          LiDAR gives 3D points but no class labels. To know which LiDAR
          points belong to the detected car (the YOLO bounding box), we
          must mathematically project the 3D points into the 2D camera
          image plane and check which ones land inside the box.

          The projection formula:
            P_camera = R × P_lidar + T   (transform to camera coordinates)
            [u, v]   = K × [X/Z, Y/Z]   (project to pixel coordinates)

        Args:
            point_cloud:  Raw LiDAR array (N, 4) in CARLA LiDAR space
                          [x=forward, y=left, z=up, intensity]
            x1,y1,x2,y2: YOLO bounding box pixel coordinates
            camera_K:     3×3 camera intrinsic matrix
                          (available from StereoDepthEstimator._K)
            camera_R:     3×3 rotation from LiDAR frame to camera frame
            camera_T:     3-element translation from LiDAR to camera frame

        Returns:
            10th-percentile depth (metres) of LiDAR points inside the bbox,
            or None if fewer than 3 points land in the box.
        """
        if point_cloud is None or len(point_cloud) < 3:
            return None

        # ── STEP 1: Transform LiDAR points to camera coordinate frame ─────
        # CARLA LiDAR: x=forward, y=left,  z=up
        # Camera frame: x=right,  y=down,  z=forward
        # camera_R and camera_T encode the known mounting-position offset
        # between the two sensors (no physical calibration needed in CARLA).
        pts_lidar  = point_cloud[:, :3].astype(np.float64)   # drop intensity
        # Matrix multiply: each row is one point; we want R @ col_vector + T
        pts_camera = (camera_R @ pts_lidar.T).T + camera_T.reshape(1, 3)

        # ── STEP 2: Discard points behind the camera (Z ≤ 0) ─────────────
        # Points with non-positive Z would project to invalid pixel locations
        # (behind the image plane). Filter them before dividing by Z.
        valid_mask = pts_camera[:, 2] > 0.1
        pts_camera = pts_camera[valid_mask]
        if len(pts_camera) == 0:
            return None

        # ── STEP 3: Project to pixel coordinates ──────────────────────────
        # Pinhole projection:
        #   u = fx * (X / Z) + cx
        #   v = fy * (Y / Z) + cy
        # camera_K is the 3×3 intrinsic matrix [[fx, 0, cx],[0, fy, cy],[0,0,1]]
        Z = pts_camera[:, 2]                                  # depth (forward)
        u = (camera_K[0, 0] * pts_camera[:, 0] / Z
             + camera_K[0, 2]).astype(int)
        v = (camera_K[1, 1] * pts_camera[:, 1] / Z
             + camera_K[1, 2]).astype(int)

        # ── STEP 4: Filter to bounding box ───────────────────────────────
        # Keep only the LiDAR points whose projected pixel falls inside
        # the YOLO bounding box. These points are on the detected object.
        in_box = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        depths_in_box = Z[in_box]

        if len(depths_in_box) < 3:
            # Too few hits inside the box — not enough for a reliable estimate.
            return None

        # 10th percentile = distance to the nearest surface of the object.
        # Using 10th (rather than minimum) rejects stray noise points.
        return float(np.percentile(depths_in_box, 10))

    def get_status(self) -> dict:
        """
        Returns a summary dict for logging and HUD display.

        Keys:
          'active'             (bool) — True if get_nearest_obstacle_distance()
                                        has been called at least once.
          'last_cluster_count' (int)  — number of obstacle clusters found in
                                        the last frame processed.
        """
        return {
            'active':             self._computed,
            'last_cluster_count': len(self._last_clusters),
        }
