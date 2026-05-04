"""
lidar_config.py — All tunable constants for the temporal LiDAR pipeline.
Import from here; do not embed magic numbers in logic modules.
"""

# ── Stage 1: ROI filter ────────────────────────────────────────────────────────
TEMPORAL_ROI_MIN_DIST_M: float = 0.5        # minimum detection range (m)
TEMPORAL_ROI_MAX_DIST_M: float = 50.0       # maximum detection range (m)
TEMPORAL_ROI_HALF_ANGLE_DEG: float = 30.0   # front 60° cone = ±30° from forward axis

# ── Stage 1: Ego-vehicle rectangular bounding box (sensor-local frame) ─────────
# Sensor at x=2.0 m forward, z=1.8 m above ground on Tesla Model3
EGO_BOX_X_MIN: float = -4.8    # rear of vehicle in sensor frame
EGO_BOX_X_MAX: float = 0.8     # front bumper + margin ahead of sensor
EGO_BOX_Y_MIN: float = -1.2    # left side + margin
EGO_BOX_Y_MAX: float = 1.2     # right side + margin
EGO_BOX_Z_MIN: float = -2.0    # below ground in sensor frame
EGO_BOX_Z_MAX: float = -0.1    # just below sensor level

# ── Stage 1: Open3D RANSAC ground removal ─────────────────────────────────────
RANSAC_DISTANCE_THRESHOLD: float = 0.2   # metres — inlier distance to fitted plane
RANSAC_N: int = 3                        # points sampled per RANSAC hypothesis
RANSAC_ITERATIONS: int = 100            # RANSAC iterations

# ── Stage 1: Open3D statistical outlier removal ───────────────────────────────
# nb_neighbors kept low (10) because single-frame clouds in the ±30° cone have
# only ~100-300 points; 20 neighbors on such sparse data removes real returns.
STAT_OUTLIER_NB_NEIGHBORS: int = 10
STAT_OUTLIER_STD_RATIO: float = 2.0
STAT_OUTLIER_MIN_POINTS: int = 50   # skip removal entirely if cloud is smaller

# ── Stage 1: Point budget guard ───────────────────────────────────────────────
MAX_POINTS_PER_FRAME: int = 20_000

# ── Stage 3: Sliding window buffer ────────────────────────────────────────────
TEMPORAL_BUFFER_SIZE: int = 10           # number of frames to accumulate

# ── Stage 4A: Pipeline A occupancy filter ────────────────────────────────────
OCCUPANCY_VOXEL_SIZE_M: float = 0.15    # 15 cm voxel grid for persistence voting
OCCUPANCY_MIN_FRAME_COUNT: int = 5      # voxel must appear in ≥5/10 frames to survive

# ── Stage 5: Open3D DBSCAN ────────────────────────────────────────────────────
# Pipeline A (10 accumulated frames, dense): strict params give clean static map
PIPELINE_A_DBSCAN_EPS: float = 0.4
PIPELINE_A_DBSCAN_MIN_POINTS: int = 15

# Pipeline B (single current frame, sparse): lenient params so distant objects
# with few returns (5-15 pts) still cluster. Matches old detector's eps=0.5/min=3.
PIPELINE_B_DBSCAN_EPS: float = 0.5
PIPELINE_B_DBSCAN_MIN_POINTS: int = 5

# ── Stage 6: Shape-based classification ──────────────────────────────────────
WALL_ANY_DIM_M: float = 30.0            # any bbox dimension > this → wall
WALL_LENGTH_M: float = 15.0             # long dimension threshold for long-thin wall
WALL_WIDTH_M: float = 1.0              # narrow dimension threshold for long-thin wall
# Min points for noise filter matches Pipeline B DBSCAN min so volume is the
# real discriminator (DBSCAN already guarantees at least PIPELINE_B_DBSCAN_MIN_POINTS).
NOISE_MIN_POINTS: int = 5
NOISE_MIN_VOLUME_M3: float = 0.05       # smaller bbox volume (m³) → noise (discarded)

# ── Stage 7: Static / dynamic matching ───────────────────────────────────────
STATIC_MATCH_DIST_M: float = 1.0        # Pipeline B centroid within this of any
                                         # Pipeline A centroid → static_obstacle

# ── Danger thresholds (mirrors existing LidarObstacleDetector scale) ──────────
BASE_EMERGENCY_DIST: float = 3.0
BASE_STOP_DIST: float = 5.0
BASE_SLOW_DIST: float = 10.0
BASE_CAUTIOUS_DIST: float = 15.0
SPEED_FACTOR: float = 0.5               # extra metres per km/h of ego speed
