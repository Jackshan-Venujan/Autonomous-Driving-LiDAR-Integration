"""
Single-Object 6D Kalman Filter
================================

Wraps filterpy.kalman.KalmanFilter with a constant-velocity kinematic model
tuned for LiDAR obstacle tracking at 10 Hz.

This module is the building block for Task 04 MultiObjectTracker.
One ObjectKalmanFilter instance is created per tracked obstacle and destroyed
when the track is deleted.

State vector:  [px, py, pz, vx, vy, vz]  (position + velocity, ego frame)
Measurement:   [px, py, pz]              (centroid from LiDAR cluster AABB)

Install dependency:
    pip install filterpy
"""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  ObjectKalmanFilter
# ══════════════════════════════════════════════════════════════════════════════

class ObjectKalmanFilter:
    """
    Single-object 6D constant-velocity Kalman filter.

    Tracks one obstacle's 3D position and velocity over time.
    Wraps filterpy.kalman.KalmanFilter with ADAS-specific configuration.

    Mathematical model
    ------------------
    STATE:  x = [px, py, pz, vx, vy, vz]^T

    TRANSITION (F, 6x6):
        px(k+1) = px(k) + vx(k)*dt
        py(k+1) = py(k) + vy(k)*dt
        pz(k+1) = pz(k) + vz(k)*dt
        vx/vy/vz assumed constant (no acceleration model)
        dt is updated EVERY frame — never assumed fixed.

    OBSERVATION (H, 3x6):
        z = [px, py, pz]  — we observe only position, not velocity.
        Velocity is derived from successive position updates.

    NOISE:
        Q (process noise)     = diag([q_pos]*3, [q_vel]*3)
        R (measurement noise) = diag([r_pos_xy, r_pos_xy, r_pos_z])

    WHY CONSTANT-VELOCITY:
        At 10 Hz LiDAR, CV offers the best accuracy/robustness trade-off.
        CA (constant-acceleration) is numerically unstable with sparse
        LiDAR measurements and adds 3 extra state dimensions.

    Usage
    -----
        kf = ObjectKalmanFilter(initial_pos, config)
        kf.predict(dt=0.1)              # call every frame
        kf.update(new_centroid_xyz)     # call only when matched
        pos = kf.position               # smoothed position
        vel = kf.velocity               # estimated velocity (ego frame)
    """

    def __init__(
        self,
        initial_position: np.ndarray,   # shape (3,) float64, [px, py, pz]
        config,                          # TrackerConfig (duck-typed to avoid circular import)
    ):
        """
        Initialise Kalman filter at first-detected centroid position.

        Args:
            initial_position: 3D centroid of first detection (metres, ego frame).
            config          : TrackerConfig providing Q, R, P noise parameters.
        """
        self._cfg = config

        # filterpy KalmanFilter(dim_x=6, dim_z=3)
        # dim_x = 6 : state  [px, py, pz, vx, vy, vz]
        # dim_z = 3 : measurement [px, py, pz]
        self._kf = KalmanFilter(dim_x=6, dim_z=3)

        # ── State transition matrix F (6x6) ──────────────────────────────────
        # Identity base; velocity-to-position blocks set in predict().
        # F = [[1 0 0 dt 0  0 ],
        #      [0 1 0 0  dt 0 ],
        #      [0 0 1 0  0  dt],
        #      [0 0 0 1  0  0 ],
        #      [0 0 0 0  1  0 ],
        #      [0 0 0 0  0  1 ]]
        # F[i, i+3] = dt is filled every predict() call because dt varies.
        self._kf.F = np.eye(6, dtype=np.float64)
        # F[0,3], F[1,4], F[2,5] are set to dt in predict()

        # ── Observation matrix H (3x6) ────────────────────────────────────────
        # We observe only px, py, pz — velocity is latent.
        # H = [[1 0 0 0 0 0],
        #      [0 1 0 0 0 0],
        #      [0 0 1 0 0 0]]
        self._kf.H = np.zeros((3, 6), dtype=np.float64)
        self._kf.H[0, 0] = 1.0   # observe px
        self._kf.H[1, 1] = 1.0   # observe py
        self._kf.H[2, 2] = 1.0   # observe pz

        # ── Process noise covariance Q (6x6) ─────────────────────────────────
        # Diagonal: assumes position and velocity noise are independent.
        # q_pos models unmodelled position drift (e.g. sensor noise accumulated).
        # q_vel models unmodelled accelerations (braking, swerving).
        # q_vel > q_pos because velocity changes faster than position per step.
        self._kf.Q = np.diag([
            config.q_pos, config.q_pos, config.q_pos,
            config.q_vel, config.q_vel, config.q_vel
        ]).astype(np.float64)

        # ── Measurement noise covariance R (3x3) ─────────────────────────────
        # XY centroid less accurate than Z centroid for 64-channel LiDAR:
        #   XY: cluster centroid shifts with occlusion and scan angle.
        #   Z:  vertical channel spacing is fixed, giving stable height range.
        # r_pos_xy ≈ 0.25 (std 0.5 m) — pessimistic to prevent filter divergence
        # r_pos_z  ≈ 0.15 (std 0.39 m) — tighter because Z is more stable.
        self._kf.R = np.diag([
            config.r_pos_xy,
            config.r_pos_xy,
            config.r_pos_z
        ]).astype(np.float64)

        # ── Initial state covariance P (6x6) ─────────────────────────────────
        # Large initial uncertainty → filter corrects aggressively on first
        # few measurements → converges to ~R levels within 3-5 updates.
        # p_pos_init  = 2.0  → std 1.41 m  (unknown first position)
        # p_vel_init  = 5.0  → std 2.24 m/s (unknown first velocity, covers 0-5 m/s)
        self._kf.P = np.diag([
            config.p_pos_init, config.p_pos_init, config.p_pos_init,
            config.p_vel_init, config.p_vel_init, config.p_vel_init
        ]).astype(np.float64)

        # ── Initial state x (6x1) ────────────────────────────────────────────
        # Position: first detection centroid.
        # Velocity: zero — unknown on first detection.
        self._kf.x = np.zeros((6, 1), dtype=np.float64)
        self._kf.x[:3, 0] = initial_position.astype(np.float64)

        # Cache last dt for external inspection
        self._last_dt: float = 0.1

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, dt: float) -> np.ndarray:
        """
        Run the Kalman predict step for a given time delta.

        Updates F with current dt before propagating state. Must be called
        EVERY frame, even when no matching detection exists (coasting track).

        IMPORTANT: dt is updated here, not at filter construction time, because
        CARLA can drop frames, making dt variable and non-constant.

        Args:
            dt: Time elapsed since last update (seconds). Must be > 0.

        Returns:
            Predicted position (3,) float64 in ego vehicle frame.
        """
        # Clamp dt to valid range to guard against CARLA timestamp glitches.
        # dt < 0.01 s: unrealistically short (sensor callback jitter).
        # dt > 1.0 s:  assume tracking gap; cap to avoid velocity divergence.
        dt = float(np.clip(dt, 0.01, 1.0))
        self._last_dt = dt

        # Update F velocity-to-position blocks with current dt.
        # Rows 0-2 (position): p(k+1) = p(k) + v(k)*dt
        self._kf.F[0, 3] = dt
        self._kf.F[1, 4] = dt
        self._kf.F[2, 5] = dt

        self._kf.predict()
        return self._kf.x[:3, 0].copy()

    def update(self, measurement: np.ndarray) -> None:
        """
        Run the Kalman update step with a new centroid measurement.

        Called only when a detection is matched to this track.
        After update, position and velocity estimates are refined.

        WHY CENTROID, NOT NEAREST POINT:
            Nearest-point distance fluctuates with LiDAR scan angle and
            partial occlusion. AABB centroid is computed from the full cluster
            geometry and is more stable. Min_distance is smoothed separately
            with EMA (in TrackState.min_dist_ema).

        Args:
            measurement: Observed 3D centroid (3,) float64 — AABB midpoint.
        """
        z = measurement.reshape(3, 1).astype(np.float64)
        self._kf.update(z)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        """
        Kalman-smoothed 3D position estimate (3,) float64 in ego frame.

        Updated every predict() call (dead-reckoning) and corrected by
        every update() call (measurement fusion).
        """
        return self._kf.x[:3, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        """
        Estimated 3D velocity vector (3,) float64 in m/s, EGO VEHICLE FRAME.

        NOTE: This is velocity RELATIVE TO THE EGO VEHICLE.
        A stationary parked car will have a non-zero velocity here when
        the ego vehicle is moving. Add ego_velocity_xyz to convert to
        world frame (done in MultiObjectTracker._build_output).

        Velocity estimate is unreliable for the first velocity_min_age frames.
        """
        return self._kf.x[3:, 0].copy()

    @property
    def position_uncertainty(self) -> float:
        """
        Scalar position uncertainty: trace of the 3x3 position block of P.

        Lower = more confident. Converges from p_pos_init to ~R trace after
        5+ update() calls. Used for confidence scoring and NN export.
        """
        return float(np.trace(self._kf.P[:3, :3]))

    # ── State export for Task 07 NN training ──────────────────────────────────

    def get_state_vector(self) -> np.ndarray:
        """
        Full 6D Kalman state [px, py, pz, vx, vy, vz] as (6,) float64.

        Exported to Task 07 for sequence model training (LSTM, Transformer).
        Temporal sequences of state vectors form the NN input features.
        """
        return self._kf.x[:, 0].copy()

    def get_covariance_diagonal(self) -> np.ndarray:
        """
        Diagonal of 6x6 P matrix as (6,) float64.

        Per-dimension uncertainty. Exported as NN input for uncertainty-aware
        models (e.g. Bayesian LSTM, evidential deep learning).
        """
        return np.diag(self._kf.P).copy()
