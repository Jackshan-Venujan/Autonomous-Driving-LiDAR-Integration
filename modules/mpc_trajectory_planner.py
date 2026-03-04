"""
MPC-Based Trajectory Planner Module
Model Predictive Control for optimal path following in autonomous driving
"""

import carla
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
import time

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available. MPC will use simplified optimization.")


# -----------------------------
# VEHICLE KINEMATIC MODEL
# -----------------------------
class KinematicBicycleModel:
    """
    Kinematic bicycle model for vehicle motion prediction.
    State: [x, y, yaw, v]
    Control: [steering_angle, acceleration]
    """
    
    def __init__(self, wheelbase: float = 2.9, max_steer: float = 0.7, max_speed: float = 50.0):
        self.L = wheelbase  # Distance between front and rear axles (meters)
        self.max_steer = max_steer  # Maximum steering angle (radians)
        self.max_speed = max_speed  # Maximum speed (m/s)
        self.min_speed = 0.0  # Minimum speed (m/s)
        self.max_accel = 3.0  # Maximum acceleration (m/s^2)
        self.max_decel = -5.0  # Maximum deceleration (m/s^2)
    
    def update(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """
        Update vehicle state using kinematic bicycle model.
        
        Args:
            state: [x, y, yaw, v] - current state
            control: [delta, a] - steering angle (rad), acceleration (m/s^2)
            dt: time step (seconds)
        
        Returns:
            new_state: [x, y, yaw, v] - predicted state
        """
        x, y, yaw, v = state
        delta, a = control
        
        # Clamp control inputs
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        a = np.clip(a, self.max_decel, self.max_accel)
        
        # Kinematic equations
        x_new = x + v * np.cos(yaw) * dt
        y_new = y + v * np.sin(yaw) * dt
        yaw_new = yaw + (v / self.L) * np.tan(delta) * dt
        v_new = v + a * dt
        
        # Clamp speed
        v_new = np.clip(v_new, self.min_speed, self.max_speed)
        
        # Normalize yaw to [-pi, pi]
        yaw_new = np.arctan2(np.sin(yaw_new), np.cos(yaw_new))
        
        return np.array([x_new, y_new, yaw_new, v_new])
    
    def predict_trajectory(self, state: np.ndarray, controls: np.ndarray, dt: float) -> np.ndarray:
        """
        Predict trajectory over horizon given sequence of controls.
        
        Args:
            state: initial state [x, y, yaw, v]
            controls: control sequence [N x 2] - [[delta1, a1], [delta2, a2], ...]
            dt: time step
        
        Returns:
            trajectory: predicted states [N+1 x 4]
        """
        N = len(controls)
        trajectory = np.zeros((N + 1, 4))
        trajectory[0] = state
        
        for i in range(N):
            trajectory[i + 1] = self.update(trajectory[i], controls[i], dt)
        
        return trajectory


# -----------------------------
# REFERENCE TRAJECTORY GENERATOR
# -----------------------------
class ReferenceTrajectoryGenerator:
    """
    Generate reference trajectory from CARLA waypoints.
    """
    
    def __init__(self, world: carla.World):
        self.world = world
        self.map = world.get_map()
    
    def get_reference_trajectory(self, vehicle: carla.Vehicle, 
                                  horizon: int, 
                                  spacing: float = 2.0,
                                  target_speed: float = 8.33) -> np.ndarray:
        """
        Generate reference trajectory from current position.
        
        Args:
            vehicle: CARLA vehicle actor
            horizon: number of waypoints to generate
            spacing: distance between waypoints (meters)
            target_speed: desired speed (m/s) - default 30 km/h
        
        Returns:
            reference: [horizon x 4] array of [x, y, yaw, v_ref]
        """
        transform = vehicle.get_transform()
        location = transform.location
        
        # Get current waypoint
        current_wp = self.map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if current_wp is None:
            return None
        
        reference = np.zeros((horizon, 4))
        wp = current_wp
        
        for i in range(horizon):
            # Get next waypoint
            next_wps = wp.next(spacing)
            if not next_wps:
                # If no more waypoints, repeat last one
                if i > 0:
                    reference[i] = reference[i - 1]
                continue
            
            wp = next_wps[0]
            
            # Extract pose
            loc = wp.transform.location
            rot = wp.transform.rotation
            yaw = math.radians(rot.yaw)
            
            # Adaptive speed based on curvature
            curvature = self._estimate_curvature(wp, spacing)
            adaptive_speed = self._speed_from_curvature(curvature, target_speed)
            
            reference[i] = [loc.x, loc.y, yaw, adaptive_speed]
        
        return reference
    
    def _estimate_curvature(self, waypoint: carla.Waypoint, lookahead: float = 5.0) -> float:
        """Estimate road curvature at waypoint."""
        try:
            next_wps = waypoint.next(lookahead)
            prev_wps = waypoint.previous(lookahead)
            
            if not next_wps or not prev_wps:
                return 0.0
            
            next_wp = next_wps[0]
            prev_wp = prev_wps[0]
            
            # Compute curvature from 3 points
            x1, y1 = prev_wp.transform.location.x, prev_wp.transform.location.y
            x2, y2 = waypoint.transform.location.x, waypoint.transform.location.y
            x3, y3 = next_wp.transform.location.x, next_wp.transform.location.y
            
            # Menger curvature formula
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            d12 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            d23 = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)
            d13 = math.sqrt((x3 - x1)**2 + (y3 - y1)**2)
            
            denom = d12 * d23 * d13
            if denom < 1e-6:
                return 0.0
            
            return 4 * area / denom
        except:
            return 0.0
    
    def _speed_from_curvature(self, curvature: float, max_speed: float) -> float:
        """Compute safe speed based on curvature."""
        if curvature < 0.01:
            return max_speed  # Straight road
        elif curvature < 0.02:
            return max_speed * 0.9
        elif curvature < 0.04:
            return max_speed * 0.7
        elif curvature < 0.08:
            return max_speed * 0.5
        else:
            return max_speed * 0.3  # Sharp turn


# -----------------------------
# MPC TRAJECTORY PLANNER
# -----------------------------
class MPCTrajectoryPlanner:
    """
    Model Predictive Control based trajectory planner.
    Optimizes steering and acceleration to follow reference trajectory.
    """
    
    def __init__(self, 
                 world: carla.World,
                 horizon: int = 10,
                 dt: float = 0.1,
                 wheelbase: float = 2.9):
        """
        Initialize MPC planner.
        
        Args:
            world: CARLA world
            horizon: prediction horizon (number of steps)
            dt: time step for prediction (seconds)
            wheelbase: vehicle wheelbase (meters)
        """
        self.world = world
        self.horizon = horizon
        self.dt = dt
        
        # Vehicle model
        self.model = KinematicBicycleModel(wheelbase=wheelbase)
        
        # Reference trajectory generator
        self.ref_generator = ReferenceTrajectoryGenerator(world)
        
        # MPC weights (tunable)
        self.w_cross_track = 10.0    # Cross-track error weight
        self.w_heading = 5.0         # Heading error weight
        self.w_speed = 1.0           # Speed tracking weight
        self.w_steer = 0.1           # Steering effort weight
        self.w_accel = 0.1           # Acceleration effort weight
        self.w_steer_rate = 10.0     # Steering rate weight (smoothness)
        self.w_accel_rate = 1.0      # Acceleration rate weight
        
        # Control limits
        self.max_steer = 0.7  # radians (~40 degrees)
        self.max_steer_rate = 0.5  # rad/s
        self.max_accel = 3.0
        self.max_decel = -5.0
        
        # Previous solution (warm start)
        self.prev_controls = None
        self.last_steer = 0.0
        self.last_accel = 0.0
        
        # Target speed (can be updated)
        self.target_speed_kmh = 30.0
        
        # External reference trajectory (from decision maker)
        self.external_reference = None
        
        # Timing
        self.last_update_time = time.time()
        
        print(f"✓ MPC Trajectory Planner initialized")
        print(f"  Horizon: {horizon} steps, dt: {dt}s")
        print(f"  scipy available: {SCIPY_AVAILABLE}")
    
    def set_external_trajectory(self, waypoints: List[Tuple[float, float, float]], 
                                 target_speed: float = 8.33):
        """
        Set external reference trajectory from decision maker.
        
        Args:
            waypoints: List of (x, y, yaw) tuples from decision maker
            target_speed: Target speed in m/s
        """
        if waypoints and len(waypoints) > 0:
            reference = np.zeros((len(waypoints), 4))
            for i, wp in enumerate(waypoints):
                reference[i] = [wp[0], wp[1], wp[2], target_speed]
            self.external_reference = reference
        else:
            self.external_reference = None
    
    def clear_external_trajectory(self):
        """Clear external trajectory, revert to lane-following."""
        self.external_reference = None
    
    def compute_control(self, vehicle: carla.Vehicle, 
                        external_waypoints: List[Tuple[float, float, float]] = None,
                        external_target_speed: float = None) -> Tuple[float, float, Dict]:
        """
        Compute optimal steering and throttle/brake using MPC.
        
        Args:
            vehicle: CARLA vehicle actor
            external_waypoints: Optional external trajectory [(x, y, yaw), ...]
            external_target_speed: Target speed for external trajectory (m/s)
        
        Returns:
            steering: optimal steering [-1, 1]
            throttle_brake: positive = throttle, negative = brake
            info: debug information
        """
        # Get current state
        state = self._get_vehicle_state(vehicle)
        if state is None:
            return 0.0, 0.0, {'error': 'Failed to get vehicle state'}
        
        # Determine reference trajectory source
        reference = None
        reference_source = 'lane_following'
        
        # Priority 1: External waypoints passed directly
        if external_waypoints and len(external_waypoints) >= 3:
            target_speed = external_target_speed if external_target_speed else (self.target_speed_kmh / 3.6)
            reference = np.zeros((len(external_waypoints), 4))
            for i, wp in enumerate(external_waypoints):
                reference[i] = [wp[0], wp[1], wp[2], target_speed]
            reference_source = 'decision_maker_direct'
        
        # Priority 2: External reference set via set_external_trajectory
        elif self.external_reference is not None and len(self.external_reference) >= 3:
            reference = self.external_reference
            reference_source = 'decision_maker_stored'
        
        # Priority 3: Generate from CARLA waypoints (lane following)
        if reference is None:
            target_speed_ms = self.target_speed_kmh / 3.6
            reference = self.ref_generator.get_reference_trajectory(
                vehicle, 
                self.horizon,
                spacing=state[3] * self.dt + 1.0,  # Adaptive spacing
                target_speed=target_speed_ms
            )
            reference_source = 'carla_waypoints'
        
        if reference is None:
            return 0.0, 0.0, {'error': 'Failed to generate reference'}
        
        # Solve MPC
        if SCIPY_AVAILABLE:
            controls, info = self._solve_mpc_scipy(state, reference)
        else:
            controls, info = self._solve_mpc_simple(state, reference)
        
        # Extract first control action
        if controls is not None and len(controls) > 0:
            delta, accel = controls[0]
            
            # Update previous controls for warm start
            self.prev_controls = controls
            self.last_steer = delta
            self.last_accel = accel
        else:
            delta = self.last_steer * 0.9
            accel = 0.0
        
        # Convert to CARLA controls
        # Steering: delta (rad) -> normalized [-1, 1]
        steering = np.clip(delta / self.max_steer, -1.0, 1.0)
        
        # Throttle/Brake
        if accel >= 0:
            throttle = np.clip(accel / self.max_accel, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-accel / abs(self.max_decel), 0.0, 1.0)
        
        info['steering'] = steering
        info['throttle'] = throttle
        info['brake'] = brake
        info['delta_rad'] = delta
        info['accel'] = accel
        info['reference'] = reference
        info['reference_source'] = reference_source
        info['current_state'] = state
        
        return steering, throttle - brake, info
    
    def _get_vehicle_state(self, vehicle: carla.Vehicle) -> Optional[np.ndarray]:
        """Get current vehicle state [x, y, yaw, v]."""
        try:
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            
            x = transform.location.x
            y = transform.location.y
            yaw = math.radians(transform.rotation.yaw)
            v = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            return np.array([x, y, yaw, v])
        except:
            return None
    
    def _solve_mpc_scipy(self, state: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Solve MPC using scipy optimization.
        """
        N = self.horizon
        
        # Initial guess (warm start from previous solution or zeros)
        if self.prev_controls is not None and len(self.prev_controls) == N:
            # Shift previous solution
            x0 = np.zeros(2 * N)
            x0[:-2] = self.prev_controls[1:].flatten()
            x0[-2:] = self.prev_controls[-1]
        else:
            x0 = np.zeros(2 * N)
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((-self.max_steer, self.max_steer))  # steering
            bounds.append((self.max_decel, self.max_accel))   # acceleration
        
        # Cost function
        def cost(u):
            controls = u.reshape(N, 2)
            trajectory = self.model.predict_trajectory(state, controls, self.dt)
            
            total_cost = 0.0
            
            for i in range(N):
                # State at time i+1
                pred_state = trajectory[i + 1]
                ref_state = reference[min(i, len(reference) - 1)]
                
                # Cross-track error (distance to reference)
                dx = pred_state[0] - ref_state[0]
                dy = pred_state[1] - ref_state[1]
                cross_track = dx**2 + dy**2
                
                # Heading error
                heading_error = pred_state[2] - ref_state[2]
                heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
                
                # Speed error
                speed_error = (pred_state[3] - ref_state[3])**2
                
                # Control effort
                steer_effort = controls[i, 0]**2
                accel_effort = controls[i, 1]**2
                
                # Control rate (smoothness)
                if i > 0:
                    steer_rate = (controls[i, 0] - controls[i-1, 0])**2
                    accel_rate = (controls[i, 1] - controls[i-1, 1])**2
                else:
                    steer_rate = (controls[i, 0] - self.last_steer)**2
                    accel_rate = (controls[i, 1] - self.last_accel)**2
                
                total_cost += (
                    self.w_cross_track * cross_track +
                    self.w_heading * heading_error**2 +
                    self.w_speed * speed_error +
                    self.w_steer * steer_effort +
                    self.w_accel * accel_effort +
                    self.w_steer_rate * steer_rate +
                    self.w_accel_rate * accel_rate
                )
            
            return total_cost
        
        # Optimize
        try:
            result = minimize(
                cost, 
                x0, 
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-4}
            )
            
            if result.success:
                controls = result.x.reshape(N, 2)
                return controls, {'success': True, 'cost': result.fun, 'iterations': result.nit}
            else:
                return None, {'success': False, 'message': result.message}
        except Exception as e:
            return None, {'success': False, 'error': str(e)}
    
    def _solve_mpc_simple(self, state: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Simplified MPC without scipy (pure pursuit + proportional control).
        Falls back to this if scipy is not available.
        """
        N = self.horizon
        controls = np.zeros((N, 2))
        
        # Pure pursuit for steering
        lookahead_idx = min(3, len(reference) - 1)
        target = reference[lookahead_idx]
        
        dx = target[0] - state[0]
        dy = target[1] - state[1]
        
        # Transform to vehicle frame
        cos_yaw = np.cos(state[2])
        sin_yaw = np.sin(state[2])
        
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        
        # Pure pursuit steering
        lookahead_dist = np.sqrt(dx**2 + dy**2)
        if lookahead_dist > 0.1:
            curvature = 2 * local_y / (lookahead_dist**2)
            delta = np.arctan(curvature * self.model.L)
        else:
            delta = 0.0
        
        # Clamp steering
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        
        # Proportional speed control
        target_speed = reference[0, 3]
        speed_error = target_speed - state[3]
        
        if speed_error > 0:
            accel = min(speed_error * 0.5, self.max_accel)
        else:
            accel = max(speed_error * 0.8, self.max_decel)
        
        # Fill control sequence
        for i in range(N):
            controls[i] = [delta, accel]
        
        return controls, {'success': True, 'method': 'pure_pursuit'}
    
    def set_target_speed(self, speed_kmh: float):
        """Set target speed in km/h."""
        self.target_speed_kmh = speed_kmh
    
    def reset(self):
        """Reset MPC state."""
        self.prev_controls = None
        self.last_steer = 0.0
        self.last_accel = 0.0


# -----------------------------
# STANDALONE TEST SCRIPT
# -----------------------------
def main():
    """Test MPC trajectory planner standalone."""
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    carla_map = world.get_map()
    
    # Spawn vehicle
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    spawn_points = carla_map.get_spawn_points()
    
    if not spawn_points:
        raise RuntimeError("No spawn points available")
    
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle")
    
    print(f"✓ Vehicle spawned: {vehicle.id}")
    
    # Create MPC planner
    mpc = MPCTrajectoryPlanner(
        world,
        horizon=10,
        dt=0.1,
        wheelbase=2.9
    )
    mpc.set_target_speed(30.0)  # 30 km/h
    
    print("▶ MPC control loop running. Ctrl+C to stop.")
    
    try:
        while True:
            loop_start = time.time()
            
            # Compute MPC control
            steering, throttle_brake, info = mpc.compute_control(vehicle)
            
            # Apply control
            control = carla.VehicleControl()
            control.steer = float(steering)
            
            if throttle_brake >= 0:
                control.throttle = float(throttle_brake)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = float(-throttle_brake)
            
            vehicle.apply_control(control)
            
            # Status
            if info.get('success'):
                state = info.get('current_state', [0, 0, 0, 0])
                print(f"\rSpeed: {state[3]*3.6:.1f} km/h | Steer: {steering:.2f} | "
                      f"Throttle: {control.throttle:.2f} | Brake: {control.brake:.2f}", end='')
            
            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = 0.05 - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n⏹ Interrupted")
    
    finally:
        print("🧹 Cleaning up...")
        try:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            time.sleep(0.2)
            vehicle.destroy()
            print("✓ Vehicle destroyed")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")


if __name__ == "__main__":
    main()
