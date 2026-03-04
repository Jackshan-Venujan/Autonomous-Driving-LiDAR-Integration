"""
Decision Making Module for Trajectory Planning
Behavioral planning and decision logic for autonomous driving
"""

import carla
import math
import numpy as np
from enum import Enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time


# -----------------------------
# DRIVING BEHAVIORS (State Machine)
# -----------------------------
class DrivingBehavior(Enum):
    """High-level driving behaviors"""
    LANE_FOLLOWING = "lane_following"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    EMERGENCY_STOP = "emergency_stop"
    TRAFFIC_LIGHT_STOP = "traffic_light_stop"
    INTERSECTION_CROSSING = "intersection_crossing"
    OVERTAKING = "overtaking"
    YIELDING = "yielding"
    PARKING = "parking"


class DrivingState(Enum):
    """Vehicle states"""
    DRIVING = "driving"
    STOPPED = "stopped"
    ACCELERATING = "accelerating"
    DECELERATING = "decelerating"
    TURNING = "turning"
    WAITING = "waiting"


# -----------------------------
# DATA STRUCTURES
# -----------------------------
@dataclass
class VehicleState:
    """Current vehicle state"""
    x: float
    y: float
    yaw: float  # radians
    speed: float  # m/s
    acceleration: float  # m/s^2
    steering_angle: float  # radians
    
    @classmethod
    def from_carla(cls, vehicle: carla.Vehicle) -> 'VehicleState':
        """Create from CARLA vehicle"""
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        control = vehicle.get_control()
        
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        return cls(
            x=transform.location.x,
            y=transform.location.y,
            yaw=math.radians(transform.rotation.yaw),
            speed=speed,
            acceleration=0.0,  # Would need history to compute
            steering_angle=control.steer * 0.7  # Approximate max steering
        )


@dataclass
class TrajectoryCandidate:
    """A candidate trajectory for evaluation"""
    waypoints: List[Tuple[float, float, float]]  # [(x, y, yaw), ...]
    target_speed: float
    behavior: DrivingBehavior
    cost: float = float('inf')
    feasible: bool = True
    
    def __lt__(self, other):
        return self.cost < other.cost


@dataclass
class EnvironmentState:
    """Perception of environment"""
    obstacles: List[Dict]  # From obstacle detector
    traffic_light: Optional[str]  # 'red', 'yellow', 'green', None
    lane_info: Optional[Dict]  # Lane detection info
    nearby_vehicles: List[Dict]  # Detected vehicles
    road_curvature: float  # Current road curvature
    speed_limit: float  # m/s
    is_intersection: bool
    is_junction: bool


# -----------------------------
# COST FUNCTIONS
# -----------------------------
class CostFunctions:
    """Cost calculation for trajectory evaluation"""
    
    # Weight configuration
    WEIGHTS = {
        'safety': 100.0,        # Collision avoidance
        'comfort': 5.0,         # Acceleration/jerk
        'efficiency': 2.0,      # Progress toward goal
        'lane_keeping': 10.0,   # Stay in lane
        'speed_limit': 8.0,     # Obey speed limits
        'smoothness': 3.0,      # Steering smoothness
        'traffic_rules': 50.0,  # Traffic light compliance
    }
    
    @staticmethod
    def safety_cost(trajectory: TrajectoryCandidate, obstacles: List[Dict]) -> float:
        """
        Calculate collision risk cost.
        Lane change trajectories get reduced cost if they avoid obstacles in current lane.
        """
        if not obstacles:
            return 0.0
        
        total_cost = 0.0
        
        # Check if any obstacle is blocking
        min_obstacle_dist = float('inf')
        has_dangerous_obstacle = False
        
        for obs in obstacles:
            obs_dist = obs.get('distance', float('inf'))
            if obs_dist < min_obstacle_dist:
                min_obstacle_dist = obs_dist
            if obs.get('is_dangerous', False) or obs.get('danger_level') in ('warning', 'stop', 'emergency'):
                has_dangerous_obstacle = True
        
        # Lane change trajectories REDUCE cost when there's an obstacle ahead
        # This encourages lane changes to avoid obstacles
        is_lane_change = trajectory.behavior in (
            DrivingBehavior.LANE_CHANGE_LEFT, 
            DrivingBehavior.LANE_CHANGE_RIGHT,
            DrivingBehavior.OBSTACLE_AVOIDANCE,
            DrivingBehavior.OVERTAKING
        )
        
        if is_lane_change and has_dangerous_obstacle:
            # Lane change when obstacle ahead = GOOD, lower cost
            if min_obstacle_dist < 30.0:
                # Reward lane change with negative cost (bonus)
                total_cost -= 50.0  # Bonus for avoiding obstacle
        
        # For lane following with obstacle ahead = BAD, higher cost
        if trajectory.behavior == DrivingBehavior.LANE_FOLLOWING:
            for obs in obstacles:
                obs_dist = obs.get('distance', float('inf'))
                
                # Higher cost for closer obstacles in lane following
                if obs_dist < 5.0:
                    total_cost += 200.0 * (5.0 - obs_dist)  # VERY high cost
                elif obs_dist < 10.0:
                    total_cost += 50.0 * (10.0 - obs_dist)
                elif obs_dist < 20.0:
                    total_cost += 15.0 * (20.0 - obs_dist)
                elif obs_dist < 30.0:
                    total_cost += 5.0 * (30.0 - obs_dist)
        
        return total_cost
    
    @staticmethod
    def comfort_cost(trajectory: TrajectoryCandidate, current_state: VehicleState) -> float:
        """Calculate comfort cost (acceleration, jerk)"""
        if len(trajectory.waypoints) < 2:
            return 0.0
        
        # Speed change penalty
        speed_diff = abs(trajectory.target_speed - current_state.speed)
        
        # Steering change penalty
        heading_changes = []
        for i in range(1, len(trajectory.waypoints)):
            h1 = trajectory.waypoints[i-1][2]
            h2 = trajectory.waypoints[i][2]
            heading_changes.append(abs(h2 - h1))
        
        avg_heading_change = np.mean(heading_changes) if heading_changes else 0.0
        
        return speed_diff * 2.0 + avg_heading_change * 10.0
    
    @staticmethod
    def efficiency_cost(trajectory: TrajectoryCandidate, target_speed: float) -> float:
        """Cost for not making progress"""
        speed_ratio = trajectory.target_speed / max(target_speed, 0.1)
        
        # Penalize going too slow
        if speed_ratio < 0.5:
            return (1.0 - speed_ratio) * 20.0
        elif speed_ratio < 0.8:
            return (1.0 - speed_ratio) * 5.0
        
        return 0.0
    
    @staticmethod
    def lane_keeping_cost(trajectory: TrajectoryCandidate, lane_center: Tuple[float, float]) -> float:
        """Cost for deviating from lane center"""
        if not lane_center:
            return 0.0
        
        total_deviation = 0.0
        for wp in trajectory.waypoints:
            dx = wp[0] - lane_center[0]
            dy = wp[1] - lane_center[1]
            deviation = math.sqrt(dx**2 + dy**2)
            total_deviation += deviation
        
        return total_deviation / max(len(trajectory.waypoints), 1)
    
    @staticmethod
    def traffic_rule_cost(trajectory: TrajectoryCandidate, traffic_light: Optional[str]) -> float:
        """Cost for violating traffic rules"""
        if traffic_light == 'red' and trajectory.target_speed > 0.5:
            return 1000.0  # Very high cost for running red
        elif traffic_light == 'yellow':
            # Penalty based on speed (should slow down)
            return trajectory.target_speed * 5.0
        
        return 0.0


# -----------------------------
# TRAJECTORY GENERATOR
# -----------------------------
class TrajectoryGenerator:
    """Generate candidate trajectories for different behaviors"""
    
    def __init__(self, world: carla.World):
        self.world = world
        self.map = world.get_map()
        
    def generate_lane_following(self, vehicle_state: VehicleState, 
                                 horizon: int = 10,
                                 spacing: float = 2.0,
                                 target_speed: float = 8.33) -> TrajectoryCandidate:
        """Generate lane following trajectory"""
        location = carla.Location(x=vehicle_state.x, y=vehicle_state.y, z=0)
        waypoint = self.map.get_waypoint(location, project_to_road=True)
        
        if waypoint is None:
            return TrajectoryCandidate([], target_speed, DrivingBehavior.LANE_FOLLOWING, feasible=False)
        
        waypoints = []
        wp = waypoint
        
        for _ in range(horizon):
            next_wps = wp.next(spacing)
            if not next_wps:
                break
            wp = next_wps[0]
            loc = wp.transform.location
            yaw = math.radians(wp.transform.rotation.yaw)
            waypoints.append((loc.x, loc.y, yaw))
        
        return TrajectoryCandidate(waypoints, target_speed, DrivingBehavior.LANE_FOLLOWING)
    
    def generate_lane_change_left(self, vehicle_state: VehicleState,
                                   horizon: int = 10,
                                   spacing: float = 2.0,
                                   target_speed: float = 8.33) -> Optional[TrajectoryCandidate]:
        """Generate left lane change trajectory"""
        location = carla.Location(x=vehicle_state.x, y=vehicle_state.y, z=0)
        waypoint = self.map.get_waypoint(location, project_to_road=True)
        
        if waypoint is None:
            return None
        
        # Check if left lane exists
        left_wp = waypoint.get_left_lane()
        if left_wp is None or left_wp.lane_type != carla.LaneType.Driving:
            return None
        
        waypoints = []
        
        # Smooth transition to left lane
        transition_steps = min(5, horizon // 2)
        
        # Current lane portion
        wp = waypoint
        for i in range(transition_steps):
            next_wps = wp.next(spacing)
            if not next_wps:
                break
            wp = next_wps[0]
            
            # Interpolate toward left lane
            left_wp_next = wp.get_left_lane()
            if left_wp_next:
                t = (i + 1) / transition_steps
                loc = wp.transform.location
                left_loc = left_wp_next.transform.location
                
                x = loc.x * (1 - t) + left_loc.x * t
                y = loc.y * (1 - t) + left_loc.y * t
                yaw = math.radians(left_wp_next.transform.rotation.yaw)
                waypoints.append((x, y, yaw))
        
        # Left lane portion
        if left_wp:
            wp = left_wp
            for _ in range(horizon - transition_steps):
                next_wps = wp.next(spacing)
                if not next_wps:
                    break
                wp = next_wps[0]
                loc = wp.transform.location
                yaw = math.radians(wp.transform.rotation.yaw)
                waypoints.append((loc.x, loc.y, yaw))
        
        if len(waypoints) < 3:
            return None
        
        return TrajectoryCandidate(waypoints, target_speed, DrivingBehavior.LANE_CHANGE_LEFT)
    
    def generate_lane_change_right(self, vehicle_state: VehicleState,
                                    horizon: int = 10,
                                    spacing: float = 2.0,
                                    target_speed: float = 8.33) -> Optional[TrajectoryCandidate]:
        """Generate right lane change trajectory"""
        location = carla.Location(x=vehicle_state.x, y=vehicle_state.y, z=0)
        waypoint = self.map.get_waypoint(location, project_to_road=True)
        
        if waypoint is None:
            return None
        
        # Check if right lane exists
        right_wp = waypoint.get_right_lane()
        if right_wp is None or right_wp.lane_type != carla.LaneType.Driving:
            return None
        
        waypoints = []
        
        # Smooth transition to right lane
        transition_steps = min(5, horizon // 2)
        
        # Current lane portion
        wp = waypoint
        for i in range(transition_steps):
            next_wps = wp.next(spacing)
            if not next_wps:
                break
            wp = next_wps[0]
            
            # Interpolate toward right lane
            right_wp_next = wp.get_right_lane()
            if right_wp_next:
                t = (i + 1) / transition_steps
                loc = wp.transform.location
                right_loc = right_wp_next.transform.location
                
                x = loc.x * (1 - t) + right_loc.x * t
                y = loc.y * (1 - t) + right_loc.y * t
                yaw = math.radians(right_wp_next.transform.rotation.yaw)
                waypoints.append((x, y, yaw))
        
        # Right lane portion
        if right_wp:
            wp = right_wp
            for _ in range(horizon - transition_steps):
                next_wps = wp.next(spacing)
                if not next_wps:
                    break
                wp = next_wps[0]
                loc = wp.transform.location
                yaw = math.radians(wp.transform.rotation.yaw)
                waypoints.append((loc.x, loc.y, yaw))
        
        if len(waypoints) < 3:
            return None
        
        return TrajectoryCandidate(waypoints, target_speed, DrivingBehavior.LANE_CHANGE_RIGHT)
    
    def generate_emergency_stop(self, vehicle_state: VehicleState) -> TrajectoryCandidate:
        """Generate emergency stop trajectory"""
        # Stay in place
        waypoints = [(vehicle_state.x, vehicle_state.y, vehicle_state.yaw)]
        return TrajectoryCandidate(waypoints, 0.0, DrivingBehavior.EMERGENCY_STOP)
    
    def generate_slow_down(self, vehicle_state: VehicleState, 
                           target_speed: float,
                           horizon: int = 5) -> TrajectoryCandidate:
        """Generate slow down trajectory"""
        # Follow current lane but slow
        traj = self.generate_lane_following(vehicle_state, horizon, 2.0, target_speed)
        traj.behavior = DrivingBehavior.YIELDING
        return traj


# -----------------------------
# DECISION MAKER
# -----------------------------
class DecisionMaker:
    """
    High-level decision making for trajectory planning.
    Evaluates multiple trajectory candidates and selects the best one.
    """
    
    def __init__(self, world: carla.World):
        self.world = world
        self.map = world.get_map()
        
        # Components
        self.trajectory_generator = TrajectoryGenerator(world)
        self.cost_functions = CostFunctions()
        
        # State
        self.current_behavior = DrivingBehavior.LANE_FOLLOWING
        self.current_state = DrivingState.DRIVING
        self.behavior_start_time = time.time()
        
        # Configuration
        self.config = {
            'min_lane_change_duration': 1.0,  # Minimum time before considering lane change (seconds)
            'safe_lane_change_gap': 10.0,     # meters gap needed for safe lane change
            'overtake_speed_threshold': 5.0,  # m/s slower than us triggers overtake consideration
            'emergency_distance': 5.0,        # meters
            'target_speed': 8.33,             # 30 km/h default
        }
        
        # History
        self.decision_history = []
        
        print("✓ Decision Maker initialized")
    
    def update_environment(self, 
                           obstacles: List[Dict],
                           traffic_light: Optional[str],
                           lane_info: Optional[Dict]) -> EnvironmentState:
        """Create environment state from perception data"""
        
        # Check road characteristics
        location = carla.Location(x=0, y=0, z=0)  # Will be updated
        is_junction = False
        road_curvature = 0.0
        
        return EnvironmentState(
            obstacles=obstacles or [],
            traffic_light=traffic_light,
            lane_info=lane_info,
            nearby_vehicles=[obs for obs in (obstacles or []) if obs.get('class') in ['car', 'truck', 'bus', 'motorcycle']],
            road_curvature=road_curvature,
            speed_limit=self.config['target_speed'],
            is_intersection=False,
            is_junction=is_junction
        )
    
    def evaluate_trajectory(self, 
                            trajectory: TrajectoryCandidate,
                            vehicle_state: VehicleState,
                            environment: EnvironmentState) -> float:
        """Evaluate a trajectory candidate and compute total cost"""
        
        if not trajectory.feasible or not trajectory.waypoints:
            return float('inf')
        
        costs = {}
        
        # Safety cost (highest priority)
        costs['safety'] = self.cost_functions.safety_cost(trajectory, environment.obstacles)
        
        # Comfort cost
        costs['comfort'] = self.cost_functions.comfort_cost(trajectory, vehicle_state)
        
        # Efficiency cost
        costs['efficiency'] = self.cost_functions.efficiency_cost(trajectory, self.config['target_speed'])
        
        # Traffic rules cost
        costs['traffic_rules'] = self.cost_functions.traffic_rule_cost(trajectory, environment.traffic_light)
        
        # Total weighted cost
        total_cost = sum(
            self.cost_functions.WEIGHTS.get(key, 1.0) * value 
            for key, value in costs.items()
        )
        
        trajectory.cost = total_cost
        return total_cost
    
    def should_consider_lane_change(self, 
                                    vehicle_state: VehicleState,
                                    environment: EnvironmentState) -> Tuple[bool, str]:
        """Determine if lane change should be considered"""
        
        # Check if we've been in current behavior long enough
        time_in_behavior = time.time() - self.behavior_start_time
        if time_in_behavior < self.config['min_lane_change_duration']:
            return False, "too_soon"
        
        # Check for obstacles/vehicles ahead in our lane
        vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person'}
        
        for obs in environment.obstacles:
            obs_class = obs.get('class', '').lower()
            distance = obs.get('distance')
            
            if distance is None:
                continue
            
            # Check for vehicles blocking path (15-35m range for lane change consideration)
            if obs_class in vehicle_classes and 10.0 < distance < 35.0:
                # Vehicle in danger zone - consider lane change
                if obs.get('is_dangerous', False) or obs.get('danger_level') in ('warning', 'stop'):
                    return True, f"obstacle_ahead_{obs_class}"
            
            # Also check nearby_vehicles list
            for vehicle in environment.nearby_vehicles:
                v_distance = vehicle.get('distance', float('inf'))
                if 10.0 < v_distance < 35.0:
                    if vehicle.get('is_dangerous', False) or vehicle.get('danger_level') in ('warning', 'stop', 'slowdown'):
                        return True, "vehicle_ahead"
        
        # Check if we need to return to original lane after overtaking
        if self.current_behavior == DrivingBehavior.OVERTAKING:
            # Check if we've passed the slow vehicle
            vehicles_behind = [v for v in environment.nearby_vehicles 
                              if v.get('distance', 0) < 0]  # Negative distance = behind
            if len(vehicles_behind) > 0:
                return True, "return_to_lane"
        
        return False, "none"
    
    def check_emergency_conditions(self, 
                                   vehicle_state: VehicleState,
                                   environment: EnvironmentState) -> bool:
        """Check if emergency stop is needed"""
        
        # Immediate collision threat
        for obs in environment.obstacles:
            if obs.get('danger_level') == 'emergency':
                return True
            if obs.get('distance', float('inf')) < self.config['emergency_distance']:
                return True
        
        # Red traffic light when very close
        if environment.traffic_light == 'red' and vehicle_state.speed > 0.5:
            # Would need distance to traffic light
            pass
        
        return False
    
    def decide(self, 
               vehicle: carla.Vehicle,
               obstacles: List[Dict] = None,
               traffic_light: Optional[str] = None,
               lane_info: Optional[Dict] = None) -> Tuple[TrajectoryCandidate, Dict]:
        """
        Main decision function.
        
        Args:
            vehicle: CARLA vehicle actor
            obstacles: Detected obstacles from perception
            traffic_light: Traffic light state
            lane_info: Lane detection info
        
        Returns:
            best_trajectory: Selected trajectory
            info: Decision information
        """
        
        # Get current state
        vehicle_state = VehicleState.from_carla(vehicle)
        environment = self.update_environment(obstacles, traffic_light, lane_info)
        
        # Check emergency conditions first
        if self.check_emergency_conditions(vehicle_state, environment):
            self.current_behavior = DrivingBehavior.EMERGENCY_STOP
            self.current_state = DrivingState.STOPPED
            trajectory = self.trajectory_generator.generate_emergency_stop(vehicle_state)
            return trajectory, {
                'behavior': self.current_behavior.value,
                'state': self.current_state.value,
                'reason': 'emergency_condition'
            }
        
        # Traffic light handling
        if traffic_light == 'red':
            self.current_behavior = DrivingBehavior.TRAFFIC_LIGHT_STOP
            self.current_state = DrivingState.DECELERATING
            trajectory = self.trajectory_generator.generate_slow_down(vehicle_state, 0.0)
            return trajectory, {
                'behavior': self.current_behavior.value,
                'state': self.current_state.value,
                'reason': 'red_light'
            }
        
        # Generate candidate trajectories
        candidates = []
        
        # Always consider lane following
        lane_follow = self.trajectory_generator.generate_lane_following(
            vehicle_state, 
            horizon=10,
            target_speed=self.config['target_speed']
        )
        candidates.append(lane_follow)
        
        # Consider lane changes if appropriate
        should_change, reason = self.should_consider_lane_change(vehicle_state, environment)
        if should_change:
            left_change = self.trajectory_generator.generate_lane_change_left(
                vehicle_state, target_speed=self.config['target_speed']
            )
            if left_change:
                candidates.append(left_change)
            
            right_change = self.trajectory_generator.generate_lane_change_right(
                vehicle_state, target_speed=self.config['target_speed']
            )
            if right_change:
                candidates.append(right_change)
        
        # Consider slowing down if obstacles present
        if environment.obstacles:
            min_dist = min(obs.get('distance', float('inf')) for obs in environment.obstacles)
            if min_dist < 30.0:
                slow_down = self.trajectory_generator.generate_slow_down(
                    vehicle_state,
                    target_speed=min(self.config['target_speed'] * 0.5, vehicle_state.speed * 0.7)
                )
                candidates.append(slow_down)
        
        # Evaluate all candidates
        for candidate in candidates:
            self.evaluate_trajectory(candidate, vehicle_state, environment)
        
        # Debug: Print candidate costs when there are multiple options
        if len(candidates) > 1:
            print(f"🔍 Decision candidates:")
            for c in candidates:
                print(f"   - {c.behavior.value}: cost={c.cost:.1f}, feasible={c.feasible}")
        
        # Select best trajectory
        feasible_candidates = [c for c in candidates if c.feasible and c.cost < float('inf')]
        
        if not feasible_candidates:
            # Fallback to emergency stop
            trajectory = self.trajectory_generator.generate_emergency_stop(vehicle_state)
            return trajectory, {
                'behavior': DrivingBehavior.EMERGENCY_STOP.value,
                'state': DrivingState.STOPPED.value,
                'reason': 'no_feasible_trajectory'
            }
        
        best_trajectory = min(feasible_candidates, key=lambda x: x.cost)
        
        # Update behavior if changed
        if best_trajectory.behavior != self.current_behavior:
            print(f"🚗 Behavior change: {self.current_behavior.value} → {best_trajectory.behavior.value}")
            self.current_behavior = best_trajectory.behavior
            self.behavior_start_time = time.time()
        
        # Update state
        if best_trajectory.target_speed < 0.5:
            self.current_state = DrivingState.STOPPED
        elif best_trajectory.target_speed < vehicle_state.speed - 1.0:
            self.current_state = DrivingState.DECELERATING
        elif best_trajectory.target_speed > vehicle_state.speed + 1.0:
            self.current_state = DrivingState.ACCELERATING
        else:
            self.current_state = DrivingState.DRIVING
        
        # Store decision
        self.decision_history.append({
            'time': time.time(),
            'behavior': best_trajectory.behavior.value,
            'cost': best_trajectory.cost,
            'candidates': len(candidates)
        })
        
        # Keep history limited
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-100:]
        
        return best_trajectory, {
            'behavior': best_trajectory.behavior.value,
            'state': self.current_state.value,
            'cost': best_trajectory.cost,
            'target_speed': best_trajectory.target_speed,
            'candidates_evaluated': len(candidates),
            'waypoints': len(best_trajectory.waypoints)
        }
    
    def set_target_speed(self, speed_kmh: float):
        """Set target speed in km/h"""
        self.config['target_speed'] = speed_kmh / 3.6
    
    def get_decision_stats(self) -> Dict:
        """Get decision statistics"""
        if not self.decision_history:
            return {}
        
        behaviors = [d['behavior'] for d in self.decision_history]
        avg_cost = np.mean([d['cost'] for d in self.decision_history])
        
        return {
            'total_decisions': len(self.decision_history),
            'average_cost': avg_cost,
            'behavior_counts': {b: behaviors.count(b) for b in set(behaviors)},
            'current_behavior': self.current_behavior.value,
            'current_state': self.current_state.value
        }


# -----------------------------
# INTEGRATION WITH MPC
# -----------------------------
class DecisionMakingMPCController:
    """
    Combines Decision Making with MPC for optimal trajectory planning.
    """
    
    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        self.world = world
        self.vehicle = vehicle
        
        # Decision maker
        self.decision_maker = DecisionMaker(world)
        
        # Import MPC planner
        from modules.mpc_trajectory_planner import MPCTrajectoryPlanner
        self.mpc_planner = MPCTrajectoryPlanner(world, horizon=10, dt=0.1)
        
        # State
        self.last_decision_time = 0
        self.decision_interval = 0.2  # Make decisions every 200ms
        self.current_trajectory = None
        
        print("✓ Decision Making + MPC Controller initialized")
    
    def compute_control(self, 
                        obstacles: List[Dict] = None,
                        traffic_light: Optional[str] = None,
                        lane_info: Optional[Dict] = None) -> Tuple[float, float, float, Dict]:
        """
        Compute control using decision making + MPC.
        
        Returns:
            steering: [-1, 1]
            throttle: [0, 1]
            brake: [0, 1]
            info: Decision and control info
        """
        
        current_time = time.time()
        
        # Make high-level decision periodically
        if current_time - self.last_decision_time >= self.decision_interval:
            self.current_trajectory, decision_info = self.decision_maker.decide(
                self.vehicle, obstacles, traffic_light, lane_info
            )
            self.last_decision_time = current_time
            
            # Update MPC target speed based on decision
            self.mpc_planner.set_target_speed(self.current_trajectory.target_speed * 3.6)
        else:
            decision_info = {
                'behavior': self.decision_maker.current_behavior.value,
                'state': self.decision_maker.current_state.value
            }
        
        # Handle emergency stop
        if self.decision_maker.current_behavior == DrivingBehavior.EMERGENCY_STOP:
            return 0.0, 0.0, 1.0, decision_info
        
        # Handle traffic light stop
        if self.decision_maker.current_behavior == DrivingBehavior.TRAFFIC_LIGHT_STOP:
            vehicle_state = VehicleState.from_carla(self.vehicle)
            if vehicle_state.speed < 0.5:
                return 0.0, 0.0, 0.5, decision_info
            else:
                return 0.0, 0.0, 0.5, decision_info
        
        # Use MPC for trajectory tracking
        steering, throttle_brake, mpc_info = self.mpc_planner.compute_control(self.vehicle)
        
        # Combine info
        combined_info = {**decision_info, **mpc_info}
        
        # Convert throttle_brake
        if throttle_brake >= 0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake
        
        return steering, throttle, brake, combined_info


# -----------------------------
# STANDALONE TEST
# -----------------------------
def main():
    """Test decision making standalone"""
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    carla_map = world.get_map()
    
    # Spawn vehicle
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    spawn_points = carla_map.get_spawn_points()
    
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle")
    
    print(f"✓ Vehicle spawned: {vehicle.id}")
    
    # Create controller
    controller = DecisionMakingMPCController(world, vehicle)
    controller.decision_maker.set_target_speed(30.0)  # 30 km/h
    
    print("▶ Decision Making + MPC loop running. Ctrl+C to stop.")
    
    try:
        while True:
            loop_start = time.time()
            
            # Simulate perception (empty in standalone test)
            obstacles = []
            traffic_light = None
            
            # Compute control
            steering, throttle, brake, info = controller.compute_control(
                obstacles, traffic_light, None
            )
            
            # Apply control
            control = carla.VehicleControl()
            control.steer = float(steering)
            control.throttle = float(throttle)
            control.brake = float(brake)
            vehicle.apply_control(control)
            
            # Status
            behavior = info.get('behavior', 'unknown')
            state = info.get('state', 'unknown')
            speed = VehicleState.from_carla(vehicle).speed * 3.6
            
            print(f"\r[{behavior}] {state} | Speed: {speed:.1f} km/h | "
                  f"Steer: {steering:.2f} | Thr: {throttle:.2f} | Brk: {brake:.2f}", end='')
            
            # Maintain loop rate
            elapsed = time.time() - loop_start
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)
    
    except KeyboardInterrupt:
        print("\n⏹ Interrupted")
    
    finally:
        print("🧹 Cleaning up...")
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(0.2)
        vehicle.destroy()
        print("✓ Done")


if __name__ == "__main__":
    main()
