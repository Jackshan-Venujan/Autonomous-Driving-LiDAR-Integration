import carla
import math
import time

# -----------------------------
# PID CLASS
# -----------------------------
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = (self.kp * error +
                  self.ki * self.integral +
                  self.kd * derivative)
        self.prev_error = error
        return output


# -----------------------------
# CONNECT TO CARLA
# -----------------------------
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
map = world.get_map()

vehicle_bp = world.get_blueprint_library().filter('model3')[0]
spawn_point = map.get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# -----------------------------
# CREATE CASCADED PID
# -----------------------------
lateral_pid = PID(kp=0.8, ki=0.0, kd=0.1)
heading_pid = PID(kp=1.5, ki=0.0, kd=0.2)

# -----------------------------
# CONTROL LOOP
# -----------------------------
dt = 0.05

while True:

    transform = vehicle.get_transform()
    location = transform.location
    yaw = math.radians(transform.rotation.yaw)

    # Get nearest waypoint (lane center reference)
    waypoint = map.get_waypoint(location)

    lane_center = waypoint.transform.location
    lane_yaw = math.radians(waypoint.transform.rotation.yaw)

    # -----------------------------
    # ERROR CALCULATION
    # -----------------------------

    dx = lane_center.x - location.x
    dy = lane_center.y - location.y

    # Lateral error (cross-track error)
    lateral_error = -math.sin(yaw) * dx + math.cos(yaw) * dy

    # Heading error
    heading_error = lane_yaw - yaw

    # Normalize heading error
    heading_error = math.atan2(math.sin(heading_error),
                               math.cos(heading_error))

    # -----------------------------
    # CASCADED CONTROL
    # -----------------------------

    # Outer loop: lateral error → desired heading correction
    heading_correction = lateral_pid.compute(lateral_error, dt)

    # Combined heading error
    total_heading_error = heading_error + heading_correction

    # Inner loop: heading error → steering
    steering = heading_pid.compute(total_heading_error, dt)

    # Clamp steering
    steering = max(min(steering, 1.0), -1.0)

    # Apply control
    control = carla.VehicleControl()
    control.throttle = 0.3
    control.steer = steering
    vehicle.apply_control(control)

    time.sleep(dt)
