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
        if dt <= 0:
            return 0.0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error +
                  self.ki * self.integral +
                  self.kd * derivative)
        self.prev_error = error
        return output


# -----------------------------
# CASCADED LANE KEEPING CONTROLLER
# (Reusable class — can also be imported into DrivingAgent)
# -----------------------------
class CascadedLaneKeepingController:
    """
    Cascaded PID lane-keeping controller.
    Outer loop: lateral (cross-track) error → desired heading correction
    Inner loop: heading error → steering output
    """

    def __init__(self, lateral_kp=0.8, lateral_ki=0.0, lateral_kd=0.1,
                 heading_kp=1.5, heading_ki=0.0, heading_kd=0.2):
        self.lateral_pid = PID(kp=lateral_kp, ki=lateral_ki, kd=lateral_kd)
        self.heading_pid = PID(kp=heading_kp, ki=heading_ki, kd=heading_kd)

    def compute_steering(self, vehicle_transform, waypoint_transform, dt):
        """
        Compute steering value [-1.0, 1.0] given vehicle and target waypoint transforms.
        """
        location = vehicle_transform.location
        yaw = math.radians(vehicle_transform.rotation.yaw)

        lane_center = waypoint_transform.location
        lane_yaw = math.radians(waypoint_transform.rotation.yaw)

        # Cross-track error (lateral offset from lane center in vehicle frame)
        dx = lane_center.x - location.x
        dy = lane_center.y - location.y
        lateral_error = -math.sin(yaw) * dx + math.cos(yaw) * dy

        # Heading error (normalized to [-pi, pi])
        heading_error = lane_yaw - yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        # Outer loop: lateral error → heading correction
        heading_correction = self.lateral_pid.compute(lateral_error, dt)

        # Inner loop: total heading error → steering
        total_heading_error = heading_error + heading_correction
        steering = self.heading_pid.compute(total_heading_error, dt)

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, steering))


# -----------------------------
# STANDALONE TEST SCRIPT
# Run this directly to test lane keeping in isolation.
# Does NOT conflict with main.py / DrivingAgent.
# -----------------------------
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    carla_map = world.get_map()   # FIX: renamed from 'map' to avoid shadowing built-in

    # Safely get blueprint
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter('vehicle.tesla.model3')  # FIX: more specific filter
    if not vehicle_bps:
        raise RuntimeError("vehicle.tesla.model3 blueprint not found")
    vehicle_bp = vehicle_bps[0]

    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available")
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])  # FIX: use try_spawn
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle — spawn point may be occupied")

    print(f"✓ Vehicle spawned: {vehicle.id}")

    controller = CascadedLaneKeepingController(
        lateral_kp=0.8, lateral_ki=0.0, lateral_kd=0.1,
        heading_kp=1.5, heading_ki=0.0, heading_kd=0.2
    )

    dt = 0.05
    print("▶ Lane keeping loop running. Ctrl+C to stop.")

    try:
        while True:
            loop_start = time.time()  # FIX: measure real elapsed time

            transform = vehicle.get_transform()
            location = transform.location

            waypoint = carla_map.get_waypoint(location)

            steering = controller.compute_steering(transform, waypoint.transform, dt)

            control = carla.VehicleControl()
            control.throttle = 0.3
            control.steer = steering
            vehicle.apply_control(control)

            # FIX: sleep for remaining time to maintain dt accurately
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Recompute actual dt for next iteration
            dt = max(0.001, time.time() - loop_start)

    except KeyboardInterrupt:
        print("\n⏹ Interrupted")

    finally:
        # FIX: always clean up
        print("🧹 Stopping and destroying vehicle...")
        try:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            time.sleep(0.2)
            vehicle.destroy()
            print("✓ Vehicle destroyed")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")


if __name__ == "__main__":
    main()