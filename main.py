"""
Main entry point for modular autonomous driving system
Clean architecture with separated modules
"""

import carla
import cv2
import time
import sys

from modules.driving_agent import DrivingAgent


class AutonomousDrivingSystem:
    """Main system coordinator"""
    
    def __init__(self):
        print("="*60)
        print("CARLA Autonomous Driving System - Modular Architecture")
        print("="*60)
        
        # Connect to CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town04')
        
        # Initial weather (can be changed later via hotkeys)
        weather = carla.WeatherParameters.ClearNoon 
        # weather = carla.WeatherParameters.ClearSunset
        # weather = carla.WeatherParameters.WetSunset
        # weather = carla.WeatherParameters.WetNoon
        # weather = carla.WeatherParameters.WetCloudyNoon
        # weather = carla.WeatherParameters.WetCloudySunset        
        # weather = carla.WeatherParameters.HardRainSunset
        # weather = carla.WeatherParameters.SoftRainNoon
        # weather = carla.WeatherParameters.SoftRainSunset

        # weather = carla.WeatherParameters.CloudyNoon
        # weather = carla.WeatherParameters.CloudySunset
        # weather = carla.WeatherParameters.WetNoon         
        # weather = carla.WeatherParameters.MidRainyNoon
        # weather = carla.WeatherParameters.MidRainSunset
        # weather = carla.WeatherParameters.HardRainNoon
        
        self.world.set_weather(weather)
        
        # Spawn ego vehicle
        bp = self.world.get_blueprint_library()
        vehicle_bp = bp.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        
        # Setup camera
        camera_bp = bp.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        
        cam_transform = carla.Transform(
            carla.Location(x=2.0, z=1.4),
            carla.Rotation(pitch=-15)
        )
        self.camera = self.world.spawn_actor(camera_bp, cam_transform, attach_to=self.vehicle)
        
        self.camera_data = None
        # Timestamps let us verify both frames were captured within
        # 100 ms of each other before using them for stereo depth.
        self.camera_timestamp = None
        self.camera.listen(lambda image: self._camera_callback(image))

        # ── RIGHT camera for stereo vision ──────────────────────────
        # Placed 0.54 m to the right of the LEFT camera (y=0.54).
        # The separation between the two lenses is called the "baseline".
        # Stereo depth works by comparing how much an object shifts
        # between the two images — more shift = object is closer.
        # All other parameters are identical to the LEFT camera.
        # ─────────────────────────────────────────────────────────────
        right_cam_transform = carla.Transform(
            carla.Location(x=2.0, y=0.54, z=1.4),
            carla.Rotation(pitch=-15)
        )
        self.right_camera = self.world.spawn_actor(
            camera_bp, right_cam_transform, attach_to=self.vehicle
        )
        # Timestamps let us verify both frames were captured within
        # 100 ms of each other before using them for stereo depth.
        self.right_camera_data = None
        self.right_camera_timestamp = None
        self.right_camera.listen(lambda image: self._right_camera_callback(image))

        # ── LiDAR sensor ────────────────────────────────────────────────
        # Mounted on the vehicle roof (z=2.4, above both cameras at z=1.4).
        # A LiDAR fires laser pulses in a 360° rotating pattern and measures
        # how long each pulse takes to return — giving exact 3D distances.
        # Each frame produces thousands of 3D points (a "point cloud").
        # Unlike cameras, LiDAR works in complete darkness.
        # ────────────────────────────────────────────────────────────────
        lidar_bp = bp.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',            '32')
        lidar_bp.set_attribute('range',               '50.0')
        lidar_bp.set_attribute('points_per_second',   '56000')
        lidar_bp.set_attribute('rotation_frequency',  '10')
        lidar_bp.set_attribute('upper_fov',           '10.0')
        lidar_bp.set_attribute('lower_fov',           '-30.0')

        lidar_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=2.4)
        )
        self.lidar_sensor = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle
        )
        self.lidar_data      = None
        self.lidar_timestamp = None
        self.lidar_sensor.listen(lambda data: self._lidar_callback(data))

        # Initialize driving agent
        self.agent = DrivingAgent(self.world, self.vehicle)
        
        print("✓ System initialized")
    
    def _camera_callback(self, image):
        """Camera callback - CARLA provides BGRA, convert to BGR for OpenCV"""
        import numpy as np
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # CARLA gives BGRA, we want BGR (drop alpha channel)
        # No need to reverse channels - OpenCV expects BGR
        array = array[:, :, :3]  # Keep BGR, drop alpha
        self.camera_data = array
        self.camera_timestamp = image.timestamp

    def _right_camera_callback(self, image):
        """Right camera callback for stereo vision.

        Identical BGRA→BGR conversion as the left camera callback.
        Stores both the frame and its CARLA timestamp so the main loop
        can verify left/right frames are within 100 ms of each other
        before passing them to the stereo depth estimator.
        """
        import numpy as np
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Keep BGR, drop alpha
        self.right_camera_data = array
        self.right_camera_timestamp = image.timestamp

    def _lidar_callback(self, data):
        """LiDAR callback — converts raw bytes to a float32 point cloud.

        CARLA LiDAR returns raw bytes. We convert to a numpy float32 array
        shaped (N, 4) where each row is [x, y, z, intensity] for one point.

        CARLA LiDAR coordinate convention:
          x = forward (away from vehicle front)
          y = left
          z = up
        This differs from the camera frame (x=right, y=down, z=forward).
        We do NOT convert here — LidarDistanceEstimator handles coordinates.
        """
        import numpy as np
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = points.reshape(-1, 4)
        self.lidar_data      = points
        self.lidar_timestamp = data.timestamp
    
    def run(self, duration=300, spawn_traffic=True):
        """Run autonomous driving"""
        print("\n" + "="*70)
        print("  CONTROLS:")
        print("  [L] = Autonomous Mode (with Traffic Light Detection)")
        print("  [M] = Manual Mode")
        print("  [V] = Toggle Lane Mask Visualization")
        print("  [T] = Toggle Lead Vehicle (test your model!)")
        print("  [N] = Night  [B] = Bright (Day)")
        print("  [W/A/S/D] = Manual throttle/brake/steering")
        print("  [Space] = Brake")
        print("  [Q] = Quit")
        print("="*70)
        print("\n⚠️  Vehicle starts in MANUAL mode")
        print("⚠️  Press [L] to enable AUTONOMOUS driving")
        print("⚠️  Press [W] to start driving manually")
        if self.agent.traffic_light_enabled:
            print("✓  Traffic Light Detection: ACTIVE\n")
        else:
            print("⚠️  Traffic Light Detection: DISABLED\n")
        
        # Wait for camera
        print("⏳ Waiting for camera...")
        wait_start = time.time()
        while self.camera_data is None:
            if time.time() - wait_start > 10:
                print("❌ Camera timeout")
                return
            time.sleep(0.1)
            self.world.tick()
        print("✓ Camera ready")
        
        # Spawn traffic
        if spawn_traffic:
            self.agent.spawn_traffic(num_vehicles=50, num_static=10)
        
        # Main loop
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                if self.camera_data is None:
                    time.sleep(0.01)
                    continue
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                # Mode switching
                if key == ord('m'):
                    self.agent.set_mode('manual')
                elif key == ord('l'):
                    self.agent.set_mode('auto')
                    self.agent.handle_roi_choice_when_auto(self.camera_data)
                # Weather/time of day hotkeys
                elif key == ord('n'):
                    self.set_night()
                elif key == ord('b'):
                    self.set_day()
                
                # NEW: Toggle lane mask visualization
                elif key == ord('v'):
                    self.agent.toggle_lane_mask_visualization()
                
                # NEW: Toggle lead vehicle
                elif key == ord('t'):
                    if self.agent.lead_vehicle.enabled:
                        self.agent.lead_vehicle.destroy()
                        print("Lead Vehicle: DISABLED")
                    else:
                        if self.agent.lead_vehicle.spawn_lead_vehicle():
                            print("Lead Vehicle: ENABLED (RED car ahead)")
                
                # ROI selection in auto mode
                if self.agent.awaiting_roi_choice:
                    self.agent.process_roi_choice_key(key)
                
                # Manual controls
                if self.agent.mode == 'manual':
                    self.agent.process_manual_keys(key)
                
                # Quit
                if key == ord('q'):
                    break
                
                # Only pass a right frame to the agent when both cameras have
                # delivered a frame AND the frames are within 100 ms of each
                # other. Frames captured at very different times would give
                # wrong stereo depth because the vehicle may have moved.
                right_frame = None
                if (self.right_camera_data is not None
                        and self.camera_timestamp is not None
                        and self.right_camera_timestamp is not None
                        and abs(self.camera_timestamp
                                - self.right_camera_timestamp) < 0.1):
                    right_frame = self.right_camera_data

                # Process frame
                result = self.agent.process_frame(self.camera_data,
                                                  right_frame=right_frame,
                                                  lidar_data=self.lidar_data)
                
                # Apply control
                self.vehicle.apply_control(result['control'])
                
                # Visualize
                vis, _ = self.agent.visualize(self.camera_data, result)
                
                if vis is not None:
                    cv2.imshow('Autonomous Driving - Modular', vis)
                
                # Status
                if frame_count % 100 == 0:
                    print(f"[{frame_count}] {self.agent.mode.upper()}: {result['decision']}")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\n⏹ Interrupted")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources - safe order"""
        print("\n🧹 Cleaning up...")
        
        # 1. Close windows first
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # 2. Stop vehicle
        try:
            if hasattr(self, 'vehicle') and self.vehicle is not None:
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0
                self.vehicle.apply_control(control)
        except:
            pass
        
        # 3. Cleanup spawned actors (agent handles this)
        try:
            if hasattr(self, 'agent'):
                self.agent.cleanup()
        except Exception as e:
            print(f"   ⚠️ Error in agent cleanup: {e}")
        
        # 4. Destroy left camera BEFORE vehicle
        try:
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.stop()  # Stop listening first
                time.sleep(0.1)  # Give it time
                self.camera.destroy()
        except Exception as e:
            print(f"   ⚠️ Camera cleanup error: {e}")

        # 4b. Destroy right (stereo) camera BEFORE vehicle
        try:
            if hasattr(self, 'right_camera') and self.right_camera is not None:
                self.right_camera.stop()
                time.sleep(0.1)
                self.right_camera.destroy()
        except Exception as e:
            print(f"   ⚠️ Right camera cleanup error: {e}")

        # 4c. Destroy LiDAR sensor BEFORE vehicle
        try:
            if hasattr(self, 'lidar_sensor') and self.lidar_sensor is not None:
                self.lidar_sensor.stop()
                time.sleep(0.1)
                self.lidar_sensor.destroy()
        except Exception as e:
            print(f"   ⚠️ LiDAR cleanup error: {e}")
        
        # 5. Destroy vehicle last
        try:
            if hasattr(self, 'vehicle') and self.vehicle is not None:
                self.vehicle.destroy()
        except Exception as e:
            print(f"   ⚠️ Vehicle cleanup error: {e}")
        
        print("✓ Cleanup complete")

    # --- Day/Night helpers ---
    def set_night(self):
        """Set world to night-like lighting and turn on vehicle lights."""
        try:
            weather = self.world.get_weather()
            # Negative/low sun altitude simulates night
            weather.sun_altitude_angle = -10.0
            # Optionally reduce scattering to make it darker
            if hasattr(weather, 'scattering_intensity'):
                weather.scattering_intensity = 0.2
            if hasattr(weather, 'mie_scattering_scale'):
                weather.mie_scattering_scale = 0.02
            if hasattr(weather, 'rayleigh_scattering_scale'):
                weather.rayleigh_scattering_scale = 0.02
            self.world.set_weather(weather)

            # Turn on vehicle lights (position + low beam)
            try:
                vls = carla.VehicleLightState
                lights = vls.Position | vls.LowBeam
                if hasattr(vls, 'Fog'):
                    lights |= vls.Fog
                self.vehicle.set_light_state(carla.VehicleLightState(lights))
            except Exception:
                pass
            print("🌙 Night mode applied")
        except Exception as e:
            print(f"⚠️ Failed to set night: {e}")

    def set_day(self):
        """Set world to a bright day preset and turn off vehicle lights."""
        try:
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
            try:
                # Turn off lights
                self.vehicle.set_light_state(carla.VehicleLightState(0))
            except Exception:
                pass
            print("☀️ Day mode applied")
        except Exception as e:
            print(f"⚠️ Failed to set day: {e}")


def main():
    """Main function"""
    try:
        system = AutonomousDrivingSystem()
        system.run(duration=300, spawn_traffic=True)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
