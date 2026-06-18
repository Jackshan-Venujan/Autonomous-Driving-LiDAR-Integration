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
        self.camera.listen(lambda image: self._camera_callback(image))

        # Rear camera — faces backward, 120° FOV for wider blind-spot coverage
        rear_camera_bp = bp.find('sensor.camera.rgb')
        rear_camera_bp.set_attribute('image_size_x', '1280')
        rear_camera_bp.set_attribute('image_size_y', '720')
        rear_camera_bp.set_attribute('fov', '120')   # wider than front (90°)
        rear_cam_transform = carla.Transform(
            carla.Location(x=-2.5, z=1.4),
            carla.Rotation(pitch=-10, yaw=180)
        )
        self.rear_camera = self.world.spawn_actor(
            rear_camera_bp, rear_cam_transform, attach_to=self.vehicle
        )
        self.rear_camera_data = None
        self.rear_camera.listen(lambda image: self._rear_camera_callback(image))

        # LiDAR sensor — roof-mounted ray-cast sensor (simulates Velodyne HDL-64E)
        lidar_bp = bp.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',          '64')
        lidar_bp.set_attribute('range',             '50')
        lidar_bp.set_attribute('points_per_second', '1300000')
        lidar_bp.set_attribute('rotation_frequency','20')
        lidar_bp.set_attribute('upper_fov',         '10')
        lidar_bp.set_attribute('lower_fov',         '-30')
        lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.lidar_data = None
        self.lidar.listen(lambda data: self._lidar_callback(data))

        # Initialize driving agent
        self.agent = DrivingAgent(self.world, self.vehicle)
        
        print("✓ System initialized")
    
    def _camera_callback(self, image):
        """Camera callback - CARLA provides BGRA, convert to BGR for OpenCV"""
        import numpy as np
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Keep BGR, drop alpha
        self.camera_data = array

    def _rear_camera_callback(self, image):
        """Rear camera callback"""
        import numpy as np
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.rear_camera_data = array

    def _lidar_callback(self, data):
        """LiDAR callback — store raw measurement; LidarProcessor handles parsing."""
        self.lidar_data = data
    
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
        print("  Rear Camera window: BLIND SPOT MONITOR (always on)")
        print("="*70)
        print("\n⚠️  Vehicle starts in MANUAL mode")
        print("⚠️  Press [L] to enable AUTONOMOUS driving")
        print("⚠️  Press [W] to start driving manually")
        if self.agent.traffic_light_enabled:
            print("✓  Traffic Light Detection: ACTIVE\n")
        else:
            print("⚠️  Traffic Light Detection: DISABLED\n")
        
        # Wait for cameras and LiDAR
        print("⏳ Waiting for sensors...")
        wait_start = time.time()
        while self.camera_data is None or self.rear_camera_data is None or self.lidar_data is None:
            if time.time() - wait_start > 10:
                print("❌ Sensor timeout")
                return
            time.sleep(0.1)
            self.world.tick()
        print("✓ Front camera, rear camera, and LiDAR ready")
        
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
                
                # Process frame (front + rear)
                result = self.agent.process_frame(
                    self.camera_data, rear_image=self.rear_camera_data
                )

                # Apply control
                self.vehicle.apply_control(result['control'])

                # LiDAR — process point cloud
                lidar_result = self.agent.process_lidar(self.lidar_data)

                # LiDAR-Camera fusion — project LiDAR into both cameras,
                # fuse with YOLO detections for accurate depth per obstacle
                front_dets = (result['obstacle_data'].get('all_detections', [])
                              if result.get('obstacle_data') else [])
                rear_dets  = (result['rear_data'].get('all_detections', [])
                              if result.get('rear_data') else [])
                fusion = self.agent.run_lidar_camera_fusion(
                    lidar_result, front_dets, rear_dets
                )

                # Log per-object fusion data at ~5 fps
                self.agent.fusion_logger.log_frame(
                    frame_count,
                    fusion['front']['detections'],
                    fusion['rear']['detections'],
                    fusion['lidar_obstacles'],
                )

                # Render BEV with camera FOV cones + class labels
                vis_bev = self.agent.visualize_bev(lidar_result, fusion=fusion)

                # Visualize — front HUD and rear window (with LiDAR overlay)
                vis, vis_rear = self.agent.visualize(
                    self.camera_data, result,
                    rear_image=self.rear_camera_data,
                    fusion=fusion,
                )

                if vis is not None:
                    cv2.imshow('Autonomous Driving - Modular', vis)

                if vis_rear is not None:
                    cv2.imshow('Rear Camera - Blind Spot Monitor', vis_rear)

                if vis_bev is not None:
                    cv2.imshow('LiDAR BEV - Obstacle Map', vis_bev)
                
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
        
        # 4. Destroy LiDAR and cameras BEFORE vehicle
        for cam_attr, label in [('lidar', 'lidar'), ('camera', 'front'), ('rear_camera', 'rear')]:
            try:
                cam = getattr(self, cam_attr, None)
                if cam is not None:
                    cam.stop()
                    time.sleep(0.05)
                    cam.destroy()
            except Exception as e:
                print(f"   ⚠️ {label} camera cleanup error: {e}")
        
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
