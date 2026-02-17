"""
Main entry point for modular autonomous driving system
Clean architecture with separated modules
"""

import carla
import cv2
import time
import sys
import subprocess

from modules.driving_agent import DrivingAgent
from modules.lidar_based_obstacle_detector import LidarManager


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
        
        # ── Attach Standard LiDAR (Phase 1) ──────────────────────────────
        # sensor.lidar.ray_cast — streams Nx4 point clouds via callback
        # Config can be customized here; defaults: 32ch, 100m range, 600k pps
        self.lidar_manager = LidarManager(
            world=self.world,
            vehicle=self.vehicle,
            config={
                'channels': 32,
                'range': 100.0,
                'points_per_second': 600000,
                'rotation_frequency': 20.0,
                'upper_fov': 10.0,
                'lower_fov': -30.0,
                'verbose': True,       # per-frame prints (set False to reduce spam)
                'save_interval': 0,    # 0=off; set e.g. 200 to save every 200th frame
                'watchdog_timeout': 5.0,
                'stats_interval': 10.0,
            },
            auto_start=True,  # attach + listen immediately
        )
        
        # Initialize driving agent (pass lidar_manager for future use)
        self.agent = DrivingAgent(self.world, self.vehicle, lidar_manager=self.lidar_manager)
        
        print("✓ System initialized (with LiDAR)")
    
    def _camera_callback(self, image):
        """Camera callback - CARLA provides BGRA, convert to BGR for OpenCV"""
        import numpy as np
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # CARLA gives BGRA, we want BGR (drop alpha channel)
        # No need to reverse channels - OpenCV expects BGR
        array = array[:, :, :3]  # Keep BGR, drop alpha
        self.camera_data = array
    
    def run(self, duration=300, spawn_traffic=True):
        """Run autonomous driving"""
        print("\n" + "="*70)
        print("  CONTROLS:")
        print("  [L] = Autonomous Mode (with Traffic Light Detection)")
        print("  [M] = Manual Mode")
        print("  [V] = Toggle Lane Mask Visualization")
        print("  [T] = Toggle Lead Vehicle (test your model!)")
        print("  [P] = Print LiDAR Stats")
        print("  [O] = Open LiDAR 3D Viewer (separate window, needs open3d)")
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
                
                # Print LiDAR throughput stats
                elif key == ord('p'):
                    self.lidar_manager.print_stats()
                
                # Launch LiDAR 3D viewer in a separate process
                elif key == ord('o'):
                    self._launch_lidar_viewer()
                
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
                
                # Process frame
                result = self.agent.process_frame(self.camera_data)
                
                # Apply control
                self.vehicle.apply_control(result['control'])
                
                # Visualize
                vis, _ = self.agent.visualize(self.camera_data, result)
                
                if vis is not None:
                    cv2.imshow('Autonomous Driving - Modular', vis)
                
                # LiDAR periodic stats (auto-prints if interval elapsed)
                self.lidar_manager.maybe_print_stats()
                
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
        
        # 3b. Shutdown LiDAR sensor BEFORE vehicle
        try:
            if hasattr(self, 'lidar_manager') and self.lidar_manager is not None:
                self.lidar_manager.shutdown()
        except Exception as e:
            print(f"   ⚠️ LiDAR cleanup error: {e}")
        
        # 4. Destroy camera BEFORE vehicle
        try:
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.stop()  # Stop listening first
                time.sleep(0.1)  # Give it time
                self.camera.destroy()
        except Exception as e:
            print(f"   ⚠️ Camera cleanup error: {e}")
        
        # 5. Destroy vehicle last
        try:
            if hasattr(self, 'vehicle') and self.vehicle is not None:
                self.vehicle.destroy()
        except Exception as e:
            print(f"   ⚠️ Vehicle cleanup error: {e}")
        
        print("✓ Cleanup complete")

    # --- LiDAR viewer launcher ---
    def _launch_lidar_viewer(self):
        """Launch LiDAR 3D viewer as a separate process.
        
        Open3D requires the main thread, so we run lidar_viewer_stage1.py
        in a subprocess. It connects to the same CARLA server and creates
        its own vehicle + LiDAR (independent from the driving system).
        
        For integrated viewing (reusing the ego vehicle's LiDAR), run
        lidar_viewer.py manually in a second terminal.
        """
        import os
        script = os.path.join(os.path.dirname(__file__), 'lidar_viewer_stage1.py')
        if not os.path.exists(script):
            print("⚠️ lidar_viewer_stage1.py not found!")
            return
        try:
            import platform
            if platform.system() == 'Windows':
                subprocess.Popen(
                    [sys.executable, script],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # Linux / macOS: start_new_session detaches the child process
                subprocess.Popen(
                    [sys.executable, script],
                    start_new_session=True
                )
            print("✓ LiDAR 3D Viewer launched in new window")
            print("  (Close that window or Ctrl+C in it to stop)")
        except Exception as e:
            print(f"⚠️ Failed to launch LiDAR viewer: {e}")

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
