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

# Shared file for LiDAR viewer IPC
LIDAR_VIEWER_DATA_FILE = '/tmp/lidar_viewer_data.npz'


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
                # High-density 360° continuous scanning
                'channels': 64,              # 64 vertical layers (was 32)
                'range': 120.0,              # 120m range (was 100m)
                'points_per_second': 1400000, # 1.4M points/sec (was 600k)
                'rotation_frequency': 10.0,   # 10Hz = 140k pts/rotation (was 20Hz)
                'upper_fov': 15.0,            # 15° above horizon (was 10°)
                'lower_fov': -25.0,           # 25° below horizon (was -30°)
                'sensor_tick': 0.0,           # Update every sim tick
                'atmosphere_attenuation_rate': 0.004,  # Realistic attenuation
                'dropoff_general_rate': 0.45,          # Point dropout rate
                'dropoff_intensity_limit': 0.8,
                'dropoff_zero_intensity': 0.4,
                'verbose': False,             # Reduce spam with high density
                'save_interval': 0,
                'watchdog_timeout': 5.0,
                'stats_interval': 10.0,
            },
            auto_start=True,  # attach + listen immediately
        )
        
        # Initialize driving agent (pass lidar_manager for future use)
        self.agent = DrivingAgent(self.world, self.vehicle, lidar_manager=self.lidar_manager)
        
        # LiDAR 3D Viewer (separate Open3D window, lazy-initialized)
        self.lidar_viewer_process = None
        self.lidar_viewer_active = False
        
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
        print("  [O] = Toggle LiDAR 3D Viewer (Open3D, shows ego vehicle's LiDAR)")
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
                
                # Toggle LiDAR 3D viewer (Open3D in separate process)
                elif key == ord('o'):
                    self._toggle_lidar_viewer()
                
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
                
                # Send LiDAR data to Open3D viewer (if active)
                if self.lidar_viewer_active:
                    lidar_data = self.lidar_manager.get_latest()
                    if lidar_data is not None:
                        self._write_lidar_data_for_viewer(
                            lidar_data['points'],
                            lidar_data['frame'],
                            lidar_data['timestamp']
                        )
                
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
        
        # 3b. Stop LiDAR viewer FIRST (before LiDAR sensor)
        try:
            self.lidar_viewer_active = False
            if hasattr(self, 'lidar_viewer_process') and self.lidar_viewer_process is not None:
                self.lidar_viewer_process.terminate()
                self.lidar_viewer_process = None
            # Clean up shared file
            import os
            if os.path.exists(LIDAR_VIEWER_DATA_FILE):
                os.remove(LIDAR_VIEWER_DATA_FILE)
        except Exception as e:
            print(f"   ⚠️ LiDAR viewer cleanup error: {e}")
        
        # 3c. Shutdown LiDAR sensor BEFORE vehicle
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

    # --- LiDAR viewer toggle ---
    def _toggle_lidar_viewer(self):
        """Toggle Open3D LiDAR viewer (shared data from ego vehicle's LiDAR).
        
        Launches a separate Python script that reads LiDAR data from a shared file.
        This approach avoids conflicts between OpenCV and Open3D GUI systems.
        """
        import os
        
        # Check if viewer process is still running
        if self.lidar_viewer_process is not None:
            poll = self.lidar_viewer_process.poll()
            if poll is None:  # Still running
                # Kill it
                self.lidar_viewer_process.terminate()
                self.lidar_viewer_process.wait(timeout=2)
                self.lidar_viewer_process = None
                self.lidar_viewer_active = False
                # Clean up shared file
                if os.path.exists(LIDAR_VIEWER_DATA_FILE):
                    os.remove(LIDAR_VIEWER_DATA_FILE)
                print("✓ LiDAR 3D Viewer stopped")
                return
            else:
                # Process already exited
                self.lidar_viewer_process = None
                self.lidar_viewer_active = False
        
        # Start new viewer
        script = os.path.join(os.path.dirname(__file__), 'core', 'lidar_viewer_script.py')
        if not os.path.exists(script):
            print(f"⚠️ Viewer script not found: {script}")
            return
        
        try:
            # Launch viewer as separate process
            self.lidar_viewer_process = subprocess.Popen(
                [sys.executable, script],
                start_new_session=True,
                stdout=None,  # Inherit stdout
                stderr=None,  # Inherit stderr
            )
            self.lidar_viewer_active = True
            print("✓ LiDAR 3D Viewer started")
            print("  Controls: [1]intensity [2]height [G]ground [R]reset [+/-]size [Q]quit")
            print("  Press [O] again to close")
        except Exception as e:
            print(f"⚠️ Failed to start LiDAR viewer: {e}")
            self.lidar_viewer_process = None
            self.lidar_viewer_active = False
    
    def _write_lidar_data_for_viewer(self, points, frame, timestamp):
        """Write LiDAR data to shared file for the viewer to read."""
        import numpy as np
        try:
            np.savez(LIDAR_VIEWER_DATA_FILE, 
                     points=points, 
                     frame=frame, 
                     timestamp=timestamp)
        except Exception:
            pass  # Ignore write errors

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
