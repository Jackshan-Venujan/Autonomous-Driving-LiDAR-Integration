import time
import traceback
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

import carla

from core.carla_spawner import CarlaSpawner
from evaluation.ground_truth_logger import GroundTruthLogger
from evaluation.metrics_engine import MetricsEngine


@dataclass
class ScenarioDefinition:
    """
    One test scenario. Traffic is fixed; weather is the independent variable.

    YAML schema (in config_experiment.yaml):
        scenarios:
          - name: "clear_noon"
            weather_preset: "ClearNoon"
            num_vehicles: 15
            num_pedestrians: 5
            num_static: 3
            duration_ticks: 500
            tag: "clear"
    """
    name: str
    weather_preset: str
    num_vehicles: int = 15
    num_pedestrians: int = 5
    num_static: int = 3
    duration_ticks: int = 500
    tag: str = 'clear'
    ego_spawn_index: int = 0


PREDEFINED_SCENARIOS: List[ScenarioDefinition] = [
    ScenarioDefinition(
        name='clear_noon',
        weather_preset='ClearNoon',
        num_vehicles=15, num_pedestrians=5, num_static=3,
        duration_ticks=500, tag='clear',
    ),
    ScenarioDefinition(
        name='hard_rain_noon',
        weather_preset='HardRainNoon',
        num_vehicles=15, num_pedestrians=5, num_static=3,
        duration_ticks=500, tag='rain',
    ),
    ScenarioDefinition(
        name='clear_night',
        weather_preset='ClearNight',
        num_vehicles=15, num_pedestrians=5, num_static=3,
        duration_ticks=500, tag='night',
    ),
    ScenarioDefinition(
        name='cloudy_sunset',
        weather_preset='CloudySunset',
        num_vehicles=15, num_pedestrians=5, num_static=3,
        duration_ticks=500, tag='cloudy_evening',
    ),
]


def scenarios_from_config(scenario_dicts: List[Dict]) -> List[ScenarioDefinition]:
    """Build ScenarioDefinition list from YAML-loaded dicts."""
    result = []
    for d in scenario_dicts:
        result.append(ScenarioDefinition(
            name=d['name'],
            weather_preset=d['weather_preset'],
            num_vehicles=d.get('num_vehicles', 15),
            num_pedestrians=d.get('num_pedestrians', 5),
            num_static=d.get('num_static', 3),
            duration_ticks=d.get('duration_ticks', 500),
            tag=d.get('tag', 'clear'),
        ))
    return result


class ScenarioRunner:
    """
    Iterates through a list of ScenarioDefinitions, configures CARLA weather
    and traffic for each, runs the experiment frame loop, and feeds data to
    MetricsEngine.

    The frame_callback is the only difference between Experiment 1 and 2 — it
    encapsulates the per-frame processing so ScenarioRunner stays sensor-agnostic.

    frame_callback signature:
        (camera_frame: np.ndarray, frame_id: int) -> Dict
        Returns the result dict from DrivingAgent.process_frame() (possibly
        enriched with stereo depth before being passed here).
    """

    def __init__(
        self,
        client: carla.Client,
        world: carla.World,
        vehicle: carla.Vehicle,
        frame_callback: Callable,
        gt_logger: GroundTruthLogger,
        metrics_engine: MetricsEngine,
        scenarios: List[ScenarioDefinition],
        get_camera_frame: Callable,       # () -> Optional[np.ndarray]
        use_synchronous_mode: bool = True,
    ):
        self._client = client
        self._world = world
        self._vehicle = vehicle
        self._frame_cb = frame_callback
        self._gt_logger = gt_logger
        self._engine = metrics_engine
        self._scenarios = scenarios
        self._get_frame = get_camera_frame
        self._use_sync = use_synchronous_mode
        self._spawner: Optional[CarlaSpawner] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> Dict[str, List[float]]:
        """
        Run every scenario sequentially.
        Returns {scenario_name: [per_frame_latency_ms]}.
        """
        if self._use_sync:
            self._set_synchronous_mode(True)

        all_latencies = {}
        try:
            for scenario in self._scenarios:
                print(f"\n[ScenarioRunner] Starting scenario: {scenario.name} ({scenario.weather_preset})")
                lats = self.run_scenario(scenario)
                all_latencies[scenario.name] = lats
                print(f"[ScenarioRunner] Completed scenario: {scenario.name} — {len(lats)} frames")
        finally:
            if self._use_sync:
                self._set_synchronous_mode(False)
            if self._spawner is not None:
                self._spawner.cleanup()

        return all_latencies

    def run_scenario(self, scenario: ScenarioDefinition) -> List[float]:
        """Run one scenario. Returns per-frame latency_ms list."""
        self._apply_weather(scenario.weather_preset)

        # Spawn NPC traffic
        self._spawner = CarlaSpawner(self._world)
        self._spawner.spawn_traffic_obstacles(
            num_vehicles=scenario.num_vehicles,
            num_pedestrians=scenario.num_pedestrians,
            num_static=scenario.num_static,
        )

        # Brief warm-up: let simulation settle
        for _ in range(10):
            if self._use_sync:
                self._world.tick()
            else:
                time.sleep(0.05)

        latencies = []
        frame_id = 0
        try:
            for _ in range(scenario.duration_ticks):
                if self._use_sync:
                    self._world.tick()

                camera_frame = self._get_frame()
                if camera_frame is None:
                    continue

                t0 = time.perf_counter()
                try:
                    result = self._frame_cb(camera_frame, frame_id)
                except Exception:
                    traceback.print_exc()
                    frame_id += 1
                    continue
                latency_ms = (time.perf_counter() - t0) * 1000.0

                frame_gt = self._gt_logger.capture_frame(frame_id, scenario.tag)
                self._engine.update(frame_id, result, frame_gt, latency_ms, scenario.tag)

                latencies.append(latency_ms)
                frame_id += 1
        finally:
            if self._spawner is not None:
                self._spawner.cleanup()
                self._spawner = None

        return latencies

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_weather(self, preset_name: str):
        """Apply a carla.WeatherParameters preset by attribute name."""
        try:
            preset = getattr(carla.WeatherParameters, preset_name)
            self._world.set_weather(preset)
            print(f"[ScenarioRunner] Weather set to: {preset_name}")
        except AttributeError:
            print(f"[ScenarioRunner] Unknown weather preset '{preset_name}', using ClearNoon")
            self._world.set_weather(carla.WeatherParameters.ClearNoon)

    def _set_synchronous_mode(self, enabled: bool):
        settings = self._world.get_settings()
        settings.synchronous_mode = enabled
        settings.fixed_delta_seconds = 0.05 if enabled else None
        self._world.apply_settings(settings)
        # Traffic manager must also be in sync mode
        tm = self._client.get_trafficmanager()
        tm.set_synchronous_mode(enabled)
