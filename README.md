This is a sub branch from New-LiDAR-Obstacle-Detector Branch upon the lane detection accuracy reduced significantly.

Current Issues in this commit
1. LiDAR measured distance is not stable, DRIVE and SLOW changes frequently causing unstable decision making. I think giving a threshold value 
2. LiDAR detection only works on automatic mode. not in manual mode.
3. Traffic Light model is not accurate, even without the traffic light, it is detecting


**Plan: Fix LiDAR DRIVE/SLOW Decision Instability** -- Found that this is NOT the problem
Context
The driving agent oscillates rapidly between DRIVE and SLOW states when an obstacle sits near the 10 m boundary. This is a classic control chattering problem: LiDAR point-cloud noise causes the measured cluster centroid distance to bounce a few centimetres either side of the threshold on every frame, flipping obstacle_action every cycle and producing erratic throttle/brake behaviour.

Two causes reinforce each other:

_classify_danger() in lidar_obstacle_detector.py uses a hard single threshold — no tolerance band around the boundary.
driving_agent.py applies the raw fused action directly with no temporal smoothing, so every noisy measurement reaches the actuators immediately.
Root issue: no hysteresis at state boundaries + no debounce on state downgrade.

Solution: Two-Layer Fix
Layer 1 — Hysteresis in LidarObstacleDetector (primary fix)
File: core/lidar_obstacle_detector.py

Add per-sector state memory and widen the "exit" threshold so the system must travel 20% past the entry threshold before it relaxes the state.

Changes:

__init__ (after line 69) — add two new instance variables:

self._sector_states: dict = {}   # {sector_str: danger_level_str}
self.hysteresis_factor: float = 1.2   # exit = entry * 1.2
_classify_danger() (lines 87–99) — replace entire method:

def _classify_danger(self, distance: float, sector: str, thresholds: dict) -> str:
    if sector in ('side_left', 'side_right'):
        return 'cautious'

    prev = self._sector_states.get(sector, 'drive')
    hf = self.hysteresis_factor

    # Entry: always upgrade to higher danger immediately (safety-first)
    if distance <= thresholds['emergency']:
        state = 'emergency_stop'
    elif distance <= thresholds['stop']:
        state = 'stop'
    elif distance <= thresholds['slow']:
        state = 'slow'
    elif distance <= thresholds['cautious']:
        state = 'cautious'
    else:
        # Above all entry thresholds — apply hysteresis before downgrading
        if prev == 'stop'  and distance <= thresholds['stop']     * hf:
            state = 'stop'
        elif prev in ('stop', 'slow') and distance <= thresholds['slow'] * hf:
            state = 'slow'
        elif prev in ('stop', 'slow', 'cautious') and distance <= thresholds['cautious'] * hf:
            state = 'cautious'
        else:
            state = 'drive'

    self._sector_states[sector] = state
    return state
Effect at 10 m boundary:

Enter SLOW when distance drops below 10 m
Exit SLOW (→ DRIVE) only when distance rises above 12 m (10 × 1.2)
Layer 2 — Action Debounce in DrivingAgent (secondary stabiliser)
File: modules/driving_agent.py

Prevent a single noisy frame from triggering a state downgrade by requiring the lower-priority action to persist for N consecutive frames before it is applied.

__init__ (after line 119, inside state variables block) — add:

self._action_history: deque = deque(maxlen=5)
self._confirmed_action: str = 'drive'
self._downgrade_frames: int = 3   # frames of lower-priority needed to confirm downgrade
After line 401 (self.obstacle_action = obstacle_action) — add a debounce call:

obstacle_action = self._debounce_action(obstacle_action)
self.obstacle_action = obstacle_action
New method _debounce_action (add near the other helper methods):

def _debounce_action(self, new_action: str) -> str:
    from core.lidar_obstacle_detector import ACTION_PRIORITY
    self._action_history.append(new_action)
    new_p = ACTION_PRIORITY.get(new_action, 0)
    cur_p = ACTION_PRIORITY.get(self._confirmed_action, 0)

    if new_p >= cur_p:
        # Upgrade or maintain: apply immediately (safety-critical path)
        self._confirmed_action = new_action
    else:
        # Downgrade: only apply if the lower action dominated the recent window
        recent_max_p = max(ACTION_PRIORITY.get(a, 0) for a in self._action_history)
        if recent_max_p <= new_p:
            self._confirmed_action = new_action
        # else: stay in confirmed_action until window clears
    return self._confirmed_action
Critical Files
File	Lines changed	Purpose
core/lidar_obstacle_detector.py	63–99	Hysteresis in _classify_danger + new instance vars
modules/driving_agent.py	~119, ~401, new method	Debounce helper + wiring
What Layer 2 Actually Does (Plain Explanation)
Layer 2 is a "vote counter" on the final control decision only. It does not touch lane detection, camera detection, LiDAR processing, or fusion.

Here's the execution order in driving_agent.py each frame:

Frame N:
  1. Lane detector runs  (line 285/324)   ← untouched
  2. YOLO/camera runs    (line 350-359)   ← untouched  
  3. LiDAR processes     (line 362-369)   ← untouched
  4. Fusion combines 2+3 (line 374-381)   ← untouched
  5. obstacle_action = fusion result      ← debounce inserted HERE (line 401)
  6. _make_control_decision(obstacle_action)  ← receives debounced value
The debounce only delays step 6 reacting to a downgrade. Steps 1–4 run identically every frame regardless.

Safety rule: upgrades (DRIVE→SLOW, SLOW→STOP) bypass the debounce and apply immediately. Only downgrades (SLOW→DRIVE) wait 3 frames. This means the system becomes MORE dangerous faster and more conservative slower — the safe direction.

Example without debounce:

Frame 1: LIDAR=9.8m → SLOW
Frame 2: LIDAR=10.1m → DRIVE   ← noise spike, throttle applied
Frame 3: LIDAR=9.9m → SLOW    ← brake reapplied
Example with debounce:

Frame 1: LIDAR=9.8m → confirmed=SLOW
Frame 2: LIDAR=10.1m → history=[SLOW,DRIVE], not 3× DRIVE yet → confirmed=SLOW
Frame 3: LIDAR=9.9m → history=[SLOW,DRIVE,SLOW] → confirmed=SLOW  (stable)
Lane detection is completely unaffected — its pipeline (lines 285–403) runs independently before the debounce point.

Why These Values
Parameter	Value	Reasoning
hysteresis_factor	1.2	20 % band = 2 m at the 10 m boundary; large enough to absorb typical LiDAR noise, small enough not to delay genuine clear-path detection
_action_history maxlen	5	~0.15–0.5 s at 10–30 FPS; short enough to remain responsive
_downgrade_frames	3	Must see lower action for 3 consecutive frames before downgrading; blocks single-frame noise spikes
Both parameters are instance variables, easy to tune without redeployment.

Verification
Run the simulator with an obstacle held stationary at ~10 m.
Watch the terminal log (frame-by-frame LIDAR= and FUSED= lines at lines 389–394 in driving_agent.py).
Before fix: FUSED alternates slow/drive every few frames.
After fix: FUSED stays slow until the obstacle is clearly beyond 12 m, then transitions cleanly to drive.
Test edge cases: obstacle approaching from 20 m → 3 m should still trigger cautious → slow → stop → emergency_stop without skipping levels.
Test clear path: obstacle removed should eventually reach drive within ~5 frames.
