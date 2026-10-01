"""Gym environment for training a LOCAL waypoint-tracking policy with PPO +
domain randomization, reusing the rest of Lab 3's navigation stack:

  - Planner.py (A*) computes the maze's start->goal waypoint route ONCE - the
    global route is known and fixed (this is Lab 4: navigation in a KNOWN
    environment), so the policy doesn't need to learn to plan, only to track.
  - EKF.py (unchanged from Lab 3) filters the noisy observation; the policy
    observes the EKF's state estimate, not the raw measurement, exactly like
    Lab 3's controller does.
  - controller_pd.py (Lab 3's hand-tuned PD controller) is NOT used for
    driving here - it's reserved as the deployment-time safety-layer fallback
    (see labs/lab4/lab4.py).

Because the global path is fixed, each training episode starts the robot at
a RANDOM point along the KNOWN route (small pose jitter) with the task of
following the remaining waypoints to the goal. This turns "solve the whole
maze" into many short, diverse local-tracking episodes covering every
corridor and turn - far more sample-efficient than replaying the full route
start-to-goal every episode, while still training on the real route (not a
synthetic one), since this lab's whole premise is a known maze.

Domain randomization: both lab1/dynamics.py (max_turn_rate=0.3, max_accel=
0.003, max_decel=0.01) and the lab3 sim plant (3.0, 0.3, 0.5) are existing,
uncalibrated GUESSES at the real robot's dynamics that disagree with each
other by 10-100x - neither has been validated against real measurements (see
the teammate's timing_report.py / collect_data.py audit on this branch, which
retracted an earlier "measured" turn rate). So rather than trusting either
guess, the TRUE simulated plant is randomized log-uniformly across the full
span between them each episode, and the EKF's process model is fixed at the
geometric-mean "split the difference" calibration - giving the EKF a
realistic, constant amount of model-mismatch noise to filter through, which
is exactly the situation a real deployment will face too.
"""
import functools

import gymnasium as gym
import numpy as np

from sphero_env.envs import SpheroEnv
from sphero_env.envs.custom_maze_full import build_occupancy_grid

from Planner import Planner
from EKF import EKF

DT = 0.3  # measured real control period (see teammate's timing_report.py)
WORLD_SIZE = 1.25
START_XY = np.array([-0.5, -0.5], dtype=np.float32)
GOAL_XY = (0.5, 0.5)
GRID_RESOLUTION = 0.125
VEL_LIMIT = 0.15

EPISODE_MAX_STEPS = 250
WAYPOINT_TOLERANCE = 0.05

OBS_DIM = 8  # [dx0, dy0, dx1, dy1, sin(heading), cos(heading), speed, collision_flag]
ACTION_DIM = 3  # policy outputs [speed_cmd, sin_raw, cos_raw] - see step() for why

# --- dynamics uncertainty band: bracket lab1's guess and lab3's sim guess ---
TURN_RATE_RANGE = (0.3, 3.0)
MAX_ACCEL_RANGE = (0.003, 0.3)
MAX_DECEL_RANGE = (0.01, 0.5)

# EKF's fixed assumed calibration: geometric mean of each range above.
NOM_MAX_TURN_RATE = float(np.sqrt(TURN_RATE_RANGE[0] * TURN_RATE_RANGE[1]))
NOM_MAX_ACCEL = float(np.sqrt(MAX_ACCEL_RANGE[0] * MAX_ACCEL_RANGE[1]))
NOM_MAX_DECEL = float(np.sqrt(MAX_DECEL_RANGE[0] * MAX_DECEL_RANGE[1]))

# EKF's fixed Q/R - lab3.py always overrides EKF.py's class defaults per
# context rather than trusting them as-is (its defaults are tuned for lab3's
# own dt/noise setup, not ours). Position has no DIRECT process noise in
# unicycle_dynamics - drift enters only via heading/speed - so Q on x/y is
# kept small; R is the mid-point of the randomized observation-noise bands
# above, the EKF's fixed "belief" about measurement quality even though the
# true noise is randomized per episode (same asymmetry as the dynamics
# calibration: the filter's assumptions are fixed, the world varies).
NOM_EKF_Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
NOM_EKF_R = np.diag([0.04 ** 2, 0.04 ** 2, 0.03 ** 2, 0.02 ** 2])

OBS_NOISE_POS_RANGE = (0.01, 0.08)
OBS_NOISE_VEL_RANGE = (0.01, 0.05)
PROCESS_NOISE_HEADING_RANGE = (0.0, 0.03)
PROCESS_NOISE_SPEED_RANGE = (0.0, 0.02)

# Reward shaping weights.
PROGRESS_GAIN = 20.0
STEP_PENALTY = 0.05
COLLISION_PENALTY = 0.5
WAYPOINT_BONUS = 5.0
GOAL_BONUS = 50.0


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def unicycle_dynamics(state, action, dt, max_turn_rate, max_accel, max_decel):
    """Same rate-limited-unicycle functional form as labs/lab1/dynamics.py,
    parameterised so domain randomization / the EKF's nominal model can each
    pick their own rate limits."""
    x, y, heading, speed = state
    speed_cmd, heading_cmd = action

    heading_error = wrap_angle(heading_cmd - heading)
    heading_new = wrap_angle(heading + np.clip(heading_error, -max_turn_rate * dt, max_turn_rate * dt))

    speed_target = speed_cmd * max(0.0, float(np.cos(heading_error)))
    speed_error = speed_target - speed
    max_step = max_accel * dt if speed_error > 0 else max_decel * dt
    speed_new = float(np.clip(speed + np.clip(speed_error, -max_step, max_step), 0.0, 1.0))

    x_new = x + speed * np.sin(heading_new) * dt
    y_new = y + speed * np.cos(heading_new) * dt
    return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)


def nominal_dynamics(state, action):
    """The EKF's fixed process model - see module docstring."""
    return unicycle_dynamics(state, action, dt=DT, max_turn_rate=NOM_MAX_TURN_RATE,
                              max_accel=NOM_MAX_ACCEL, max_decel=NOM_MAX_DECEL)


def build_features(est, waypoints, wp_index, collision_flag):
    """The policy's actual input: position relative to the next two waypoints
    (not absolute position - keeps the policy a reusable local tracker rather
    than something tied to one absolute spot in the world), sin/cos heading,
    speed, collision flag. Shared between training (maze_rl_env.py) and
    deployment (labs/lab4/lab4.py) so they build the exact same features from
    the EKF's state estimate."""
    target0 = waypoints[min(wp_index, len(waypoints) - 1)]
    target1 = waypoints[min(wp_index + 1, len(waypoints) - 1)]
    dx0, dy0 = target0[0] - est[0], target0[1] - est[1]
    dx1, dy1 = target1[0] - est[0], target1[1] - est[1]
    return np.array([dx0, dy0, dx1, dy1, np.sin(est[2]), np.cos(est[2]),
                      est[3], collision_flag], dtype=np.float32)


class MazeRLEnv(gym.Wrapper):
    def __init__(self, render_mode=None, seed=None, randomize=True):
        self.randomize = randomize
        occupancy_grid = build_occupancy_grid()
        env = SpheroEnv(
            dt=DT,
            max_steps=100_000,  # this wrapper truncates episodes itself
            vel_limit=VEL_LIMIT,
            world_width=WORLD_SIZE,
            world_height=WORLD_SIZE,
            goal_pos=GOAL_XY,
            goal_tolerance=0.1,
            occupancy_grid=occupancy_grid,
            grid_resolution=GRID_RESOLUTION,
            render_mode=render_mode,
            window_size=(800, 800),
        )
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

        # The policy outputs [speed_cmd, sin_raw, cos_raw], NOT [speed_cmd,
        # heading_cmd] directly. heading_cmd is an absolute angle with a
        # discontinuity at +-pi, and SB3 clips every sampled action to this
        # Box's bounds BEFORE calling step() (stable_baselines3's
        # OnPolicyAlgorithm.collect_rollouts does this internally, upstream
        # of anything this env can control) - so if the action space's
        # second dimension were heading_cmd itself, any raw network output
        # landing outside [-pi, pi] (routine for an unbounded linear output,
        # and exactly the values needed near this maze's many +-pi-heading
        # corridors) gets silently clamped to the boundary instead of
        # wrapped, corrupting the command. Outputting sin/cos instead and
        # reconstructing heading_cmd = atan2(sin_raw, cos_raw) in step()
        # sidesteps the discontinuity entirely: atan2 returns a valid angle
        # for ANY (sin_raw, cos_raw), clipped or not.
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.5, -1.5], dtype=np.float32),
            high=np.array([VEL_LIMIT, 1.5, 1.5], dtype=np.float32),
        )

        self.planner = Planner(map=occupancy_grid, dt=DT, resolution=GRID_RESOLUTION)
        start_state = np.array([START_XY[0], START_XY[1], 0.0, 0.0])
        self.full_waypoints = self.planner.plan(start_state, GOAL_XY, margin_cells=0)

        self._rng = np.random.default_rng(seed)
        self.ekf = None
        self.local_waypoints = None
        self.wp_index = 0
        self._prev_wp_dist = None
        self._steps = 0

    def _randomize_true_dynamics(self):
        if not self.randomize:
            self.env.dynamics = nominal_dynamics
            self.env.obs_noise_std_pos = 0.02
            self.env.obs_noise_std_vel = 0.02
            self.env.process_noise_std_heading = 0.0
            self.env.process_noise_std_speed = 0.0
            return

        def loguniform(lo, hi):
            return float(np.exp(self._rng.uniform(np.log(lo), np.log(hi))))

        self.env.dynamics = functools.partial(
            unicycle_dynamics, dt=DT,
            max_turn_rate=loguniform(*TURN_RATE_RANGE),
            max_accel=loguniform(*MAX_ACCEL_RANGE),
            max_decel=loguniform(*MAX_DECEL_RANGE),
        )
        self.env.obs_noise_std_pos = self._rng.uniform(*OBS_NOISE_POS_RANGE)
        self.env.obs_noise_std_vel = self._rng.uniform(*OBS_NOISE_VEL_RANGE)
        self.env.process_noise_std_heading = self._rng.uniform(*PROCESS_NOISE_HEADING_RANGE)
        self.env.process_noise_std_speed = self._rng.uniform(*PROCESS_NOISE_SPEED_RANGE)

    def _build_obs(self, est, collision_flag):
        return build_features(est, self.local_waypoints, self.wp_index, collision_flag)

    def reset(self, *, seed=None, options=None):
        self._randomize_true_dynamics()
        self.env.reset(seed=seed)

        # Random start point along the KNOWN route - see module docstring.
        start_idx = int(self._rng.integers(0, len(self.full_waypoints) - 1))
        self.local_waypoints = self.full_waypoints[start_idx + 1:]
        self.wp_index = 0

        anchor = self.full_waypoints[start_idx]
        target0 = self.local_waypoints[0]
        bearing = float(np.arctan2(target0[0] - anchor[0], target0[1] - anchor[1]))

        jitter_xy = self._rng.normal(0.0, 0.03, size=2)
        jitter_heading = self._rng.uniform(-np.pi / 6, np.pi / 6)
        x0 = float(anchor[0] + jitter_xy[0])
        y0 = float(anchor[1] + jitter_xy[1])
        heading0 = wrap_angle(bearing + jitter_heading)

        self.env.state_true[:] = np.array([x0, y0, heading0, 0.0], dtype=np.float32)
        self.env.state_odom[:] = np.array([x0, y0, heading0, 0.0], dtype=np.float32)

        self.ekf = EKF(dt=DT, dynamics_fn=nominal_dynamics)
        self.ekf.state_est = np.array([x0, y0, heading0, 0.0], dtype=float)
        self.ekf.P = np.eye(4) * 1e-3
        self.ekf.Q = NOM_EKF_Q.copy()
        self.ekf.R = NOM_EKF_R.copy()

        self._prev_wp_dist = float(np.hypot(target0[0] - x0, target0[1] - y0))
        self._steps = 0

        obs = self._build_obs(self.ekf.state_est, collision_flag=0.0)
        return obs, {}

    def step(self, action):
        # action = [speed_cmd, sin_raw, cos_raw] (see action_space comment in
        # __init__ for why). Reconstruct the real 2-dim env action here;
        # atan2 gives a valid heading for any sin_raw/cos_raw, so this is
        # unaffected by SB3's own clip-to-action-space-bounds step upstream.
        action = np.asarray(action, dtype=np.float32)
        # speed_cmd is non-negative by convention across every controller in
        # this repo (labs 1-3 all clip to [0.00, vel_limit], never negative) -
        # dynamics() treats negative speed_cmd as equivalent to "stop" anyway
        # (speed_target = speed_cmd * max(0, cos(...))), so allowing negative
        # values here would just be a large chunk of the action space that
        # all means the same thing ("don't move") - not a range worth giving
        # the policy to get stuck in.
        speed_cmd = float(np.clip(action[0], 0.0, VEL_LIMIT))
        heading_cmd = float(np.arctan2(action[1], action[2]))
        env_action = np.array([speed_cmd, heading_cmd], dtype=np.float32)

        self.ekf.predict(env_action)
        raw_obs, _, terminated, truncated, info = self.env.step(env_action)
        est, _ = self.ekf.update(raw_obs[:4], max_innovation=None)  # sim: measurement model is exact
        self._steps += 1

        collision_flag = float(raw_obs[4]) if len(raw_obs) > 4 else 0.0

        target0 = self.local_waypoints[self.wp_index]
        dist_now = float(np.hypot(target0[0] - est[0], target0[1] - est[1]))
        reward = PROGRESS_GAIN * (self._prev_wp_dist - dist_now) - STEP_PENALTY
        self._prev_wp_dist = dist_now

        if info.get("collision", False):
            reward -= COLLISION_PENALTY
        if info.get("out_of_bounds", False):
            reward -= COLLISION_PENALTY
            terminated = True

        reached_final = False
        if dist_now < WAYPOINT_TOLERANCE:
            reward += WAYPOINT_BONUS
            self.wp_index += 1
            if self.wp_index >= len(self.local_waypoints):
                reached_final = True
                reward += GOAL_BONUS
                terminated = True
            else:
                next_target = self.local_waypoints[self.wp_index]
                self._prev_wp_dist = float(np.hypot(next_target[0] - est[0], next_target[1] - est[1]))

        if self._steps >= EPISODE_MAX_STEPS:
            truncated = True

        obs = self._build_obs(est, collision_flag)
        info = dict(info)
        info["wp_index"] = self.wp_index
        info["reached_final"] = reached_final
        return obs, reward, terminated, truncated, info
