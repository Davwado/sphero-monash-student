"""Lab 4: the Lab 3 A* planner and EKF, with a learned residual dynamics model used for
the EKF's predict step, for braking (predicted coast distance), and as the simulator's
physics in --sim. The speed control is a loop on measured speed rather than Lab 3's PD.

Train first (from labs/lab4); every real run and speed test is logged to logs/:
    python train_dynamics.py logs/speed_test_*.csv logs/lab4_real_*.csv
Then:
    python lab4.py --sim
    python lab4.py
"""
import argparse
import csv
import os
import sys
import time
from contextlib import ExitStack, contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lab3"))

from sphero_env.robot.connect import scan_and_connect  # noqa: E402
from sphero_unsw.sphero_edu import SpheroEduAPI  # noqa: E402
from sphero_env.robot.robot import Robot  # noqa: E402
from sphero_env.envs import SpheroEnv  # noqa: E402
from sphero_env.envs.custom_maze_full import build_occupancy_grid  # noqa: E402

from Planner import Planner  # noqa: E402
from EKF import EKF, dynamics as analytic_dynamics, wrap_angle  # noqa: E402
from train_dynamics import SPEED_DT, make_learned_dynamics, measured_speed  # noqa: E402

# Replace with your actual student ID before submitting.
STUDENT_ID = "your_id_here"

WEIGHTS = os.path.join(HERE, "dyn_weights.npz")
LOG_DIR = os.path.join(HERE, "logs")

# Seconds per step on the real robot. Must match STEP_PERIOD in examples/teleop.py,
# because the learned model predicts one step of that length.
STEP_PERIOD = 0.3

SIM_POS_NOISE = 0.01
REAL_POS_NOISE = 0.01

SEED = 0
MAX_STEPS = 1500
START_XY = np.array([-0.5, -0.5])
WAYPOINT_TOLERANCE = 0.03
MAX_STEPS_PER_WAYPOINT = 200
MAX_REPLANS = 10

# Safety: if the estimate sits more than MAX_EST_GAP from the odometry for this many
# steps in a row, the model no longer matches reality: resync and use the analytic model.
MAX_EST_GAP = 0.10
MAX_GAP_STEPS = 3

# Safety: the real robot's collision flag can miss a ball pinned on a wall, so treat
# STUCK_STEPS forward commands that move it less than STUCK_DIST in total as a collision.
STUCK_STEPS = 8
STUCK_DIST = 0.03

# At a waypoint where the path turns, only count it once the ball has stopped nearby;
# coasting past a corner lines the ball up with the wall instead of the next corridor.
CORNER_ANGLE = np.radians(30)
SETTLED_MOVE = 0.01

# Speed loop. At a fixed low command the real ball either stalls for 5-13 steps or keeps
# accelerating to ~7 cm/step, so the command is adjusted every step to hold TARGET_STEP of
# movement per step, measured from odometry. Commands below MIN_DRIVE_SPEED don't roll it.
TARGET_STEP = 0.02
SPEED_GAIN = 0.15
START_CMD = 0.008
MIN_DRIVE_SPEED = 0.007
MAX_CMD = 0.015
LOOKAHEAD = 0.10

# Turn in place (Lab 3's rule): above TURN_THRESHOLD of heading error, stop and turn,
# until within TURN_EXIT.
TURN_THRESHOLD = np.radians(35)
TURN_EXIT = np.radians(10)

# Braking: stop driving once the model predicts the ball would coast to within
# COAST_MARGIN of the stop point anyway.
COAST_MARGIN = 0.01
COAST_HORIZON = 10

# Only count "driven forward but not moving" as stuck when a wall is this close to the
# ball's centre; in open corridor it is just the motor stalling at low speed.
WALL_CHECK = 0.07

VERBOSE = True

occupancy = build_occupancy_grid()
learned_dynamics = make_learned_dynamics(WEIGHTS)
# Swapped for analytic_dynamics by --analytic, to compare against the Lab 3 model.
model = learned_dynamics


def make_sim_env():
    return SpheroEnv(
        dt=STEP_PERIOD,
        max_steps=MAX_STEPS,
        vel_limit=0.15,
        world_width=1.25,
        world_height=1.25,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        occupancy_grid=occupancy,
        grid_resolution=0.125,
        dynamics=model,
        # Real odometry noise is ~1 cm/step: the learned model's held-out x error (1.4 cm)
        # bounds it. Lab 3's 5 cm swamped the 2-5 cm the ball moves per 0.3 s step.
        obs_noise_std_pos=SIM_POS_NOISE,
        # The model turns speed into distance with dt=SPEED_DT rather than Lab 3's 0.1,
        # so scale the speed noise to keep Lab 3's per-step odometry drift.
        process_noise_std_speed=0.005 * 0.1 / SPEED_DT,
        process_noise_std_heading=0.01,
        obs_noise_std_vel=0.025,
        render_mode="human",
        window_size=(800, 800),
    )


def make_real_env(api):
    return Robot(
        api=api,
        dt=0.1,
        max_steps=MAX_STEPS,
        vel_limit=0.15,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        # Robot adds artificial Gaussian noise (default 5 cm) to what step() returns,
        # on top of the real odometry. Switched off: the real odometry is ~1 cm.
        obs_noise_std_pos=0.0,
        obs_noise_std_vel=0.0,
        render_mode="human",
        window_size=(800, 800),
    )


@contextmanager
def managed_env(sim: bool):
    if sim:
        env = make_sim_env()
        env.set_log_path(os.path.join(LOG_DIR, "lab4_sim.csv"))
        env.start_logging()
        try:
            yield env
        finally:
            env.stop_logging()
            env.close()
    else:
        with ExitStack() as stack:
            toy, _ = scan_and_connect()
            print(f"Selected: {toy.name}")
            api = stack.enter_context(SpheroEduAPI(toy))
            api.reset_aim()
            env = make_real_env(api)
            # One file per run, so every real run can be kept as training data.
            kind = "learned" if model is learned_dynamics else "analytic"
            env.set_log_path(os.path.join(LOG_DIR, f"lab4_real_{kind}_{time.strftime('%Y%m%d-%H%M%S')}.csv"))
            env.start_logging()
            try:
                yield env
            finally:
                env.close()
                env.stop_logging()


def make_ekf(is_sim):
    ekf = EKF(dt=SPEED_DT, dynamics_fn=model)
    # Q from the learned model's held-out error. It is measured along/sideways to the
    # ball's heading, so use the same average position variance for both x and y.
    rmse = np.maximum(np.load(WEIGHTS)["rmse"], 1e-3)
    pos_q = float(np.mean(rmse[:2] ** 2))
    ekf.Q = np.diag([pos_q, pos_q, rmse[2] ** 2, rmse[3] ** 2])
    # Lab 3 used 0.02 (~14 cm) on the real robot, which made the filter believe the model
    # over the odometry: a ball stuck on a wall was "estimated" all the way to the goal.
    pos_var = (SIM_POS_NOISE if is_sim else REAL_POS_NOISE) ** 2
    heading_var = 0.025 ** 2 if is_sim else 0.05
    # Measured speed is a difference of two positions divided by SPEED_DT.
    ekf.R = np.diag([pos_var, pos_var, heading_var, 2 * pos_var / SPEED_DT ** 2])
    ekf.state_est = np.array([START_XY[0], START_XY[1], 0.0, 0.0])
    ekf.P = np.eye(4) * 1e-3
    return ekf


def safe_action(action, heading, vel_limit):
    action = np.asarray(action, dtype=float)
    if not np.all(np.isfinite(action)):
        return np.array([0.0, heading], dtype=np.float32)
    speed = float(np.clip(action[0], 0.0, vel_limit))
    if 0.0 < speed < MIN_DRIVE_SPEED:
        speed = MIN_DRIVE_SPEED
    return np.array([speed, action[1]], dtype=np.float32)


def lookahead_point(pos, a, b):
    """Aim LOOKAHEAD metres ahead along segment a->b instead of at b itself, so the ball
    is pulled back onto the line rather than swinging its heading around near b."""
    ab = b - a
    length = float(np.hypot(*ab))
    if length < 1e-6:
        return b
    u = ab / length
    s = float(np.clip((pos - a) @ u, 0.0, length)) + LOOKAHEAD
    return b if s >= length else a + u * s


def coast_distance(est):
    """How far the model predicts the ball rolls if told to stop now."""
    state = np.asarray(est, dtype=float)
    action = np.array([0.0, state[2]])
    total = 0.0
    for _ in range(COAST_HORIZON):
        nxt = np.asarray(model(state, action), dtype=float)
        total += float(np.hypot(*(nxt[:2] - state[:2])))
        state = nxt
    return total


class Driver:
    """Heading from the aim point; speed from a loop on measured movement per step."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cmd = 0.0
        self.turning = False
        self.turn_target = 0.0

    def act(self, est, aim, stop_dist=None):
        heading = est[2]
        desired = float(np.arctan2(aim[0] - est[0], aim[1] - est[1]))

        if self.turning and abs(wrap_angle(self.turn_target - heading)) >= TURN_EXIT:
            return np.array([0.0, self.turn_target])
        self.turning = False
        if abs(wrap_angle(desired - heading)) > TURN_THRESHOLD:
            self.turning, self.turn_target, self.cmd = True, desired, 0.0
            return np.array([0.0, desired])

        if stop_dist is not None and coast_distance(est) >= stop_dist - COAST_MARGIN:
            self.cmd = 0.0
            return np.array([0.0, desired])

        moving = est[3] * SPEED_DT
        if self.cmd == 0.0:
            self.cmd = START_CMD if moving < TARGET_STEP else 0.0
        else:
            self.cmd += SPEED_GAIN * (TARGET_STEP - moving)
            if self.cmd < MIN_DRIVE_SPEED:
                self.cmd = 0.0
        self.cmd = min(self.cmd, MAX_CMD)
        return np.array([self.cmd, desired])


def near_wall(planner, pos):
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        row, col = planner.world_to_occ(pos + WALL_CHECK * np.array([np.sin(a), np.cos(a)]))
        if occupancy[row, col]:
            return True
    return False


def corner_flags(waypoints):
    # The first waypoint is the current cell's centre (after a replan, settle there first)
    # and the last is the goal: both are stop points.
    flags = [True] * len(waypoints)
    for i in range(1, len(waypoints) - 1):
        a = np.asarray(waypoints[i]) - np.asarray(waypoints[i - 1])
        b = np.asarray(waypoints[i + 1]) - np.asarray(waypoints[i])
        turn = abs(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
        flags[i] = bool(turn > CORNER_ANGLE)
    return flags


def control_loop(env):
    env.reset(seed=SEED)
    is_sim = isinstance(env, SpheroEnv)

    if is_sim:
        env.state_true[0:4] = np.array([START_XY[0], START_XY[1], 0.0, 0.0])
        env.state_odom[0:4] = np.array([START_XY[0], START_XY[1], 0.0, 0.0])
        frame_offset = np.zeros(2)
    else:
        frame_offset = START_XY - np.asarray(env.state_odom[:2], dtype=float)
        print(f"Shifting robot odometry by {tuple(np.round(frame_offset, 3))} into the map frame")

    ekf = make_ekf(is_sim)
    est = ekf.state_est.copy()
    learned_active = model is learned_dynamics
    gap_steps = 0
    prev_raw_xy = START_XY.copy()
    last_step_time = 0.0

    planner = Planner(map=occupancy, dt=env.dt)
    waypoints = planner.plan(est, env.goal_pos, margin_cells=0)
    corners = corner_flags(waypoints)
    driver = Driver()
    print(f"Planned {len(waypoints)} waypoints")

    os.makedirs(LOG_DIR, exist_ok=True)
    suffix = "" if model is learned_dynamics else "_analytic"
    csv_file = open(os.path.join(HERE, f"{STUDENT_ID}_lab4{suffix}.csv"), "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["sim_x", "sim_y", "real_x", "real_y"])

    def step(action):
        """Pace, act, then filter. Returns (collided, raw measurement, action sent)."""
        nonlocal est, prev_raw_xy, last_step_time, learned_active, gap_steps
        action = safe_action(action, est[2], env.vel_limit)

        if not is_sim:
            wait = STEP_PERIOD - (time.time() - last_step_time)
            if wait > 0:
                time.sleep(wait)
            last_step_time = time.time()

        ekf.predict(action)
        obs, _, _, _, info = env.step(action)

        raw = np.asarray(obs, dtype=float)[:4].copy()
        raw[0:2] += frame_offset
        raw[3] = measured_speed(prev_raw_xy, raw[:2])
        prev_raw_xy = raw[:2].copy()

        est = ekf.update(raw)[0].copy()

        gap_steps = gap_steps + 1 if np.hypot(*(est[:2] - raw[:2])) > MAX_EST_GAP else 0
        if gap_steps >= MAX_GAP_STEPS:
            print("  *** SAFETY: estimate drifted from the odometry - resyncing"
                  f"{' and switching to the analytic model' if learned_active else ''} ***")
            learned_active = False
            ekf.dynamics_fn = None
            ekf.state_est = raw.copy()
            ekf.P = np.eye(4) * 1e-2
            est = raw.copy()
            gap_steps = 0

        env.render()
        sim_xy = env.state_true[0:2] if is_sim else ("", "")
        writer.writerow([sim_xy[0], sim_xy[1], est[0], est[1]])

        if VERBOSE:
            print(f"  raw=({raw[0]:.3f},{raw[1]:.3f}) est=({est[0]:.3f},{est[1]:.3f}) "
                  f"spd={est[3]:.3f} cmd=({action[0]:.3f},{np.degrees(action[1]):.0f}deg)"
                  f"{'' if learned_active else ' [analytic]'}")

        collided = bool(info.get("collision", False)) or (len(obs) > 4 and bool(obs[4]))
        return collided, raw, action

    steps = 0
    replans = 0
    reached_goal = False
    loop_start = time.time()
    try:
        wp_index = 0
        pushing = []  # raw positions over the current run of forward commands
        while wp_index < len(waypoints) and steps < MAX_STEPS:
            waypoint = np.asarray(waypoints[wp_index], dtype=float)
            seg_start = np.asarray(waypoints[wp_index - 1], dtype=float) if wp_index else est[:2].copy()
            wp_steps = 0
            collided = False

            while wp_steps < MAX_STEPS_PER_WAYPOINT and steps < MAX_STEPS:
                to_wp = float(np.hypot(*(est[:2] - waypoint)))
                if corners[wp_index] and to_wp < WAYPOINT_TOLERANCE:
                    # Within tolerance of a stop point: stop and hold heading while it settles.
                    # Steering at a target ~1 cm away aims wherever the position noise points.
                    driver.reset()
                    action = np.array([0.0, est[2]])
                else:
                    action = driver.act(est, lookahead_point(est[:2], seg_start, waypoint),
                                        to_wp if corners[wp_index] else None)
                collided, raw, sent = step(action)
                wp_steps += 1
                steps += 1

                pushing = pushing + [raw[:2]] if sent[0] >= MIN_DRIVE_SPEED else []
                if len(pushing) > STUCK_STEPS and near_wall(planner, raw[:2]):
                    if np.hypot(*(pushing[-1] - pushing[-1 - STUCK_STEPS])) < STUCK_DIST:
                        print(f"  *** SAFETY: ball pinned on a wall - treating as a collision ***")
                        collided = True
                if collided:
                    break
                near = np.hypot(est[0] - waypoint[0], est[1] - waypoint[1]) < WAYPOINT_TOLERANCE
                settled = est[3] * SPEED_DT < SETTLED_MOVE
                # A ~5 cm step can jump right over a 3 cm circle, so a pass-through
                # waypoint also counts once the ball is beyond it along the segment.
                ab = waypoint - seg_start
                passed = ab @ ab > 1e-12 and (est[:2] - seg_start) @ ab >= ab @ ab
                if (near and settled) or (not corners[wp_index] and (near or passed)):
                    print(f"Reached waypoint {wp_index}: {waypoint}")
                    reached_goal = wp_index == len(waypoints) - 1
                    break

            if reached_goal:
                print("Goal reached.")
                break

            if collided and replans < MAX_REPLANS:
                # No reverse back-off: at 0.3 s steps it flung the ball across the maze.
                # The new plan starts at the centre of the current cell, so the robot
                # first settles back there, away from the wall.
                print(f"Collision near waypoint {wp_index} - replanning from the current cell")
                pushing = []
                try:
                    waypoints = planner.plan(est, env.goal_pos, margin_cells=0)
                    corners = corner_flags(waypoints)
                    wp_index = 0
                    replans += 1
                    driver.reset()
                    print(f"Replanned {len(waypoints)} waypoints (replan #{replans})")
                    continue
                except (ValueError, RuntimeError) as e:
                    print(f"Replan failed: {e} - continuing with the old plan")

            if wp_steps >= MAX_STEPS_PER_WAYPOINT:
                print(f"Timed out on waypoint {wp_index}: {waypoint}")
            wp_index += 1

        if not reached_goal:
            d = np.hypot(est[0] - env.goal_pos[0], est[1] - env.goal_pos[1])
            print(f"Goal not reached. Final distance: {d:.3f} m")
    finally:
        if steps:
            timing = f"{steps} steps, {(time.time() - loop_start) / steps:.3f} s per step on average"
            print(timing)
            with open(os.path.join(LOG_DIR, "lab4_timing.txt"), "a") as f:
                f.write(f"{'sim' if is_sim else 'real'}: {timing}\n")
        if is_sim:
            d = np.hypot(env.state_true[0] - env.goal_pos[0], env.state_true[1] - env.goal_pos[1])
            print(f"True (sim) final distance to goal: {d:.3f} m")
        csv_file.close()
        env.emergency_stop()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run in the simulator")
    parser.add_argument("--analytic", action="store_true",
                        help="Use the Lab 3 analytic model instead of the learned one (for comparison)")
    args = parser.parse_args(argv)
    if args.analytic:
        global model
        model = analytic_dynamics
    with managed_env(args.sim) as env:
        control_loop(env)


if __name__ == "__main__":
    main()
