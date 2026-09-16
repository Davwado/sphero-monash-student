# Import necessary libraries
from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

import argparse
import csv
import numpy as np

from types import SimpleNamespace

from Planner import *
from EKF import EKF
from sphero_env.envs.custom_maze_full import build_occupancy_grid

import controller

from contextlib import ExitStack, contextmanager

LAB1_SEED = 0
MAX_STEPS = 5000
map = build_occupancy_grid()

# Replace with your actual student ID before submitting.
STUDENT_ID = "your_id_here"

START_XY = np.array([-0.5, -0.5])

SIM_DT = 0.1

SIM_MAX_TURN_RATE = 3.0
SIM_MAX_ACCEL = 0.3
SIM_MAX_DECEL = 0.5

# Set False to silence the per-step diagnostic print.
VERBOSE = True

# Warn when the filtered estimate and the raw reading disagree by more than
# this. Not a rejection gate - just a flag, so a broken sensor shows up in
# the log instead of being silently smoothed into a fake clean run.
DIVERGENCE_WARN = 0.25


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def dynamics(state, action):
    """Rate-limited unicycle plant for the simulator.

    action: [speed_cmd, heading_cmd]
    state:  [x, y, heading, speed]
    """
    x, y, heading, speed = state
    speed_cmd, heading_cmd = action

    heading_error = wrap_angle(heading_cmd - heading)
    max_turn = SIM_MAX_TURN_RATE * SIM_DT
    heading_new = wrap_angle(heading + np.clip(heading_error, -max_turn, max_turn))

    speed_target = speed_cmd * max(0.0, float(np.cos(heading_error)))
    speed_error = speed_target - speed
    max_step = (SIM_MAX_ACCEL if speed_error > 0 else SIM_MAX_DECEL) * SIM_DT
    speed_new = float(np.clip(speed + np.clip(speed_error, -max_step, max_step), 0.0, 1.0))

    x_new = x + speed_new * np.sin(heading_new) * SIM_DT
    y_new = y + speed_new * np.cos(heading_new) * SIM_DT
    return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)


def make_sim_env():
    return SpheroEnv(
        dt=SIM_DT,
        max_steps=MAX_STEPS,
        vel_limit=0.15,
        world_width=1.25,
        world_height=1.25,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        occupancy_grid=map,
        grid_resolution=0.125,
        dynamics=dynamics,
        obs_noise_std_pos=0.05,
        process_noise_std_speed=0.005,
        process_noise_std_heading=0.01,
        obs_noise_std_vel=0.025,
        render_mode="human",
        window_size=(800, 800),
    )


def make_real_env(api):
    return Robot(
        api=api,
        dt=0.1,
        max_steps=5000,
        vel_limit=0.15,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        render_mode="human",
        window_size=(800, 800),
    )


def _fast_managed_api():
    """Lazy import so the fast_comms path is only pulled in when --fast-comms
    is actually passed - it lives alongside lab2, not lab3.

    Unlike lab1/lab2, control_loop() below actually reads info["collision"]
    (and obs[4]) to trigger the escape/replan logic when the ball hits a
    maze wall - so unlike fast_comms' locator-only default, this needs
    accelerometer/velocity/gyroscope streamed too, or Robot._sense_collision()
    silently always reports no collision and the escape logic never fires.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab2", "fast_comms"))
    from fast_link import fast_managed_api
    return fast_managed_api(sensors=("locator", "accelerometer", "velocity", "gyroscope"))


@contextmanager
def managed_env(sim: bool, fast_comms: bool = False):
    if sim:
        sim_env = make_sim_env()
        sim_env.set_log_path("logs/lab3_sim.csv")
        sim_env.start_logging()
        try:
            yield sim_env
        finally:
            sim_env.stop_logging()
            sim_env.close()
    elif fast_comms:
        # See labs/lab2/fast_comms/fast_link.py - drives through a low-latency
        # BLE path instead of SpheroEduAPI, but presents the same interface
        # Robot expects from `api`, so make_real_env() below is unchanged.
        with _fast_managed_api() as api:
            real_env = make_real_env(api)
            real_env.set_log_path("logs/lab3_real.csv")

            real_env.start_logging()
            try:
                yield real_env
            finally:
                real_env.close()
                real_env.stop_logging()
    else:
        with ExitStack() as stack:
            selected_toy, _ = scan_and_connect()
            print(f"Selected: {selected_toy.name}")

            api = stack.enter_context(SpheroEduAPI(selected_toy))
            api.reset_aim()
            real_env = make_real_env(api)
            real_env.set_log_path("logs/lab3_real.csv")

            real_env.start_logging()
            try:
                yield real_env
            finally:
                real_env.close()
                real_env.stop_logging()


def control_loop(control_env):

    obs, _ = control_env.reset(seed=LAB1_SEED)

    is_sim = isinstance(control_env, SpheroEnv)

    if is_sim:
        control_env.state_true[0:3] = np.array([-0.5, -0.5, 0.0])
        control_env.state_odom[0:3] = np.array([-0.5, -0.5, 0.0])
        frame_offset = np.zeros(2)
    else:
        frame_offset = START_XY - np.asarray(control_env.state_odom[:2], dtype=float)
        print(f"Robot odom origin is map {tuple(np.round(-frame_offset, 3))}; "
              f"shifting readings by {tuple(np.round(frame_offset, 3))}")

    def to_map(o):
        """Reading -> map frame. Heading is assumed already aligned (0 rad =
        +y): place the ball on the start plate facing +y before reset_aim()."""
        m = np.asarray(o, dtype=float)[:4].copy()
        m[0:2] += frame_offset
        return m

    rng = np.random.default_rng(LAB1_SEED)

    planner = Planner(map=map, dt=control_env.dt)

    start_state = np.array([-0.5, -0.5, 0.0, 0.0])

    if is_sim:
        ekf = EKF(dt=control_env.dt, dynamics_fn=dynamics)
        ekf.Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
        ekf.R = np.diag([0.05**2, 0.05**2, 0.025**2, 0.025**2])
    else:
        # Real robot: ~10:1 R/Q on position. Enough to smooth the step-to-step
        # wobble, but still anchored to the measurement over time.
        #
        # A previous 500:1 ratio was a mistake: it made the filter effectively
        # deaf, so when the odometry broke (position jumped 0.7m after the
        # ball was picked up) the estimate simply integrated the motion model
        # and drew the path that had been COMMANDED, reporting "Goal reached"
        # for a run where the ball had gone the wrong way. A filter that
        # can't be contradicted turns a sensor failure into a silent one.
        ekf = EKF(dt=2.95)
        ekf.Q = np.diag([0.002, 0.002, 0.005, 0.005])
        ekf.R = np.diag([0.02, 0.02, 0.05, 0.05])

    ekf.state_est = start_state.astype(float).copy()
    ekf.P = np.eye(4) * 1e-3
    est = ekf.state_est.copy()

    waypoints = planner.plan(start_state, control_env.goal_pos, margin_cells=0)

    print(f"Planned {len(waypoints)} waypoints")
    for i, wp in enumerate(waypoints):
        print(f"  wp{i}: {wp}")

    steps = 0

    # Loosened from 0.02: a tight radius makes the ball spin in place at each
    # waypoint, because arctan2(dx, dy) gets very sensitive as dist shrinks.
    # Still well inside the corridor half-width (grid_resolution/2 = 6.25cm).
    WAYPOINT_TOLERANCE = 0.05

    MAX_STEPS_PER_WAYPOINT = int(60.0 / control_env.dt)
    MAX_REPLANS = 10
    replans = 0
    reached_goal = False
    warned_divergence = False

    csv_file = open(f"{STUDENT_ID}_lab3.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["sim_x", "sim_y", "real_x", "real_y"])

    def log_row():
        if is_sim:
            sim_xy = control_env.state_true[0:2]
        else:
            sim_xy = ("", "")
        csv_writer.writerow([sim_xy[0], sim_xy[1], est[0], est[1]])

    try:
        wp_index = 0
        while wp_index < len(waypoints) and steps < MAX_STEPS:
            waypoint = waypoints[wp_index]
            wp_steps = 0
            collided = False

            wp_target = SimpleNamespace(goal_pos=waypoint, vel_limit=control_env.vel_limit)

            while wp_steps < MAX_STEPS_PER_WAYPOINT and steps < MAX_STEPS:
                action = controller.compute_action(wp_target, est, steps)
                ekf.predict(action)
                obs, _, terminated, truncated, info = control_env.step(action)
                est = ekf.update(to_map(obs))[0]
                control_env.render()
                log_row()

                raw = to_map(obs)

                if VERBOSE:
                    print(f"  wp{wp_index} s{wp_steps}: "
                          f"raw=({raw[0]:.3f},{raw[1]:.3f}) "
                          f"est=({est[0]:.3f},{est[1]:.3f}) "
                          f"hdg={np.degrees(est[2]):.0f}° spd={est[3]:.3f} "
                          f"-> tgt=({waypoint[0]:.2f},{waypoint[1]:.2f}) "
                          f"cmd_spd={action[0]:.3f} "
                          f"cmd_hdg={np.degrees(action[1]):.0f}°")

                gap = np.hypot(raw[0]-est[0], raw[1]-est[1])
                if gap > DIVERGENCE_WARN and not warned_divergence:
                    print(f"  *** WARNING: estimate and raw reading disagree by "
                          f"{gap:.3f}m. The odometry has probably broken (this "
                          f"happens if the ball is picked up mid-run). Anything "
                          f"logged after this point is unreliable. ***")
                    warned_divergence = True

                wp_steps += 1
                steps += 1

                collided = info.get("collision", False) if isinstance(info, dict) else False
                if not collided and len(obs) > 4:
                    collided = bool(obs[4])
                if collided:
                    break

                dist_to_goal_sq = (est[0]-control_env.goal_pos[0])**2 + (est[1]-control_env.goal_pos[1])**2
                if dist_to_goal_sq < control_env.goal_tolerance**2:
                    reached_goal = True
                    break

                dist_to_wp_sq = (est[0]-waypoint[0])**2 + (est[1]-waypoint[1])**2
                if dist_to_wp_sq < WAYPOINT_TOLERANCE**2:
                    print(f"Reached waypoint {wp_index}: {waypoint}")
                    break

            if reached_goal:
                print("Goal reached.")
                if warned_divergence:
                    print("  (NOTE: a divergence warning fired earlier in this "
                          "run, so this result may not reflect where the ball "
                          "physically ended up.)")
                break

            if collided and replans < MAX_REPLANS:
                print(f"Collision near wp{wp_index}, step {wp_steps} - escaping")

                pos_before = np.array([est[0], est[1]])

                for _ in range(8):
                    straight_back = np.array([-0.08, est[2]], dtype=np.float32)
                    ekf.predict(straight_back)
                    obs, _, terminated, truncated, info = control_env.step(straight_back)
                    est = ekf.update(to_map(obs))[0]
                    control_env.render()
                    log_row()
                    steps += 1

                moved = np.hypot(est[0]-pos_before[0], est[1]-pos_before[1])

                # Zero movement means the ball is being HELD (picked up), not
                # wedged against a wall. Skipping the waypoint here throws
                # away a leg of the path and makes the next leg cut a
                # diagonal across the maze. Wait and retry the SAME waypoint.
                if moved < 0.005:
                    print(f"  ball didn't move ({moved:.3f}m) - held or stalled, "
                          f"waiting rather than skipping")
                    for _ in range(20):
                        hold = np.array([0.0, est[2]], dtype=np.float32)
                        ekf.predict(hold)
                        obs, _, terminated, truncated, info = control_env.step(hold)
                        est = ekf.update(to_map(obs))[0]
                        control_env.render()
                        log_row()
                        steps += 1
                    controller.reset()
                    continue

                try:
                    waypoints = planner.plan(est, control_env.goal_pos, margin_cells=0)
                    wp_index = 0
                    controller.reset()
                    replans += 1
                    print(f"Replanned {len(waypoints)} waypoints "
                          f"(replan #{replans}) from ({est[0]:.3f}, {est[1]:.3f})")
                    continue
                except (ValueError, RuntimeError) as e:
                    print(f"Replan failed: {e} - continuing with old plan")

            if wp_steps >= MAX_STEPS_PER_WAYPOINT:
                print(f"Timed out on waypoint {wp_index}: {waypoint} "
                      f"(stuck at {est[0]:.3f}, {est[1]:.3f})")

            wp_index += 1

        if not reached_goal:
            final_dist = np.hypot(est[0]-control_env.goal_pos[0], est[1]-control_env.goal_pos[1])
            print(f"Path complete but goal not reached. Final distance: {final_dist:.3f} m")

    finally:
        csv_file.close()
        control_env.emergency_stop()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run simulation")
    parser.add_argument("--fast-comms", action="store_true",
                        help="Real robot only: drive through labs/lab2/fast_comms' low-latency "
                             "BLE path instead of SpheroEduAPI (see fast_link.py)")
    args = parser.parse_args(argv)

    with managed_env(args.sim, fast_comms=args.fast_comms) as control_env:
        control_loop(control_env)


if __name__ == "__main__":
    main()
