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


@contextmanager
def managed_env(sim: bool):
    if sim:
        sim_env = make_sim_env()
        sim_env.set_log_path("logs/lab3_sim.csv")
        sim_env.start_logging()
        try:
            yield sim_env
        finally:
            sim_env.stop_logging()
            sim_env.close()
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
        ekf = EKF(dt=2.95)
        ekf.Q = np.diag([0.01, 0.01, 0.02, 0.01])
        ekf.R = np.diag([0.01, 0.01, 0.04, 0.02])

    ekf.state_est = start_state.astype(float).copy()
    ekf.P = np.eye(4) * 1e-3
    est = ekf.state_est.copy()

    waypoints = planner.plan(start_state, control_env.goal_pos, margin_cells=0)

    print(f"Planned {len(waypoints)} waypoints")
    for i, wp in enumerate(waypoints):
        print(f"  wp{i}: {wp}")

    steps = 0
    WAYPOINT_TOLERANCE = 0.02
    MAX_STEPS_PER_WAYPOINT = int(60.0 / control_env.dt)
    MAX_REPLANS = 10
    replans = 0
    reached_goal = False

    collisions_at_wp = {}
    MAX_COLLISIONS_PER_WP = 2

    # Alternate escape direction each time, so a failed escape isn't simply
    # repeated. Last run reversed to the identical position twice running.
    escape_sign = 1

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
                break

            if collided:
                key = (round(float(waypoint[0]), 3), round(float(waypoint[1]), 3))
                collisions_at_wp[key] = collisions_at_wp.get(key, 0) + 1

                if collisions_at_wp[key] > MAX_COLLISIONS_PER_WP:
                    print(f"Waypoint {waypoint} has collided "
                          f"{collisions_at_wp[key]} times - skipping it")
                    wp_index += 1
                    controller.reset()
                    continue

                if replans < MAX_REPLANS:
                    print(f"Collision near wp{wp_index}, step {wp_steps} - escaping")

                    pos_before = np.array([est[0], est[1]])

                    # Reverse FIRST to break contact. Pivoting while wedged
                    # against a wall does nothing - last run's escape left
                    # the ball at the identical position twice running.
                    for _ in range(8):
                        straight_back = np.array([-0.08, est[2]], dtype=np.float32)
                        ekf.predict(straight_back)
                        obs, _, terminated, truncated, info = control_env.step(straight_back)
                        est = ekf.update(to_map(obs))[0]
                        control_env.render()
                        log_row()
                        steps += 1

                    # Now that contact should be broken, turn away and back
                    # off further, alternating side each escape so a failed
                    # attempt isn't simply repeated.
                    escape_heading = wrap_angle(est[2] + escape_sign * np.pi / 2)
                    escape_sign *= -1

                    for _ in range(6):
                        away = np.array([-0.08, escape_heading], dtype=np.float32)
                        ekf.predict(away)
                        obs, _, terminated, truncated, info = control_env.step(away)
                        est = ekf.update(to_map(obs))[0]
                        control_env.render()
                        log_row()
                        steps += 1

                    moved = np.hypot(est[0]-pos_before[0], est[1]-pos_before[1])
                    if moved < 0.02:
                        print(f"  escape moved only {moved:.3f}m - ball is wedged, "
                              f"skipping this waypoint")
                        wp_index += 1
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
    args = parser.parse_args(argv)

    with managed_env(args.sim) as control_env:
        control_loop(control_env)


if __name__ == "__main__":
    main()