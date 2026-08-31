import time
import argparse
import importlib
import numpy as np
import pygame
from contextlib import ExitStack, contextmanager

from EKF import *
import controller

from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.connect import scan_and_connect
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv


# Square path: (0,0) -> (0,0.5) -> (0.5,0.5) -> (0.5,0) -> (0,0)
WAYPOINTS = [
    (0.0, 0.0),
    (0.0, 0.5),
    (0.5, 0.5),
    (0.5, 0.0),
    (0.0, 0.0),
]
START_INDEX = 1   # skip (0,0), we start sitting on it


def set_goal(env, robot_env, waypoint):
    """Point the environments at the current waypoint so controller.py sees it."""
    env.goal_pos = np.array(waypoint, dtype=float)
    if robot_env is not None:
        robot_env.goal_pos = np.array(waypoint, dtype=float)


def control_loop(env, ekf, robot_env=None, action=None, moving=False):
    """
    Step both environments, run the EKF predict/update cycle, and visualise
    the belief state.
    """
    sim_obs, _, _, _, sim_info = env.step(action)

    if robot_env is not None:
        robot_obs, _, _, _, robot_info = robot_env.step(action)

    if robot_env is not None and not moving:
        robot_env.emergency_stop()

    # EKF predict and update
    ekf.predict(action)
    if robot_env is not None:
        ekf.update(robot_obs)
    else:
        ekf.update(sim_obs)

    env.vis.set_belief(ekf.state_est, ekf.P)
    env.render()


def make_sim_env():
    return SpheroEnv(
        dt=0.1,
        max_steps=5000,
        vel_limit=0.15,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        occupancy_grid=None,
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
def managed_sim_env():
    env = make_sim_env()
    env.set_log_path("logs/lab2_sim.csv")
    env.reset()
    env.start_logging()
    try:
        yield env
    finally:
        env.stop_logging()
        env.close()


@contextmanager
def managed_robot_env():
    selected_toy, _ = scan_and_connect()
    print(f"Selected: {selected_toy.name}")

    with SpheroEduAPI(selected_toy) as api:
        api.reset_aim()
        robot_env = make_real_env(api)
        robot_env.set_log_path("logs/lab2_robot.csv")
        robot_env.reset()
        robot_env.start_logging()
        try:
            yield robot_env
        finally:
            robot_env.stop_logging()
            robot_env.emergency_stop()
            robot_env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lab 2: EKF state estimation with waypoint navigation")
    parser.add_argument("--sim", action="store_true",
                        help="Run simulator-only mode (no robot connection)")
    return parser.parse_args()


def main(sim_only=False):
    with ExitStack() as stack:
        env = stack.enter_context(managed_sim_env())
        robot_env = stack.enter_context(managed_robot_env()) if not sim_only else None

        env.render()

        ekf = EKF(dt=0.1)

        # Seed the EKF with the real starting pose if we have a robot
        if robot_env is not None:
            start = np.asarray(robot_env.get_odom_state(), dtype=float)
            ekf.state_est = np.array([start[0], start[1], start[2], 0.0], dtype=float)

        wp_index = START_INDEX
        set_goal(env, robot_env, WAYPOINTS[wp_index])

        stack.callback(pygame.quit)
        stack.callback(lambda: print("Stopped. Lab 2 closed."))

        mode_text = "Simulator only" if sim_only else "Robot + Simulator"
        print(f"\nWaypoint navigation ready ({mode_text}):")
        print("  R     = reload controller.py and restart the path")
        print("  SPACE = emergency stop")
        print("  Q     = quit\n")
        print(f"Heading to waypoint {wp_index}: {WAYPOINTS[wp_index]}")

        running = True
        moving = False
        step = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        if robot_env is not None:
                            robot_env.emergency_stop()
                        moving = False
                        print("Emergency stop")

                    elif event.key == pygame.K_r:
                        # Hot-reload controller.py so gain edits take effect
                        importlib.reload(controller)
                        wp_index = START_INDEX
                        set_goal(env, robot_env, WAYPOINTS[wp_index])
                        step = 0
                        print("Reloaded controller.py, restarting path")
                        print(f"Heading to waypoint {wp_index}: {WAYPOINTS[wp_index]}")

            if wp_index >= len(WAYPOINTS):
                # Path complete - hold position
                action = np.array([0.0, float(ekf.state_est[2])], dtype=np.float32)
                moving = False
            else:
                # Drive off the EKF belief, not a single noisy reading
                result = controller.compute_action(env, ekf.state_est, step)

                # controller.py returns ["Stop", "Stop"] once it's within its
                # goal tolerance - use that as the arrival signal.
                if isinstance(result, list):
                    print(f"Reached waypoint {wp_index}: {WAYPOINTS[wp_index]}  "
                          f"(est: {ekf.state_est[0]:.3f}, {ekf.state_est[1]:.3f})")
                    wp_index += 1

                    if wp_index < len(WAYPOINTS):
                        set_goal(env, robot_env, WAYPOINTS[wp_index])
                        print(f"Heading to waypoint {wp_index}: {WAYPOINTS[wp_index]}")
                        result = controller.compute_action(env, ekf.state_est, step)
                    else:
                        print("Path complete.")
                        result = None

                if result is None or isinstance(result, list):
                    action = np.array([0.0, float(ekf.state_est[2])], dtype=np.float32)
                    moving = False
                else:
                    action = np.asarray(result, dtype=np.float32)
                    moving = True

            control_loop(env, ekf=ekf, robot_env=robot_env, action=action, moving=moving)
            step += 1

            time.sleep(0.01 if robot_env is not None else 0.1)


if __name__ == "__main__":
    args = parse_args()
    main(sim_only=args.sim)