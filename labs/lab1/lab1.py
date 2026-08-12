# Import necessary libraries
from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

import argparse
import time
import numpy as np
import dynamics as dyn
from dynamics import *

from contextlib import ExitStack, contextmanager

LAB1_SEED = 0

def make_sim_env():
    sim_dt = 0.1
    return SpheroEnv(
        dt=sim_dt,
        max_steps=5000,
        vel_limit=0.15,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        occupancy_grid=None,
        dynamics=make_dynamics(dt=sim_dt),
        obs_noise_std_pos=0.00,
        process_noise_std_speed=0.00,
        process_noise_std_heading=0.00,
        obs_noise_std_vel=0,
        render_mode="human",
        window_size=(800, 800),
    )

def make_real_env(api):
    return Robot(
        api=api,
        dt=0.1,
        max_steps=5000,
        vel_limit=0.015,
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
        sim_env.set_log_path("logs/lab1_sim.csv")
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
            real_env = make_real_env(api)
            real_env.set_log_path("logs/lab1_real.csv")

            real_env.start_logging()
            try:
                yield real_env
            finally:
                real_env.close()
                real_env.stop_logging()

def control_loop(control_env):

    obs, _ = control_env.reset(seed=LAB1_SEED)
    rng = np.random.default_rng(LAB1_SEED)

    start_time = time.perf_counter()
    last_step_time = start_time
    step_count = 0
    min_step_time = float('inf')
    max_step_time = 0.0

    # Bool for goal reached
    goal_reached = False

    while goal_reached == False:

        #action = rng.uniform(low=-1.0, high=1.0, size=2)  # Random action for testing

        ## Step the environment with the action and render the result
        x_meas, y_meas, heading_meas, speed_meas, collision_flag = obs
        
        # Decides action based on the observation
        # Checks if ball is within the goal tolerance
        if x_meas < control_env.goal_pos[0] - control_env.goal_tolerance or x_meas > control_env.goal_pos[0] + control_env.goal_tolerance or y_meas < control_env.goal_pos[1] - control_env.goal_tolerance or y_meas > control_env.goal_pos[1] + control_env.goal_tolerance:
            # Ball is outside the goal tolerance, move towards the goal

            # Calculates the heading angle towards the goal
            heading_angle = np.arctan2(control_env.goal_pos[0] - x_meas, control_env.goal_pos[1] - y_meas)

            # Creates action
            action = [0.015, heading_angle]

        else:
            input("Done. Press Enter to exit...")
            print("Ball is within the goal tolerance. Stopping the robot.")

            # Ball is within the goal tolerance, stop moving
            action = [0.0, 0.0]  # Stop moving

            # Exit the loop if the ball is within the goal tolerance
            goal_reached = True
            

        # Updates values
        obs, _, terminated, truncated, info = control_env.step(action)
        
        # Execute the action in the environment
        control_env.render()

        current_time = time.perf_counter()
        step_duration = current_time - last_step_time
        last_step_time = current_time
        min_step_time = min(min_step_time, step_duration)
        max_step_time = max(max_step_time, step_duration)

        step_count += 1
        if step_count % 10 == 0:
            total_time = current_time - start_time
            actual_rate = step_count / total_time
            print(f"actual average sampling rate = {actual_rate:.3f} Hz ({step_count} steps in {total_time:.3f} s)")
            print(f"step duration min={min_step_time:.3f}s max={max_step_time:.3f}s")

    total_time = time.perf_counter() - start_time
    if step_count > 0:
        actual_rate = step_count / total_time
        print(f"actual average sampling rate = {actual_rate:.3f} Hz ({step_count} steps in {total_time:.3f} s)")
        print(f"step duration min={min_step_time:.3f}s max={max_step_time:.3f}s")

    control_env.emergency_stop()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run simulation")
    args = parser.parse_args(argv)

    with managed_env(args.sim) as control_env:
        # Print the environment timestep once (avoid printing inside dynamics)
        print("env.dt =", control_env.dt, "s ->", 1.0 / control_env.dt, "Hz")
        print("target polling rate =", dyn.TARGET_POLL_HZ, "Hz")
        control_loop(control_env)

if __name__ == "__main__":
    main()
