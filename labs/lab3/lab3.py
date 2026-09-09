from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

import argparse
import numpy as np

from Planner import *
from EKF import dynamics

from sphero_env.envs.custom_maze_full import build_occupancy_grid


from contextlib import ExitStack, contextmanager

LAB1_SEED = 0
MAX_STEPS = 5000
map = build_occupancy_grid()


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# def dynamics(state, action):
#     """
#     Unicycle model with a turn-RATE command (not absolute heading):
#         action: [speed_cmd, turn_rate_cmd]
#     """
#     x, y, heading, speed = state
#     speed_cmd, turn_rate_cmd = action
#     heading_new = wrap_angle(heading + turn_rate_cmd * 0.1)
#     speed_new = np.clip(speed + speed_cmd * 0.1, 0, 1.0)
#     x_new = x + speed_new * np.sin(heading_new) * 0.1
#     y_new = y + speed_new * np.cos(heading_new) * 0.1
#     return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)


class Controller:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.MAX_DECEL = 0.01   # must match EKF.py
        selfDT = 2.95    

        # --- NEW TUNING PARAMETERS FOR LEGO SURFACE ---
        self.MAX_ACCEL_STEP = 0.02  # Max change in speed per step (TUNE THIS: lower = less slip)
        self.MAX_DECEL_STEP = 0.04  # Sphero can usually brake slightly harder than it accelerates

    # def compute_action(self, state, waypoint):
    def compute_action(self, env, obs, step):
        """Return action = [speed_cmd, heading_cmd] for the current observation."""
        # Default: random action for testing. Replace with your control law.
        # return _rng.uniform(low=-1.0, high=1.0, size=2)
        global hold_heading

        
        
        # --- P-controller-to-goal skeleton (uncomment and tune) ---
        dx = env.goal_pos[0] - obs[0]
        dy = env.goal_pos[1] - obs[1]
        current_speed = obs[3]

        dist = np.hypot(dx, dy)
        KP = 0.09
        KD = 0.45 # tune this
        # brake_dist = 1.5*(current_speed ** 2) / (2 * MAX_DECEL * DT)

        if dist < 0.055:
            return ["Stop","Stop"]  # Goal Reached

        else:
            heading_cmd = np.arctan2(dx, dy)   # 0 rad = +y convention -> atan2(dx, dy)
            hold_heading = heading_cmd #Update previous angle 

            # Calculate shortest angular error between current and desired heading
            # Normalizes the difference to be between -pi and pi
            heading_error = (heading_cmd - obs[2] + np.pi) % (2 * np.pi) - np.pi
            
            # 4. Calculate Raw Speed Command (Your PD logic)
            raw_speed_cmd = KP * dist - KD * current_speed
            
            # 5. Heading-Coupled Speed Limiting
            # Slow down if we need to turn. The larger the turn, the slower we go.
            # np.pi/4 (45 degrees) is used as a scaling factor here.
            turn_penalty = max(0.0, 1.0 - (abs(heading_error) / (np.pi / 4)))
            raw_speed_cmd *= turn_penalty
            target_speed = np.clip(raw_speed_cmd, 0.00, env.vel_limit)
            
            # 6. Slew Rate Limiting (Anti-Slip)
            # Don't let the requested speed jump too fast from the previous speed
            speed_cmd = np.clip(
                target_speed, 
                prev_speed_cmd - self.MAX_DECEL_STEP, 
                prev_speed_cmd + self.MAX_ACCEL_STEP
            )
            
            # Save for next time step
            prev_speed_cmd = speed_cmd

        return np.array([speed_cmd, heading_cmd])



def make_sim_env():
    return SpheroEnv(
        dt=0.1,
        max_steps=5000,
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

    control_env.state_true[0:3] = np.array([-0.5, -0.5, 0.0])
    control_env.state_odom[0:3] = np.array([-0.5, -0.5, 0.0])
    rng = np.random.default_rng(LAB1_SEED)

    controller = Controller(dt=control_env.dt)
    planner = Planner(map=control_env.occupancy_grid, dt=control_env.dt)

    # Plan from the pose we just set above, not the stale obs returned by
    # reset() before the overwrite.
    start_state = np.array([-0.5, -0.5, 0.0, 0.0])

    # margin_cells=0: this maze's corridors are only as wide as a single
    # connector cell, so any wall inflation blocks the only free passage.
    # Wall-clipping is instead handled at runtime via collision + replan.
    waypoints = planner.plan(start_state, control_env.goal_pos, margin_cells=0)

    print(f"Planned {len(waypoints)} waypoints")
    for i, wp in enumerate(waypoints):
        print(f"  wp{i}: {wp}")

    steps = 0
    MAX_STEPS_PER_WAYPOINT = 200
    MAX_REPLANS = 10
    replans = 0
    reached_goal = False

    wp_index = 0
    while wp_index < len(waypoints) and steps < MAX_STEPS:
        waypoint = waypoints[wp_index]
        wp_steps = 0
        collided = False

        while wp_steps < MAX_STEPS_PER_WAYPOINT and steps < MAX_STEPS:
            action = controller.compute_action(obs, waypoint)
            obs, _, terminated, truncated, info = control_env.step(action)
            control_env.render()

            wp_steps += 1
            steps += 1

            collided = info.get("collision", False) if isinstance(info, dict) else False
            if not collided and len(obs) > 4:
                collided = bool(obs[4])
            if collided:
                break

            dist_to_goal_sq = (obs[0]-control_env.goal_pos[0])**2 + (obs[1]-control_env.goal_pos[1])**2
            if dist_to_goal_sq < control_env.goal_tolerance**2:
                reached_goal = True
                break

            dist_to_wp_sq = (obs[0]-waypoint[0])**2 + (obs[1]-waypoint[1])**2
            if dist_to_wp_sq < control_env.goal_tolerance**2:
                print(f"Reached waypoint {wp_index}: {waypoint}")
                break

        if reached_goal:
            print("Goal reached.")
            break

        if collided and replans < MAX_REPLANS:
            print(f"Collision near wp{wp_index}, step {wp_steps} - backing off and replanning")

            # Back off for several steps, not just one, so the ball
            # actually clears the wall before replanning
            for _ in range(5):
                obs, _, terminated, truncated, info = control_env.step(
                    np.array([-0.1, 0.0], dtype=np.float32))
                control_env.render()
                steps += 1

            try:
                waypoints = planner.plan(obs, control_env.goal_pos, margin_cells=0)
                wp_index = 0
                replans += 1
                print(f"Replanned {len(waypoints)} waypoints "
                      f"(replan #{replans}) from ({obs[0]:.3f}, {obs[1]:.3f})")
                continue
            except (ValueError, RuntimeError) as e:
                print(f"Replan failed: {e} - continuing with old plan")

        if wp_steps >= MAX_STEPS_PER_WAYPOINT:
            print(f"Timed out on waypoint {wp_index}: {waypoint} "
                  f"(stuck at {obs[0]:.3f}, {obs[1]:.3f})")

        wp_index += 1

    if not reached_goal:
        final_dist = np.hypot(obs[0]-control_env.goal_pos[0], obs[1]-control_env.goal_pos[1])
        print(f"Path complete but goal not reached. Final distance: {final_dist:.3f} m")

    control_env.emergency_stop()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run simulation")
    args = parser.parse_args(argv)

    with managed_env(args.sim) as control_env:
        control_loop(control_env)


if __name__ == "__main__":
    main()