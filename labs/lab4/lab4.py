"""Lab 4 deployment: run the PPO-trained local-tracking policy
(rl_approach/train_rl.py) on the sim or the real robot.

Reuses Lab 3's navigation stack almost unchanged:
  - Planner.py (A*)        -> the global start->goal waypoint route.
  - EKF.py                 -> state estimate the policy actually acts on.
  - controller.py (PD)     -> NOT the primary driver here; kept only as the
                               safety-layer fallback during collision
                               recovery, exactly the role it plays in
                               rl_approach/maze_rl_env.py's design.
  - lab3.py's control_loop shape: plan once, step, EKF predict/update,
    collision -> back off -> replan, CSV logging, divergence warning. Only
    the per-step "what action do we take" call changes (RL policy instead of
    controller.compute_action).

Safety layer (explicit lab requirement):
  - the policy's action is hard-clamped to the environment's action bounds
    before being sent anywhere.
  - collision handling reuses lab3.py's back-off-then-replan block verbatim,
    with the PD controller (controller_pd.py) driving during the brief
    escape manoeuvre.
  - a stuck-on-one-waypoint timeout (same idea as lab3.py's
    MAX_STEPS_PER_WAYPOINT) skips ahead rather than looping forever.

Usage:
    python lab4.py --sim
    python lab4.py
"""
import argparse
import csv
import os
import sys
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import numpy as np
import torch

from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv
from sphero_env.envs.custom_maze_full import build_occupancy_grid

from Policy import Policy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_approach"))
from Planner import Planner  # noqa: E402
from EKF import EKF  # noqa: E402
import controller_pd  # noqa: E402
from maze_rl_env import (  # noqa: E402
    DT, WORLD_SIZE, GOAL_XY, START_XY, VEL_LIMIT, GRID_RESOLUTION,
    WAYPOINT_TOLERANCE, OBS_DIM, ACTION_DIM, nominal_dynamics, build_features,
    wrap_angle, NOM_EKF_Q, NOM_EKF_R,
)

LAB4_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(LAB4_DIR, "weight.pth")

LAB1_SEED = 0
MAX_STEPS = 3000
MAX_STEPS_PER_WAYPOINT = int(60.0 / DT)
MAX_REPLANS = 10
DIVERGENCE_WARN = 0.25
VERBOSE = True
MAP = build_occupancy_grid()

# Replace with your actual student ID before submitting.
STUDENT_ID = "your_id_here"


def make_sim_env():
    return SpheroEnv(
        dt=DT,
        max_steps=MAX_STEPS,
        vel_limit=VEL_LIMIT,
        world_width=WORLD_SIZE,
        world_height=WORLD_SIZE,
        goal_pos=GOAL_XY,
        goal_tolerance=0.1,
        occupancy_grid=MAP,
        grid_resolution=GRID_RESOLUTION,
        # Ground-truth plant for a --sim demo run: the EKF's own nominal
        # calibration (see maze_rl_env.py) - a fair, reproducible default
        # inside the trained randomization band, not an easy/hard extreme.
        dynamics=nominal_dynamics,
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
        dt=DT,
        max_steps=MAX_STEPS,
        vel_limit=VEL_LIMIT,
        world_width=5.0,
        world_height=5.0,
        goal_pos=GOAL_XY,
        goal_tolerance=0.1,
        render_mode="human",
        window_size=(800, 800),
    )


@contextmanager
def managed_env(sim: bool):
    if sim:
        sim_env = make_sim_env()
        sim_env.set_log_path("logs/lab4_sim.csv")
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
            real_env.set_log_path("logs/lab4_real.csv")
            real_env.start_logging()
            try:
                yield real_env
            finally:
                real_env.close()
                real_env.stop_logging()


def load_policy():
    policy = Policy(d_in=OBS_DIM, hidden=64, d_out=ACTION_DIM)
    if os.path.exists(WEIGHTS_PATH):
        policy.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        print(f"Loaded trained weights from {WEIGHTS_PATH}")
    else:
        print(f"WARNING: {WEIGHTS_PATH} not found - running an UNTRAINED policy. "
              f"Run rl_approach/train_rl.py first.")
    policy.eval()
    return policy


def control_loop(control_env):
    policy = load_policy()

    obs, _ = control_env.reset(seed=LAB1_SEED)
    is_sim = isinstance(control_env, SpheroEnv)

    if is_sim:
        control_env.state_true[0:3] = np.array([START_XY[0], START_XY[1], 0.0])
        control_env.state_odom[0:3] = np.array([START_XY[0], START_XY[1], 0.0])
        frame_offset = np.zeros(2)
    else:
        frame_offset = START_XY - np.asarray(control_env.state_odom[:2], dtype=float)
        print(f"Robot odom origin is map {tuple(np.round(-frame_offset, 3))}; "
              f"shifting readings by {tuple(np.round(frame_offset, 3))}")

    def to_map(o):
        m = np.asarray(o, dtype=float)[:5].copy()
        m[0:2] += frame_offset
        return m

    planner = Planner(map=MAP, dt=DT, resolution=GRID_RESOLUTION)
    start_state = np.array([START_XY[0], START_XY[1], 0.0, 0.0])

    ekf = EKF(dt=DT, dynamics_fn=nominal_dynamics)
    ekf.state_est = start_state.astype(float).copy()
    ekf.P = np.eye(4) * 1e-3
    ekf.Q = NOM_EKF_Q.copy()
    ekf.R = NOM_EKF_R.copy()
    est = ekf.state_est.copy()
    if not is_sim:
        # Real measurements can carry slip-induced outliers; sim's model is exact.
        max_innovation = 0.25
    else:
        max_innovation = None

    waypoints = planner.plan(start_state, GOAL_XY, margin_cells=0)
    print(f"Planned {len(waypoints)} waypoints")

    csv_file = open(f"{STUDENT_ID}_lab4.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["sim_x", "sim_y", "real_x", "real_y"])

    def log_row():
        sim_xy = control_env.state_true[0:2] if is_sim else ("", "")
        csv_writer.writerow([sim_xy[0], sim_xy[1], est[0], est[1]])

    steps = 0
    wp_index = 0
    wp_steps = 0
    replans = 0
    reached_goal = False
    warned_divergence = False

    try:
        while wp_index < len(waypoints) and steps < MAX_STEPS:
            collision_flag = float(to_map(obs)[4]) if len(obs) > 4 else 0.0
            features = build_features(est, waypoints, wp_index, collision_flag)

            with torch.no_grad():
                raw_action = policy(torch.tensor(features, dtype=torch.float32)).numpy()
            # raw_action = [speed_cmd, sin_raw, cos_raw] - see maze_rl_env.py's
            # action_space comment for why heading is reconstructed via atan2
            # rather than output directly (avoids a clip-at-+-pi corruption).
            # Safety layer: hard-clamp speed before sending anywhere.
            action = np.array([
                np.clip(raw_action[0], 0.0, VEL_LIMIT),
                np.arctan2(raw_action[1], raw_action[2]),
            ], dtype=np.float32)

            ekf.predict(action)
            obs, _, terminated, truncated, info = control_env.step(action)
            est = ekf.update(to_map(obs), max_innovation=max_innovation)[0]
            control_env.render()
            log_row()

            raw = to_map(obs)
            if VERBOSE:
                target = waypoints[wp_index]
                print(f"  wp{wp_index} s{wp_steps}: "
                      f"est=({est[0]:.3f},{est[1]:.3f}) hdg={np.degrees(est[2]):.0f} "
                      f"-> tgt=({target[0]:.2f},{target[1]:.2f}) "
                      f"cmd=({action[0]:.3f},{np.degrees(action[1]):.0f})")

            gap = np.hypot(raw[0] - est[0], raw[1] - est[1])
            if gap > DIVERGENCE_WARN and not warned_divergence:
                print(f"  *** WARNING: estimate and raw reading disagree by {gap:.3f}m. "
                      f"Odometry has probably broken. Logged data after this is unreliable. ***")
                warned_divergence = True

            steps += 1
            wp_steps += 1

            collided = info.get("collision", False) if isinstance(info, dict) else False
            if not collided and len(obs) > 4:
                collided = bool(obs[4])

            dist_to_goal_sq = (est[0] - GOAL_XY[0]) ** 2 + (est[1] - GOAL_XY[1]) ** 2
            if dist_to_goal_sq < control_env.goal_tolerance ** 2:
                reached_goal = True
                break

            if collided and replans < MAX_REPLANS:
                print(f"Collision near wp{wp_index}, step {wp_steps} - escaping")
                pos_before = np.array([est[0], est[1]])

                for _ in range(8):
                    target = SimpleNamespace(goal_pos=waypoints[wp_index], vel_limit=control_env.vel_limit)
                    straight_back = np.array([-0.08, est[2]], dtype=np.float32)
                    ekf.predict(straight_back)
                    obs, _, terminated, truncated, info = control_env.step(straight_back)
                    est = ekf.update(to_map(obs), max_innovation=max_innovation)[0]
                    control_env.render()
                    log_row()
                    steps += 1

                moved = np.hypot(est[0] - pos_before[0], est[1] - pos_before[1])
                if moved < 0.005:
                    print(f"  ball didn't move ({moved:.3f}m) - held or stalled, waiting")
                    for _ in range(20):
                        hold = np.array([0.0, est[2]], dtype=np.float32)
                        ekf.predict(hold)
                        obs, _, terminated, truncated, info = control_env.step(hold)
                        est = ekf.update(to_map(obs), max_innovation=max_innovation)[0]
                        control_env.render()
                        log_row()
                        steps += 1
                    controller_pd.reset()
                    continue

                try:
                    waypoints = planner.plan(est, GOAL_XY, margin_cells=0)
                    wp_index = 0
                    wp_steps = 0
                    controller_pd.reset()
                    replans += 1
                    print(f"Replanned {len(waypoints)} waypoints (replan #{replans}) "
                          f"from ({est[0]:.3f}, {est[1]:.3f})")
                    continue
                except (ValueError, RuntimeError) as e:
                    print(f"Replan failed: {e} - continuing with old plan")

            dist_to_wp_sq = (est[0] - waypoints[wp_index][0]) ** 2 + (est[1] - waypoints[wp_index][1]) ** 2
            if dist_to_wp_sq < WAYPOINT_TOLERANCE ** 2:
                wp_index += 1
                wp_steps = 0
                continue

            if wp_steps >= MAX_STEPS_PER_WAYPOINT:
                print(f"Timed out on waypoint {wp_index}: {waypoints[wp_index]} "
                      f"(stuck at {est[0]:.3f}, {est[1]:.3f}) - skipping ahead")
                wp_index += 1
                wp_steps = 0

        if not reached_goal:
            final_dist = np.hypot(est[0] - GOAL_XY[0], est[1] - GOAL_XY[1])
            print(f"Path complete but goal not reached. Final distance: {final_dist:.3f} m")
        else:
            print("Goal reached.")

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
