# Import necessary libraries
from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

import argparse
import importlib
import time
import traceback

import numpy as np
import pygame

from dynamics import *
import controller

from contextlib import ExitStack, contextmanager

IDLE_STATUS = "IDLE  R:run  SPACE:stop  Q:quit"

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
        obs_noise_std_pos=0.00,
        process_noise_std_speed=0.00,
        process_noise_std_heading=0.00,
        obs_noise_std_vel=0,
        render_mode="human",
        window_size=(1000, 1000),
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
    # The Bluetooth connection (real robot) is opened ONCE here and stays open
    # for the whole session; runs are started/stopped from the pygame window.
    if sim:
        sim_env = make_sim_env()
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
            try:
                yield real_env
            finally:
                real_env.stop_logging()
                real_env.close()


def _safe_stop(env):
    """Emergency-stop without letting a dead BLE link raise."""
    try:
        env.emergency_stop()
    except Exception:
        traceback.print_exc()


def run_one(env, sim: bool, run_idx: int, steps: int, seed: int) -> str:
    """Execute one control run. Returns "idle" (back to menu) or "quit"."""
    # Hot-reload the controller so edits take effect without reconnecting.
    try:
        importlib.reload(controller)
    except Exception:
        traceback.print_exc()
        print("controller.py failed to load - fix it and press R again.")
        return "idle"

    # New CSV per run.
    env.stop_logging()
    env.set_log_path(f"logs/lab1_{'sim' if sim else 'real'}_run{run_idx:03d}.csv")
    env.start_logging()

    env.vis.reset()
    env.vis.set_hud(run=run_idx, status="RUNNING  SPACE:abort  Q:quit")
    print(f"Run {run_idx}: {steps} steps (seed={seed})")

    result = "idle"
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(steps):
            aborted = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return "quit"
                    if event.key in (pygame.K_SPACE, pygame.K_ESCAPE):
                        print("Run aborted.")
                        aborted = True
            if aborted:
                break

            try:
                action = controller.compute_action(obs, step)
            except Exception:
                traceback.print_exc()
                print("controller.compute_action crashed - run aborted.")
                break

            try:
                obs, _, terminated, truncated, info = env.step(action)
            except Exception:
                # Covers BLE timeouts/disconnects on the real robot. We never
                # auto-reconnect; stop and drop back to idle.
                traceback.print_exc()
                print("env.step failed - run aborted.")
                break

            env.render()
            if terminated or truncated:
                print(f"Episode ended at step {step} "
                      f"(terminated={terminated}, truncated={truncated}).")
                break
    except KeyboardInterrupt:
        print("Run interrupted (Ctrl+C) - robot stopped, back to idle.")
    finally:
        _safe_stop(env)
        env.stop_logging()
        env.vis.set_hud(status=IDLE_STATUS)

    return result


def run_session(env, sim: bool, steps: int, seed: int):
    """Idle loop: window stays open and connected; R starts a run."""
    run_idx = 1
    env.vis.set_hud(run=0, status=IDLE_STATUS)
    print("Ready. In the pygame window: R = run, SPACE = stop, Q = quit.")

    last_keepalive = time.time()
    while True:
        try:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return
                    if event.key == pygame.K_SPACE:
                        _safe_stop(env)
                    if event.key == pygame.K_r:
                        if run_one(env, sim, run_idx, steps, seed) == "quit":
                            return
                        run_idx += 1
                        last_keepalive = time.time()

            # Keep the BLE link warm while idle so the toy doesn't sleep.
            if not sim and time.time() - last_keepalive > 3.0:
                last_keepalive = time.time()
                try:
                    env.api.get_location()
                except Exception:
                    traceback.print_exc()

            time.sleep(0.1)
        except KeyboardInterrupt:
            _safe_stop(env)
            print("Ctrl+C: robot stopped. Press Q in the window to quit.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run simulation")
    parser.add_argument("--steps", type=int, default=100,
                        help="Steps per run (lab spec: 100)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Reset seed used for every run")
    args = parser.parse_args(argv)

    with managed_env(args.sim) as control_env:
        run_session(control_env, args.sim, args.steps, args.seed)

if __name__ == "__main__":
    main()
