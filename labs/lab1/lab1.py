# Import necessary libraries
from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

import argparse
import csv
import importlib
import math
import os
import shutil
import time
import traceback

import numpy as np
import pygame

from dynamics import *
import controller

from contextlib import ExitStack, contextmanager

IDLE_STATUS = "IDLE  R:run P:play S:save"
REPLAY_HELP = ("Replay: LEFT/RIGHT step (hold to scroll), wheel/drag bar to scrub, "
               "HOME/END jump, SPACE play/pause, S save, ESC back, Q quit")

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
        sensor_interval_ms=50,   # sensor stream rate (50 ms = 20 Hz)
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
            api.reset_aim()
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


def _handle_common_keys(env, event):
    """Events that work everywhere: window resize/maximise, +/- HUD text size."""
    if event.type == pygame.VIDEORESIZE:
        env.vis.handle_resize(event.w, event.h)
        return True
    if event.type != pygame.KEYDOWN:
        return False
    if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
        env.vis.set_hud_font_size(env.vis.hud_font_size + 2)
        return True
    if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        env.vis.set_hud_font_size(env.vis.hud_font_size - 2)
        return True
    return False


def _latest_csv_path(sim: bool) -> str:
    return f"logs/lab1_{'sim' if sim else 'real'}_latest.csv"


def save_last_run(sim: bool):
    """Copy the scratch CSV of the last run into logs/saved/ (explicit save)."""
    src = _latest_csv_path(sim)
    if not os.path.exists(src):
        print("Nothing to save yet - run something first (R).")
        return
    os.makedirs("logs/saved", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = f"logs/saved/lab1_{'sim' if sim else 'real'}_{stamp}.csv"
    shutil.copy2(src, dest)
    print(f"Saved run to {dest}  (replay later with: python lab1.py --replay {dest})")


def run_one(env, sim: bool, run_idx: int, steps: int, seed: int):
    """Execute one control run.

    Returns ("idle" | "quit", frames) where frames is the per-step history
    used for replay (P key).
    """
    frames = []
    # Hot-reload the controller so edits take effect without reconnecting.
    try:
        importlib.reload(controller)
    except Exception:
        traceback.print_exc()
        print("controller.py failed to load - fix it and press R again.")
        return "idle", frames

    # Scratch CSV, overwritten every run (press S to keep a copy).
    env.stop_logging()
    env.set_log_path(_latest_csv_path(sim))
    env.start_logging()

    env.vis.reset()
    env.vis.set_hud(run=run_idx, status="RUNNING  SPACE:abort")
    print(f"Run {run_idx}: {steps} steps (seed={seed})")

    result = "idle"
    try:
        obs, _ = env.reset(seed=seed)

        # controller.py is hot-reloaded above, so its GOAL can change between
        # runs. reset() re-points the marker at the env's own goal_pos, so push
        # the controller's goal in afterwards to keep the yellow dot, the logged
        # setpoint and the goal-reached test all tracking the same target.
        goal = getattr(controller, "GOAL", None)
        if goal is not None:
            env.goal_pos = np.asarray(goal, dtype=np.float32)
            env.vis.set_goal(env.goal_pos)
            env.set_current_setpoint(env.goal_pos)

        for step in range(steps):
            aborted = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit", frames
                if _handle_common_keys(env, event):
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return "quit", frames
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

            frames.append({
                "gt": np.array(info["state_true"], dtype=np.float32),
                "odom": np.array(info["state_odom"], dtype=np.float32),
                "action": (float(info["speed_cmd"]), float(info["heading_cmd"])),
                "collision": 1.0 if info.get("collision") else 0.0,
                "step": step + 1,
            })

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

    return result, frames


def replay(env, sim: bool, frames, can_return: bool = True) -> str:
    """Scrub through a recorded run like a video.

    Returns "idle" (back to the session) or "quit". With can_return=False
    (file replay mode) ESC quits too.
    """
    if not frames:
        print("No run recorded yet - press R to run first.")
        return "idle"

    n = len(frames)
    i = 0
    playing = False
    scrubbing = False
    last_advance = 0.0
    dt = getattr(env, "dt", 0.1)
    print(REPLAY_HELP)
    pygame.key.set_repeat(300, 40)  # hold arrows to scroll

    def seek_from_mouse(mx):
        bar = env.vis.scrub_bar_rect()
        frac = (mx - bar.x) / max(1, bar.width)
        return int(round(min(1.0, max(0.0, frac)) * (n - 1)))

    try:
        while True:
            frame = frames[i]
            env.vis.set_trajectories(
                gt=[(f["gt"][0], f["gt"][1]) for f in frames[:i + 1] if f["gt"] is not None],
                odom=[(f["odom"][0], f["odom"][1]) for f in frames[:i + 1] if f["odom"] is not None],
            )
            env.vis.hud_action = frame["action"]
            env.vis.hud_collision = frame["collision"]
            env.vis.hud_step = frame["step"]
            env.vis.set_hud(status=f"REPLAY {'>' if playing else '||'} {i + 1}/{n}")
            env.vis.render(frame["gt"], frame["odom"], scrub=(i, n, playing))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if _handle_common_keys(env, event):
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return "quit"
                    if event.key in (pygame.K_ESCAPE, pygame.K_p):
                        return "idle" if can_return else "quit"
                    if event.key == pygame.K_RIGHT:
                        i = min(n - 1, i + 1); playing = False
                    if event.key == pygame.K_LEFT:
                        i = max(0, i - 1); playing = False
                    if event.key == pygame.K_HOME:
                        i = 0; playing = False
                    if event.key == pygame.K_END:
                        i = n - 1; playing = False
                    if event.key == pygame.K_SPACE:
                        if i == n - 1:
                            i = 0  # replay from the start
                        playing = not playing
                        last_advance = time.time()
                    if event.key == pygame.K_s and can_return:
                        save_last_run(sim)
                if event.type == pygame.MOUSEWHEEL:
                    i = min(n - 1, max(0, i + (1 if event.y < 0 else -1)))
                    playing = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    bar = env.vis.scrub_bar_rect().inflate(0, 14)
                    if bar.collidepoint(event.pos):
                        scrubbing = True
                        playing = False
                        i = seek_from_mouse(event.pos[0])
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    scrubbing = False
                if event.type == pygame.MOUSEMOTION and scrubbing:
                    i = seek_from_mouse(event.pos[0])

            if playing and time.time() - last_advance >= dt:
                last_advance = time.time()
                if i < n - 1:
                    i += 1
                else:
                    playing = False
    except KeyboardInterrupt:
        return "idle" if can_return else "quit"
    finally:
        pygame.key.set_repeat()  # disable key repeat outside replay
        env.vis.reset()
        env.vis.set_hud(status=IDLE_STATUS)


def _csv_float(row, col):
    try:
        return float(row.get(col, "nan") or "nan")
    except ValueError:
        return float("nan")


def load_frames_from_csv(path):
    """Rebuild replay frames from a saved run CSV (see Visualiser._columns)."""
    frames = []
    with open(path, newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            odom = [_csv_float(row, c) for c in ("odom_x", "odom_y", "heading", "speed")]
            gt = [_csv_float(row, c) for c in ("gt_x", "gt_y", "gt_heading", "gt_speed")]
            action = (_csv_float(row, "speed_cmd"), _csv_float(row, "heading_cmd"))
            step = _csv_float(row, "step")
            collision = _csv_float(row, "collision")

            def clean(state):
                if math.isnan(state[0]) or math.isnan(state[1]):
                    return None
                return np.array([state[0], state[1],
                                 0.0 if math.isnan(state[2]) else state[2],
                                 0.0 if math.isnan(state[3]) else state[3]],
                                dtype=np.float32)

            frames.append({
                "gt": clean(gt),
                "odom": clean(odom),
                "action": None if math.isnan(action[0]) or math.isnan(action[1]) else action,
                "collision": 0.0 if math.isnan(collision) else collision,
                "step": idx + 1 if math.isnan(step) else int(step),
            })
    print(f"Loaded {len(frames)} frames from {path}")
    return frames


def run_session(env, sim: bool, steps: int, seed: int):
    """Idle loop: window stays open and connected; R starts a run."""
    run_idx = 1
    last_frames = []
    env.vis.set_hud(run=0, status=IDLE_STATUS)
    print("Ready. In the pygame window: R = run, P = replay last run, "
          "S = save last run, SPACE = stop, +/- = text size, Q = quit.")

    last_keepalive = time.time()
    while True:
        try:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if _handle_common_keys(env, event):
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return
                    if event.key == pygame.K_SPACE:
                        _safe_stop(env)
                    if event.key == pygame.K_p:
                        if replay(env, sim, last_frames) == "quit":
                            return
                        last_keepalive = time.time()
                    if event.key == pygame.K_s:
                        save_last_run(sim)
                    if event.key == pygame.K_r:
                        result, frames = run_one(env, sim, run_idx, steps, seed)
                        if result == "quit":
                            return
                        if frames:
                            last_frames = frames
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
    parser.add_argument("--font-size", type=int, default=22,
                        help="HUD text size (also +/- keys while running)")
    parser.add_argument("--replay", type=str, default=None, metavar="CSV",
                        help="Replay a saved run CSV (no robot connection)")
    args = parser.parse_args(argv)

    if args.replay:
        # Pure playback: no Bluetooth, no controller - just the window.
        frames = load_frames_from_csv(args.replay)
        view_env = make_sim_env()
        view_env.vis.set_hud_font_size(args.font_size)
        try:
            replay(view_env, True, frames, can_return=False)
        finally:
            view_env.close()
        return

    with managed_env(args.sim) as control_env:
        control_env.vis.set_hud_font_size(args.font_size)
        run_session(control_env, args.sim, args.steps, args.seed)

if __name__ == "__main__":
    main()
