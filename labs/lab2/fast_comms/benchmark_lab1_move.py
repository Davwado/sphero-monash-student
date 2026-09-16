"""Lab 1 equivalent (0,0) -> (0.5,0.5) move, driven through the fast_comms
low-latency link instead of Robot/SpheroEduAPI.

This is a comms-layer swap, not a new control approach: it imports and
calls labs/lab1/controller.py's actual compute_action() unmodified, with the
same goal/vel_limit lab1.py uses (goal_pos=(0.5,0.5), vel_limit=0.15), so the
resulting run is a fair benchmark against lab1's existing real-robot logs -
same controller, same task, only the transport underneath is different.

Logs a CSV using the same column names Visualiser/compare_runs.py expect
(odom_x, odom_y, heading, speed, heading_cmd, speed_cmd, step), plus a
wall-clock `t` column - since fast_comms isn't throttled to a fixed loop
rate, timestamps are how you see the real achieved control rate and, unlike
every real log we had before this, a genuine measured per-step dt.

Does NOT touch sphero_env/Robot/lab1.py/lab2.py - entirely standalone.

Runs until the controller reports the goal is reached (or --max-time /
--steps runs out, or you Ctrl+C - either way the ball is stopped and
whatever was recorded gets saved).

Usage:
    python labs/lab2/fast_comms/benchmark_lab1_move.py
    python labs/lab2/fast_comms/benchmark_lab1_move.py --max-time 120 --out logs/fastcomms_run.csv
    python labs/lab2/fast_comms/benchmark_lab1_move.py --max-hz 30   # throttle for a controlled comparison
"""
import argparse
import csv
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_link  # noqa: E402

LAB1_DIR = Path(__file__).resolve().parents[2] / "lab1"
sys.path.insert(0, str(LAB1_DIR))
import controller as lab1_controller  # noqa: E402  - reuse lab1's actual control law

GOAL_POS = (0.5, 0.5)   # matches lab1.py's make_real_env()
VEL_LIMIT = 0.15        # matches lab1.py's make_real_env()
DEFAULT_OUT = "logs/fastcomms_lab1_equiv.csv"


class _GoalEnv:
    """Minimal stand-in for the `env` argument controller.py expects -
    it only ever reads .goal_pos and .vel_limit."""
    def __init__(self, goal_pos, vel_limit):
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.vel_limit = vel_limit


def _action_to_command(speed_cmd: float, heading_cmd: float):
    """Same conversion Robot.step() uses: rad/(m/s) action -> hardware
    heading_deg/speed_raw, so the fast path drives identically to lab1."""
    if speed_cmd < 0:
        heading_deg = int(np.degrees(heading_cmd + np.pi) % 360)
        speed_raw = int(np.clip(-speed_cmd / VEL_LIMIT * 255, 0, 255))
    else:
        heading_deg = int(np.degrees(heading_cmd) % 360)
        speed_raw = int(np.clip(speed_cmd / VEL_LIMIT * 255, 0, 255))
    return heading_deg, speed_raw


def run(steps: int | None, max_time: float, out_path: str, max_hz: float | None):
    env = _GoalEnv(GOAL_POS, VEL_LIMIT)

    toy = fast_link.connect()
    with toy:
        link = fast_link.FastSpheroLink(toy)
        link.configure_sensors(sensors=("locator",), interval_ms=33)

        # Let the first locator reading land before starting.
        for _ in range(50):
            if link.get_location() is not None:
                break
            time.sleep(0.05)
        else:
            print("WARNING: no locator reading yet - proceeding anyway.")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        rows = []

        # Track our own last-commanded heading/speed for the observation,
        # exactly like Robot.step() does (get_heading()/get_speed() on the
        # stock API are readbacks of the last command, not independent
        # measurements - so there's nothing lost by tracking it ourselves).
        cur_heading_rad = 0.0
        cur_speed_mps = 0.0

        min_period = (1.0 / max_hz) if max_hz else 0.0
        t0 = time.time()
        stopped = False
        step_iter = range(steps) if steps is not None else itertools.count()

        try:
            for step in step_iter:
                t = time.time() - t0
                if t >= max_time:
                    print(f"Hit --max-time ({max_time:.0f}s) before reaching the goal.")
                    break

                loc = link.get_location()
                x_m = (loc["x"] / 100.0) if loc else 0.0
                y_m = (loc["y"] / 100.0) if loc else 0.0

                obs = np.array([x_m, y_m, cur_heading_rad, cur_speed_mps, 0.0], dtype=np.float32)
                action = lab1_controller.compute_action(env, obs, step)

                if isinstance(action, list) and action[0] == "Stop":
                    stopped = True
                    link.fast_stop(np.degrees(cur_heading_rad))
                    rows.append({
                        "odom_x": x_m, "odom_y": y_m,
                        "heading": cur_heading_rad, "speed": 0.0,
                        "heading_cmd": cur_heading_rad, "speed_cmd": 0.0,
                        "step": step + 1, "t": t,
                    })
                    print(f"Goal reached at step {step + 1} (t={t:.3f}s): "
                          f"x={x_m:.3f} y={y_m:.3f}")
                    break

                speed_cmd, heading_cmd = float(action[0]), float(action[1])
                heading_deg, speed_raw = _action_to_command(speed_cmd, heading_cmd)
                link.fast_drive(heading_deg, speed_raw)

                cur_heading_rad = heading_cmd
                cur_speed_mps = speed_cmd

                rows.append({
                    "odom_x": x_m, "odom_y": y_m,
                    "heading": heading_cmd, "speed": speed_cmd,
                    "heading_cmd": heading_cmd, "speed_cmd": speed_cmd,
                    "step": step + 1, "t": t,
                })

                if min_period:
                    elapsed = time.time() - t0 - t
                    if elapsed < min_period:
                        time.sleep(min_period - elapsed)
            else:
                # step_iter exhausted (only possible when --steps was given)
                print(f"Ran out of steps ({steps}) before reaching the goal.")
        except KeyboardInterrupt:
            print("\nInterrupted - stopping the ball and saving what was recorded.")
        finally:
            if not stopped:
                link.fast_stop(np.degrees(cur_heading_rad))

        if not rows:
            print("Nothing recorded (stopped before the first row) - no CSV written.")
            return

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Logged {len(rows)} rows to {out_path}")

        if len(rows) > 1:
            dts = np.diff([r["t"] for r in rows])
            print(f"Achieved control rate: mean dt={dts.mean() * 1000:.1f}ms  "
                  f"({1 / dts.mean():.1f} Hz),  min={dts.min() * 1000:.1f}ms  "
                  f"max={dts.max() * 1000:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None,
                        help="Optional cap on control steps (default: unbounded - runs until the "
                             "goal, --max-time, or Ctrl+C)")
    parser.add_argument("--max-time", type=float, default=60.0,
                        help="Safety cutoff in seconds (default: 60s) - the real bound for an "
                             "unthrottled loop, since step count doesn't map to a fixed duration")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
                        help="CSV output path")
    parser.add_argument("--max-hz", type=float, default=None,
                        help="Optional throttle for a controlled-rate comparison (default: unthrottled)")
    args = parser.parse_args()
    run(args.steps, args.max_time, args.out, args.max_hz)


if __name__ == "__main__":
    main()
