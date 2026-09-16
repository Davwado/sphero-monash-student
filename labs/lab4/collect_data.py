"""Lab 4 - excitation run for fitting a dynamics model.

This is NOT a driving task. There is no goal and no controller. The point is
to command the ball through the states a working controller never visits -
hard turns from standstill, full-speed stops, reversals, near-stall crawling -
so a fitted model sees the transients it will later be asked to predict.

Why scripted rather than teleop: identification data needs KNOWN excitation.
Hand-driving produces whatever the driver felt like, concentrated in whatever
the driver is good at, and impossible to reproduce on a second surface. The
segments below are the same every run, so a whiteboard set and a plate set are
directly comparable (that comparison is the whole point of --surface).

Run on an OPEN surface. The maze corridors are too narrow for this.

    python collect_data.py --surface whiteboard --battery 0.9 --workspace 1.2 0.8
    python collect_data.py --sim          # dry run, no robot, checks the schedule

Output:
    logs/lab4_dyn_<surface>_<timestamp>.csv       transitions (with t_wall)
    logs/lab4_dyn_<surface>_<timestamp>.meta.json surface/battery/segment index
"""
import argparse
import json
import os
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime

import numpy as np

from sphero_env.envs import SpheroEnv
from sphero_env.robot.connect import scan_and_connect
from sphero_env.robot.robot import Robot
from sphero_unsw.sphero_edu import SpheroEduAPI

# Speed cap for collection. Deliberately at the lab-3 vel_limit rather than
# higher: the model only needs to be valid over the range the robot will
# actually be driven in, and a ball that gets away from you on a table is a
# ball on the floor.
SPEED_CAP = 0.15

# --- measured on the whiteboard table, Phase 1, 2026-09-16 ------------------
# From labs/lab4/logs/sphero_teleop_log.csv via timing_report.py. Re-measure
# on a different surface or comms path; these are observations, not constants.
#
#   control period   0.300 s (3.3 Hz), p10/p90 0.196/0.316 - regular
#   turn rate        0.70 rad/s median, 1.07 max (forward only, reversal
#                    artefacts excluded - robot.py:420 flips commanded heading
#                    by pi for negative speed, which reads as a fake 180 turn)
#   locator refresh  1% stale at sensor_interval_ms=150 against a 300ms loop
#
# For reference, both existing models were wrong: lab1/dynamics.py assumes
# 0.3 rad/s (2x too slow), the lab3 sim assumes 3.0 rad/s (4x too fast).
MEASURED_STEP_S = 0.300
MEASURED_TURN_RATE = 0.70

# Steps held at each SPEED command. The ball must be given long enough to
# actually reach steady state, otherwise every sample is a transient and the
# model never sees the asymptote it is supposed to converge to. 8 steps at
# 0.3s is ~2.4s, comfortably longer than the observed speed response.
HOLD_STEPS = 8

# Turns are held by ANGLE, not by a fixed step count. At 0.70 rad/s a step
# turns ~12 deg, so a flat 8-step hold completes a 15 deg command with time
# to spare but cuts a 180 deg command off at the halfway point - capturing
# only transient and never the steady state it is supposed to settle into.
TURN_SETTLE_FACTOR = 1.5   # overshoot allowance beyond the ideal-rate estimate
MIN_TURN_HOLD = 4

DEG = np.pi / 180.0


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def hold_for_turn(angle_rad):
    """Steps to complete a turn of this size and then settle."""
    ideal = abs(angle_rad) / (MEASURED_TURN_RATE * MEASURED_STEP_S)
    return max(MIN_TURN_HOLD, int(np.ceil(ideal * TURN_SETTLE_FACTOR)))


# ---------------------------------------------------------------- schedule --

def build_schedule(speed_cap=SPEED_CAP):
    """Return [(segment_name, [(speed_cmd, heading_cmd_offset, n_steps), ...])].

    heading_cmd_offset is RELATIVE to the heading the segment started at, so
    the schedule is orientation-independent - it does the same manoeuvre
    wherever the ball happens to be pointing when the segment begins.
    """
    v = speed_cap
    sched = []

    # --- heading channel, from standstill -------------------------------
    # Isolates turn rate and turn lag with translation held at zero. These are
    # the parameters the three dynamics models in this repo disagree about
    # most, and they cost no space at all to identify.
    #
    # Repeated rather than held longer. A held command gives ONE transient and
    # then n-1 steady-state samples, and the transient is the informative
    # part - a dry run put 131 of 370 samples in the aligned/near-zero bin
    # for exactly this reason. Holds still need to be long enough to reach
    # steady state on the real robot (whose turn rate is unknown and likely
    # far slower than the sim's), so the fix is more repeats, not shorter holds.
    seg = []
    for _ in range(3):
        prev = 0.0
        for d in (15, -15, 45, -45, 90, -90, 180, -180):
            seg.append((0.0, d * DEG, hold_for_turn((d * DEG) - prev)))
            # Settling back is itself a turn of the same size, in the other
            # direction - so it needs the same allowance, and doubles as data.
            seg.append((0.0, 0.0, hold_for_turn(d * DEG)))
            prev = 0.0
    sched.append(("heading_standstill", seg))

    # --- speed channel, straight line -----------------------------------
    # Accelerate to each level and hold, then brake hard. Accel and decel are
    # separate parameters in dynamics() (MAX_ACCEL vs MAX_DECEL) and only a
    # hard stop distinguishes them.
    seg = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        seg.append((v * frac, 0.0, HOLD_STEPS * 2))
        seg.append((0.0, 0.0, HOLD_STEPS))       # brake to rest
    sched.append(("speed_straight", seg))

    # --- low-speed regime -----------------------------------------------
    # Near stall the ball behaves differently - traction and the motor
    # deadband both matter here, and this is where a linear fit breaks.
    seg = []
    for frac in (0.05, 0.1, 0.15, 0.2):
        seg.append((v * frac, 0.0, HOLD_STEPS * 2))
    seg.append((0.0, 0.0, HOLD_STEPS))
    sched.append(("speed_lowspeed", seg))

    # --- turning under way ----------------------------------------------
    # The coupling term: dynamics() models speed_target as
    # speed_cmd * cos(heading_error), and that factor is only observable when
    # speed and heading change at the same time.
    seg = []
    for _ in range(2):
        for d in (30, -30, 60, -60, 90, -90):
            seg.append((v * 0.75, 0.0, HOLD_STEPS))                  # get moving
            seg.append((v * 0.75, d * DEG, hold_for_turn(d * DEG)))  # turn under way
    seg.append((0.0, 0.0, HOLD_STEPS))
    sched.append(("turn_under_way", seg))

    # --- reversals -------------------------------------------------------
    # 180 while moving: the worst case for any model, and the manoeuvre a
    # replanning controller will ask for after a collision.
    seg = []
    for _ in range(3):
        seg.append((v * 0.6, 0.0, HOLD_STEPS * 2))
        seg.append((v * 0.6, 180 * DEG, hold_for_turn(180 * DEG)))
    seg.append((0.0, 0.0, HOLD_STEPS))
    sched.append(("reversal", seg))

    return sched


# ------------------------------------------------------------ environments --

def make_real_env(api):
    return Robot(
        api=api,
        dt=0.1,
        max_steps=100000,
        vel_limit=SPEED_CAP,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.0, 0.0),
        goal_tolerance=0.1,
        render_mode=None,
        window_size=(800, 800),
    )


def make_sim_env():
    """Dry-run environment. Only there to prove the schedule executes and the
    log comes out with the right schema - the sim's dynamics are exactly what
    this whole exercise is trying to replace, so its DATA is worthless here."""
    return SpheroEnv(
        dt=0.1,
        max_steps=100000,
        vel_limit=SPEED_CAP,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.0, 0.0),
        goal_tolerance=0.1,
        occupancy_grid=None,
        render_mode=None,
        window_size=(800, 800),
    )


@contextmanager
def managed_env(sim, log_path):
    if sim:
        env = make_sim_env()
        env.set_log_path(log_path)
        env.reset(seed=0)
        env.start_logging()
        try:
            yield env
        finally:
            env.stop_logging()
            env.close()
    else:
        with ExitStack() as stack:
            selected_toy, _ = scan_and_connect()
            print(f"Selected: {selected_toy.name}")
            api = stack.enter_context(SpheroEduAPI(selected_toy))
            api.reset_aim()
            env = make_real_env(api)
            env.set_log_path(log_path)
            env.reset()
            env.start_logging()
            try:
                yield env
            finally:
                env.emergency_stop()
                env.close()
                env.stop_logging()


# ------------------------------------------------------------------ guards --

class WorkspaceGuard:
    """Stops the run if the ball leaves - or is about to leave - a box centred
    on where it started.

    A whiteboard table has edges. Odometry is the only thing that knows where
    the ball is, and it drifts - so this is a soft guard, not a safety
    interlock. Keep a hand near the ball regardless.

    Two things make a position-only check too late. The guard can only run
    AFTER a step has been applied, and Phase 1 measured up to 0.22m of travel
    in a single 0.3s step. A 0.10m margin is therefore inside one step's
    stopping distance: by the time the check fires the ball is already over
    the edge. Hence a wider default margin AND a one-step lookahead using the
    current heading and speed.
    """

    # Wider than one step's worst-case travel measured in Phase 1 (0.22m).
    DEFAULT_MARGIN = 0.25

    def __init__(self, origin_xy, width, height, margin=DEFAULT_MARGIN,
                 lookahead_s=MEASURED_STEP_S):
        self.origin = np.asarray(origin_xy, dtype=float)
        self.half_w = max(0.0, width / 2.0 - margin)
        self.half_h = max(0.0, height / 2.0 - margin)
        self.lookahead_s = lookahead_s
        if self.half_w <= 0 or self.half_h <= 0:
            print(f"  !! table ({width:.2f} x {height:.2f} m) is not much bigger "
                  f"than the {margin:.2f} m margin - the guard will fire "
                  f"immediately. Use a larger surface or lower --speed-cap.")

    def _outside(self, xy):
        d = np.asarray(xy, dtype=float) - self.origin
        return abs(d[0]) > self.half_w or abs(d[1]) > self.half_h

    def breached(self, xy, heading=None, speed=None):
        """True if the ball is outside the box, or projects outside it within
        one step at its current heading and speed."""
        if self._outside(xy):
            return True
        if heading is None or speed is None or not np.isfinite(speed):
            return False
        # Heading convention: 0 rad -> +y, pi/2 -> +x (see sphero_env).
        step = np.array([np.sin(heading), np.cos(heading)]) * speed * self.lookahead_s
        return self._outside(np.asarray(xy, dtype=float) + step)

    def describe(self):
        return (f"workspace +-{self.half_w:.2f} x +-{self.half_h:.2f} m "
                f"about ({self.origin[0]:.2f}, {self.origin[1]:.2f}), "
                f"{self.lookahead_s:.2f}s lookahead")


# -------------------------------------------------------------------- main --

def run_segment(env, name, commands, guard, base_heading, step_counter):
    """Execute one segment. Returns (start_step, end_step, aborted)."""
    start = step_counter[0]
    aborted = False

    for speed_cmd, heading_offset, n_steps in commands:
        heading_cmd = wrap_angle(base_heading + heading_offset)
        action = np.array([speed_cmd, heading_cmd], dtype=np.float32)

        for _ in range(n_steps):
            obs, _, terminated, truncated, info = env.step(action)
            step_counter[0] += 1

            odom = np.asarray(info.get("state_odom", obs), dtype=float)
            xy = odom[:2]
            odom_h = odom[2] if odom.size > 2 else None
            odom_v = odom[3] if odom.size > 3 else None
            if guard is not None and guard.breached(xy, odom_h, odom_v):
                print(f"  !! workspace breached at ({xy[0]:.2f}, {xy[1]:.2f}) "
                      f"- stopping segment '{name}'")
                env.emergency_stop()
                aborted = True
                break
            if terminated or truncated:
                aborted = True
                break
        if aborted:
            break

    env.emergency_stop()
    return start, step_counter[0], aborted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", action="store_true",
                        help="Dry run against the simulator - checks the schedule and "
                             "log schema without a robot. The DATA is not usable.")
    parser.add_argument("--surface", default="unknown",
                        help="Surface tag, e.g. whiteboard / plates. Recorded in the "
                             "sidecar and used in the filename - never pool surfaces.")
    parser.add_argument("--battery", type=float, default=None,
                        help="Battery fraction 0-1 at the start of the run. Sphero "
                             "torque sags as it drains and mimics a traction change.")
    parser.add_argument("--workspace", nargs=2, type=float, metavar=("W", "H"),
                        default=None,
                        help="Usable table size in metres. Without this there is no "
                             "edge guard.")
    parser.add_argument("--speed-cap", type=float, default=SPEED_CAP)
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"logs/lab4_dyn_{args.surface}_{stamp}"
    log_path = f"{base}.csv"
    meta_path = f"{base}.meta.json"

    schedule = build_schedule(args.speed_cap)
    total_steps = sum(n for _, seg in schedule for _, _, n in seg)
    print(f"Schedule: {len(schedule)} segments, {total_steps} steps total")
    for name, seg in schedule:
        print(f"  {name}: {sum(n for _, _, n in seg)} steps")

    if args.workspace is None and not args.sim:
        print("\n*** No --workspace given: the edge guard is DISABLED. On a table, "
              "pass --workspace W H. ***\n")

    segments_meta = []

    with managed_env(args.sim, log_path) as env:
        odom = np.asarray(env.get_odom_state(), dtype=float)
        origin = odom[:2].copy()
        base_heading = float(odom[2])

        guard = None
        if args.workspace is not None:
            guard = WorkspaceGuard(origin, args.workspace[0], args.workspace[1])
            print(guard.describe())

        print(f"Start pose: ({origin[0]:.3f}, {origin[1]:.3f}) "
              f"heading {np.degrees(base_heading):.0f} deg")

        step_counter = [0]
        t_start = time.perf_counter()

        try:
            for name, commands in schedule:
                print(f"\n-- segment: {name}")
                s, e, aborted = run_segment(env, name, commands, guard,
                                            base_heading, step_counter)
                segments_meta.append({
                    "name": name, "start_step": s, "end_step": e, "aborted": aborted,
                })
                print(f"   steps {s}..{e}{'  (ABORTED)' if aborted else ''}")

                if aborted:
                    print("   recentre the ball and press Enter to continue, "
                          "or Ctrl-C to stop")
                    try:
                        input()
                    except EOFError:
                        break
                    odom = np.asarray(env.get_odom_state(), dtype=float)
                    if guard is not None:
                        guard.origin = odom[:2].copy()
                    base_heading = float(odom[2])
        except KeyboardInterrupt:
            print("\nInterrupted - stopping cleanly")
        finally:
            env.emergency_stop()

        elapsed = time.perf_counter() - t_start

    meta = {
        "surface": args.surface,
        "battery": args.battery,
        "notes": args.notes,
        "sim": bool(args.sim),
        "speed_cap": args.speed_cap,
        "hold_steps": HOLD_STEPS,
        "comms": "sim" if args.sim else "SpheroEduAPI",
        "aim_reset": not args.sim,
        "timestamp": stamp,
        "steps": step_counter[0],
        "elapsed_s": round(elapsed, 2),
        "segments": segments_meta,
        "log": os.path.basename(log_path),
    }
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {log_path}")
    print(f"Wrote {meta_path}")
    if step_counter[0]:
        print(f"{step_counter[0]} steps in {elapsed:.1f}s "
              f"({elapsed / step_counter[0]:.3f}s per step)")
    if args.sim:
        print("\n(--sim: schedule and schema only. Collect on the real robot for "
              "data worth fitting.)")


if __name__ == "__main__":
    main()
