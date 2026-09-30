"""Polling-rate / reliability comparison: stock SpheroEduAPI vs fast_comms.

Runs the lab 1 task (P-controller drive to a goal 0.5 m away in x and y, using
labs/lab1/controller.py unmodified) through lab 1's own Robot.step() twice -
once on the stock SpheroEduAPI path, once on the fast_comms link - so the ONLY
difference between the two runs is the transport underneath. Headless (no
pygame window) so rendering doesn't pollute the timings.

Per run it measures:
  Speed        control-loop rate (Hz), step-time mean/median/p95/p99/max, jitter
  Latency      per-API-call time (set_heading, set_speed, get_location, ...) so
               you can see WHERE the stock path loses its time
  Freshness    unique position samples/s, the age of the position the controller
               acted on, and the longest gap with no new position. (A loop
               faster than the sensor stream re-reads the same sample, so
               'fraction of steps with a new position' is informational only.)
  Reliability  API exceptions, worst stale-data gap,
               collisions flagged, whether the goal was reached / timed out
  Control      time & steps to goal, final distance, overshoot, path length
               and path efficiency (straight-line / travelled)

The goal is set RELATIVE to wherever the ball starts (start + goal offset), so
you can just put the ball somewhere with ~0.7 m of clear space in +x/+y and
run each mode again without worrying about the locator origin.

Usage:
    python labs/lab2/fast_comms/polling_test.py                    # stock, then fast (prompts between)
    python labs/lab2/fast_comms/polling_test.py --mode fast
    python labs/lab2/fast_comms/polling_test.py --runs 3           # 3 runs of each mode
    python labs/lab2/fast_comms/polling_test.py --mode fast --max-hz 50   # throttled to ~sensor rate
    python labs/lab2/fast_comms/polling_test.py --report           # re-summarise saved runs only
    python labs/lab2/fast_comms/polling_test.py --dry-run          # simulated robot, no hardware (checks the script)

Outputs (default logs/polling_test/): <mode>_run<N>.csv (per-step),
<mode>_run<N>.json (summary), comparison.png, and a printed comparison table.
"""
import argparse
import contextlib
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "labs" / "lab1"))

import controller as lab1_controller  # noqa: E402  - lab1's actual control law

VEL_LIMIT = 0.15
GOAL_OFFSET = (0.5, 0.5)
DEFAULT_OUT = ROOT / "logs" / "polling_test"
MODES = ("stock", "fast")


# --------------------------------------------------------------------------- #
# Timing / failure instrumentation around whatever `api` Robot is given
# --------------------------------------------------------------------------- #
class TimedApi:
    """Transparent proxy: times every callable on the wrapped api and counts
    exceptions (which are swallowed per-step by the test loop, not here)."""

    def __init__(self, api):
        object.__setattr__(self, "_api", api)
        object.__setattr__(self, "calls", {})       # name -> [durations]
        object.__setattr__(self, "errors", {})      # name -> count

    def __getattr__(self, name):
        attr = getattr(self._api, name)
        if not callable(attr):
            return attr

        def timed(*a, **k):
            t = time.perf_counter()
            try:
                return attr(*a, **k)
            except Exception:
                self.errors[name] = self.errors.get(name, 0) + 1
                raise
            finally:
                self.calls.setdefault(name, []).append(time.perf_counter() - t)
        return timed


def pace(t_next):
    """Wait until perf_counter() reaches t_next. time.sleep() on Windows only
    has ~15 ms resolution, so sleep the bulk and spin the last stretch."""
    while True:
        rem = t_next - time.perf_counter()
        if rem <= 0:
            return
        if rem > 0.02:
            time.sleep(rem - 0.015)

# --------------------------------------------------------------------------- #
# Simulated robot for --dry-run (validates the script, NOT the hardware)
# --------------------------------------------------------------------------- #
class FakeApi:
    """Fake SpheroEduAPI: each call costs `call_s` (blocking, like stock BLE),
    and the locator only refreshes every `sensor_s`."""

    def __init__(self, call_s, sensor_s, drop_prob=0.0, seed=0):
        self.call_s, self.sensor_s, self.drop_prob = call_s, sensor_s, drop_prob
        self.rng = np.random.default_rng(seed)
        self.x = self.y = 0.0
        self.cx = self.cy = 0.0
        self._h, self._s = 0, 0
        self._t_phys = self._t_sensor = time.perf_counter()

    def _tick(self):
        now = time.perf_counter()
        v = self._s / 255.0 * VEL_LIMIT
        dt = now - self._t_phys
        self.x += v * np.sin(np.radians(self._h)) * dt
        self.y += v * np.cos(np.radians(self._h)) * dt
        self._t_phys = now
        if now - self._t_sensor >= self.sensor_s:
            self.cx, self.cy, self._t_sensor = self.x, self.y, now

    def _cost(self):
        time.sleep(self.call_s)
        if self.rng.random() < self.drop_prob:
            raise TimeoutError("simulated BLE timeout")
        self._tick()

    def set_heading(self, h): self._cost(); self._h = int(h)
    def set_speed(self, s): self._cost(); self._s = int(s)
    def get_heading(self): self._tick(); return self._h
    def get_speed(self): self._tick(); return self._s
    def get_location(self): self._tick(); return {"x": self.cx * 100, "y": self.cy * 100}
    def get_acceleration(self): self._tick(); return {"x": 0.0, "y": 0.0, "z": 1.0}
    def get_velocity(self): self._tick(); return {"x": 0.0, "y": 0.0}
    def get_gyroscope(self): self._tick(); return {"x": 0.0, "y": 0.0, "z": 0.0}
    def get_orientation(self): self._tick(); return {"pitch": 0, "roll": 0, "yaw": self._h}


@contextlib.contextmanager
def open_api(mode, dry_run):
    if dry_run:
        yield (FakeApi(0.045, 0.150, drop_prob=0.01) if mode == "stock"
               else FakeApi(0.0005, 0.033))
        return
    if mode == "fast":
        from fast_link import fast_managed_api
        with fast_managed_api(sensors=("locator", "accelerometer", "velocity", "gyroscope")) as api:
            yield api
        return
    from sphero_env.robot.connect import scan_and_connect
    from sphero_unsw.sphero_edu import SpheroEduAPI
    toy, _ = scan_and_connect()
    print(f"Selected: {toy.name}")
    with SpheroEduAPI(toy) as api:
        api.reset_aim()
        yield api


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def run_once(mode, run_idx, out_dir, max_time, settle_s, dry_run, max_hz=None):
    from sphero_env.robot.robot import Robot

    print(f"\n=== {mode.upper()} run {run_idx} ===")
    with open_api(mode, dry_run) as raw_api:
        api = TimedApi(raw_api)
        env = Robot(api=api, dt=0.1, max_steps=100000, vel_limit=VEL_LIMIT,
                    settle_steps=5, world_width=5.0, world_height=5.0,
                    goal_pos=GOAL_OFFSET, goal_tolerance=0.1,
                    obs_noise_std_pos=0.0, obs_noise_std_vel=0.0,
                    render_mode=None)
        env.reset()
        api.calls.clear(); api.errors.clear()      # exclude connect/settle from stats

        start = np.array(env.state_odom[:2], dtype=float)
        goal = start + np.array(GOAL_OFFSET)
        env.set_goal(goal)
        straight = float(np.linalg.norm(goal - start))
        print(f"Start ({start[0]:.3f},{start[1]:.3f}) -> goal ({goal[0]:.3f},{goal[1]:.3f}); "
              f"max {max_time:.0f}s. Ctrl+C aborts and stops the ball.")

        rows, step_errors, consecutive_errors = [], 0, 0
        status = "timeout"
        last_pos, last_change_t = None, None
        t0 = time.perf_counter()
        t_stop = None
        period = 1.0 / max_hz if max_hz else 0.0
        t_next = time.perf_counter()
        obs = np.array([*start, 0.0, 0.0, 0.0], dtype=np.float32)
        action = np.array([0.0, 0.0], dtype=np.float32)

        try:
            while True:
                t = time.perf_counter() - t0
                if t_stop is None and t >= max_time:
                    break
                if t_stop is not None and t - t_stop >= settle_s:
                    break

                if period:
                    pace(t_next)
                    t_next = max(t_next + period, time.perf_counter() - period)
                tc = time.perf_counter()
                # Age of the position the controller is about to act on
                # (time since it last changed, as observed by this loop).
                age = (tc - last_change_t) if last_change_t is not None else 0.0
                if t_stop is None:
                    action = lab1_controller.compute_action(env, obs, len(rows))
                    if isinstance(action, list):        # ["Stop", "Stop"]
                        status = "goal_reached"
                        t_stop = t
                        action = np.array([0.0, float(obs[2])], dtype=np.float32)
                else:
                    action = np.array([0.0, float(obs[2])], dtype=np.float32)
                ctrl_s = time.perf_counter() - tc

                ts = time.perf_counter()
                ok = True
                try:
                    obs, _, _, _, info = env.step(action)
                    consecutive_errors = 0
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    ok = False
                    step_errors += 1
                    consecutive_errors += 1
                    print(f"  step error ({type(e).__name__}: {e})")
                    if consecutive_errors >= 10:
                        status = "comms_failure"
                        break
                step_s = time.perf_counter() - ts

                pos = np.array(env.state_odom[:2], dtype=float)
                changed = last_pos is None or not np.array_equal(pos, last_pos)
                if changed:
                    last_pos, last_change_t = pos.copy(), time.perf_counter()
                info = info if ok else {}
                rows.append({
                    "step": len(rows) + 1, "t": t, "step_s": step_s, "ctrl_s": ctrl_s,
                    "x": pos[0], "y": pos[1],
                    "dist_goal": float(np.linalg.norm(goal - pos)),
                    "speed_cmd": float(action[0]), "heading_cmd": float(action[1]),
                    "fresh": int(changed), "age_s": age,
                    "collision": int(bool(info.get("collision", False))) if ok else 0,
                    "ok": int(ok), "settling": int(t_stop is not None),
                })
        except KeyboardInterrupt:
            status = "aborted"
            print("\nAborted - stopping the ball.")
        finally:
            with contextlib.suppress(Exception):
                env.emergency_stop()

        call_stats = {n: {"n": len(d), "mean_ms": 1e3 * float(np.mean(d)),
                          "p95_ms": 1e3 * float(np.percentile(d, 95)),
                          "max_ms": 1e3 * float(np.max(d)),
                          "errors": api.errors.get(n, 0)}
                      for n, d in api.calls.items() if d}
        with contextlib.suppress(Exception):
            env.vis.close()

    return summarise(mode, run_idx, rows, call_stats, start, goal, straight,
                     status, step_errors, api.errors, max_hz), rows


def summarise(mode, run_idx, rows, call_stats, start, goal, straight, status, step_errors, api_errors, max_hz=None):
    if len(rows) < 3:
        return {"mode": mode, "run": run_idx, "status": status, "steps": len(rows)}
    t = np.array([r["t"] for r in rows])
    dt = np.diff(t)
    step_ms = np.array([r["step_s"] for r in rows]) * 1e3
    xy = np.array([[r["x"], r["y"]] for r in rows])
    fresh = np.array([r["fresh"] for r in rows], bool)
    age = np.array([r["age_s"] for r in rows]) * 1e3
    moving = np.array([r["speed_cmd"] > 0 for r in rows])
    settling = np.array([r["settling"] for r in rows], bool)
    drive = ~settling

    # Gaps between successive position changes during the drive phase (the
    # ball's sensor stream as this loop sees it; wall-clock, so comparable
    # between a 5 Hz and a 480 Hz loop). Once the ball is stationary the
    # position legitimately stops changing, so the settle phase is excluded.
    change_t = t[fresh & drive]
    gaps = np.diff(change_t) * 1e3 if len(change_t) > 2 else np.array([np.nan])
    longest = float(np.nanmax(gaps)) / 1e3
    writes = sum(call_stats.get(n, {}).get("n", 0) for n in ("set_heading", "set_speed"))

    path = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    direction = (goal - start) / max(straight, 1e-9)
    overshoot = float(max(0.0, np.max((xy - start) @ direction) - straight))
    t_goal = next((r["t"] for r in rows if r["settling"]), None)

    d = dt[drive[1:]] if drive[1:].any() else dt
    mv_fresh = fresh[moving]
    return {
        "mode": mode, "run": run_idx, "status": status,
        "steps": len(rows), "duration_s": float(t[-1]),
        "time_to_goal_s": t_goal, "steps_to_goal": int(drive.sum()) if t_goal else None,
        "loop_hz": float(1.0 / d.mean()),
        "dt_mean_ms": float(d.mean() * 1e3), "dt_median_ms": float(np.median(d) * 1e3),
        "dt_p95_ms": float(np.percentile(d, 95) * 1e3), "dt_p99_ms": float(np.percentile(d, 99) * 1e3),
        "dt_max_ms": float(d.max() * 1e3), "dt_jitter_std_ms": float(d.std() * 1e3),
        "step_call_mean_ms": float(step_ms.mean()),
        "fresh_fraction_moving": float(mv_fresh.mean()) if mv_fresh.size else None,
        "unique_pos_per_s": float(fresh[drive].sum() / max(t[drive][-1] - t[0], 1e-9)),
        "data_age_mean_ms": float(age[moving & drive].mean()) if (moving & drive).any() else None,
        "data_age_p95_ms": float(np.percentile(age[moving & drive], 95)) if (moving & drive).any() else None,
        "longest_stale_ms": float(longest * 1e3),
        "sample_gap_median_ms": float(np.nanmedian(gaps)), "sample_gap_p95_ms": float(np.nanpercentile(gaps, 95)),
        "ble_writes_per_s": float(writes / max(t[-1], 1e-9)), "max_hz": max_hz,
        "step_errors": step_errors, "api_errors": dict(api_errors),
        "collisions": int(sum(r["collision"] for r in rows)),
        "final_dist_m": float(rows[-1]["dist_goal"]),
        "min_dist_m": float(min(r["dist_goal"] for r in rows)),
        "overshoot_m": overshoot, "path_len_m": path,
        "path_efficiency": float(straight / path) if path > 1e-6 else None,
        "call_stats": call_stats,
    }


def save_run(out_dir, mode, run_idx, summary, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{mode}_run{run_idx}"
    if rows:
        import csv
        with open(f"{stem}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(f"{stem}.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Saved {stem}.csv / .json")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
METRICS = [  # (key, label, unit, better: 'high'|'low'|None)
    ("loop_hz", "Control loop rate", "Hz", "high"),
    ("dt_mean_ms", "Step period, mean", "ms", "low"),
    ("dt_median_ms", "Step period, median", "ms", "low"),
    ("dt_p95_ms", "Step period, p95", "ms", "low"),
    ("dt_p99_ms", "Step period, p99", "ms", "low"),
    ("dt_max_ms", "Step period, worst", "ms", "low"),
    ("dt_jitter_std_ms", "Step period jitter (std)", "ms", "low"),
    ("unique_pos_per_s", "Fresh position samples", "/s", "high"),
    ("fresh_fraction_moving", "Steps with a new position (moving)", "frac", None),
    ("sample_gap_median_ms", "Time between new positions, median", "ms", "low"),
    ("sample_gap_p95_ms", "Time between new positions, p95", "ms", "low"),
    ("longest_stale_ms", "Longest gap with no new position", "ms", "low"),
    ("data_age_mean_ms", "Position age at decision, mean*", "ms", None),
    ("data_age_p95_ms", "Position age at decision, p95*", "ms", None),
    ("ble_writes_per_s", "BLE command writes", "/s", None),
    ("step_errors", "Failed steps (exceptions)", "", "low"),
    ("collisions", "Collisions flagged", "", "low"),
    ("time_to_goal_s", "Time to goal", "s", "low"),
    ("steps_to_goal", "Control steps to goal", "", None),
    ("final_dist_m", "Final distance to goal", "m", "low"),
    ("overshoot_m", "Overshoot past goal", "m", "low"),
    ("path_efficiency", "Path efficiency (1 = straight)", "", "high"),
]


def load_summaries(out_dir):
    by_mode = {m: [] for m in MODES}
    for p in sorted(glob.glob(str(out_dir / "*_run*.json"))):
        s = json.load(open(p))
        if s.get("mode") in by_mode and "loop_hz" in s:
            by_mode[s["mode"]].append(s)
    return by_mode


def mean_of(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def fmt(v):
    if v is None:
        return "-"
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}" if abs(v) < 0.01 else f"{v:.2f}"


def print_report(by_mode):
    modes = [m for m in MODES if by_mode[m]]
    if not modes:
        print("No saved runs to report.")
        return
    print("\n" + "=" * 86)
    hdr = f"{'Metric':<38}" + "".join(f"{m + f' (n={len(by_mode[m])})':>17}" for m in modes)
    if len(modes) == 2:
        hdr += f"{'fast vs stock':>17}"
    print(hdr)
    print("-" * 86)
    for key, label, unit, better in METRICS:
        vals = {m: mean_of(by_mode[m], key) for m in modes}
        line = f"{label + (f' [{unit}]' if unit else ''):<38}" + "".join(f"{fmt(vals[m]):>17}" for m in modes)
        if len(modes) == 2 and vals["stock"] and vals["fast"]:
            ratio = vals["fast"] / vals["stock"]
            line += f"{ratio:>15.2f}x" + ("" if better is None else "  +" if (ratio > 1) == (better == "high") else "  -")
        print(line)
    print("=" * 86)
    print("(+/- = fast is better/worse than stock.  Means across runs.  Goal outcomes:)")
    print("* age = time since the position last changed, seen from the loop. It is a lower bound: a slow\n"
          "  loop reads the sample right after it arrives (so ~0), and it hides samples missed between steps -\n"
          "  use 'time between new positions' for the real sensor-data cadence. Drive phase only.")
    for m in modes:
        outcomes = [r["status"] for r in by_mode[m]]
        print(f"  {m:<6} " + ", ".join(f"{o}" for o in outcomes))

    print("\nPer-API-call latency (mean ms / p95 ms / max ms, mean across runs):")
    for m in modes:
        names = sorted({n for r in by_mode[m] for n in r["call_stats"]})
        print(f"  {m}:")
        for n in names:
            cs = [r["call_stats"][n] for r in by_mode[m] if n in r["call_stats"]]
            errs = sum(c["errors"] for c in cs)
            print(f"    {n:<18} {np.mean([c['mean_ms'] for c in cs]):8.2f} "
                  f"{np.mean([c['p95_ms'] for c in cs]):8.2f} {np.max([c['max_ms'] for c in cs]):8.2f}"
                  f"   calls={sum(c['n'] for c in cs)}" + (f"  ERRORS={errs}" if errs else ""))


def make_plot(by_mode, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot.")
        return
    colors = {"stock": "#d9534f", "fast": "#2b8cbe"}
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for m in MODES:
        for i, s in enumerate(by_mode[m]):
            p = out_dir / f"{m}_run{s['run']}.csv"
            if not p.exists():
                continue
            d = np.genfromtxt(p, delimiter=",", names=True)
            lab = m if i == 0 else None
            c = colors[m]
            drive = d["settling"] == 0
            ax[0, 0].plot(d["t"], d["dist_goal"], color=c, alpha=.8, label=lab)
            ax[0, 1].plot(d["x"] - d["x"][0], d["y"] - d["y"][0], color=c, alpha=.8, label=lab)
            ax[0, 2].hist(np.diff(d["t"])[drive[1:]] * 1e3, bins=np.logspace(-0.5, 3, 50),
                          color=c, alpha=.5, label=lab)
            ax[1, 0].plot(d["t"][drive], d["age_s"][drive] * 1e3, color=c, alpha=.6, label=lab)
            ax[1, 1].plot(d["t"], d["speed_cmd"], color=c, alpha=.8, label=lab)
    ax[0, 1].plot([0, GOAL_OFFSET[0]], [0, GOAL_OFFSET[1]], "k--", lw=.8, label="straight line")
    ax[0, 1].plot(*GOAL_OFFSET, "k*", ms=12)
    for a, t, xl, yl in [(ax[0, 0], "Distance to goal", "time (s)", "m"),
                         (ax[0, 1], "Path (relative to start)", "x (m)", "y (m)"),
                         (ax[0, 2], "Step period distribution (log x)", "ms", "steps"),
                         (ax[1, 0], "Age of position at each decision (drive phase)", "time (s)", "ms"),
                         (ax[1, 1], "Commanded speed", "time (s)", "m/s")]:
        a.set_title(t); a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=.3); a.legend()
    ax[0, 1].set_aspect("equal", "datalim")
    ax[0, 2].set_xscale("log")

    a = ax[1, 2]
    names = sorted({n for m in MODES for r in by_mode[m] for n in r["call_stats"]
                    if n.startswith(("set_", "get_"))})
    x = np.arange(len(names))
    for j, m in enumerate(MODES):
        vals = [np.mean([r["call_stats"][n]["mean_ms"] for r in by_mode[m] if n in r["call_stats"]] or [0])
                for n in names]
        a.bar(x + (j - .5) * .38, vals, .38, color=colors[m], label=m)
    a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels(names, rotation=40, ha="right")
    a.set_title("Mean latency per API call (log)"); a.set_ylabel("ms"); a.grid(alpha=.3, axis="y"); a.legend()
    fig.suptitle("Lab 1 drive-to-goal: stock SpheroEduAPI vs fast_comms")
    fig.tight_layout()
    path = out_dir / "comparison.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nPlot saved to {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("stock", "fast", "both"), default="both")
    ap.add_argument("--runs", type=int, default=1, help="Runs per mode (default 1)")
    ap.add_argument("--max-time", type=float, default=60.0, help="Per-run timeout in s (default 60)")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="Seconds to keep sampling after the controller says Stop, to measure "
                         "final resting error (default 1.5)")
    ap.add_argument("--max-hz", type=float, default=None,
                    help="Throttle the control loop to this rate (default: unthrottled). Use ~50 with "
                         "--mode fast to see the link without spinning far faster than the sensors update")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true", help="Only summarise previously saved runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="Use a simulated robot (fake latencies) to check the script without hardware; "
                         "results are NOT hardware measurements")
    a = ap.parse_args()

    out_dir = a.out / "dry_run" if a.dry_run else a.out
    if not a.report:
        modes = MODES if a.mode == "both" else (a.mode,)
        for mi, mode in enumerate(modes):
            for r in range(1, a.runs + 1):
                if not a.dry_run and (mi > 0 or r > 1):
                    input(f"\nReset the ball to a start with ~0.7 m clear in +x/+y, "
                          f"then press ENTER for {mode} run {r}...")
                summary, rows = run_once(mode, r, out_dir, a.max_time, a.settle, a.dry_run, a.max_hz)
                save_run(out_dir, mode, r, summary, rows)
    by_mode = load_summaries(out_dir)
    print_report(by_mode)
    make_plot(by_mode, out_dir)


if __name__ == "__main__":
    main()
