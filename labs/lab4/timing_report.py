"""Lab 4 Phase 1 - measure the real control period from a log.

Three constants in this repo are guesses about time, and they disagree:

    labs/lab1/dynamics.py   dt = 0.1
    labs/lab3/EKF.py        dt = 2.95   (used as a PREDICTION timestep)
    labs/lab3/lab3.py       SIM_DT = 0.1

Whichever is right, the EKF propagates the state by dt every step, so being
wrong by 30x means the prediction runs 30x too far and the filter has to be
beaten back into line with inflated Q. This script replaces the guess with a
measurement, using the t_wall column.

It separates two rates that are easy to conflate:

  - CONTROL PERIOD: how often step() actually returns. Sets the EKF dt.
  - LOCATOR REFRESH: how often the position reading actually CHANGES. If the
    control loop runs faster than this, the extra steps carry stale readings -
    logs/fastcomms_lab1_equiv.csv is 87% duplicate poses for this reason, and
    those rows look like data without being data.

Usage:
    python timing_report.py logs/teleop_sphero.csv
    python timing_report.py logs/api.csv logs/fastcomms.csv     # compare paths
"""
import argparse
import csv
import os

import numpy as np

# Poses closer than this are the same reading, not motion.
SAME_POSE_M = 1e-6


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: empty")
    if "t_wall" not in rows[0]:
        raise SystemExit(
            f"{path}: no t_wall column. This log predates the Visualiser change; "
            f"re-record before trusting any timing number from it.")
    return rows


def _col(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out)


def report(path):
    rows = load(path)
    t = _col(rows, "t_wall")
    x, y = _col(rows, "odom_x"), _col(rows, "odom_y")
    spd_cmd = _col(rows, "speed_cmd")

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise SystemExit(f"{path}: no usable t_wall differences")

    moved = np.hypot(np.diff(x), np.diff(y)) > SAME_POSE_M
    commanded = spd_cmd[:-1] > 1e-3
    # Only judge staleness where the ball was actually asked to move; a
    # duplicate pose while commanded to stop is correct behaviour, not a
    # dropped packet.
    judged = commanded & np.isfinite(np.diff(x))
    stale_frac = float((~moved & judged).sum() / max(judged.sum(), 1))

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"  rows                {len(rows)}")
    print(f"  duration            {t[-1] - t[0]:.1f}s")
    print(f"\n  CONTROL PERIOD (diff of t_wall)")
    print(f"    median            {np.median(dt):.4f}s   -> {1/np.median(dt):.1f} Hz")
    print(f"    mean              {dt.mean():.4f}s")
    print(f"    p10 / p90         {np.percentile(dt, 10):.4f}s / {np.percentile(dt, 90):.4f}s")
    print(f"    min / max         {dt.min():.4f}s / {dt.max():.4f}s")

    spread = np.percentile(dt, 90) / max(np.percentile(dt, 10), 1e-9)
    if spread > 3:
        print(f"    NOTE: p90/p10 = {spread:.1f}x - the period is highly irregular, "
              f"so any single dt constant is wrong most of the time. Prefer "
              f"passing the per-step dt from t_wall into predict().")

    print(f"\n  LOCATOR REFRESH")
    print(f"    steps commanded to move      {int(judged.sum())}")
    print(f"    of those, pose unchanged     {int((~moved & judged).sum())} "
          f"({100*stale_frac:.0f}%)")
    if stale_frac > 0.25:
        eff = np.median(dt) / max(1 - stale_frac, 1e-6)
        print(f"    -> logging faster than the locator updates. Effective unique-"
              f"sample period is nearer {eff:.3f}s ({1/eff:.1f} Hz).")
        print(f"       Sampling here inflates the row count without adding "
              f"information.")
    else:
        print(f"    -> healthy; most steps carry a fresh reading.")

    # In the simulator, ground truth and odometry diverge (process noise is
    # applied to odom only). On the real robot state_true IS the odometry, so
    # the two columns are identical. That difference identifies the source,
    # which matters because t_wall on a sim log measures how fast the loop
    # EXECUTED - headless that is ~60Hz - not the simulated timestep. Taking
    # an EKF dt from a sim log would be worse than the guess it replaces.
    gt_x, gt_y = _col(rows, "gt_x"), _col(rows, "gt_y")
    both = np.isfinite(gt_x) & np.isfinite(x)
    is_sim = bool(both.sum()) and float(
        np.nanmax(np.abs(gt_x[both] - x[both]) + np.abs(gt_y[both] - y[both]))) > 1e-6

    if is_sim:
        print(f"\n  SOURCE: simulator (ground truth diverges from odometry).")
        print(f"    t_wall here is loop EXECUTION speed, not the simulated dt "
              f"(that is SIM_DT, set in lab3.py). Do NOT take an EKF dt from "
              f"this log - re-run on the real robot.")
    else:
        print(f"\n  SUGGESTED EKF dt   {np.median(dt):.3f}   "
              f"(currently 2.95 on the real branch in lab3.py)")

    return {"path": path, "median_dt": float(np.median(dt)),
            "stale_frac": stale_frac, "n": len(rows)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args(argv)

    results = [report(p) for p in args.logs if os.path.exists(p)]

    if len(results) > 1:
        print("\n=== comparison ===")
        for r in results:
            print(f"  {os.path.basename(r['path']):<40} "
                  f"{r['median_dt']:.4f}s  stale {100*r['stale_frac']:.0f}%")
        best = min(results, key=lambda r: r["median_dt"])
        print(f"\n  Fastest control period: {os.path.basename(best['path'])}")
        print("  Faster is only better if its stale fraction stays low - a high "
              "rate over a locator that has not refreshed is duplicate rows, not "
              "more data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
