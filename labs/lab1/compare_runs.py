"""Compare a sim run and a real run from their CSV logs.

Usage (from the repo root or labs/lab1):
    python labs/lab1/compare_runs.py
    python labs/lab1/compare_runs.py path/to/sim.csv path/to/real.csv -o out.png
    python labs/lab1/compare_runs.py --overlay-saved-real
    python labs/lab1/compare_runs.py --saved-real logs/saved/lab1_real_*.csv

Defaults to logs/lab1_sim_latest.csv and logs/lab1_real_latest.csv.
Produces one figure: overlaid XY trajectories, distance-to-goal, speeds and
headings vs step, plus the per-step sim-vs-real position RMSE (the lab's
calibration metric). Works with any CSVs written by the Visualiser logger,
including files saved with the S key.

--overlay-saved-real (or --saved-real PATH [PATH ...]) additionally draws
every past real run from logs/saved/ as faint gray background traces, so you
can see run-to-run real-robot variability alongside the sim/real comparison.
"""
import argparse
import csv
import glob
import math
import os

import numpy as np
import matplotlib.pyplot as plt

GOAL = (0.5, 0.5)
GOAL_TOL = 0.1


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(name, "nan") or "nan"))
            except ValueError:
                out.append(float("nan"))
        return np.array(out)

    return {
        "x": col("odom_x"), "y": col("odom_y"),
        "heading": col("heading"), "speed": col("speed"),
        "heading_cmd": col("heading_cmd"), "speed_cmd": col("speed_cmd"),
        "n": len(rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sim_csv", nargs="?", default="logs/lab1_sim_latest.csv")
    ap.add_argument("real_csv", nargs="?", default="logs/lab1_real_latest.csv")
    ap.add_argument("-o", "--out", default="logs/compare_sim_real.png")
    ap.add_argument("--no-show", action="store_true", help="Save the PNG only")
    ap.add_argument("--overlay-saved-real", action="store_true",
                     help="Overlay every past real run in logs/saved/ (lab1_real_*.csv) "
                          "as faint background traces")
    ap.add_argument("--saved-real", nargs="*", default=None,
                     help="Explicit saved real CSVs to overlay (overrides --overlay-saved-real's auto-glob)")
    args = ap.parse_args()

    if os.path.abspath(args.sim_csv) == os.path.abspath(args.real_csv):
        print("WARNING: sim and real point at the SAME file - RMSE will be 0 "
              "and the comparison meaningless.")
    sim, real = load(args.sim_csv), load(args.real_csv)

    saved_paths = []
    if args.saved_real:
        saved_paths = args.saved_real
    elif args.overlay_saved_real:
        saved_paths = sorted(glob.glob("logs/saved/lab1_real_*.csv"))
    # Don't re-plot whichever file is already the primary "real" trace.
    saved_paths = [p for p in saved_paths if os.path.abspath(p) != os.path.abspath(args.real_csv)]
    saved_runs = [(p, load(p)) for p in saved_paths]
    if saved_paths:
        print(f"Overlaying {len(saved_runs)} saved real run(s) from logs/saved/")

    n = min(sim["n"], real["n"])
    rmse = float(np.sqrt(np.nanmean(
        (sim["x"][:n] - real["x"][:n]) ** 2 + (sim["y"][:n] - real["y"][:n]) ** 2)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"sim: {os.path.basename(args.sim_csv)}  vs  real: {os.path.basename(args.real_csv)}"
        f"   |   position RMSE over {n} steps: {rmse:.3f} m",
        fontsize=11)

    ax = axes[0][0]
    for i, (p, s) in enumerate(saved_runs):
        ax.plot(s["x"], s["y"], "-", color="lightgray", alpha=0.5, lw=1, zorder=0,
                label="saved real runs" if i == 0 else None)
    ax.plot(sim["x"], sim["y"], ".-", color="tab:green", label="sim")
    ax.plot(real["x"], real["y"], ".-", color="tab:blue", label="real")
    ax.plot(sim["x"][0], sim["y"][0], "s", color="tab:green")
    ax.plot(real["x"][0], real["y"][0], "s", color="tab:blue")
    ax.add_patch(plt.Circle(GOAL, GOAL_TOL, fill=False, color="gold", lw=2))
    ax.plot(*GOAL, "*", color="gold", ms=14, label="goal")
    ax.set_title("Trajectories (squares = start)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[0][1]
    for i, (p, s) in enumerate(saved_runs):
        ax.plot(np.hypot(s["x"] - GOAL[0], s["y"] - GOAL[1]), color="lightgray", alpha=0.5,
                lw=1, zorder=0, label="saved real runs" if i == 0 else None)
    ax.plot(np.hypot(sim["x"] - GOAL[0], sim["y"] - GOAL[1]), color="tab:green", label="sim")
    ax.plot(np.hypot(real["x"] - GOAL[0], real["y"] - GOAL[1]), color="tab:blue", label="real")
    ax.axhline(GOAL_TOL, color="gold", ls="--", label=f"tolerance {GOAL_TOL} m")
    ax.set_title("Distance to goal")
    ax.set_xlabel("step"); ax.set_ylabel("dist [m]"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1][0]
    for i, (p, s) in enumerate(saved_runs):
        ax.plot(s["speed"], color="lightgray", alpha=0.5, lw=1, zorder=0,
                label="saved real runs" if i == 0 else None)
    ax.plot(sim["speed"], color="tab:green", label="sim measured")
    ax.plot(real["speed"], color="tab:blue", label="real measured")
    ax.plot(sim["speed_cmd"], color="tab:green", ls=":", alpha=0.7, label="sim cmd")
    ax.plot(real["speed_cmd"], color="tab:blue", ls=":", alpha=0.7, label="real cmd")
    ax.set_title("Speed")
    ax.set_xlabel("step"); ax.set_ylabel("speed [m/s]"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1][1]
    for i, (p, s) in enumerate(saved_runs):
        ax.plot(np.degrees(s["heading"]), color="lightgray", alpha=0.5, lw=1, zorder=0,
                label="saved real runs" if i == 0 else None)
    ax.plot(np.degrees(sim["heading"]), color="tab:green", label="sim measured")
    ax.plot(np.degrees(real["heading"]), color="tab:blue", label="real measured")
    ax.plot(np.degrees(sim["heading_cmd"]), color="tab:green", ls=":", alpha=0.7, label="sim cmd")
    ax.plot(np.degrees(real["heading_cmd"]), color="tab:blue", ls=":", alpha=0.7, label="real cmd")
    ax.set_title("Heading")
    ax.set_xlabel("step"); ax.set_ylabel("heading [deg]"); ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"RMSE (sim vs real, {n} steps): {rmse:.3f} m  (lab spec: <= 0.20 m)")
    print(f"Saved {args.out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
