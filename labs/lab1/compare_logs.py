"""Compare simulator vs real robot logs: x/y position, heading, and speed."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_SIM_LOG = Path(__file__).resolve().parents[2] / "logs" / "lab1_sim.csv"
DEFAULT_REAL_LOG = Path(__file__).resolve().parents[2] / "logs" / "lab1_real.csv"


def plot_comparison(sim_csv: Path, real_csv: Path):
    sim = pd.read_csv(sim_csv)
    real = pd.read_csv(real_csv)

    sim_t = sim.index * 0.1
    real_t = real.index * 0.1

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(sim["gt_x"], sim["gt_y"], label="Sim")
    ax.plot(real["gt_x"], real["gt_y"], label="Real")
    ax.scatter(sim["setpoint_x"].iloc[0], sim["setpoint_y"].iloc[0], c="black", marker="x", label="Goal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("XY Trajectory")
    ax.legend()
    ax.axis("equal")

    ax = axes[0, 1]
    ax.plot(sim_t, sim["gt_x"], label="Sim x")
    ax.plot(real_t, real["gt_x"], label="Real x")
    ax.plot(sim_t, sim["gt_y"], label="Sim y", linestyle="--")
    ax.plot(real_t, real["gt_y"], label="Real y", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("X/Y Position vs Time")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(sim_t, sim["heading"], label="Sim")
    ax.plot(real_t, real["heading"], label="Real")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heading (rad)")
    ax.set_title("Heading vs Time")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(sim_t, sim["speed"], label="Sim")
    ax.plot(real_t, real["speed"], label="Real")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed vs Time")
    ax.legend()

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", type=Path, default=DEFAULT_SIM_LOG, help="Path to simulator log CSV")
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL_LOG, help="Path to real robot log CSV")
    args = parser.parse_args()

    plot_comparison(args.sim, args.real)


if __name__ == "__main__":
    main()
