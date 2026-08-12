"""Compare simulator vs real robot logs: x/y position, heading, and speed."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_SIM_LOG = Path(__file__).resolve().parents[2] / "logs" / "lab1_sim.csv"
DEFAULT_REAL_LOG = Path(__file__).resolve().parents[2] / "logs" / "lab1_real.csv"


def get_timestamps(df: pd.DataFrame, csv_path: Path):
    """Return the logged per-row time column ('t'), dropping rows without it.

    Sim logs t = step_count * dt (evenly paced); real logs t = actual
    wall-clock elapsed time (unevenly paced, since real steps aren't rate
    limited to dt). There is no valid way to reconstruct either after the
    fact, so logs missing 't' entirely can't be time-aligned.
    """
    if "t" not in df.columns or df["t"].isna().all():
        raise SystemExit(
            f"'{csv_path}' has no per-row timestamps ('t' column). "
            "This log predates timestamp recording -- rerun lab1.py to "
            "regenerate it before comparing timescales."
        )
    valid = df["t"].notna().to_numpy()
    return df.index.to_numpy()[valid], df["t"].to_numpy()[valid]


def plot_comparison(sim_csv: Path, real_csv: Path):
    sim = pd.read_csv(sim_csv)
    real = pd.read_csv(real_csv)

    sim_idx, sim_t = get_timestamps(sim, sim_csv)
    real_idx, real_t = get_timestamps(real, real_csv)
    sim = sim.iloc[sim_idx]
    real = real.iloc[real_idx]

    # Sim logs far more points than the real robot (sim isn't rate-limited
    # to dt, the real loop is slowed by Bluetooth round-trips), so only
    # keep real rows whose timestamp actually falls within sim's time range
    # -- otherwise the real trace would extend past what sim covers and
    # mislead the time-axis comparison.
    in_range = (real_t >= sim_t.min()) & (real_t <= sim_t.max())
    real = real[in_range]
    real_t = real_t[in_range]

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
