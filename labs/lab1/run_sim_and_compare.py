"""Run a 100-step sim, then compare it against the last real run.

Runs lab1's actual sim env + controller headlessly (no pygame window, no
keypresses needed), writes logs/lab1_sim_latest.csv, then opens the
compare_runs.py plot against logs/lab1_real_latest.csv so you can look at
the result.

Usage (from anywhere):
    python labs/lab1/run_sim_and_compare.py
    python labs/lab1/run_sim_and_compare.py --steps 100 --seed 0
"""
import argparse
import os
import sys

# Headless pygame - no window needed to run the sim loop.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame  # noqa: E402  (after SDL_VIDEODRIVER is set)

import lab1  # noqa: E402
import compare_runs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true",
                         help="Save the comparison PNG only, don't pop up the plot")
    parser.add_argument("--overlay-saved-real", action="store_true",
                         help="Also overlay every past real run in logs/saved/ as faint background traces")
    args = parser.parse_args()

    pygame.init()  # normally done implicitly by run_session()'s first render()

    sim_env = lab1.make_sim_env()
    try:
        result, frames = lab1.run_one(sim_env, sim=True, run_idx=1,
                                       steps=args.steps, seed=args.seed)
        print(f"Sim run finished: {len(frames)} rows logged "
              f"to logs/lab1_sim_latest.csv")
    finally:
        sim_env.stop_logging()
        sim_env.close()

    compare_argv = [sys.argv[0]]
    if args.no_show:
        compare_argv.append("--no-show")
    if args.overlay_saved_real:
        compare_argv.append("--overlay-saved-real")
    sys.argv = compare_argv
    compare_runs.main()


if __name__ == "__main__":
    main()
