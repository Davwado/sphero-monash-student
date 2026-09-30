"""Train a residual dynamics model from real-robot logs (numpy only, no torch).

    learned(state, action) = analytic_dynamics(state, action) + residual(state, action)

Usage (from labs/lab4); the last file is held out for validation:
    python train_dynamics.py ../../logs/real_run1_v015.csv ../../logs/real_run3.csv ../../logs/real_run4.csv

Then in lab4.py (state speed must come from measured_speed(), not the robot's reading):
    from train_dynamics import make_learned_dynamics
    ekf = EKF(dt=2.95, dynamics_fn=make_learned_dynamics("dyn_weights.npz"))
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab3"))
from EKF import dynamics as analytic_dynamics, wrap_angle  # noqa: E402


# Only speed and commanded speed. Nothing map-relative (the model would learn "the ball
# drifts towards +y" from where it happened to be driven), and no heading error: the
# robot reports its commanded heading, so heading error is ~0 in all the data and any
# weight on it is fitted to noise.
def features(state, action):
    speed = state[3]
    speed_cmd = action[0]
    return np.array([1.0, speed, speed_cmd, speed * speed_cmd])


def to_body(dx, dy, heading):
    """World displacement -> (along heading, sideways). Heading 0 = +y, pi/2 = +x."""
    s, c = np.sin(heading), np.cos(heading)
    return dx * s + dy * c, dx * c - dy * s


def to_world(along, side, heading):
    s, c = np.sin(heading), np.cos(heading)
    return along * s + side * c, along * c - side * s


def _f(row, col):
    try:
        return float(row[col])
    except (KeyError, ValueError):
        return float("nan")


# The robot's reported speed just echoes the command (it reads 0 while the ball coasts),
# so speed is measured from odometry instead. Dividing the per-step distance by the
# analytic model's dt keeps it in the units dynamics() expects: x += speed * sin(h) * dt.
SPEED_DT = 2.95


PIN_STEPS = 6
PIN_DIST = 0.02


def measured_speed(prev_xy, xy):
    return float(np.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1])) / SPEED_DT


def load_transitions(path):
    """Return (states, actions, next_states) for consecutive, clean rows of one CSV."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    S = np.array([[_f(r, "odom_x"), _f(r, "odom_y"), _f(r, "heading"), np.nan] for r in rows])
    S[1:, 3] = np.hypot(np.diff(S[:, 0]), np.diff(S[:, 1])) / SPEED_DT
    A = np.array([[_f(r, "speed_cmd"), _f(r, "heading_cmd")] for r in rows])
    C = np.array([_f(r, "collision") for r in rows]) > 0
    # Reversing flips the reported heading by 180 deg, which the analytic model can't represent.
    ok = ~np.isnan(S).any(1) & ~np.isnan(A).any(1) & (A[:, 0] >= 0)
    # A ball pinned on a wall (driven forward but not moving) is the maze, not the robot's
    # physics, and the collision flag often misses it: drop those windows too.
    moved = np.hypot(S[:, 0] - np.roll(S[:, 0], PIN_STEPS), S[:, 1] - np.roll(S[:, 1], PIN_STEPS))
    pushing = np.convolve(A[:, 0] > 0.01, np.ones(PIN_STEPS), "full")[:len(rows)] >= PIN_STEPS
    pinned = pushing & (moved < PIN_DIST)
    pinned[:PIN_STEPS] = False
    for i in np.where(pinned)[0]:
        pinned[max(0, i - PIN_STEPS):i] = True
    # Measured speed at row i spans rows i-1..i, so a collision there taints it too.
    keep = [i for i in range(1, len(rows) - 1)
            if ok[i] and ok[i + 1] and not (C[i - 1] or C[i] or C[i + 1] or pinned[i] or pinned[i + 1])]
    return S[keep], A[keep], S[[i + 1 for i in keep]]


def residual_targets(S, A, S2):
    R = np.zeros_like(S2)
    for i in range(len(S)):
        pred = np.asarray(analytic_dynamics(S[i], A[i]), dtype=float)
        R[i] = S2[i] - pred
        R[i, 0], R[i, 1] = to_body(R[i, 0], R[i, 1], pred[2])
        R[i, 2] = wrap_angle(R[i, 2])
    return R


def fit_ridge(Phi, R, alpha):
    mu, sd = Phi[:, 1:].mean(0), Phi[:, 1:].std(0) + 1e-8
    Z = np.hstack([np.ones((len(Phi), 1)), (Phi[:, 1:] - mu) / sd])
    reg = alpha * np.eye(Z.shape[1])
    reg[0, 0] = 0.0
    W = np.linalg.solve(Z.T @ Z + reg, Z.T @ R)
    return W, mu, sd


def apply_ridge(Phi, W, mu, sd, z_clip=None):
    Z = (Phi[:, 1:] - mu) / sd
    if z_clip is not None:
        Z = np.clip(Z, -z_clip, z_clip)
    return np.hstack([np.ones((len(Phi), 1)), Z]) @ W


def make_learned_dynamics(weights_path, max_pos_step=0.1, z_clip=3.0):
    """Analytic dynamics + learned residual, with safety fallbacks.

    Inputs are clamped to +/- z_clip standard deviations of the training data so the
    model never extrapolates far, and it falls back to the analytic model if the
    residual is NaN or its position part exceeds max_pos_step metres.
    """
    d = np.load(weights_path)
    W, mu, sd = d["W"], d["mu"], d["sd"]

    def learned(state, action):
        base = np.asarray(analytic_dynamics(state, action), dtype=float)
        try:
            res = apply_ridge(features(state, action)[None, :], W, mu, sd, z_clip)[0]
        except Exception:
            return base.astype(np.float32)
        if not np.all(np.isfinite(res)) or np.hypot(res[0], res[1]) > max_pos_step:
            return base.astype(np.float32)
        out = base + res
        out[0], out[1] = base[:2] + to_world(res[0], res[1], base[2])
        out[2] = wrap_angle(out[2])
        out[3] = float(np.clip(out[3], 0.0, 1.0))
        return out.astype(np.float32)

    return learned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="real-robot log CSVs (globs allowed)")
    ap.add_argument("--out", default="dyn_weights.npz")
    args = ap.parse_args()

    paths = list(dict.fromkeys(p for pat in args.csvs for p in sorted(glob.glob(pat))))
    if not paths:
        sys.exit("No CSVs matched.")

    per_file = []
    for p in paths:
        S, A, S2 = load_transitions(p)
        print(f"{os.path.basename(p)}: {len(S)} usable transitions")
        if len(S) > 0:
            per_file.append((S, A, S2))
    if not per_file:
        sys.exit("No usable transitions.")

    if len(per_file) >= 2:
        train, val = per_file[:-1], per_file[-1:]
        print("Validation = last file")
    else:
        S, A, S2 = per_file[0]
        cut = int(0.8 * len(S))
        train, val = [(S[:cut], A[:cut], S2[:cut])], [(S[cut:], A[cut:], S2[cut:])]
        print("Only one file: validation = last 20% of it (more files = a more honest check)")

    def build(sets):
        S = np.vstack([s[0] for s in sets]); A = np.vstack([s[1] for s in sets]); S2 = np.vstack([s[2] for s in sets])
        return S, A, S2, np.array([features(s, a) for s, a in zip(S, A)]), residual_targets(S, A, S2)

    _, _, _, Phi_tr, R_tr = build(train)
    _, _, _, Phi_va, R_va = build(val)
    print(f"train={len(Phi_tr)} val={len(Phi_va)}")

    rmse = lambda M: np.sqrt((M ** 2).mean(0))
    base = rmse(R_va)
    best = None
    for alpha in [0.1, 1, 10, 100, 1000]:
        W, mu, sd = fit_ridge(Phi_tr, R_tr, alpha)
        err = rmse(R_va - apply_ridge(Phi_va, W, mu, sd))
        score = float(err.sum())
        print(f"alpha={alpha:<6} val RMSE [along sideways heading speed] = {np.round(err, 4)}")
        if best is None or score < best[0]:
            best = (score, alpha, err)
    _, alpha, err = best
    use = err < base
    names = np.array(["along", "sideways", "heading", "speed"])
    print(f"\nAnalytic only  : {np.round(base, 4)}")
    print(f"With residual  : {np.round(err, 4)}   (alpha={alpha})")
    print(f"Correction kept for: {', '.join(names[use]) or 'nothing'}"
          f"{'   (off for: ' + ', '.join(names[~use]) + ')' if (~use).any() else ''}")
    print(f"Final model    : {np.round(np.where(use, err, base), 4)}")
    if not use.any():
        print("Residual does NOT beat the analytic model on held-out data - collect more/varied data.")

    W, mu, sd = fit_ridge(np.vstack([Phi_tr, Phi_va]), np.vstack([R_tr, R_va]), alpha)
    W[:, ~use] = 0.0
    # Held-out per-step error of the final model; lab4.py uses it as the EKF's Q.
    np.savez(args.out, W=W, mu=mu, sd=sd, rmse=np.where(use, err, base))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
