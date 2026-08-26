#!/usr/bin/env python3
"""
calibrate_noise.py  -  Turn a logged robot run into EKF starting values.

Reads a CSV log (lab1-style OR lab2 automark-style) and reports:

  1. R  - measurement-noise covariance, from  (measured - ground_truth) residuals.
  2. Q  - a process-noise ballpark, from one-step prediction residuals of your
          own dynamics() model (imported from EKF.py in the same folder).
  3. Position Mahalanobis / chi-square calibration, from the EKF estimate + its
          logged covariance vs ground truth  -  the exact metric the automarker
          uses (2 DoF, 95% gate). Also saves a NIS-style plot next to the CSV.

This does NOT implement or run the EKF for you - it only measures data and, for
part 3, reads the estimate/covariance your filter already logged. Parts 1-2 work
on any run with ground-truth columns; part 3 needs a run where the filter was
actually producing estimates (non-nan est/cov columns).

Usage:
    python calibrate_noise.py <path-to-log.csv>
    python calibrate_noise.py <path-to-log.csv> --no-plot
"""

import argparse
import os
import sys
import numpy as np


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def wrap_angle(a):
    """Normalise angle(s) to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def col(data, *names):
    """Return the first matching column (by name) as a float array, or None."""
    for n in names:
        if n in data.dtype.names:
            return np.asarray(data[n], dtype=float)
    return None


def finite_mask(*arrays):
    """Boolean mask where every supplied array is finite (non-nan/inf)."""
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def fmt_cov(M, labels):
    """Pretty-print a covariance matrix with row/col labels."""
    w = max(len(l) for l in labels)
    head = " " * (w + 2) + "  ".join(f"{l:>12}" for l in labels)
    lines = [head]
    for i, l in enumerate(labels):
        row = "  ".join(f"{M[i, j]:12.6g}" for j in range(M.shape[1]))
        lines.append(f"{l:>{w}}  {row}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 1. Measurement noise R
# ----------------------------------------------------------------------------
def measure_R(data):
    print("\n" + "=" * 70)
    print("1. MEASUREMENT NOISE  R   =  cov( measured - ground_truth )")
    print("=" * 70)

    channels = []  # (label, residual_array)

    # position: raw odometry vs ground truth
    ox, oy = col(data, "odom_x"), col(data, "odom_y")
    gx, gy = col(data, "gt_x"), col(data, "gt_y")
    if ox is not None and gx is not None:
        channels.append(("x", ox - gx))
        channels.append(("y", oy - gy))

    # heading (wrapped) and speed
    h, gh = col(data, "heading"), col(data, "gt_heading")
    if h is not None and gh is not None:
        channels.append(("heading", wrap_angle(h - gh)))
    sp, gsp = col(data, "speed"), col(data, "gt_speed")
    if sp is not None and gsp is not None:
        channels.append(("speed", sp - gsp))

    if not channels:
        print("  (no measured-vs-ground-truth column pairs found - skipping)")
        return

    labels = [c[0] for c in channels]
    resid = np.vstack([c[1] for c in channels])
    mask = np.all(np.isfinite(resid), axis=0)
    n = int(mask.sum())
    if n < 3:
        print(f"  Only {n} usable rows - too few to estimate R. Log a longer run.")
        return
    resid = resid[:, mask]

    R = np.cov(resid)  # full covariance across channels
    R = np.atleast_2d(R)
    print(f"\n  Using {n} rows.\n")
    print("  Per-channel std (sqrt of variance):")
    for i, l in enumerate(labels):
        print(f"    {l:>8}:  std = {np.sqrt(R[i, i]):.6g}   var = {R[i, i]:.6g}")
    print("\n  Full R (with cross-covariances):")
    print(fmt_cov(R, labels))
    print("\n  Diagonal R you can paste (independence assumption):")
    diag = ", ".join(f"{R[i, i]:.6g}" for i in range(len(labels)))
    print(f"    self.R = np.diag([{diag}])   # order: {labels}")


# ----------------------------------------------------------------------------
# 2. Process noise Q  (one-step prediction residuals of dynamics())
# ----------------------------------------------------------------------------
def measure_Q(data, dynamics):
    print("\n" + "=" * 70)
    print("2. PROCESS NOISE  Q   =  var( gt_next - dynamics(gt_now, action) )")
    print("=" * 70)

    if dynamics is None:
        print("  dynamics() could not be imported from EKF.py - skipping Q.")
        print("  (run this script from inside the lab2 folder so 'import EKF' works)")
        return

    gx, gy = col(data, "gt_x"), col(data, "gt_y")
    gh, gs = col(data, "gt_heading"), col(data, "gt_speed")
    hc, sc = col(data, "heading_cmd"), col(data, "speed_cmd")
    if any(v is None for v in (gx, gy, gh, gs, hc, sc)):
        print("  Missing gt_* or *_cmd columns - cannot compute prediction residuals.")
        return

    labels = ["x", "y", "heading", "speed"]
    resids = {l: [] for l in labels}
    used = 0
    for k in range(len(gx) - 1):
        state = np.array([gx[k], gy[k], gh[k], gs[k]], dtype=float)
        action = np.array([sc[k], hc[k]], dtype=float)   # dynamics wants [speed_cmd, heading_cmd]
        nxt = np.array([gx[k + 1], gy[k + 1], gh[k + 1], gs[k + 1]], dtype=float)
        if not (np.all(np.isfinite(state)) and np.all(np.isfinite(action))
                and np.all(np.isfinite(nxt))):
            continue
        pred = np.asarray(dynamics(state, action), dtype=float)
        r = nxt - pred
        r[2] = wrap_angle(r[2])  # heading residual wrapped
        for i, l in enumerate(labels):
            resids[l].append(r[i])
        used += 1

    if used < 3:
        print(f"  Only {used} usable transitions - too few. Log a longer run.")
        return

    print(f"\n  Using {used} one-step transitions.\n")
    print("  Per-state prediction-residual variance (Q ballpark):")
    qvals = []
    for l in labels:
        arr = np.asarray(resids[l])
        v = float(np.var(arr))
        qvals.append(v)
        print(f"    {l:>8}:  std = {np.sqrt(v):.6g}   var = {v:.6g}")
    diag = ", ".join(f"{v:.6g}" for v in qvals)
    print("\n  Diagonal Q ballpark you can paste (then tune via NIS):")
    print(f"    self.Q = np.diag([{diag}])   # order: {labels}")
    print("\n  NOTE: this is a starting scale only. Q absorbs un-modelled effects,")
    print("        so finalise it with the chi-square loop in instructions.md.")


# ----------------------------------------------------------------------------
# 3. Position Mahalanobis / chi-square calibration  (the automarker metric)
# ----------------------------------------------------------------------------
def measure_calibration(data, csv_path, make_plot):
    print("\n" + "=" * 70)
    print("3. POSITION CALIBRATION   (Mahalanobis / chi-square, 2 DoF)")
    print("=" * 70)

    # estimate: est_* (lab1) or real_* (automark);  truth: gt_* or sim_*
    ex = col(data, "est_x", "real_x")
    ey = col(data, "est_y", "real_y")
    tx = col(data, "gt_x", "sim_x")
    ty = col(data, "gt_y", "sim_y")
    cxx = col(data, "cov_x", "P_xx")
    cyy = col(data, "cov_y", "P_yy")
    cxy = col(data, "cov_xy", "P_xy")

    if any(v is None for v in (ex, ey, tx, ty, cxx, cyy, cxy)):
        print("  Missing estimate / truth / covariance columns - skipping.")
        print("  (this part needs a run where your filter logged est + covariance)")
        return

    upper = 5.991  # chi2 0.95 quantile, 2 DoF (upper gate)
    lower = 0.103   # chi2 0.05 quantile, 2 DoF (lower gate)

    d2_list, steps = [], []
    for k in range(len(ex)):
        e = np.array([ex[k] - tx[k], ey[k] - ty[k]])
        P = np.array([[cxx[k], cxy[k]], [cxy[k], cyy[k]]])
        if not (np.all(np.isfinite(e)) and np.all(np.isfinite(P))):
            continue
        try:
            d2 = float(e @ np.linalg.inv(P) @ e)
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(d2):
            d2_list.append(d2)
            steps.append(k)

    if len(d2_list) < 3:
        print(f"  Only {len(d2_list)} usable rows (est/cov mostly nan?).")
        print("  Run this again on a log where the EKF was producing estimates.")
        return

    d2 = np.asarray(d2_list)
    mean_d2 = float(np.mean(d2))          # NEES: expected ~2 for 2 DoF if calibrated
    mean_d = float(np.mean(np.sqrt(d2)))  # mean Mahalanobis *distance*
    pass_rate = float(np.mean(d2 <= upper))

    print(f"\n  Using {len(d2)} rows.\n")
    print(f"  Mean Mahalanobis distance (sqrt):     {mean_d:.3f}   (marker threshold <= 4.0)")
    print(f"  Mean squared / NEES (compare to 2):   {mean_d2:.3f}   (~2.0 = well calibrated)")
    print(f"  Chi-square pass rate (d2 <= 5.991):   {pass_rate:.3f}   (marker threshold >= 0.90)")
    if mean_d2 > 2.5:
        print("  -> NEES high: filter OVERCONFIDENT. Increase Q (and/or R).")
    elif mean_d2 < 1.5:
        print("  -> NEES low:  filter PESSIMISTIC.   Decrease Q (and/or R).")
    else:
        print("  -> NEES in a healthy range.")

    if not make_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(steps, d2, marker=".", lw=1, label="NIS / Mahalanobis$^2$")
        ax.axhline(upper, ls="--", color="tab:red", label="95% upper gate (5.991)")
        ax.axhline(lower, ls="--", color="tab:orange", label="5% lower gate (0.103)")
        ax.axhline(2.0, ls=":", color="gray", label="expected mean (2 DoF)")
        ax.set_xlabel("step")
        ax.set_ylabel("squared Mahalanobis distance")
        ax.set_title(f"Position NIS  -  pass rate {pass_rate:.2f}, mean NEES {mean_d2:.2f}")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        out = os.path.splitext(csv_path)[0] + "_nis.png"
        fig.savefig(out, dpi=130)
        print(f"\n  NIS plot saved to: {out}")
    except Exception as e:
        print(f"\n  (plot skipped: {e})")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Estimate EKF R/Q and check calibration from a log CSV.")
    ap.add_argument("csv", help="path to the log CSV")
    ap.add_argument("--no-plot", action="store_true", help="skip the NIS plot")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"File not found: {args.csv}")

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    if data.dtype.names is None:
        sys.exit("Could not parse a header row from the CSV.")
    print(f"Loaded {len(data)} rows from {args.csv}")
    print(f"Columns: {', '.join(data.dtype.names)}")

    # Try to import the student's dynamics() from EKF.py next to the CSV or cwd.
    dynamics = None
    for cand in (os.path.dirname(os.path.abspath(args.csv)), os.getcwd()):
        if os.path.exists(os.path.join(cand, "EKF.py")):
            sys.path.insert(0, cand)
            try:
                from EKF import dynamics as _dyn
                dynamics = _dyn
            except Exception as e:
                print(f"(could not import dynamics from EKF.py: {e})")
            break

    measure_R(data)
    measure_Q(data, dynamics)
    measure_calibration(data, args.csv, make_plot=not args.no_plot)
    print()


if __name__ == "__main__":
    main()
