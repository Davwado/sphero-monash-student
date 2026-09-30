"""Lab 4 - fit the learned dynamics model to logged robot runs.

    python train_dynamics_torch.py logs/*.csv ../../logs/lab4_dyn_*.csv
    python train_dynamics_torch.py --out weights/dynamics.pt logs/*.csv

Loads and filters logs with dataset.py (dedup, stale reads, collisions, dt
outliers - see there), fits learned_dynamics.ResidualDynamics, and reports
held-out one-step error against the untrained model, which is the analytic
base on its own. That comparison is the number that matters:

  - learned beats baseline  -> the model captures something the hand-written
                               dynamics miss. Ship the weights.
  - learned does not        -> the data is the problem, not the model. Collect
                               better data rather than growing the network.

Validation is a BLOCK split: the last --val-frac of each file is held out.
Adjacent steps of one run are nearly identical, so a random split would put
near-copies of the validation samples into training and report a fit that
would not survive a new run.

What gets learned by default is limited by what the logs actually measure -
see "Which channels can be learned" in learned_dynamics.py.
"""
import argparse
import glob
import math
import os
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime

import numpy as np
import torch

import dataset
from learned_dynamics import (CHANNELS, DTYPE, DynamicsConfig, ResidualDynamics,
                              save, wrap_angle)

# Loss is computed on each channel divided by these, so a 1 cm position error
# and a ~6 deg heading error weigh about the same once those channels are on.
LOSS_SCALE = {"x": 0.01, "y": 0.01, "heading": 0.1, "speed": 0.01}


def expand(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    return [p for p in out if os.path.exists(p)]


def block_split(transitions, val_frac, min_per_file=10):
    """Last val_frac of each file -> validation; files too short -> train only."""
    by_file = defaultdict(list)
    for t in transitions:
        by_file[t["source"]].append(t)
    train, val = [], []
    for src, ts in by_file.items():
        n_val = int(len(ts) * val_frac) if len(ts) >= min_per_file else 0
        train += ts[:len(ts) - n_val]
        val += ts[len(ts) - n_val:]
    return train, val


def to_tensors(ts):
    return (torch.tensor(np.stack([t["state"] for t in ts]), dtype=DTYPE),
            torch.tensor(np.stack([t["action"] for t in ts]), dtype=DTYPE),
            torch.tensor(np.stack([t["next_state"] for t in ts]), dtype=DTYPE),
            torch.tensor([t["dt"] for t in ts], dtype=DTYPE))


def channel_errors(pred, target):
    err = pred - target
    err[..., 2] = wrap_angle(err[..., 2])
    return err


def loss_fn(model, batch, channels):
    s, a, s1, dt = batch
    err = channel_errors(model(s, a, dt), s1)
    idx = [CHANNELS.index(c) for c in channels]
    scale = torch.tensor([LOSS_SCALE[c] for c in channels], dtype=DTYPE)
    return ((err[..., idx] / scale) ** 2).sum(-1).mean()


def position_rmse(model, batch):
    s, a, s1, dt = batch
    with torch.no_grad():
        err = channel_errors(model(s, a, dt), s1)
    return float(torch.sqrt((err[..., 0] ** 2 + err[..., 1] ** 2).mean()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="CSV logs or globs")
    ap.add_argument("--out", default="weights/dynamics.pt")
    ap.add_argument("--surface", default=None,
                    help="override surface tag (dataset.py refuses to pool surfaces silently)")
    ap.add_argument("--assume-dt", type=float, default=None,
                    help="dt for legacy logs with no t_wall column - a guess the model inherits")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--learn-channels", nargs="+", default=["x", "y"], choices=CHANNELS,
                    help="ONLY add heading/speed once measured yaw/encoder speed are logged")
    ap.add_argument("--fit-params", nargs="*", default=["speed_scale"],
                    choices=["max_turn_rate", "max_accel", "max_decel", "speed_scale"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)

    paths = expand(args.logs)
    if not paths:
        raise SystemExit("No log files matched.")
    print(f"Reading {len(paths)} file(s)")
    transitions, stats = dataset.build(paths, args.surface, args.assume_dt)
    stats.pop("_medians", None)
    rejected = {k: v for k, v in stats.items() if v and not k.startswith("kept_")}
    if rejected:
        print("  rejected: " + ", ".join(f"{k}={v}" for k, v in sorted(rejected.items())))
    if not transitions:
        raise SystemExit("No usable transitions after filtering.")

    for label, key in (("surface", "surface"), ("comms path", "comms")):
        counter = Counter(t[key] for t in transitions)
        print(f"  {label}: {dict(counter)}")
        if len(counter) > 1:
            print(f"  WARNING: more than one {label} - pooled data mixes regimes "
                  f"(see dataset.py). 'unknown' means a log with no .meta.json.")

    # speed_scale is only identifiable from steps where the ball was asked to
    # move. Stationary data fits anything.
    moving = sum(1 for t in transitions if t["state"][3] > 1e-3)
    print(f"  {len(transitions)} transitions, {moving} with the ball commanded to move")
    if moving < 50:
        print("  WARNING: very few moving transitions - speed_scale and the "
              "position residual are barely constrained. Treat any fit as a "
              "pipeline check, not a model.")

    train, val = block_split(transitions, args.val_frac)
    print(f"  split: {len(train)} train / {len(val)} validation (block split per file)")
    if not val:
        print("  WARNING: no validation data (files too short) - metrics below are "
              "TRAINING error and will flatter the model.")
    tr = to_tensors(train)
    va = to_tensors(val) if val else tr

    config = DynamicsConfig(hidden=args.hidden,
                            learn_channels=tuple(args.learn_channels),
                            fit_params=tuple(args.fit_params))
    baseline = ResidualDynamics(config)
    model = ResidualDynamics(replace(config))

    base_rmse = position_rmse(baseline, va)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, best_epoch = math.inf, None, 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model, tr, args.learn_channels)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model, va, args.learn_channels))
        if v < best:
            best, best_epoch = v, epoch
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        if epoch % max(1, args.epochs // 6) == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:>5}  train {loss.item():9.4f}  val {v:9.4f}")

    model.load_state_dict(best_state)
    model.eval()
    learned_rmse = position_rmse(model, va)
    scale = float(model.physical("speed_scale").detach())

    print(f"\nBest validation loss at epoch {best_epoch}")
    print(f"One-step position RMSE on {'validation' if val else 'TRAINING'} data:")
    print(f"  analytic base (untrained)  {base_rmse * 100:7.2f} cm")
    print(f"  learned                    {learned_rmse * 100:7.2f} cm")
    print(f"  fitted speed_scale         {scale:7.3f}")

    if learned_rmse >= base_rmse:
        print("\n  The learned model does NOT beat the analytic base. The data is the "
              "problem, not the model - collect better data rather than growing "
              "the network. Weights not saved.")
        return 1

    gain = 100 * (1 - learned_rmse / base_rmse)
    print(f"  -> {gain:.0f}% lower error than the analytic base")

    model.config.meta = {
        "trained_on": [os.path.basename(p) for p in paths],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_train": len(train),
        "n_val": len(val),
        "n_moving": moving,
        "val_pos_rmse_m": learned_rmse,
        "baseline_pos_rmse_m": base_rmse,
        "speed_scale": scale,
        "validation_is_training_data": not val,
    }
    save(model, args.out)
    print(f"\nWrote {args.out}")
    print(f"Check it drops in:  python learned_dynamics.py --check --weights {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
