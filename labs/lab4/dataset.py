"""Lab 4 - turn logged runs into (state, action, next_state, dt) transitions.

Every filter here exists because of something found in this repo's own logs:

  - file-hash dedup: all 12 files in labs/lab2/logs/saved/ are byte-identical.
    One 83-row run copied 13 times, with filenames ("test-track", "track2")
    implying two surfaces that were never actually two surfaces. Counting
    those as 996 samples would have silently inflated the dataset 12x.
  - stale-reading drop: logs/fastcomms_lab1_equiv.csv is 87% consecutive
    duplicate positions - logged faster than the locator refreshes. Those
    rows are not transitions; fitting on them teaches the model that commands
    do nothing.
  - collision drop: contact is a different regime. The env clamps position
    and zeros speed on impact, so those rows describe the wall, not the ball.
  - dt sanity: without t_wall (added to Visualiser for exactly this) slip and
    latency are indistinguishable, and a fitted model needs dt.

Filters report what they rejected. A filter that silently drops most of the
data looks identical to a filter that drops none.

    python dataset.py logs/lab4_dyn_whiteboard_*.csv
    python dataset.py --surface whiteboard --out data/whiteboard.npz logs/*.csv
"""
import argparse
import csv
import glob
import hashlib
import json
import os
from collections import Counter

import numpy as np

# A transition whose dt is wildly off the run's median is a stall, a dropped
# BLE packet, or a logging hiccup - not a sample of the plant.
DT_TOLERANCE = 3.0

# Below this the "movement" is locator quantisation, not motion.
MIN_MOVEMENT_M = 1e-6


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta(csv_path):
    """Sidecar written by collect_data.py, if present."""
    meta_path = csv_path[:-4] + ".meta.json" if csv_path.endswith(".csv") else None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _f(row, key):
    v = row.get(key, "")
    if v is None or v == "":
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def _truthy(row, key):
    v = str(row.get(key, "")).strip().lower()
    return v in ("true", "1", "1.0", "yes")


def segment_of(step, segments):
    for s in segments:
        if s["start_step"] <= step < s["end_step"]:
            return s["name"], s.get("aborted", False)
    return "unsegmented", False


def transitions_from_file(path, stats, surface_override=None, assume_dt=None):
    """Extract transitions from one log. Returns a list of dicts."""
    meta = load_meta(path)
    segments = meta.get("segments", [])
    surface = surface_override or meta.get("surface") or "unknown"
    comms = meta.get("comms", "unknown")

    with open(path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        stats["file_too_short"] += 1
        return []

    has_t = "t_wall" in rows[0]
    if not has_t and assume_dt is None:
        stats["file_no_timestamp"] += 1
        return []

    out = []
    raw_dts = []

    for a, b in zip(rows, rows[1:]):
        x, y = _f(a, "odom_x"), _f(a, "odom_y")
        nx, ny = _f(b, "odom_x"), _f(b, "odom_y")
        heading, speed = _f(a, "heading"), _f(a, "speed")
        nheading, nspeed = _f(b, "heading"), _f(b, "speed")
        speed_cmd, heading_cmd = _f(a, "speed_cmd"), _f(a, "heading_cmd")

        if any(np.isnan(v) for v in
               (x, y, nx, ny, heading, speed, nheading, nspeed, speed_cmd, heading_cmd)):
            stats["nan_row"] += 1
            continue

        # Contact is a different regime - drop the step into the wall and the
        # step out of it, since the latter starts from a clamped state.
        if _truthy(a, "collision") or _truthy(b, "collision"):
            stats["collision"] += 1
            continue

        if has_t:
            dt = _f(b, "t_wall") - _f(a, "t_wall")
        else:
            dt = assume_dt
        if np.isnan(dt) or dt <= 0:
            stats["bad_dt"] += 1
            continue

        # Don't form a transition across a segment boundary: the commanded
        # action changes discontinuously there and the pair straddles two
        # different manoeuvres.
        step_a, step_b = _f(a, "step"), _f(b, "step")
        if not np.isnan(step_a) and segments:
            seg_a, aborted_a = segment_of(int(step_a), segments)
            seg_b, _ = segment_of(int(step_b), segments) if not np.isnan(step_b) else (seg_a, False)
            if seg_a != seg_b:
                stats["segment_boundary"] += 1
                continue
            if aborted_a:
                stats["aborted_segment"] += 1
                continue
        else:
            seg_a = "unsegmented"

        # Stale locator read = the whole pose repeated bit-for-bit, i.e. the
        # same packet twice. Testing position alone would also throw away the
        # ball being commanded and genuinely NOT moving - stall, deadband,
        # wedged against something - and those samples are informative: they
        # are what tells a model where the motor deadband is. That case is
        # kept and flagged instead.
        repeated_pose = (abs(nx - x) < MIN_MOVEMENT_M and
                         abs(ny - y) < MIN_MOVEMENT_M and
                         abs(nheading - heading) < MIN_MOVEMENT_M and
                         abs(nspeed - speed) < MIN_MOVEMENT_M)
        if repeated_pose:
            stats["stale_reading"] += 1
            continue

        stalled = (np.hypot(nx - x, ny - y) < MIN_MOVEMENT_M and speed_cmd > 1e-3)
        if stalled:
            stats["kept_stalled"] += 1

        raw_dts.append(dt)
        out.append({
            "state": np.array([x, y, heading, speed]),
            "action": np.array([speed_cmd, heading_cmd]),
            "next_state": np.array([nx, ny, nheading, nspeed]),
            "dt": dt,
            "surface": surface,
            "comms": comms,
            "segment": seg_a,
            "source": os.path.basename(path),
        })

    # dt outliers, judged against this file's own median (each comms path has
    # its own natural rate - a global threshold would reject a whole file).
    if out:
        med = float(np.median(raw_dts))
        kept = []
        for t in out:
            if med > 0 and not (med / DT_TOLERANCE <= t["dt"] <= med * DT_TOLERANCE):
                stats["dt_outlier"] += 1
            else:
                kept.append(t)
        out = kept
        stats["_medians"].append((os.path.basename(path), med, len(out)))

    return out


def build(paths, surface=None, assume_dt=None):
    stats = Counter()
    stats["_medians"] = []
    seen_hashes = {}
    transitions = []

    for path in paths:
        h = file_hash(path)
        if h in seen_hashes:
            stats["duplicate_file"] += 1
            print(f"  SKIP (byte-identical to {seen_hashes[h]}): {path}")
            continue
        seen_hashes[h] = os.path.basename(path)
        transitions += transitions_from_file(path, stats, surface, assume_dt)

    return transitions, stats


def coverage_report(transitions):
    """The actual measure of whether a collection run was any good.

    Sample count means little - 5000 samples of driving in a straight line
    identify one parameter. What matters is whether the (heading error, speed)
    plane is covered, because that is the space the model is asked to predict
    over.
    """
    if not transitions:
        print("\nNo transitions - nothing to report.")
        return

    herr = np.array([abs(wrap_angle(t["action"][1] - t["state"][2])) for t in transitions])
    spd = np.array([t["action"][0] for t in transitions])

    herr_bins = np.array([0, 10, 30, 60, 90, 180]) * np.pi / 180.0
    spd_max = max(float(spd.max()), 1e-6)
    spd_bins = np.linspace(0, spd_max, 5)

    counts, _, _ = np.histogram2d(herr, spd, bins=[herr_bins, spd_bins])

    print("\nExcitation coverage (rows = |heading error|, cols = speed_cmd):")
    header = "            " + "".join(
        f"{spd_bins[i]:.2f}-{spd_bins[i+1]:.2f} ".rjust(12) for i in range(len(spd_bins) - 1))
    print(header)
    labels = ["0-10deg", "10-30deg", "30-60deg", "60-90deg", "90-180deg"]
    empty = 0
    for i, lab in enumerate(labels):
        cells = ""
        for j in range(counts.shape[1]):
            n = int(counts[i, j])
            if n == 0:
                empty += 1
            cells += f"{n:>12d}"
        print(f"  {lab:<10}{cells}")

    total = counts.size
    print(f"\n  {empty}/{total} bins empty.")
    if empty > total // 2:
        print("  WARNING: most of the state space is unvisited. The model will be "
              "extrapolating wherever it matters. Collect more excitation "
              "(harder turns, more speed variety) rather than training longer.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="CSV log paths or globs")
    ap.add_argument("--surface", default=None,
                    help="Override the surface tag. Logs from different surfaces "
                         "must not be pooled without one.")
    ap.add_argument("--assume-dt", type=float, default=None,
                    help="Fallback dt for legacy logs with no t_wall column. A guess, "
                         "and the model inherits the error - re-collect if you can.")
    ap.add_argument("--out", default=None, help="Write an .npz of the arrays")
    args = ap.parse_args(argv)

    paths = []
    for p in args.logs:
        paths += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("No log files matched.")
        return 1

    print(f"Reading {len(paths)} file(s)")
    transitions, stats = build(paths, args.surface, args.assume_dt)

    medians = stats.pop("_medians", [])
    # kept_* counters are diagnostics, not rejections - listing them under
    # "Rejected" reads as data loss that never happened.
    kept_flags = {k: v for k, v in stats.items() if k.startswith("kept_") and v}
    rejected = {k: v for k, v in stats.items() if not k.startswith("kept_") and v}

    print("\nRejected:")
    if not rejected:
        print("  nothing")
    for k, v in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {v}")

    if kept_flags:
        print("\nKept but flagged:")
        for k, v in sorted(kept_flags.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<20} {v}")
        if "kept_stalled" in kept_flags:
            print("  (commanded but did not move - deadband/stall samples, "
                  "informative rather than broken)")

    if medians:
        print("\nPer-file median dt:")
        for name, med, n in medians:
            print(f"  {name:<45} {med:.3f}s  ({n} kept)")

    surfaces = Counter(t["surface"] for t in transitions)
    comms = Counter(t["comms"] for t in transitions)
    print(f"\nKept {len(transitions)} transitions")
    print(f"  surfaces: {dict(surfaces)}")
    print(f"  comms:    {dict(comms)}")

    if len(surfaces) > 1:
        print("\n  WARNING: more than one surface in this set. Traction changes "
              "acceleration, top speed and turn authority, so a pooled fit is "
              "wrong on both. Split by surface or pass --surface.")
    if len(comms) > 1:
        print("\n  WARNING: more than one comms path in this set - the step rates "
              "differ, so these are not comparable samples.")

    coverage_report(transitions)

    if args.out and transitions:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(
            args.out,
            state=np.stack([t["state"] for t in transitions]),
            action=np.stack([t["action"] for t in transitions]),
            next_state=np.stack([t["next_state"] for t in transitions]),
            dt=np.array([t["dt"] for t in transitions]),
            surface=np.array([t["surface"] for t in transitions]),
            segment=np.array([t["segment"] for t in transitions]),
        )
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
