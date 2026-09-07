"""Archive the current lab2 log as the next numbered, tagged copy.

Usage (from repo root or labs/lab2):
    python labs/lab2/archive_log.py robot square_path
    python labs/lab2/archive_log.py sim baseline
"""
import argparse
import re
import shutil
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent / "logs"
SAVED_DIR = LOGS_DIR / "saved"
FILE_TYPES = ("sim", "robot")


def next_iteration(file_type: str) -> int:
    pattern = re.compile(rf"^lab2_{file_type}_(\d+)_.*\.csv$")
    existing = [
        int(m.group(1))
        for f in SAVED_DIR.glob(f"lab2_{file_type}_*.csv")
        if (m := pattern.match(f.name))
    ]
    return max(existing, default=0) + 1


def archive(file_type: str, tag: str) -> Path:
    src = LOGS_DIR / f"lab2_{file_type}.csv"
    if not src.exists():
        raise FileNotFoundError(f"Log not found: {src}")

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    n = next_iteration(file_type)
    safe_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", tag).strip("_") or "run"
    dest = SAVED_DIR / f"lab2_{file_type}_{n}_{safe_tag}.csv"
    shutil.copy2(src, dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_type", choices=FILE_TYPES, help="Which log to archive")
    parser.add_argument("tag", help="Short label describing this run (e.g. square_path)")
    args = parser.parse_args()

    dest = archive(args.file_type, args.tag)
    print(f"Copied lab2_{args.file_type}.csv -> {dest}")


if __name__ == "__main__":
    main()
