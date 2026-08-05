"""Archive the current lab1_<type>.csv log as the next numbered iteration."""

import argparse
import re
import shutil
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
FILE_TYPES = ("sim", "real")


def next_iteration_path(file_type: str) -> Path:
    pattern = re.compile(rf"^lab1_{file_type}(\d+)\.csv$")
    existing = [
        int(m.group(1))
        for f in LOGS_DIR.glob(f"lab1_{file_type}*.csv")
        if (m := pattern.match(f.name))
    ]
    next_n = max(existing, default=0) + 1
    return LOGS_DIR / f"lab1_{file_type}{next_n}.csv"


def archive(file_type: str) -> Path:
    main_log = LOGS_DIR / f"lab1_{file_type}.csv"
    if not main_log.exists():
        raise FileNotFoundError(f"Main log not found: {main_log}")

    dest = next_iteration_path(file_type)
    shutil.copy2(main_log, dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_type", choices=FILE_TYPES, help="Which log to archive")
    args = parser.parse_args()

    dest = archive(args.file_type)
    print(f"Copied lab1_{args.file_type}.csv -> {dest}")


if __name__ == "__main__":
    main()
