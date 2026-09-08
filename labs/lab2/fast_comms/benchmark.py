"""Benchmark: fast_link's direct-write drive path vs stock spherov2.

Measures achieved commands/sec for both paths so you can see the actual
speedup on your hardware before deciding whether to wire this into lab2.

Does NOT touch sphero_env/Robot/lab1.py/lab2.py - entirely standalone.

Usage:
    python labs/lab2/fast_comms/benchmark.py
    python labs/lab2/fast_comms/benchmark.py --duration 5 --toy-name SB-1234
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_link  # noqa: E402


def benchmark_stock(toy, duration: float) -> float:
    """Drive commands via the stock spherov2 path (ack-wait + cmd_safe_interval)."""
    from spherov2.utils import ToyUtil

    count = 0
    heading = 0
    end = time.time() + duration
    while time.time() < end:
        ToyUtil.roll_start(toy, heading, 0)  # speed 0 - just measuring command rate, not moving
        heading = (heading + 1) % 360
        count += 1
    return count / duration


def benchmark_fast(link: "fast_link.FastSpheroLink", duration: float) -> float:
    """Drive commands via the direct-write fast path (no ack-wait, no throttle)."""
    count = 0
    heading = 0
    end = time.time() + duration
    while time.time() < end:
        link.fast_drive(heading, 0)  # speed 0 - just measuring command rate, not moving
        heading = (heading + 1) % 360
        count += 1
    return count / duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Seconds to benchmark each path for")
    parser.add_argument("--toy-name", type=str, default=None,
                        help="Specific toy name to connect to (skips scan-all)")
    parser.add_argument("--skip-stock", action="store_true",
                        help="Skip the slow stock-path benchmark (just test fast_link)")
    args = parser.parse_args()

    print("Scanning for BOLT...")
    toy = fast_link.connect(toy_name=args.toy_name)
    print(f"Found: {toy}")

    with toy:
        link = fast_link.FastSpheroLink(toy)

        if not args.skip_stock:
            print(f"\nBenchmarking STOCK path for {args.duration}s "
                  "(this will be slow - that's the point)...")
            stock_hz = benchmark_stock(toy, args.duration)
            print(f"Stock path:  {stock_hz:.2f} commands/sec")

        print(f"\nBenchmarking FAST path for {args.duration}s...")
        fast_hz = benchmark_fast(link, args.duration)
        print(f"Fast path:   {fast_hz:.2f} commands/sec")

        if not args.skip_stock:
            print(f"\nSpeedup: {fast_hz / stock_hz:.1f}x")

        # Leave the ball stopped.
        link.fast_stop()


if __name__ == "__main__":
    main()
