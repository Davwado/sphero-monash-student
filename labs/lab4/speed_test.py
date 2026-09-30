"""Measure how far the real ball rolls per step at low speed commands, to find the
slowest DRIVE_SPEED that still moves it. Needs ~1 m of clear floor in front of and
behind the ball (it drives back and forth). Run from labs/lab4:
    python speed_test.py
The log also works as training data for train_dynamics.py.
"""
import os
import time
from contextlib import ExitStack

import numpy as np

from lab4 import LOG_DIR, STEP_PERIOD, make_real_env
from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI

SPEEDS = [0.004, 0.006, 0.008, 0.010, 0.012, 0.015]
DRIVE_STEPS = 8
STOP_STEPS = 6


def main():
    with ExitStack() as stack:
        toy, _ = scan_and_connect()
        print(f"Selected: {toy.name}")
        api = stack.enter_context(SpheroEduAPI(toy))
        api.reset_aim()
        env = make_real_env(api)
        env.set_log_path(os.path.join(LOG_DIR, f"speed_test_{time.strftime('%Y%m%d-%H%M%S')}.csv"))
        env.start_logging()
        env.reset()
        last = 0.0

        def step(speed, heading):
            nonlocal last
            time.sleep(max(0.0, STEP_PERIOD - (time.time() - last)))
            last = time.time()
            obs = env.step(np.array([speed, heading], dtype=np.float32))[0]
            return np.asarray(obs[:2], dtype=float)

        try:
            pos = step(0.0, 0.0)
            for k, speed in enumerate(SPEEDS):
                heading = 0.0 if k % 2 == 0 else np.pi
                moves = []
                for _ in range(DRIVE_STEPS):
                    new = step(speed, heading)
                    moves.append(np.hypot(*(new - pos)))
                    pos = new
                coast = 0.0
                for _ in range(STOP_STEPS):
                    new = step(0.0, heading)
                    coast += np.hypot(*(new - pos))
                    pos = new
                print(f"speed {speed:.3f}: {np.median(moves[-4:]) * 100:4.1f} cm/step once rolling, "
                      f"coasted {coast * 100:4.1f} cm after stopping")
        finally:
            env.emergency_stop()
            env.close()
            env.stop_logging()


if __name__ == "__main__":
    main()
