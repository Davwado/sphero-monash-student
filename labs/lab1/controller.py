"""Lab 1 controller — edit this file, save, then press R in the pygame window.

lab1.py hot-reloads this module before every run, so you can tune your
controller without restarting the program (and without reconnecting to the
Sphero over Bluetooth).

Interfaces:
    obs    = [x (m), y (m), heading (rad), speed (m/s), collision_flag (0/1)]
    action = [speed_cmd (m/s, env clips to +/-0.15), heading_cmd (rad)]

Heading convention: 0 rad points along +y ("up" in the window), +pi/2 points
along +x. heading_cmd is a desired absolute heading, not a turn rate.

Note: module-level state (like _rng below) is re-created on every reload, so
each run starts fresh and runs with the same controller code are repeatable.
"""
import numpy as np

GOAL = np.array([-1,0])
hold_heading = 0.0

# _rng = np.random.default_rng(0)


def compute_action(obs, step):
    """Return action = [speed_cmd, heading_cmd] for the current observation."""
    # Default: random action for testing. Replace with your control law.
    # return _rng.uniform(low=-1.0, high=1.0, size=2)

    # --- P-controller-to-goal skeleton (uncomment and tune) ---
    dx = GOAL[0] - obs[0]
    dy = GOAL[1] - obs[1]
    heading_cmd = np.arctan2(dx, dy)   # 0 rad = +y convention -> atan2(dx, dy)
    dist = np.hypot(dx, dy)
    KP = 0.05
    if dist < 0.05:
        print("Goal reached!")
        return np.array([0.0, hold_heading])  # Stop moving, keep heading
    else:
        hold_heading = heading_cmd #Update previous angle 
        speed_cmd = np.clip(KP * dist, 0.0, 0.1)
        return np.array([speed_cmd, heading_cmd])
