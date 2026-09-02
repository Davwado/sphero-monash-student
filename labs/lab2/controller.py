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

hold_heading = 0

MAX_DECEL = 0.01   # must match EKF.py
DT = 2.95    

def compute_action(env, obs, step):
    """Return action = [speed_cmd, heading_cmd] for the current observation."""
    # Default: random action for testing. Replace with your control law.
    # return _rng.uniform(low=-1.0, high=1.0, size=2)
    global hold_heading

    
    
    # --- P-controller-to-goal skeleton (uncomment and tune) ---
    dx = env.goal_pos[0] - obs[0]
    dy = env.goal_pos[1] - obs[1]
    current_speed = obs[3]

    dist = np.hypot(dx, dy)
    KP = 0.09
    KD = 0.40 # tune this
    brake_dist = 2*(current_speed ** 2) / (2 * MAX_DECEL * DT)
    if dist < 0.02 + brake_dist :
        # print("Goal reached!")
        return ["Stop","Stop"]
    else:
        heading_cmd = np.arctan2(dx, dy)   # 0 rad = +y convention -> atan2(dx, dy)
        hold_heading = heading_cmd #Update previous angle 
        speed_cmd = np.clip(KP * dist  - KD * current_speed, 0.00, env.vel_limit)  # Avoid zero speed (stuck) and clip to max speed
        return np.array([speed_cmd, heading_cmd])
