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

# --- TUNING PARAMETERS FOR LEGO SURFACE ---
# The plant tracks speed_cmd as a speed TARGET, so KD divides the gain rather
# than damping it: the loop settles at v = KP*dist/(1+KD). At the old KP=0.09
# that capped out at 0.062*dist ~ 0.05 m/s, a third of vel_limit, which is why
# a full run took ~1300 steps. KP=0.87 puts v at vel_limit one plate (0.25m)
# out. Tuned in sim - re-check on the real robot before trusting it there.
KP = 0.2
KD = 0.45  # tune this
# Arrival radius. Must stay well inside the corridor half-width
# (grid_resolution/2 = 0.0625m), or the ball turns for the next waypoint while
# still far enough off-centre to clip the corner. Also has to sit below
# lab3.py's WAYPOINT_TOLERANCE, or the loop stalls just outside the radius.
GOAL_DIST_TOL = 0.02
MAX_ACCEL_STEP = 0.02  # Max change in speed per step (TUNE THIS: lower = less slip)
MAX_DECEL_STEP = 0.04  # Sphero can usually brake slightly harder than it accelerates

# Persisted across steps for slew-rate limiting; re-created on every hot reload.
prev_speed_cmd = 0.0


def compute_action(env, obs, step):
    """Return action = [speed_cmd, heading_cmd] for the current observation."""
    global prev_speed_cmd

    dx = env.goal_pos[0] - obs[0]
    dy = env.goal_pos[1] - obs[1]
    current_speed = obs[3]
    dist = np.hypot(dx, dy)

    if dist < GOAL_DIST_TOL:
        prev_speed_cmd = 0.0
        return np.array([0.0, obs[2]])  # Goal reached: stop, hold current heading

    heading_cmd = np.arctan2(dx, dy)  # 0 rad = +y convention -> atan2(dx, dy)

    # Shortest angular error between current and desired heading, in [-pi, pi)
    heading_error = (heading_cmd - obs[2] + np.pi) % (2 * np.pi) - np.pi

    # Raw speed command (PD control on distance/speed)
    raw_speed_cmd = KP * dist - KD * current_speed

    # Heading-coupled speed limiting: slow down the more we need to turn.
    # np.pi/4 (45 degrees) is used as a scaling factor here.
    turn_penalty = max(0.0, 1.0 - (abs(heading_error) / (np.pi / 4)))
    raw_speed_cmd *= turn_penalty
    target_speed = np.clip(raw_speed_cmd, 0.0, env.vel_limit)

    # Slew rate limiting (anti-slip): don't let the requested speed jump too
    # fast from the previous one
    speed_cmd = np.clip(
        target_speed,
        prev_speed_cmd - MAX_DECEL_STEP,
        prev_speed_cmd + MAX_ACCEL_STEP,
    )

    prev_speed_cmd = speed_cmd

    return np.array([speed_cmd, heading_cmd])
