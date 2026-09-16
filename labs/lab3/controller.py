"""Lab 3 controller — brake-distance PD controller with slew-rate limiting.

Tuned for slow, methodical waypoint-to-waypoint motion: the ball should
approach each waypoint, settle, pivot, then move off gently. Top speed is
capped well below the env's vel_limit because momentum is what causes both
the overshoot and the wall contact - a slower ball has time to correct.

Interfaces:
    obs    = [x (m), y (m), heading (rad), speed (m/s), collision_flag (0/1)]
    action = [speed_cmd (m/s, env clips to +/-0.15), heading_cmd (rad)]

Heading convention: 0 rad points along +y ("up" in the window), +pi/2 points
along +x. heading_cmd is a desired absolute heading, not a turn rate.
"""
import numpy as np

hold_heading = 0

MAX_DECEL = 0.01
DT = 2.95

KP = 0.18
KD = 0.45

# Hard cap on commanded speed, below the env's vel_limit of 0.15. This is the
# main knob for "slower": KP only controls the command until it clips, so
# capping here bounds top speed on the long legs regardless of distance.
SPEED_CAP = 0.06

# Arrival radius. Kept below lab3.py's WAYPOINT_TOLERANCE (0.02) so the
# controller doesn't park just outside the acceptance radius and stall.
GOAL_DIST_TOL = 0.015

# Brake-distance multiplier. Higher = starts slowing earlier. Raised from 1.5
# because the ball was consistently overshooting - it couldn't shed momentum
# in the distance it was allowing itself.
BRAKE_GAIN = 3.0

# Slew-rate limits on the speed COMMAND, per step. Without these, arriving at
# a waypoint and switching to the next makes dist jump ~13x in one step with
# nothing damping it, producing a lurch at exactly the moment the ball is
# also being told to turn for the next leg. ACCEL lowered to 0.008 so the
# ramp takes ~8 steps rather than 3.
MAX_ACCEL_STEP = 0.008
MAX_DECEL_STEP = 0.04

prev_speed_cmd = 0.0


def reset():
    """Call after a replan so heading/slew history doesn't carry over."""
    global hold_heading, prev_speed_cmd
    hold_heading = 0
    prev_speed_cmd = 0.0


def compute_action(env, obs, step):
    """Return action = [speed_cmd, heading_cmd] for the current observation."""
    global hold_heading, prev_speed_cmd

    dx = env.goal_pos[0] - obs[0]
    dy = env.goal_pos[1] - obs[1]
    current_speed = obs[3]

    dist = np.hypot(dx, dy)
    brake_dist = BRAKE_GAIN * (current_speed ** 2) / (2 * MAX_DECEL * DT)

    if dist < GOAL_DIST_TOL + brake_dist:
        # Arrived (or close enough that we should be coasting in): stop,
        # hold current heading. Numeric (not ["Stop","Stop"]) because
        # control_env.step(action) unpacks action as floats.
        prev_speed_cmd = 0.0
        return np.array([0.0, obs[2]])

    heading_cmd = np.arctan2(dx, dy)   # 0 rad = +y convention
    hold_heading = heading_cmd

    # Turn before driving. Squared so misalignment bites harder: at 45 deg
    # off, speed drops to 50% rather than 71%, so the pivot finishes before
    # the ball builds momentum into a corridor wall.
    heading_error = (heading_cmd - obs[2] + np.pi) % (2 * np.pi) - np.pi
    align = max(0.0, np.cos(heading_error)) ** 2

    speed_limit = min(SPEED_CAP, env.vel_limit)
    target_speed = np.clip((KP * dist - KD * current_speed) * align,
                           0.00, speed_limit)

    # Ramp toward the target rather than jumping to it.
    speed_cmd = np.clip(
        target_speed,
        prev_speed_cmd - MAX_DECEL_STEP,
        prev_speed_cmd + MAX_ACCEL_STEP,
    )
    prev_speed_cmd = speed_cmd

    return np.array([speed_cmd, heading_cmd])