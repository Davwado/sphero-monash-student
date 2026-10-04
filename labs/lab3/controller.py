"""Lab 3 controller — PD with an explicit turn-in-place phase.

Speed note: SPEED_CAP was previously lowered to 0.06 to stop the ball
crashing, but that made things worse, not better. At 0.06 m/s with dt=0.1
the ball moves ~0.3cm per step while the position reading wobbles by ~5cm
(obs_noise_std_pos=0.05), so the signal was ~15x smaller than the noise and
the controller was effectively steering on noise alone - arctan2 of a
5cm wobble over a 0.3cm real displacement produces wild heading commands.
Driving at the full vel_limit gives ~1.5cm/step, a far better ratio against
the same noise.

Interfaces:
    obs    = [x (m), y (m), heading (rad), speed (m/s), collision_flag (0/1)]
    action = [speed_cmd (m/s), heading_cmd (rad)]

Heading convention: 0 rad points along +y, +pi/2 points along +x.
"""
import numpy as np

MAX_DECEL = 0.01
DT = 2.95

KP = 0.18
KD = 0.45

# Real motion must outrun the position noise or the controller steers on
# nothing but noise. See module docstring.
SPEED_CAP = 0.15

BRAKE_GAIN = 3.0

MAX_ACCEL_STEP = 0.008
MAX_DECEL_STEP = 0.04

# Enter a turn-in-place phase above this heading error (~35 deg), and stay
# in it until within TURN_EXIT (~10 deg). The gap is deliberate hysteresis:
# a single threshold would let the ball flip in and out of turning on noise.
TURN_THRESHOLD = np.radians(35)
TURN_EXIT = np.radians(10)

# Inside this radius, stop recomputing the bearing every step and lock the
# heading instead. arctan2(dx, dy) gets very sensitive as dist shrinks: a few
# cm of position noise near the target swings the commanded bearing by tens
# of degrees, which the turn-phase logic above then reacts to - the ball
# starts steering toward a wobbling setpoint right as it's trying to line up
# on a waypoint, cutting across the true corner instead of tracking into it
# straight. Locking the heading once committed removes that noise source;
# it does NOT change when the hard stop above fires, so the ball still comes
# to rest at each waypoint exactly as before.
COMMIT_RADIUS = 0.08

prev_speed_cmd = 0.0
turning = False
turn_target = 0.0
committed_heading = None


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def reset():
    """Call after a replan so slew/turn/commit state doesn't carry over."""
    global prev_speed_cmd, turning, turn_target, committed_heading
    prev_speed_cmd = 0.0
    turning = False
    turn_target = 0.0
    committed_heading = None


def compute_action(env, obs, step):
    """Return action = [speed_cmd, heading_cmd] for the current observation.

    env.goal_tolerance is the "reached" radius to brake within - lab3.py
    passes its WAYPOINT_TOLERANCE through here (see wp_target in
    control_loop()) rather than this module keeping its own separate copy.
    They used to be two independent constants that had to be kept in sync by
    hand; when they drifted apart (this one tighter than the outer loop's),
    the outer loop always advanced wp_index before this branch ever
    triggered, so the ball was still actively driving - never actually
    settling - at every intermediate waypoint.

    env.allow_stop (default True) gates the hard stop below. lab3.py sets it
    False for a waypoint on a straight run (see _is_in_line()) and steers
    those at the NEXT waypoint rather than the current one - so dist here
    stays large the whole time and this branch simply never fires, letting
    the ball cruise straight through instead of braking for a point it
    isn't actually going to turn at.
    """
    global prev_speed_cmd, turning, turn_target, committed_heading

    dx = env.goal_pos[0] - obs[0]
    dy = env.goal_pos[1] - obs[1]
    current_speed = obs[3]
    heading = obs[2]

    dist = np.hypot(dx, dy)
    brake_dist = BRAKE_GAIN * (current_speed ** 2) / (2 * MAX_DECEL * DT)

    allow_stop = getattr(env, "allow_stop", True)
    if allow_stop and dist < env.goal_tolerance + brake_dist:
        prev_speed_cmd = 0.0
        turning = False
        committed_heading = None
        return np.array([0.0, heading])

    # --- commit phase -------------------------------------------------------
    if dist < COMMIT_RADIUS:
        if committed_heading is None:
            committed_heading = np.arctan2(dx, dy)
        heading_error = wrap_angle(committed_heading - heading)
        align = max(0.0, np.cos(heading_error)) ** 2

        speed_limit = min(SPEED_CAP, env.vel_limit)
        target_speed = np.clip((KP * dist - KD * current_speed) * align,
                               0.00, speed_limit)
        speed_cmd = np.clip(
            target_speed,
            prev_speed_cmd - MAX_DECEL_STEP,
            prev_speed_cmd + MAX_ACCEL_STEP,
        )
        prev_speed_cmd = speed_cmd
        return np.array([speed_cmd, committed_heading])

    committed_heading = None
    desired_heading = np.arctan2(dx, dy)

    # --- turn phase -------------------------------------------------------
    if turning:
        # Hold the ORIGINAL target throughout the turn. Recomputing it each
        # step makes the ball chase a moving setpoint.
        if abs(wrap_angle(turn_target - heading)) < TURN_EXIT:
            turning = False
        else:
            prev_speed_cmd = 0.0
            return np.array([0.0, turn_target])

    if abs(wrap_angle(desired_heading - heading)) > TURN_THRESHOLD:
        turning = True
        turn_target = desired_heading
        prev_speed_cmd = 0.0
        return np.array([0.0, turn_target])

    # --- drive phase ------------------------------------------------------
    heading_error = wrap_angle(desired_heading - heading)
    align = max(0.0, np.cos(heading_error)) ** 2

    speed_limit = min(SPEED_CAP, env.vel_limit)
    target_speed = np.clip((KP * dist - KD * current_speed) * align,
                           0.00, speed_limit)

    speed_cmd = np.clip(
        target_speed,
        prev_speed_cmd - MAX_DECEL_STEP,
        prev_speed_cmd + MAX_ACCEL_STEP,
    )
    prev_speed_cmd = speed_cmd

    return np.array([speed_cmd, desired_heading])
