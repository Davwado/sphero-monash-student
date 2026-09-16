"""Lab 3 controller — PD with turn-in-place and corner-commit phases.

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

SPEED_CAP = 0.15
GOAL_DIST_TOL = 0.015
BRAKE_GAIN = 3.0

MAX_ACCEL_STEP = 0.008
MAX_DECEL_STEP = 0.04

TURN_THRESHOLD = np.radians(35)
TURN_EXIT = np.radians(10)

# Inside this radius, stop recomputing the bearing and drive straight on the
# heading we already have. Two reasons:
#  - the loop accepts arrival at WAYPOINT_TOLERANCE (5cm) short of the plate
#    centre, so without this the ball starts turning for the next leg while
#    still 5cm inside the current plate, and cuts the corner. The ball has
#    real diameter, so a cut corner means clipping the wall.
#  - arctan2(dx, dy) gets very sensitive as dist shrinks: at 3cm out, a 1cm
#    wobble swings the commanded bearing 30+ degrees, which is what made the
#    ball hunt in place when the tolerance itself was tightened instead.
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
    """Return action = [speed_cmd, heading_cmd] for the current observation."""
    global prev_speed_cmd, turning, turn_target, committed_heading

    dx = env.goal_pos[0] - obs[0]
    dy = env.goal_pos[1] - obs[1]
    current_speed = obs[3]
    heading = obs[2]

    dist = np.hypot(dx, dy)
    brake_dist = BRAKE_GAIN * (current_speed ** 2) / (2 * MAX_DECEL * DT)

    if dist < GOAL_DIST_TOL + brake_dist:
        prev_speed_cmd = 0.0
        turning = False
        committed_heading = None
        return np.array([0.0, heading])

    # --- commit phase -----------------------------------------------------
    # Close to the waypoint: lock the heading and drive straight through
    # rather than re-aiming at a target that's nearly underneath us.
    if dist < COMMIT_RADIUS:
        if committed_heading is None:
            committed_heading = np.arctan2(dx, dy)

        speed_limit = min(SPEED_CAP, env.vel_limit)
        target_speed = np.clip(KP * dist - KD * current_speed,
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