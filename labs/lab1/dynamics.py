import numpy as np

def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi  # Normalize to [-pi, pi)

# Rate limits (rad/s, m/s^2) finangled against logs/lab1_real_latest.csv.
MAX_TURN_RATE = 0.3    # how fast the ball can change heading
MAX_ACCEL = 0.003       # accelerating toward the target speed
MAX_DECEL = 0.01        # braking toward the target speed


def dynamics(state, action):
        """
        Unicycle model: heading and speed both move toward their commanded
        targets at a limited rate (this is what gives the ball momentum),
        and top speed drops off while turning sharply
        (speed_target = speed_cmd * cos(heading_error)).

        action: [speed_cmd, heading_cmd]
        state:  [x, y, heading, speed]
        """
        x, y, heading, speed = state
        speed_cmd, heading_cmd = action
        dt = 0.1  # real per-step time, calibrated against logs/lab1_real_latest.csv

        # Compute new heading and speed ratelimited towards the commanded targets
        heading_error = wrap_angle(heading_cmd - heading)
        heading_new = wrap_angle(heading + np.clip(heading_error, -MAX_TURN_RATE * dt, MAX_TURN_RATE * dt))

        # Compute new speed, rate limited towards the target speed
        speed_target = speed_cmd * max(0.0, float(np.cos(heading_error)))

        speed_error = speed_target - speed
        max_step = MAX_ACCEL * dt if speed_error > 0 else MAX_DECEL * dt
        # Calcs the new speeds
        speed_new = speed + np.clip(speed_error, -max_step, max_step)
        speed_new = float(np.clip(speed_new, 0.0, 1.0))

        # Uses the speed carried to this step, not speed_new, so position
        # lags speed by one step (step 1 stays at the reset position)
        x_new = x + speed * np.sin(heading_new) * dt
        y_new = y + speed * np.cos(heading_new) * dt

        # Return the new state as a numpy array of float32
        return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)