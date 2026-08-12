import numpy as np
import time

def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi  # Normalize to [-pi, pi)

# Target polling configuration: adjust to measured robot sampling rate (Hz)
# The environment calls dynamics() twice per env.step (true + odom) when a
# custom dynamics is provided, so we split the extra wait across both calls.
TARGET_POLL_HZ = 4.3
_DYNAMICS_CALLS_PER_STEP = 2

def make_dynamics(dt: float = 0.1, target_poll_hz: float = TARGET_POLL_HZ):
    """Return a dynamics(state, action) function using the provided timestep."""
    dt = float(dt)
    desired_dt = 1.0 / float(target_poll_hz)
    extra_dt = max(0.0, desired_dt - dt)
    per_call_sleep = extra_dt / float(max(1, _DYNAMICS_CALLS_PER_STEP))

    def dynamics(state, action):
        """Compute the next state given current state and action using the base dynamics without noise."""
        if per_call_sleep > 0:
            time.sleep(per_call_sleep)

        x, y, heading, speed = state
        speed_cmd, heading_cmd = action

        # Simple unicycle model dynamics
        max_turn_rate = 0.17
        heading_delta = np.clip(heading_cmd - heading, -max_turn_rate, max_turn_rate)
        heading_new = wrap_angle(heading + heading_delta)
        speed_new = np.clip(speed + speed_cmd * dt, 0, 1.0)

        x_new = x + speed_new * np.sin(heading_new) * dt
        y_new = y + speed_new * np.cos(heading_new) * dt

        return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)

    return dynamics
