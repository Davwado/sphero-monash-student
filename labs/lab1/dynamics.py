import numpy as np

def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi  # Normalize to [-pi, pi)

def dynamics(state, action):
        """
        Compute the next state given current state and action using the base dynamics without noise.
        This can be rewritten to improve the model.

        The action (input) will be:
            action = [speed, turn_rate_cmd]
        The returned state (output) will be:
            state = [x_new, y_new, heading_new, speed_new]
        """
        x, y, heading, speed = state
        speed_cmd, turn_rate_cmd = action
        # Simple unicycle model dynamics
        heading_new = wrap_angle(heading + turn_rate_cmd * 0.1)
        speed_new = np.clip(speed + speed_cmd * 0.1, 0, 1.0)
        x_new = x + speed_new * np.sin(heading_new) * 0.1
        y_new = y + speed_new * np.cos(heading_new) * 0.1
        return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)

