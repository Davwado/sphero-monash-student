import numpy as np

### Custom dynamics function for the Sphero robot - replace this with the one you developed in Lab 1
def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi  # Normalize to [-pi, pi)

def dynamics(state, action):
        """
        Compute the next state given current state and action using the base dynamics without noise.
        This is a bad model of the robot.

        The action (input) will be:
            action = [speed, turn_rate_cmd]
        The returned state (output) will be:
            state = [x_new, y_new, heading_new, speed_new]
        """
        x, y, heading, speed = state
        speed_cmd, turn_rate_cmd = action
        # Simple unicycle model dynamics
        heading_new = wrap_angle(heading + turn_rate_cmd * 0.1)
        speed_new = np.clip(speed_cmd, -2.0, 2.0)
        x_new = x + speed_new * np.sin(heading_new) * 0.1
        y_new = y + speed_new * np.cos(heading_new) * 0.1
        return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)

### EKF class to track odometry and perform state estimation for the Sphero robot
# Complete this class to implement the EKF algorithm for state estimation based on your dynamics and measurement models.

class EKF:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.state_est = np.zeros(4)  # [x, y, heading, speed]
        self.P = np.eye(4) * 0.1  # Initial covariance
        self.Q = np.diag([0.001, 0.001, 0.001, 0.001])  # Process noise covariance
        self.R = np.diag([0.005, 0.005])  # Measurement noise covariance

    def predict(self, action):
        """
        Predict the next state and covariance given a control action.

        The action (input) is the same format as in dynamics():
            action = [speed, turn_rate_cmd]
        Returns:
            self.state_est : the predicted state, same format as the state
                             returned by dynamics(): [x, y, heading, speed]
            self.P         : the predicted covariance, a 4x4 matrix whose rows
                             and columns correspond to [x, y, heading, speed]
        """
        # Predict the next state using the dynamics function
        self.state_est = dynamics(self.state_est, action)

        # Update the covariance matrix using a simple linear approximation of the dynamics
        self.P = self.P # Replace this with a proper Jacobian-based update
        return self.state_est, self.P

    def update(self, measurement):
        """
        Correct the state estimate with a new measurement.

        The measurement (input) is in the same state format:
            measurement = [x, y, heading, speed]
        Returns:
            self.state_est : the corrected state, same format as the state
                             returned by dynamics(): [x, y, heading, speed]
            self.P         : the corrected covariance, a 4x4 matrix whose rows
                             and columns correspond to [x, y, heading, speed]
        """
        # Update the state estimate with a new measurement

        self.state_est = self.state_est  # Replace this with a proper Kalman gain update
        self.P = self.P  # Replace this with a proper covariance update

        return self.state_est, self.P

