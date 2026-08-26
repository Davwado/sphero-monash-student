import numpy as np

### Custom dynamics function for the Sphero robot - replace this with the one you developed in Lab 1
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
        dt = 2.95  # real per-step time, calibrated against logs/lab1_real_latest.csv

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
            action = [speed, heading_cmd]
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
