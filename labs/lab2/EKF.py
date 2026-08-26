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

    def jacobian(self, action):
        """
        Compute the Jacobian of the dynamics function with respect to the state.

        This is used in the EKF prediction step to linearize the dynamics around
        the current state estimate.
        """
        # add jacobian here. this will be a matrix of partial deivatives of state variables (x,y,heaeding,speed) wrt each other 
        x, y, heading, speed = self.state_est[0], self.state_est[1], self.state_est[2], self.state_est[3]
        # dtheta'/dtheta within limits inside trun_rate a = 0
        if (heading > -MAX_TURN_RATE) and (heading < MAX_TURN_RATE):
            a = 1
        else:
            a = 0

        #dv'/dv = b
        if (speed > -MAX_TURN_RATE) and (speed < MAX_TURN_RATE):
            b = 0
        else:
            b = 1

        heading_error = wrap_angle(action[1] - heading) 
        # c = dv'/dtheta
        if ((speed <= -MAX_TURN_RATE) and (speed >= MAX_TURN_RATE) and np.cos(heading_error) > 0):
            c = action[0]*np.sin(heading_error)
        else:
            0

        J = np.array([[1, 0, speed*np.cos(heading)*self.dt * a, np.sin(heading)], 
                             [0, 1, -speed*np.sin(heading)*self.dt * a, 0],
                             [0, 0, a, 0],
                             [0, 0, b, b]])

        return J  # Identity matrix as a placeholder
    
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
        self.P = self.jacobian(self, action) @ self.P @ self.jacobian(self, action).T + self.Q  # Replace this with a proper Jacobian calculation
        return self.state_est, self.P

    def update(self, measurement):
        """
        Correct the state estimate with a new measurement.
        """
        # Measurement model is identity: measurement directly observes the state
        H = np.eye(4)

        # 1. Innovation: difference between actual measurement and predicted state
        innovation = measurement - H @ self.state_est

        # 2. Innovation covariance
        S = H @ self.P @ H.T + self.R

        # 3. Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # 4. Correct the state estimate
        self.state_est = self.state_est + K @ innovation

        # 5. Correct the covariance
        self.P = (np.eye(4) - K @ H) @ self.P

        return self.state_est, self.P
