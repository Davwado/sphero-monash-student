import numpy as np

### Custom dynamics function for the Sphero robot - replace this with the one you developed in Lab 1
def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi  # Normalize to [-pi, pi)

# Rate limits (rad/s, m/s^2) finangled against logs/lab1_real_latest.csv.
MAX_TURN_RATE = 0.3  #  how fast the ball can change heading
MAX_ACCEL = 0.003    # accelerating toward the target speed
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

class EKF:
    def __init__(self, dt=2.95):
        # dt must match the timestep used inside dynamics(), otherwise the
        # state propagates on one timestep and the covariance on another.
        self.dt = dt
        self.state_est = np.zeros(4)  # [x, y, heading, speed]
        self.P = np.eye(4) * 0.1  # Initial covariance

        # Process noise: how much we distrust dynamics(). Slippery plastic
        # surface, so heading and speed are the least reliable predictions.
        self.Q = np.diag([0.002, 0.002, 0.04, 0.005])

        # Measurement noise: variance of each observed state.
        # x, y from obs_noise_std_pos=0.05 -> 0.05**2 = 0.0025
        self.R = np.diag([0.005, 0.005, 0.08, 0.02])

    def jacobian(self, action):
        """
        Compute the Jacobian of the dynamics function with respect to the state,
        linearized around the current state estimate and the given action.
        """
        x, y, heading, speed = self.state_est
        speed_cmd, heading_cmd = action
        dt = self.dt

        heading_error = wrap_angle(heading_cmd - heading)
        turn_clipped = np.clip(heading_error, -MAX_TURN_RATE * dt, MAX_TURN_RATE * dt)
        heading_new = wrap_angle(heading + turn_clipped)

        # a = d(heading_new)/d(heading). 1 when the turn-rate clip is inactive
        # (heading tracks the command), 0 when saturated.
        a = 0.0 if abs(heading_error) > MAX_TURN_RATE * dt else 1.0

        # b = d(speed_new)/d(speed). 1 when the accel/decel clip is inactive.
        cos_he = float(np.cos(heading_error))
        speed_target = speed_cmd * max(0.0, cos_he)
        speed_error = speed_target - speed
        max_step = MAX_ACCEL * dt if speed_error > 0 else MAX_DECEL * dt
        b = 0.0 if abs(speed_error) > max_step else 1.0

        # c = d(speed_new)/d(heading), via speed_target's dependence on
        # heading_error. Only applies when speed isn't saturated and the
        # cos term hasn't been clamped to zero.
        if b == 1.0 and cos_he > 0:
            c = speed_cmd * np.sin(heading_error)
        else:
            c = 0.0

        J = np.array([
            [1, 0,  speed * np.cos(heading_new) * dt * a,  np.sin(heading_new) * dt],
            [0, 1, -speed * np.sin(heading_new) * dt * a,  np.cos(heading_new) * dt],
            [0, 0,  a,                                     0],
            [0, 0,  c,                                     b],
        ])

        return J

    def predict(self, action):
        """
        Predict the next state and covariance given a control action.

        action = [speed_cmd, heading_cmd]
        Returns:
            self.state_est : predicted state [x, y, heading, speed]
            self.P         : predicted covariance, 4x4
        """
        # Jacobian must be evaluated at the PRIOR state, before it is overwritten
        J = self.jacobian(action)

        self.state_est = dynamics(self.state_est, action)
        self.P = J @ self.P @ J.T + self.Q

        return self.state_est, self.P

    def update(self, measurement):
        """
        Correct the state estimate with a new measurement.
        measurement: [x, y, heading, speed, ...] - extra entries are ignored.
        """
        # Measurement model is identity: we observe all four states directly
        H = np.eye(4)

        # 1. Innovation: difference between actual measurement and prediction
        innovation = np.asarray(measurement, dtype=float)[:4] - H @ self.state_est
        innovation[2] = wrap_angle(innovation[2])  # heading is circular

        # 2. Innovation covariance
        S = H @ self.P @ H.T + self.R

        # 3. Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # 4. Correct the state estimate
        self.state_est = self.state_est + K @ innovation
        self.state_est[2] = wrap_angle(self.state_est[2])

        # 5. Correct the covariance
        self.P = (np.eye(4) - K @ H) @ self.P

        return self.state_est, self.P