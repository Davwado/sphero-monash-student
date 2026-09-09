from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

import argparse
import numpy as np

from types import SimpleNamespace

from Planner import *
from EKF import EKF
from sphero_env.envs.custom_maze_full import build_occupancy_grid

import controller

from contextlib import ExitStack, contextmanager

LAB1_SEED = 0
MAX_STEPS = 5000
map = build_occupancy_grid()

# Where the robot is physically placed at the start, in map coordinates.
START_XY = np.array([-0.5, -0.5])

# --- Simulator timing ------------------------------------------------------
# Note SpheroEnv ignores its own dt whenever a custom dynamics= is passed (see
# sphero_env.py step(): self.dt only feeds _base_dynamics), so dt is only a
# real knob because dynamics() below actually reads SIM_DT.
#
# Nothing paces the loop against wall-clock time - the visualiser just caps
# drawing at 60fps - so one step is one rendered frame and SIM_DT sets how
# much simulated time each frame covers. A full run is ~134s of simulated
# time, so at SIM_DT=0.1 it plays back in ~22 real seconds.
SIM_DT = 0.1

# Plant limits for the SIMULATED robot, in physical units (rad/s, m/s^2).
# Deliberately NOT the constants from EKF.py: those (MAX_TURN_RATE=0.3,
# MAX_ACCEL=0.003) were fitted to the real robot's ~2.95s bluetooth command
# cadence, and at any sane sim timestep they make the ball crawl - 0.003 m/s^2
# needs ~50s of simulated time just to reach the 0.15 m/s speed limit.
SIM_MAX_TURN_RATE = 3.0   # rad/s   -> 90 deg turn in ~0.5s
SIM_MAX_ACCEL = 0.3       # m/s^2   -> 0 to vel_limit in ~0.5s
SIM_MAX_DECEL = 0.5       # m/s^2


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def dynamics(state, action):
    """Rate-limited unicycle plant for the simulator.

    Same shape as EKF.dynamics - absolute heading command, top speed rolled
    off by cos(heading_error) - but driven by SIM_DT and the sim plant limits
    above, so simulated time advances uniformly no matter what SIM_DT is set to.

    action: [speed_cmd, heading_cmd]
    state:  [x, y, heading, speed]
    """
    x, y, heading, speed = state
    speed_cmd, heading_cmd = action

    heading_error = wrap_angle(heading_cmd - heading)
    max_turn = SIM_MAX_TURN_RATE * SIM_DT
    heading_new = wrap_angle(heading + np.clip(heading_error, -max_turn, max_turn))

    speed_target = speed_cmd * max(0.0, float(np.cos(heading_error)))
    speed_error = speed_target - speed
    max_step = (SIM_MAX_ACCEL if speed_error > 0 else SIM_MAX_DECEL) * SIM_DT
    speed_new = float(np.clip(speed + np.clip(speed_error, -max_step, max_step), 0.0, 1.0))

    x_new = x + speed_new * np.sin(heading_new) * SIM_DT
    y_new = y + speed_new * np.cos(heading_new) * SIM_DT
    return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)


def make_sim_env():
    return SpheroEnv(
        dt=SIM_DT,
        max_steps=MAX_STEPS,
        vel_limit=0.15,
        world_width=1.25,
        world_height=1.25,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        occupancy_grid=map,
        grid_resolution=0.125,
        dynamics=dynamics,
        obs_noise_std_pos=0.05,
        process_noise_std_speed=0.005,
        process_noise_std_heading=0.01,
        obs_noise_std_vel=0.025,
        render_mode="human",
        window_size=(800, 800),
    )


def make_real_env(api):
    return Robot(
        api=api,
        dt=0.1,
        max_steps=5000,
        vel_limit=0.15,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        render_mode="human",
        window_size=(800, 800),
    )


@contextmanager
def managed_env(sim: bool):
    if sim:
        sim_env = make_sim_env()
        sim_env.set_log_path("logs/lab3_sim.csv")
        sim_env.start_logging()
        try:
            yield sim_env
        finally:
            sim_env.stop_logging()
            sim_env.close()
    else:
        with ExitStack() as stack:
            selected_toy, _ = scan_and_connect()
            print(f"Selected: {selected_toy.name}")

            api = stack.enter_context(SpheroEduAPI(selected_toy))
            api.reset_aim()
            real_env = make_real_env(api)
            real_env.set_log_path("logs/lab3_real.csv")

            real_env.start_logging()
            try:
                yield real_env
            finally:
                real_env.close()
                real_env.stop_logging()


def control_loop(control_env):

    obs, _ = control_env.reset(seed=LAB1_SEED)

    is_sim = isinstance(control_env, SpheroEnv)

    # The plan lives in MAP coordinates (start plate at START_XY). The real
    # Sphero's get_location() is relative to wherever it was switched on and
    # aimed, i.e. it reads (0,0) at the start plate - a different frame.
    #
    # In sim the two are reconciled by force-writing the start pose below. On
    # hardware that write is silently discarded, because Robot.step() rebuilds
    # state_odom from the API on every single step. The result last run: the
    # robot read itself at (0,0), was told to drive to wp0 = (-0.5,-0.5), and
    # set off at -138 deg - straight out of the start plate to the southwest.
    # So record the offset between the two frames and shift every reading.
    if is_sim:
        control_env.state_true[0:3] = np.array([-0.5, -0.5, 0.0])
        control_env.state_odom[0:3] = np.array([-0.5, -0.5, 0.0])
        frame_offset = np.zeros(2)
    else:
        frame_offset = START_XY - np.asarray(control_env.state_odom[:2], dtype=float)
        print(f"Robot odom origin is map {tuple(np.round(-frame_offset, 3))}; "
              f"shifting readings by {tuple(np.round(frame_offset, 3))}")

    def to_map(o):
        """Reading -> map frame. Heading is assumed already aligned (0 rad =
        +y): place the ball on the start plate facing +y before reset_aim()."""
        m = np.asarray(o, dtype=float)[:4].copy()
        m[0:2] += frame_offset
        return m

    rng = np.random.default_rng(LAB1_SEED)

    # Plan against the module-level maze, not control_env.occupancy_grid: only
    # SpheroEnv carries that attribute, so reading it off the env crashes on
    # hardware. It is the same grid the sim is constructed with either way.
    planner = Planner(map=map, dt=control_env.dt)

    # Plan from the pose we just set above, not the stale obs returned by
    # reset() before the overwrite.
    start_state = np.array([-0.5, -0.5, 0.0, 0.0])

    # obs carries 0.05m of position noise, comparable to the gaps being driven
    # through, so navigating straight off it makes the waypoint test trip on
    # noise spikes and hands the controller a jittering bearing. Filter it.
    # Sim only: on hardware this would put an unvalidated filter in the loop
    # you already tuned against raw readings, so leave that path as tested.
    if is_sim:
        ekf = EKF(dt=control_env.dt, dynamics_fn=dynamics)
        # The sim's plant model is known exactly, so trust it and let the
        # filter do real smoothing. EKF.py's defaults assume the much less
        # certain real-robot model and barely filter at all here.
        ekf.Q = np.diag([1e-5, 1e-5, 1e-5, 1e-5])
        ekf.R = np.diag([0.05**2, 0.05**2, 0.025**2, 0.025**2])
        ekf.state_est = start_state.astype(float).copy()
        ekf.P = np.eye(4) * 1e-4
        est = ekf.state_est.copy()
    else:
        ekf = None
        est = to_map(obs)

    # margin_cells=0: this maze's corridors are only as wide as a single
    # connector cell, so any wall inflation blocks the only free passage.
    # Wall-clipping is instead handled at runtime via collision + replan.
    waypoints = planner.plan(start_state, control_env.goal_pos, margin_cells=0)

    print(f"Planned {len(waypoints)} waypoints")
    for i, wp in enumerate(waypoints):
        print(f"  wp{i}: {wp}")

    steps = 0

    # A waypoint must be hit far more tightly than the goal. Corridors here are
    # one occupancy cell wide, so the robot centre only has grid_resolution/2 =
    # 0.0625m of lateral room. Accepting a waypoint at goal_tolerance (0.1m)
    # let it turn for the next one while still 0.1m off-centre, which aims it
    # diagonally into the corridor wall - that's the "drove into a wall".
    # Must also stay above controller.GOAL_DIST_TOL, or the controller parks
    # just outside the acceptance radius and the loop stalls.
    WAYPOINT_TOLERANCE = 0.03

    # Budget in simulated time, not raw steps, so it doesn't silently become a
    # different limit whenever SIM_DT changes.
    MAX_STEPS_PER_WAYPOINT = int(60.0 / control_env.dt)
    MAX_REPLANS = 10
    replans = 0
    reached_goal = False

    wp_index = 0
    while wp_index < len(waypoints) and steps < MAX_STEPS:
        waypoint = waypoints[wp_index]
        wp_steps = 0
        collided = False

        # controller.compute_action() steers toward env.goal_pos, so hand it
        # a lightweight stand-in whose goal_pos is the current waypoint.
        wp_target = SimpleNamespace(goal_pos=waypoint, vel_limit=control_env.vel_limit)

        while wp_steps < MAX_STEPS_PER_WAYPOINT and steps < MAX_STEPS:
            action = controller.compute_action(wp_target, est, steps)
            if ekf is not None:
                ekf.predict(action)
            obs, _, terminated, truncated, info = control_env.step(action)
            est = ekf.update(to_map(obs))[0] if ekf is not None else to_map(obs)
            control_env.render()

            wp_steps += 1
            steps += 1

            collided = info.get("collision", False) if isinstance(info, dict) else False
            if not collided and len(obs) > 4:
                collided = bool(obs[4])
            if collided:
                break

            dist_to_goal_sq = (est[0]-control_env.goal_pos[0])**2 + (est[1]-control_env.goal_pos[1])**2
            if dist_to_goal_sq < control_env.goal_tolerance**2:
                reached_goal = True
                break

            dist_to_wp_sq = (est[0]-waypoint[0])**2 + (est[1]-waypoint[1])**2
            if dist_to_wp_sq < WAYPOINT_TOLERANCE**2:
                print(f"Reached waypoint {wp_index}: {waypoint}")
                break

        if reached_goal:
            print("Goal reached.")
            break

        if collided and replans < MAX_REPLANS:
            print(f"Collision near wp{wp_index}, step {wp_steps} - backing off and replanning")

            # Back off for several steps, not just one, so the ball
            # actually clears the wall before replanning
            back_off = np.array([-0.1, 0.0], dtype=np.float32)
            for _ in range(5):
                if ekf is not None:
                    ekf.predict(back_off)
                obs, _, terminated, truncated, info = control_env.step(back_off)
                est = ekf.update(to_map(obs))[0] if ekf is not None else to_map(obs)
                control_env.render()
                steps += 1

            try:
                waypoints = planner.plan(est, control_env.goal_pos, margin_cells=0)
                wp_index = 0
                replans += 1
                print(f"Replanned {len(waypoints)} waypoints "
                      f"(replan #{replans}) from ({est[0]:.3f}, {est[1]:.3f})")
                continue
            except (ValueError, RuntimeError) as e:
                print(f"Replan failed: {e} - continuing with old plan")

        if wp_steps >= MAX_STEPS_PER_WAYPOINT:
            print(f"Timed out on waypoint {wp_index}: {waypoint} "
                  f"(stuck at {est[0]:.3f}, {est[1]:.3f})")

        wp_index += 1

    if not reached_goal:
        final_dist = np.hypot(est[0]-control_env.goal_pos[0], est[1]-control_env.goal_pos[1])
        print(f"Path complete but goal not reached. Final distance: {final_dist:.3f} m")

    control_env.emergency_stop()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run simulation")
    args = parser.parse_args(argv)

    with managed_env(args.sim) as control_env:
        control_loop(control_env)


if __name__ == "__main__":
    main()