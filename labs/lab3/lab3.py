from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

import argparse
import csv
import time
import numpy as np
import pygame

from types import SimpleNamespace

from Planner import *
from EKF import EKF
from sphero_env.envs.custom_maze_full import build_occupancy_grid

import controller

from contextlib import ExitStack, contextmanager

LAB1_SEED = 0
MAX_STEPS = 500000
map = build_occupancy_grid()

# Replace with your actual student ID before submitting.
STUDENT_ID = "your_id_here"

START_XY = np.array([-0.5, -0.5])

SIM_DT = 0.1

SIM_MAX_TURN_RATE = 3.0
SIM_MAX_ACCEL = 0.3
SIM_MAX_DECEL = 0.5

VERBOSE = True
DIVERGENCE_WARN = 0.25

# Single source of truth for both tolerances, used across this file, the
# visualiser and controller.py (which reads WAYPOINT_TOLERANCE off the
# wp_target namespace each call - see waypoint_reached()/control_loop()
# below - instead of keeping its own separate copy of the value).
WAYPOINT_TOLERANCE = 0.05   # per-waypoint "reached" radius
GOAL_TOLERANCE = 0.05        # final-goal "reached" radius
WAYPOINT_CONFIRM_STEPS = 3


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def waypoint_reached(pos, wp, prev_wp, tolerance=WAYPOINT_TOLERANCE, require_radius=True):
    """True once `pos` is within `tolerance` of `wp` AND on the far side of
    it relative to the direction of travel (prev_wp -> wp) - a semicircle
    on the departure side, not a full circle. Falls back to a plain circle
    when prev_wp is None (first waypoint of a (re)plan - no direction yet)
    or prev_wp/wp coincide (degenerate direction).

    require_radius=False drops the distance check and keeps only the
    far-side-of-the-plane test. Used for in-line waypoints (see
    _is_in_line()): those get steered at the NEXT waypoint, not this one, so
    the ball can cross the small tolerance circle in 1-2 steps at cruising
    speed - too fast for WAYPOINT_CONFIRM_STEPS to accumulate before it's
    already past the circle, which left the run stalled on a "reached"
    check that could never fire until the per-waypoint timeout forced it.
    The plane crossing alone is reliable here because motion past an
    in-line waypoint is monotonic (still heading the same direction, one
    corridor width wide) - there's no risk of a stray crossing far off to
    the side the way a full circle-drop would risk near a real turn.
    """
    pos = np.asarray(pos, dtype=float)[:2]
    wp = np.asarray(wp, dtype=float)[:2]
    to_pos = pos - wp
    if require_radius and np.dot(to_pos, to_pos) >= tolerance ** 2:
        return False
    if prev_wp is None:
        return True
    direction = wp - np.asarray(prev_wp, dtype=float)[:2]
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-9:
        return True
    return float(np.dot(to_pos, direction)) >= 0.0


def _aim_point(waypoint, prev_waypoint, tolerance=WAYPOINT_TOLERANCE, overshoot_frac=0.5):
    """Point to steer the controller toward for an INTERMEDIATE waypoint -
    pushed past the true waypoint, along the same approach direction
    waypoint_reached()'s semicircle uses, by overshoot_frac * tolerance
    (0.5 = the middle of the zone). controller.py brakes based on distance
    to whatever target it's given, with no notion of "this one's not the
    real stop" - aiming at the true waypoint makes it decelerate as if
    arriving for good at every intermediate stop. Aiming past it (but still
    inside the reached-zone) means it's still driving, not braking, right
    up to the point waypoint_reached() (which always checks the TRUE
    waypoint + tolerance, unaffected by this) declares arrival and cuts
    over to the next leg.

    Falls back to the true waypoint when there's no direction to push along
    (first waypoint of a plan) - use waypoint itself for the FINAL waypoint
    (the real goal), where actually stopping is correct.
    """
    wp = np.asarray(waypoint, dtype=float)[:2]
    if prev_waypoint is None:
        return wp
    direction = wp - np.asarray(prev_waypoint, dtype=float)[:2]
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-9:
        return wp
    return wp + (direction / dir_norm) * (tolerance * overshoot_frac)


def _is_in_line(waypoints, i, cos_threshold=0.999):
    """True if waypoint i sits on a straight run - the direction arriving at
    it (waypoints[i-1] -> waypoints[i]) matches the direction leaving it
    (waypoints[i] -> waypoints[i+1]). False for the first/last waypoint of a
    plan (no direction on one side) or a genuine turn/dead-end reversal.

    The planner deliberately keeps every plate centre rather than thinning
    to turn-points (see Planner.plan()'s docstring), so most intermediate
    waypoints along a corridor are exactly this case - there's no reason to
    brake for one, only for an actual turn or the final goal.
    """
    if i <= 0 or i >= len(waypoints) - 1:
        return False
    incoming = np.asarray(waypoints[i], dtype=float)[:2] - np.asarray(waypoints[i - 1], dtype=float)[:2]
    outgoing = np.asarray(waypoints[i + 1], dtype=float)[:2] - np.asarray(waypoints[i], dtype=float)[:2]
    in_norm = np.linalg.norm(incoming)
    out_norm = np.linalg.norm(outgoing)
    if in_norm < 1e-9 or out_norm < 1e-9:
        return False
    return float(np.dot(incoming / in_norm, outgoing / out_norm)) > cos_threshold


def _waypoint_zones(waypoints):
    """Build the (wp_xy, dir_xy_or_None) list set_waypoint_zones() expects,
    matching waypoint_reached()'s notion of direction (previous waypoint ->
    this one; None for the first waypoint in the list)."""
    zones = []
    for i, wp in enumerate(waypoints):
        d = None if i == 0 else np.asarray(wp, dtype=float)[:2] - np.asarray(waypoints[i - 1], dtype=float)[:2]
        zones.append((wp, d))
    return zones


def dynamics(state, action):
    """Rate-limited unicycle plant for the simulator.

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
        goal_tolerance=GOAL_TOLERANCE,
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
    env = Robot(
        api=api,
        dt=0.1,
        max_steps=500000,
        vel_limit=0.15,
        # Visualiser scales its drawing to fit world_width x world_height
        # into the window (see visualiser.py: scale = min(w/world_width,
        # h/world_height)). The maze only spans ~1.1m, so the old 5.0x5.0
        # here (left over from lab1/lab2's much bigger open-world layout)
        # drew it tiny and centred in the window - looked like the ball
        # started "in the middle" instead of at a corner of the maze.
        # Matches make_sim_env()'s world size so both windows scale the same.
        world_width=1.25,
        world_height=1.25,
        goal_pos=(0.5, 0.5),
        goal_tolerance=GOAL_TOLERANCE,
        render_mode="human",
        window_size=(800, 800),
    )
    env.vis.set_occupancy(map, 0.125)
    return env


def _fast_managed_api():
    """Lazy import so the fast_comms path is only pulled in when --fast-comms
    is actually passed - it lives alongside lab2, not lab3.

    Unlike lab1/lab2, control_loop() below actually reads info["collision"]
    (and obs[4]) to trigger the escape/replan logic when the ball hits a
    maze wall - so unlike fast_comms' locator-only default, this needs
    accelerometer/velocity/gyroscope streamed too, or Robot._sense_collision()
    silently always reports no collision and the escape logic never fires.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab2", "fast_comms"))
    from fast_link import fast_managed_api
    return fast_managed_api(sensors=("locator", "accelerometer", "velocity", "gyroscope"))


@contextmanager
def managed_env(sim: bool, fast_comms: bool = False):
    if sim:
        sim_env = make_sim_env()
        sim_env.set_log_path("logs/lab3_sim.csv")
        sim_env.start_logging()
        try:
            yield sim_env
        finally:
            sim_env.stop_logging()
            sim_env.close()
    elif fast_comms:
        # See labs/lab2/fast_comms/fast_link.py - drives through a low-latency
        # BLE path instead of SpheroEduAPI, but presents the same interface
        # Robot expects from `api`, so make_real_env() below is unchanged.
        with _fast_managed_api() as api:
            real_env = make_real_env(api)
            real_env.set_log_path("logs/lab3_real.csv")
            real_env.start_logging()
            try:
                yield real_env
            finally:
                real_env.stop_logging()
                real_env.close()
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

    def render_shifted():
        """Plain control_env.render() draws Robot's raw state_true/state_odom,
        which are in the ROBOT's own odometry frame - not shifted into map
        frame like everything else here (est, to_map(obs)). Without this,
        the ball's on-screen position is offset from the maze overlay by
        -frame_offset even though the EKF/planner are using the correct
        (shifted) position internally. is_sim's frame_offset is zeros, so
        this is a no-op there - safe to call unconditionally."""
        control_env.vis.render(to_map(control_env.state_true), to_map(control_env.state_odom))

    def fixup_traj_point():
        """Robot.step()/SpheroEnv.step() call vis.record(gt_state=self.state_true,
        odom_state=self.state_odom, ...) internally at the end of step() -
        appending the RAW (unshifted) position to the trajectory line. That's
        a separate code path from render_shifted() above (which only fixes
        what's drawn for the CURRENT position), so without this the green/
        blue trajectory LINE still traces through the wrong frame even
        though the ball dot is drawn correctly - looks like the line starts
        in the middle of the maze instead of at the start corner.

        No public API to give record() a different point, so this corrects
        the just-appended entry directly. Call right after every
        control_env.step(...). No-op for sim (frame_offset is zeros)."""
        vis = control_env.vis
        if vis._gt_traj:
            vis._gt_traj[-1] = tuple(to_map(control_env.state_true)[:2])
        if vis._odom_traj:
            vis._odom_traj[-1] = tuple(to_map(control_env.state_odom)[:2])

    rng = np.random.default_rng(LAB1_SEED)

    planner = Planner(map=map, dt=control_env.dt)

    start_state = np.array([-0.5, -0.5, 0.0, 0.0])

    if is_sim:
        ekf = EKF(dt=control_env.dt, dynamics_fn=dynamics)
        ekf.Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
        ekf.R = np.diag([0.05**2, 0.05**2, 0.025**2, 0.025**2])
    else:
        # Slippery surface. Slip breaks the MODEL, not the sensor: when the
        # ball slides, dynamics() predicts a displacement that didn't happen,
        # so the prediction is the untrustworthy part. Q therefore goes up
        # (5x on position, 4x on speed vs the previous tuning) while R stays
        # put, shifting the filter from ~10:1 toward ~2:1 - it now leans
        # noticeably harder on the measurements.
        #
        # Speed gets the largest bump because slip shows up there first: the
        # wheels turn at the commanded rate while the ball doesn't actually
        # accelerate, so predicted speed runs ahead of real speed.
        ekf = EKF(dt=2.95)
        ekf.Q = np.diag([0.01, 0.01, 0.015, 0.02])
        ekf.R = np.diag([0.02, 0.02, 0.05, 0.05])

    ekf.state_est = start_state.astype(float).copy()
    ekf.P = np.eye(4) * 1e-3
    est = ekf.state_est.copy()

    _last_predict_t = [time.time()]

    def timed_predict(action):
        """ekf.predict(), but on real hardware ekf.dt is set to the ACTUALLY
        measured wall-clock time since the previous call first.

        EKF.py's dynamics() used to hardcode dt=2.95s (calibrated against
        the old, much slower blocking SpheroEduAPI loop) regardless of
        self.dt - so the filter always predicted as if 2.95s had elapsed
        every step, no matter how fast the real loop actually ran. After
        this session's comms speedups that's wildly wrong (predicting
        15-20x more travel than actually happened), which drove `est` far
        ahead of the real position almost immediately, then got the
        estimate permanently stuck once the gap exceeded EKF.update()'s
        outlier-rejection gate (a real run's log showed exactly this:
        est frozen ~44cm from odometry, P growing unbounded). Measuring the
        real per-step time removes the need to guess a fixed constant here
        that would just go stale again the next time the loop gets faster.
        No-op for sim - ekf.dt stays SIM_DT, matching the fixed-step sim
        clock rather than wall-clock time.
        """
        if not is_sim:
            now = time.time()
            ekf.dt = max(now - _last_predict_t[0], 1e-3)
            _last_predict_t[0] = now
        ekf.predict(action)

    waypoints = planner.plan(start_state, control_env.goal_pos, margin_cells=0)
    control_env.vis.set_waypoints(waypoints)
    control_env.vis.set_waypoint_zones(_waypoint_zones(waypoints), WAYPOINT_TOLERANCE)

    print(f"Planned {len(waypoints)} waypoints")
    for i, wp in enumerate(waypoints):
        print(f"  wp{i}: {wp}")

    steps = 0
    MAX_STEPS_PER_WAYPOINT = int(60.0 / control_env.dt)
    MAX_REPLANS = 10
    replans = 0
    reached_goal = False
    warned_divergence = False

    csv_file = open(f"{STUDENT_ID}_lab3.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["sim_x", "sim_y", "real_x", "real_y"])

    def log_row():
        if is_sim:
            sim_xy = control_env.state_true[0:2]
        else:
            sim_xy = ("", "")
        csv_writer.writerow([sim_xy[0], sim_xy[1], est[0], est[1]])

    loop_start = time.time()
    try:
        wp_index = 0
        while wp_index < len(waypoints) and steps < MAX_STEPS:
            waypoint = waypoints[wp_index]
            prev_waypoint = waypoints[wp_index - 1] if wp_index > 0 else None
            wp_steps = 0
            wp_confirm_count = 0
            collided = False

            # Three cases for where to steer, and whether this leg should
            # ever brake to a stop. waypoint_reached() below always checks
            # the TRUE waypoint for advancing wp_index, regardless of what
            # we're steering toward.
            is_final_waypoint = wp_index == len(waypoints) - 1
            in_line = (not is_final_waypoint) and _is_in_line(waypoints, wp_index)

            if is_final_waypoint:
                # The real goal - steer at it directly, and it's the one
                # place we actually want to come to rest.
                aim_xy = waypoint
            elif in_line:
                # Straight run (see _is_in_line()) - aim past this waypoint
                # at the NEXT one instead of slowing for it. That keeps the
                # steering target ~0.25m away the whole time (not the ~2.5cm
                # _aim_point() overshoot used for turns below), so the PD
                # law never approaches zero speed while cruising through a
                # corridor - only controller.py's commit phase kicks in,
                # and only once genuinely close to a real turn or the goal.
                aim_xy = waypoints[wp_index + 1]
            else:
                # A genuine turn - steer a little past it (see _aim_point())
                # so the controller doesn't brake as if arriving for good.
                aim_xy = _aim_point(waypoint, prev_waypoint)

            wp_target = SimpleNamespace(goal_pos=aim_xy, vel_limit=control_env.vel_limit,
                                         goal_tolerance=WAYPOINT_TOLERANCE,
                                         allow_stop=is_final_waypoint or not in_line)

            while wp_steps < MAX_STEPS_PER_WAYPOINT and steps < MAX_STEPS:
                was_turning = controller.turning
                action = controller.compute_action(wp_target, est, steps)
                if controller.turning and not was_turning:
                    # Mark where the ball was when the controller committed to
                    # a turn-in-place, using est (its input) - the position it
                    # was actually deciding from, not where it ends up after.
                    control_env.vis.add_turn_point(est[:2])
                timed_predict(action)
                obs, _, terminated, truncated, info = control_env.step(action)
                fixup_traj_point()
                est = ekf.update(to_map(obs))[0]
                control_env.update_estimate(est, ekf.P)
                render_shifted()
                log_row()

                raw = to_map(obs)

                if VERBOSE:
                    print(f"  wp{wp_index} s{wp_steps}: "
                          f"raw=({raw[0]:.3f},{raw[1]:.3f}) "
                          f"est=({est[0]:.3f},{est[1]:.3f}) "
                          f"hdg={np.degrees(est[2]):.0f}° spd={est[3]:.3f} "
                          f"-> tgt=({waypoint[0]:.2f},{waypoint[1]:.2f}) "
                          f"cmd_spd={action[0]:.3f} "
                          f"cmd_hdg={np.degrees(action[1]):.0f}°")

                gap = np.hypot(raw[0]-est[0], raw[1]-est[1])
                if gap > DIVERGENCE_WARN and not warned_divergence:
                    print(f"  *** WARNING: estimate and raw reading disagree by "
                          f"{gap:.3f}m. The odometry has probably broken (this "
                          f"happens if the ball is picked up mid-run). Anything "
                          f"logged after this point is unreliable. ***")
                    warned_divergence = True

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

                if waypoint_reached(est[:2], waypoint, prev_waypoint, require_radius=not in_line):
                    wp_confirm_count += 1
                    if wp_confirm_count >= WAYPOINT_CONFIRM_STEPS:
                        print(f"Reached waypoint {wp_index}: {waypoint}")
                        break
                else:
                    wp_confirm_count = 0

            if reached_goal:
                print("Goal reached.")
                if warned_divergence:
                    print("  (NOTE: a divergence warning fired earlier in this "
                          "run, so this result may not reflect where the ball "
                          "physically ended up.)")
                break

            if collided and replans < MAX_REPLANS:
                print(f"Collision near wp{wp_index}, step {wp_steps} - escaping")

                pos_before = np.array([est[0], est[1]])

                for _ in range(8):
                    straight_back = np.array([-0.08, est[2]], dtype=np.float32)
                    timed_predict(straight_back)
                    obs, _, terminated, truncated, info = control_env.step(straight_back)
                    fixup_traj_point()
                    est = ekf.update(to_map(obs))[0]
                    control_env.update_estimate(est, ekf.P)
                    render_shifted()
                    log_row()
                    steps += 1

                moved = np.hypot(est[0]-pos_before[0], est[1]-pos_before[1])

                if moved < 0.005:
                    print(f"  ball didn't move ({moved:.3f}m) - held or stalled, "
                          f"waiting rather than skipping")
                    for _ in range(20):
                        hold = np.array([0.0, est[2]], dtype=np.float32)
                        timed_predict(hold)
                        obs, _, terminated, truncated, info = control_env.step(hold)
                        fixup_traj_point()
                        est = ekf.update(to_map(obs))[0]
                        control_env.update_estimate(est, ekf.P)
                        render_shifted()
                        log_row()
                        steps += 1
                    controller.reset()
                    continue

                try:
                    waypoints = planner.plan(est, control_env.goal_pos, margin_cells=0)
                    control_env.vis.set_waypoints(waypoints)
                    control_env.vis.set_waypoint_zones(_waypoint_zones(waypoints), WAYPOINT_TOLERANCE)
                    wp_index = 0
                    controller.reset()
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

    finally:
        if steps:
            timing = f"{steps} steps, {(time.time() - loop_start) / steps:.3f} s per step on average"
            print(timing)
            with open("logs/lab3_timing.txt", "a") as f:
                f.write(f"{'sim' if is_sim else 'real'}: {timing}\n")
        csv_file.close()
        control_env.emergency_stop()

    # Keep the window open (still showing the final trajectory/maze) until
    # the user closes it or presses a key, instead of it vanishing the
    # instant the run finishes.
    print("Run finished - close the window or press any key to exit.")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                waiting = False
        render_shifted()
        pygame.time.wait(50)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run simulation")
    parser.add_argument("--fast-comms", action="store_true",
                        help="Real robot only: drive through labs/lab2/fast_comms' low-latency "
                             "BLE path instead of SpheroEduAPI (see fast_link.py)")
    args = parser.parse_args(argv)

    with managed_env(args.sim, fast_comms=args.fast_comms) as control_env:
        control_loop(control_env)


if __name__ == "__main__":
    main()