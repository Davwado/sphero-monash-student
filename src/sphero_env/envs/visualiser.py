import os
import csv
import math
import numpy as np
import pygame

class Visualiser:
    """
    Handles all rendering and logging for both simulator and real robot.
    Supports:
      - Map/occupancy rendering
      - Ground truth (when available)
      - Odometry
      - State estimates + uncertainty
      - Unified CSV logging
    """
    def __init__(self, world_width, world_height, goal_pos, render_mode=None, window_size=(600,600), occupancy_grid=None, grid_resolution=0.1, csv_path=None):
        self.world_width = world_width
        self.world_height = world_height
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.render_mode = render_mode
        self.window_size = window_size
        self.occupancy_grid = occupancy_grid
        self.grid_resolution = grid_resolution
        self.csv_path = csv_path
        self._file = None
        self._writer = None
        self._columns = [
            "est_x", "est_y", "odom_x", "odom_y", "gt_x", "gt_y",
            "cov_x", "cov_y", "cov_xy", "cov_yx",
            "heading_cmd", "speed_cmd", "heading", "speed",
            "setpoint_x", "setpoint_y",
            # Appended columns (keep at the end for backwards compatibility);
            # these make a saved CSV a complete replay file for lab1 --replay.
            "gt_heading", "gt_speed", "collision", "step"
        ]
        self._gt_traj = []
        self._odom_traj = []
        self._est_traj = []
        self._overlay_real_traj = None
        self._overlay_odom_traj = None
        self._overlay_true_traj = None
        self.belief_mean = None
        self.belief_cov = None
        self.visual_occupancy_grid = None
        self.distance_map = None
        self.screen = None
        self.clock = None
        self._font = None
        self._help_font = None
        # HUD state (commanded vs measured values shown live on the window)
        self.hud_font_size = 22      # adjustable via set_hud_font_size / +/- keys
        self._hud_font = None
        self.hud_action = None       # (speed_cmd, heading_cmd) from last record()
        self.hud_collision = 0.0
        self.hud_step = 0
        self.hud_run = 0
        self.hud_status = ""

    def start_logging(self):
        if not self.csv_path or self._writer is not None:
            return
        dirpath = os.path.dirname(self.csv_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._columns)
        self._file.flush()

    def stop_logging(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

    def reset(self):
        self._gt_traj.clear()
        self._odom_traj.clear()
        self._est_traj.clear()
        self._overlay_real_traj = None
        self._overlay_odom_traj = None
        self._overlay_true_traj = None
        self.belief_mean = None
        self.belief_cov = None
        self.hud_action = None
        self.hud_collision = 0.0
        self.hud_step = 0

    def set_hud(self, run=None, status=None):
        """Update session-level HUD fields (run counter, status line)."""
        if run is not None:
            self.hud_run = int(run)
        if status is not None:
            self.hud_status = str(status)

    def set_hud_font_size(self, size):
        """Set the HUD text size (clamped to 12-48). Layout scales with it."""
        self.hud_font_size = int(max(12, min(48, size)))
        if self.screen is not None:
            self._hud_font = pygame.font.SysFont(None, self.hud_font_size)

    def set_trajectories(self, gt=None, odom=None, est=None):
        """Replace the drawn trajectories (used by replay to show a prefix)."""
        if gt is not None:
            self._gt_traj = list(gt)
        if odom is not None:
            self._odom_traj = list(odom)
        if est is not None:
            self._est_traj = list(est)

    def scrub_bar_rect(self):
        """Screen rect of the replay scrub bar (shared with mouse hit-tests)."""
        w, h = self.window_size
        return pygame.Rect(20, h - 34, w - 40, 14)

    def handle_resize(self, w, h):
        """React to the window being resized/maximised: everything is drawn
        from self.window_size, so updating it rescales the whole view."""
        self.window_size = (max(200, int(w)), max(200, int(h)))
        if self.screen is not None:
            self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)

    def set_belief(self, mean, cov):
        self.belief_mean = np.asarray(mean, dtype=np.float32)
        self.belief_cov = np.asarray(cov, dtype=np.float32)

    def set_goal(self, goal_pos):
        self.goal_pos = np.array(goal_pos, dtype=np.float32)

    def set_occupancy(self, grid, resolution, visual_grid=None):
        self.occupancy_grid = grid
        self.grid_resolution = resolution
        if visual_grid is not None:
            self.visual_occupancy_grid = visual_grid

    def set_distance_map(self, dist_map):
        self.distance_map = dist_map

    def set_overlay_trajectories(self, real=None, odom=None, true=None):
        def _clean(traj):
            if traj is None:
                return None
            arr = np.asarray(traj, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("Trajectory must have shape (N,2)")
            return arr

        self._overlay_real_traj = _clean(real)
        self._overlay_odom_traj = _clean(odom)
        self._overlay_true_traj = _clean(true)

    def record(self, gt_state, odom_state, action, est_state=None, est_cov=None, setpoint=None,
               collision=None, step_count=None):
        # Stash latest values for the HUD overlay
        self.hud_action = None if action is None else (float(action[0]), float(action[1]))
        if collision is not None:
            self.hud_collision = float(collision)
        if step_count is not None:
            self.hud_step = int(step_count)
        # Append to trajectories
        if gt_state is not None:
            self._gt_traj.append(tuple(gt_state[:2]))
        if odom_state is not None:
            self._odom_traj.append(tuple(odom_state[:2]))
        if est_state is not None:
            self._est_traj.append(tuple(est_state[:2]))
        # Prepare log row
        row = {
            "gt_x": float(gt_state[0]) if gt_state is not None else np.nan,
            "gt_y": float(gt_state[1]) if gt_state is not None else np.nan,
            "odom_x": float(odom_state[0]) if odom_state is not None else np.nan,
            "odom_y": float(odom_state[1]) if odom_state is not None else np.nan,
            "est_x": float(est_state[0]) if est_state is not None else np.nan,
            "est_y": float(est_state[1]) if est_state is not None else np.nan,
            "heading_cmd": float(action[1]) if action is not None else np.nan,
            "speed_cmd": float(action[0]) if action is not None else np.nan,
            "heading": float(odom_state[2]) if odom_state is not None and len(odom_state) > 2 else np.nan,
            "speed": float(odom_state[3]) if odom_state is not None and len(odom_state) > 3 else np.nan,
        }
        if setpoint is not None:
            setpoint_arr = np.asarray(setpoint, dtype=np.float32).reshape(-1)
            if setpoint_arr.shape[0] >= 2:
                row["setpoint_x"] = float(setpoint_arr[0])
                row["setpoint_y"] = float(setpoint_arr[1])
            else:
                row["setpoint_x"] = np.nan
                row["setpoint_y"] = np.nan
        else:
            row["setpoint_x"] = np.nan
            row["setpoint_y"] = np.nan
        # Covariance
        if est_cov is not None:
            cov = np.asarray(est_cov, dtype=np.float32)
            if cov.shape == (4, 4):
                cov2 = cov[0:2, 0:2]
            elif cov.shape == (2, 2):
                cov2 = cov
            else:
                cov2 = np.full((2,2), np.nan)
            row["cov_x"] = float(cov2[0,0])
            row["cov_y"] = float(cov2[1,1])
            row["cov_xy"] = float(cov2[0,1])
            row["cov_yx"] = float(cov2[1,0])
        else:
            row["cov_x"] = row["cov_y"] = row["cov_xy"] = row["cov_yx"] = np.nan
        # Replay columns
        row["gt_heading"] = float(gt_state[2]) if gt_state is not None and len(gt_state) > 2 else np.nan
        row["gt_speed"] = float(gt_state[3]) if gt_state is not None and len(gt_state) > 3 else np.nan
        row["collision"] = float(collision) if collision is not None else np.nan
        row["step"] = int(step_count) if step_count is not None else np.nan
        # Write
        if self._writer is not None:
            self._writer.writerow([row.get(col, np.nan) for col in self._columns])
            self._file.flush()

    def get_traj(self):
        return [x for x, _ in self._odom_traj], [y for _, y in self._odom_traj], None

    def _init_pygame(self):
        if self.screen is not None:
            return
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Sphero Visualiser")
        self.clock = pygame.time.Clock()
        self._font = pygame.font.SysFont(None, 18)
        self._hud_font = pygame.font.SysFont(None, self.hud_font_size)
        self._help_font = pygame.font.SysFont(None, 16)

    def render(self, gt_state, odom_state, scrub=None):
        """Draw a frame. `scrub=(frame_idx, total_frames)` adds a replay
        scrub bar at the bottom of the window."""
        if self.render_mode != "human":
            return
        self._init_pygame()
        width, height = self.window_size
        cx, cy = width // 2, height // 2
        # Colors
        bg_color = (20, 20, 30)
        axis_color = (80, 80, 80)
        border_color = (100, 100, 255)
        true_color = (0, 200, 0)
        odom_color = (50, 150, 255)
        goal_color = (255, 215, 0)
        obstacle_color = (200, 200, 200)
        belief_color = (255, 0, 255)
        heading_len = 0.15
        self.screen.fill(bg_color)
        # Axes
        pygame.draw.line(self.screen, axis_color, (0, cy), (width, cy), 1)
        pygame.draw.line(self.screen, axis_color, (cx, 0), (cx, height), 1)
        # Scaling
        scale_x = width / self.world_width
        scale_y = height / self.world_height
        scale = min(scale_x, scale_y) * 0.9
        rect_w = self.world_width * scale
        rect_h = self.world_height * scale
        rect = pygame.Rect(0, 0, rect_w, rect_h)
        rect.center = (cx, cy)
        pygame.draw.rect(self.screen, border_color, rect, 1)
        def world_to_screen(xw, yw):
            px = cx + xw * scale
            py = cy - yw * scale
            return int(px), int(py)
        def draw_polyline(traj, color, width=2):
            if traj is None or len(traj) < 2:
                return
            pts = [world_to_screen(float(x), float(y)) for x, y in traj]
            pygame.draw.lines(self.screen, color, False, pts, width)
        def draw_last_point(traj, color, radius=5, width=0):
            if traj is None or len(traj) == 0:
                return
            x, y = traj[-1]
            px, py = world_to_screen(float(x), float(y))
            pygame.draw.circle(self.screen, color, (px, py), radius, width)
        # Draw occupancy grid
        if self.occupancy_grid is not None:
            h, w = self.occupancy_grid.shape
            cell_screen_w = self.grid_resolution * scale
            cell_screen_h = self.grid_resolution * scale
            for i in range(h):
                for j in range(w):
                    if self.occupancy_grid[i, j] == 0:
                        continue
                    x_cell = (j - w / 2.0 + 0.5) * self.grid_resolution
                    y_cell = -(i - h / 2.0 + 0.5) * self.grid_resolution
                    px, py = world_to_screen(x_cell, y_cell)
                    cell_rect = pygame.Rect(
                        px - cell_screen_w / 2,
                        py - cell_screen_h / 2,
                        cell_screen_w,
                        cell_screen_h,
                    )
                    pygame.draw.rect(self.screen, obstacle_color, cell_rect)
        # Draw estimated occupancy overlay
        if self.visual_occupancy_grid is not None:
            vh, vw = self.visual_occupancy_grid.shape
            overlay_surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
            cell_screen_w = self.grid_resolution * scale
            cell_screen_h = self.grid_resolution * scale
            v_grid = np.asarray(self.visual_occupancy_grid, dtype=np.float32)
            v_max = float(v_grid.max())
            if v_max > 1e-12:
                v_norm = v_grid / v_max
            else:
                v_norm = v_grid
            for i in range(vh):
                for j in range(vw):
                    intensity = float(v_norm[i, j])
                    if intensity < 1e-4:
                        continue
                    alpha = int(np.clip(intensity * 220, 15, 220))
                    color = (255, 140, 0, alpha)
                    x_cell = (j - vw / 2.0 + 0.5) * self.grid_resolution
                    y_cell = -(i - vh / 2.0 + 0.5) * self.grid_resolution
                    px, py = world_to_screen(x_cell, y_cell)
                    cell_rect = pygame.Rect(
                        px - cell_screen_w / 2,
                        py - cell_screen_h / 2,
                        cell_screen_w,
                        cell_screen_h,
                    )
                    pygame.draw.rect(overlay_surface, color, cell_rect)
            self.screen.blit(overlay_surface, (0, 0))
        # Draw distance map heatmap
        if self.distance_map is not None and self.occupancy_grid is not None:
            h, w = self.distance_map.shape
            finite_dists = self.distance_map[np.isfinite(self.distance_map)]
            max_dist = np.max(finite_dists) if finite_dists.size > 0 else 1.0
            heatmap_surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
            for i in range(h):
                for j in range(w):
                    if self.occupancy_grid[i, j] == 0:
                        dist = self.distance_map[i, j]
                        if np.isfinite(dist):
                            ratio = dist / max_dist
                            color = (int(255 * ratio), 0, int(255 * (1 - ratio)), 100)
                        else:
                            color = (128, 128, 128, 100)
                        x_cell = (j - w / 2.0 + 0.5) * self.grid_resolution
                        y_cell = -(i - h / 2.0 + 0.5) * self.grid_resolution
                        px, py = world_to_screen(x_cell, y_cell)
                        cell_screen_w = self.grid_resolution * scale
                        cell_screen_h = self.grid_resolution * scale
                        rect = pygame.Rect(
                            px - cell_screen_w / 2,
                            py - cell_screen_h / 2,
                            cell_screen_w,
                            cell_screen_h
                        )
                        pygame.draw.rect(heatmap_surface, color, rect)
            self.screen.blit(heatmap_surface, (0, 0))
        # Draw goal
        gx, gy = self.goal_pos
        goal_px, goal_py = world_to_screen(gx, gy)
        pygame.draw.circle(self.screen, goal_color, (goal_px, goal_py), 7)
        # Draw trajectory overlays. If explicit overlays are not set, fall back to live trajectories.
        draw_polyline(self._overlay_real_traj, (255, 120, 120), width=2)
        draw_polyline(self._overlay_odom_traj if self._overlay_odom_traj is not None else self._odom_traj, (80, 180, 255), width=2)
        draw_polyline(self._overlay_true_traj if self._overlay_true_traj is not None else self._gt_traj, (120, 255, 120), width=2)
        draw_polyline(self._est_traj, (255, 0, 255), width=2)
        draw_last_point(self._overlay_real_traj, (255, 120, 120), radius=4)
        # True pose
        if gt_state is not None:
            x_true, y_true, heading_true, _ = gt_state
            true_px, true_py = world_to_screen(x_true, y_true)
            pygame.draw.circle(self.screen, true_color, (true_px, true_py), 6)
            tx2 = x_true + heading_len * np.sin(heading_true)
            ty2 = y_true + heading_len * np.cos(heading_true)
            tpx2, tpy2 = world_to_screen(tx2, ty2)
            pygame.draw.line(self.screen, true_color, (true_px, true_py), (tpx2, tpy2), 2)
        # Odom pose
        if odom_state is not None:
            x_o, y_o, heading_o, _ = odom_state
            odom_px, odom_py = world_to_screen(x_o, y_o)
            pygame.draw.circle(self.screen, odom_color, (odom_px, odom_py), 5)
            ox2 = x_o + heading_len * np.sin(heading_o)
            oy2 = y_o + heading_len * np.cos(heading_o)
            opx2, opy2 = world_to_screen(ox2, oy2)
            pygame.draw.line(self.screen, odom_color, (odom_px, odom_py), (opx2, opy2), 2)
            # Ghost line: commanded heading (slightly longer than the measured one)
            if self.hud_action is not None:
                _, heading_cmd = self.hud_action
                gx2 = x_o + 0.20 * np.sin(heading_cmd)
                gy2 = y_o + 0.20 * np.cos(heading_cmd)
                gpx2, gpy2 = world_to_screen(gx2, gy2)
                pygame.draw.line(self.screen, (255, 165, 0), (odom_px, odom_py), (gpx2, gpy2), 1)
        # Belief mean + covariance ellipse
        if self.belief_mean is not None and self.belief_cov is not None:
            mean = self.belief_mean
            cov = self.belief_cov
            if mean.shape[0] == 4:
                mean_pos = mean[0:2]
            else:
                mean_pos = mean
            if cov.shape == (4, 4):
                cov_pos = cov[0:2, 0:2]
            else:
                cov_pos = cov
            bx, by = float(mean_pos[0]), float(mean_pos[1])
            bpx, bpy = world_to_screen(bx, by)
            pygame.draw.circle(self.screen, belief_color, (bpx, bpy), 4, 1)
            self._draw_covariance_ellipse(self.screen, mean_pos, cov_pos, world_to_screen, belief_color, n_std=2.0)
        # Legend
        font = self._font
        legend_lines = [
            "Green: ground truth",
            "Blue: odometry",
            "Magenta: estimate",
            "Orange: commanded",
        ]
        for idx, text in enumerate(legend_lines):
            surf = font.render(text, True, (220, 220, 220))
            self.screen.blit(surf, (5, 5 + 18 * idx))
        self._draw_hud(gt_state, odom_state)
        if scrub is not None:
            self._draw_scrub_bar(*scrub)
        pygame.display.flip()
        self.clock.tick(60)

    def _draw_scrub_bar(self, idx, total, playing=False):
        """Video-style timeline with a player control strip above it.

        The strip is justified across the full bar width: drawn icons
        (step, jump, play/pause, scrub) plus keycap-styled letter keys.
        """
        scr = self.screen
        bar = self.scrub_bar_rect()
        orange = (255, 165, 0)
        icon_col = (220, 220, 220)
        text_col = (200, 200, 205)
        cap_bg = (52, 52, 72)
        cap_border = (115, 115, 145)
        row_y = bar.y - 26          # top of the control strip
        ih = 12                     # icon height
        iy = row_y + 1              # icon top (visually centered vs 16px text)
        font = self._help_font

        # --- icon painters: each returns its width ---------------------------
        def icon_step(x):
            pygame.draw.polygon(scr, icon_col, [(x + 8, iy), (x + 8, iy + ih), (x, iy + ih // 2)])
            pygame.draw.polygon(scr, icon_col, [(x + 12, iy), (x + 12, iy + ih), (x + 20, iy + ih // 2)])
            return 20

        def icon_jump(x):
            pygame.draw.rect(scr, icon_col, (x, iy, 2, ih))
            pygame.draw.polygon(scr, icon_col, [(x + 12, iy), (x + 12, iy + ih), (x + 4, iy + ih // 2)])
            pygame.draw.polygon(scr, icon_col, [(x + 16, iy), (x + 16, iy + ih), (x + 24, iy + ih // 2)])
            pygame.draw.rect(scr, icon_col, (x + 26, iy, 2, ih))
            return 28

        def icon_play_pause(x):
            if playing:  # show pause while playing
                pygame.draw.rect(scr, orange, (x, iy, 4, ih))
                pygame.draw.rect(scr, orange, (x + 7, iy, 4, ih))
            else:        # show play while paused
                pygame.draw.polygon(scr, orange, [(x, iy), (x, iy + ih), (x + 11, iy + ih // 2)])
            return 11

        def icon_scrub(x):
            cy = iy + ih // 2
            pygame.draw.line(scr, icon_col, (x, cy), (x + 22, cy), 2)
            pygame.draw.circle(scr, orange, (x + 13, cy), 5)
            return 22

        def keycap(label):
            t = font.render(label, True, orange)
            w = t.get_width() + 10

            def draw(x):
                r = pygame.Rect(x, row_y - 1, w, 16)
                pygame.draw.rect(scr, cap_bg, r, border_radius=4)
                pygame.draw.rect(scr, cap_border, r, 1, border_radius=4)
                scr.blit(t, (x + 5, row_y + 1))
                return w
            return draw

        # --- segments: (icon painter or None, icon width, label text) --------
        segments = [
            (icon_step, 20, "step"),
            (icon_jump, 28, "Home/End"),
            (icon_play_pause, 11, "Space"),
            (icon_scrub, 22, "wheel / drag"),
            (keycap("S"), font.size("S")[0] + 10, "save"),
            (keycap("Esc"), font.size("Esc")[0] + 10, "back"),
            (keycap("Q"), font.size("Q")[0] + 10, "quit"),
        ]
        counter = font.render(f"{idx + 1} / {total}", True, orange)

        # Justify: distribute leftover width evenly between segments + counter.
        labels = [font.render(s[2], True, text_col) for s in segments]
        widths = [s[1] + 5 + l.get_width() for s, l in zip(segments, labels)]
        widths.append(counter.get_width())
        gap = max(8, (bar.width - sum(widths)) // (len(widths) - 1))
        x = bar.x
        for (painter, iw, _), label in zip(segments, labels):
            painter(x)
            scr.blit(label, (x + iw + 5, row_y))
            x += iw + 5 + label.get_width() + gap
        scr.blit(counter, (bar.right - counter.get_width(), row_y))

        # --- the timeline itself ---------------------------------------------
        pygame.draw.rect(scr, (40, 40, 55), bar)
        frac = 0.0 if total <= 1 else idx / (total - 1)
        fill = bar.copy()
        fill.width = max(0, int(bar.width * frac))
        pygame.draw.rect(scr, (90, 90, 130), fill)
        pygame.draw.rect(scr, (140, 140, 140), bar, 1)
        handle_x = bar.x + int(bar.width * frac)
        pygame.draw.circle(scr, orange, (handle_x, bar.centery), 9)

    def _draw_arrow(self, surface, center, heading_rad, length, color, width=2):
        """Draw a small arrow at `center` pointing along `heading_rad`.

        Heading convention matches the world view: 0 rad = up (+y),
        +pi/2 = right (+x). Screen y grows downward, hence (sin, -cos).
        """
        cx, cy = center
        dx = math.sin(heading_rad)
        dy = -math.cos(heading_rad)
        tip = (cx + dx * length, cy + dy * length)
        pygame.draw.line(surface, color, (cx, cy), tip, width)
        # Arrowhead: two short lines swept back from the tip
        head_len = max(4, length * 0.35)
        ang = math.atan2(dy, dx)
        for offset in (2.6, -2.6):
            hx = tip[0] + head_len * math.cos(ang + offset)
            hy = tip[1] + head_len * math.sin(ang + offset)
            pygame.draw.line(surface, color, tip, (hx, hy), width)

    def _draw_hud(self, gt_state, odom_state):
        """Semi-transparent panel (top-right) with commanded vs measured values.

        All layout scales with self.hud_font_size (see set_hud_font_size).
        """
        cmd_color = (255, 165, 0)
        mea_color = (50, 150, 255)
        text_color = (220, 220, 220)
        dim_color = (140, 140, 140)
        alert_color = (255, 60, 60)

        fs = self.hud_font_size
        rh = fs                      # row height
        pad = max(6, fs // 3)
        gap = int(0.4 * rh)
        comp_r = int(1.6 * fs)
        panel_w = int(12 * fs)
        # Height: header+status, 4 value rows, pos+goal+collision, compass.
        panel_h = (pad + 2 * rh + gap + 4 * rh + gap + 3 * rh + int(0.5 * rh)
                   + fs + 2 * comp_r + pad)
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 140))
        font = self._hud_font
        arrow_len = max(8, int(0.5 * fs))

        def row(y, label, color, arrow_heading=None):
            panel.blit(font.render(label, True, color), (pad, y))
            if arrow_heading is not None:
                self._draw_arrow(panel, (panel_w - pad - arrow_len, y + rh // 2 - 2),
                                 arrow_heading, arrow_len, color)

        y = pad
        row(y, f"Run {self.hud_run}   Step {self.hud_step}", text_color); y += rh
        if self.hud_status:
            row(y, self.hud_status, text_color)
        y += rh + gap

        if self.hud_action is not None:
            speed_cmd, heading_cmd = self.hud_action
            row(y, f"Cmd hdg: {math.degrees(heading_cmd) % 360:6.1f} deg", cmd_color, heading_cmd)
        else:
            row(y, "Cmd hdg:     -- deg", dim_color)
        y += rh
        if odom_state is not None:
            row(y, f"Mea hdg: {math.degrees(float(odom_state[2])) % 360:6.1f} deg", mea_color,
                float(odom_state[2]))
        else:
            row(y, "Mea hdg:     -- deg", dim_color)
        y += rh
        if self.hud_action is not None:
            row(y, f"Cmd spd: {self.hud_action[0]:+.3f} m/s", cmd_color)
        else:
            row(y, "Cmd spd:     -- m/s", dim_color)
        y += rh
        if odom_state is not None:
            row(y, f"Mea spd: {float(odom_state[3]):+.3f} m/s", mea_color)
        else:
            row(y, "Mea spd:     -- m/s", dim_color)
        y += rh + gap

        if odom_state is not None:
            x_o, y_o = float(odom_state[0]), float(odom_state[1])
            dist = math.hypot(x_o - float(self.goal_pos[0]), y_o - float(self.goal_pos[1]))
            row(y, f"Pos: ({x_o:+.2f}, {y_o:+.2f}) m", text_color); y += rh
            row(y, f"Goal dist: {dist:.2f} m", text_color); y += rh
        else:
            row(y, "Pos:  --", dim_color); y += rh
            row(y, "Goal dist:  --", dim_color); y += rh
        if self.hud_collision > 0.5:
            row(y, "COLLISION", alert_color)
        else:
            row(y, "no contact", dim_color)
        y += rh + int(0.5 * rh)

        # Compass: commanded (orange) vs measured (blue) heading needles
        comp_cx, comp_cy = panel_w // 2, y + fs + comp_r
        pygame.draw.circle(panel, dim_color, (comp_cx, comp_cy), comp_r, 1)
        n_surf = font.render("N", True, dim_color)
        panel.blit(n_surf, (comp_cx - n_surf.get_width() // 2, comp_cy - comp_r - fs))
        pygame.draw.line(panel, dim_color, (comp_cx, comp_cy - comp_r),
                         (comp_cx, comp_cy - comp_r + max(4, fs // 4)), 1)
        if odom_state is not None:
            self._draw_arrow(panel, (comp_cx, comp_cy), float(odom_state[2]),
                             comp_r - max(6, fs // 3), mea_color, width=2)
        if self.hud_action is not None:
            self._draw_arrow(panel, (comp_cx, comp_cy), self.hud_action[1],
                             comp_r - max(6, fs // 3), cmd_color, width=2)

        self.screen.blit(panel, (self.window_size[0] - panel_w - 8, 8))

    def close(self):
        self.stop_logging()
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    def _draw_covariance_ellipse(self, surface, mean_pos, cov_pos, world_to_screen, color, n_std=2.0):
        cov_pos = 0.5 * (cov_pos + cov_pos.T)
        eigvals, eigvecs = np.linalg.eigh(cov_pos)
        eigvals = np.maximum(eigvals, 1e-12)
        r1 = n_std * math.sqrt(eigvals[0])
        r2 = n_std * math.sqrt(eigvals[1])
        order = np.argsort(eigvals)
        v = eigvecs[:, order[1]]
        angle = math.atan2(v[1], v[0])
        points = []
        for t in np.linspace(0, 2 * math.pi, num=40, endpoint=True):
            x_local = r1 * math.cos(t)
            y_local = r2 * math.sin(t)
            x_rot = x_local * math.cos(angle) - y_local * math.sin(angle)
            y_rot = x_local * math.sin(angle) + y_local * math.cos(angle)
            x_world = mean_pos[0] + x_rot
            y_world = mean_pos[1] + y_rot
            px, py = world_to_screen(x_world, y_world)
            points.append((px, py))
        if len(points) >= 2:
            pygame.draw.polygon(surface, color, points, width=1)

# LabVisualiser wrapper for labs
class LabVisualiser:
    def __init__(self, sim_env, real_env=None, csv_path=None):
        self.sim_env = sim_env
        self.real_env = real_env
        self.real_mode = real_env is not None
        self.csv_path = csv_path

        if self.csv_path is not None and hasattr(self.sim_env, "vis"):
            self.sim_env.vis.csv_path = self.csv_path

        # Expose model params
        for attr in ["dt", "vel_limit", "max_turn_rate", "heading_tolerance", "speed_scale", "goal_pos"]:
            setattr(self, attr, getattr(sim_env, attr, None))
    def start_logging(self):
        if self.csv_path is not None:
            self.sim_env.vis.csv_path = self.csv_path
        self.sim_env.vis.start_logging()
    def stop_logging(self):
        self.sim_env.vis.stop_logging()
    def reset(self):
        self.sim_env.vis.reset()
        if self.real_mode:
            self.sim_env.reset()
    def update(self, action, obs, info, ekf_mean=None, ekf_cov=None):
        # In dual-env setups, mirror action to the render env.
        if self.real_mode and self.real_env is not None and self.sim_env is not self.real_env:
            self.sim_env.step(action)
            gt = self.sim_env.state_true
        else:
            gt = info.get("state_true")
        odom = info.get("state_odom")
        setpoint = info.get("setpoint_xy") if isinstance(info, dict) else None
        self.sim_env.vis.record(gt, odom, action, ekf_mean, ekf_cov, setpoint=setpoint)
        if ekf_mean is not None and ekf_cov is not None:
            self.sim_env.vis.set_belief(ekf_mean, ekf_cov)
        # Overlay trajectories
        sim_odom_traj = self.sim_env.vis._odom_traj
        sim_gt_traj = self.sim_env.vis._gt_traj

        real_traj = None
        if self.real_mode:
            real_x, real_y, _ = self.real_env.get_live_traj()
            if len(real_x) > 0:
                real_traj = np.column_stack([np.array(real_x), np.array(real_y)])

        self.sim_env.vis.set_overlay_trajectories(real=real_traj, odom=sim_odom_traj, true=sim_gt_traj)

    def step(self, action, obs, info, ekf_mean=None, ekf_cov=None):
        """Backward-compatible helper: update visualiser state then render via env API."""
        self.update(action, obs, info, ekf_mean=ekf_mean, ekf_cov=ekf_cov)
        # Route rendering through the environment API (Gym-style) rather than calling visualiser directly.
        self.sim_env.render()
