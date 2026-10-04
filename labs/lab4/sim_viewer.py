"""Run controls and replay for the Lab 4 window.

The sim has no hardware to wait on, so it runs as fast as pygame draws. With
pace=True this paces each step to the control period (STEP_PERIOD) at an
adjustable playback speed, without touching dt or the physics. On the real robot
(pace=False) it never blocks - the robot paces itself - and only listens for a stop
key. Either way every step is recorded for a scrubbable replay.

Sim:     SPACE pause/play   RIGHT/N single step (paused)   UP/DOWN speed x2 / /2
         0 reset to 1x   +/- HUD text size   Q/ESC stop the run
Real:    SPACE/ESC/Q stop the run (the robot is stopped)
Replay:  LEFT/RIGHT step (hold to scroll)   wheel/drag bar to scrub   HOME/END jump
         SPACE play/pause   UP/DOWN speed   Q/ESC back
"""
import time

import pygame

SPEEDS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
LIVE_HELP = "SPC pause  N step  UP/DN speed"
REAL_HELP = "SPC/ESC stop"
REPLAY_HELP = "UP/DN speed  ESC back"


class StopRun(Exception):
    """Raised from the viewer when the user stops the run from the window."""


class SimViewer:
    def __init__(self, env, period, speed=1.0, start_paused=False, pace=True, on_idle=None):
        self.env = env
        self.vis = env.vis
        self.period = period
        self.speed_idx = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - speed))
        self.start_paused = start_paused
        self.pace = pace
        self.on_idle = on_idle      # called while the replay sits open (e.g. BLE keepalive)
        self.new_run()

    def new_run(self):
        """Forget the previous run's frames and timing."""
        self.paused = self.start_paused if self.pace else False
        self.frames = []
        self._last = None

    @property
    def speed(self):
        return SPEEDS[self.speed_idx]

    # ------------------------------------------------------------------ live

    def after_step(self):
        """Call once per step, after env.render(). Snapshots the frame for replay,
        then (sim) blocks until it is time for the next step, or (real) just checks
        for a stop key. Raises StopRun to end the run."""
        self._snapshot()
        if not self.pace:
            self.vis.set_hud(status=f"REAL RUN  step {len(self.frames)}\n{REAL_HELP}")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise StopRun
                if self.handle_common(event) or event.type != pygame.KEYDOWN:
                    continue
                if event.key in (pygame.K_q, pygame.K_ESCAPE, pygame.K_SPACE):
                    raise StopRun
            return
        due = (self._last or time.perf_counter()) + self.period / self.speed
        step_once = False
        while True:
            self._status_live()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise StopRun
                if self.handle_common(event) or event.type != pygame.KEYDOWN:
                    continue
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    raise StopRun
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    due = time.perf_counter() + self.period / self.speed
                if event.key in (pygame.K_RIGHT, pygame.K_n) and self.paused:
                    step_once = True
            if step_once or (not self.paused and time.perf_counter() >= due):
                break
            self._redraw_live()
        self._last = time.perf_counter()

    def _status_live(self):
        state = "PAUSED" if self.paused else "PLAYING"
        self.vis.set_hud(status=f"SIM {state} {self.speed:g}x  step {len(self.frames)}\n{LIVE_HELP}")

    def _redraw_live(self):
        self.vis.render(self.env.state_true, self.env.state_odom)

    # ---------------------------------------------------------------- replay

    def _snapshot(self):
        v = self.vis
        self.frames.append({
            "n_gt": len(v._gt_traj), "n_odom": len(v._odom_traj), "n_est": len(v._est_traj),
            "gt": self.env.state_true.copy(), "odom": self.env.state_odom.copy(),
            "belief": (None if v.belief_mean is None else
                       (v.belief_mean.copy(), v.belief_cov.copy())),
            "action": v.hud_action, "collision": v.hud_collision, "step": v.hud_step,
        })

    def replay(self):
        """Scrub through the last run like a video. Returns "quit" if the window was
        closed, otherwise "idle" (ESC/Q pressed)."""
        frames, v = self.frames, self.vis
        if not frames or v.screen is None:
            print("No run to replay yet.")
            return "idle"
        gt, odom, est = list(v._gt_traj), list(v._odom_traj), list(v._est_traj)
        n, i = len(frames), len(frames) - 1
        playing, scrubbing, last_advance = False, False, 0.0
        print("Replay: LEFT/RIGHT step, drag the bar, SPACE play, UP/DOWN speed, ESC back")
        pygame.key.set_repeat(300, 40)

        def seek(mx):
            bar = v.scrub_bar_rect()
            return int(round(min(1.0, max(0.0, (mx - bar.x) / max(1, bar.width))) * (n - 1)))

        try:
            while True:
                f = frames[i]
                v.set_trajectories(gt=gt[:f["n_gt"]], odom=odom[:f["n_odom"]], est=est[:f["n_est"]])
                if f["belief"] is not None:
                    v.set_belief(*f["belief"])
                v.hud_action, v.hud_collision, v.hud_step = f["action"], f["collision"], f["step"]
                v.set_hud(status=f"REPLAY {'>' if playing else '||'} {i + 1}/{n}  "
                                 f"{self.speed:g}x\n{REPLAY_HELP}")
                v.render(f["gt"], f["odom"], scrub=(i, n, playing))

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return "quit"
                    if self.handle_common(event):
                        continue
                    if event.type == pygame.KEYDOWN:
                        k = event.key
                        if k in (pygame.K_q, pygame.K_ESCAPE):
                            return "idle"
                        if k == pygame.K_RIGHT:
                            i, playing = min(n - 1, i + 1), False
                        if k == pygame.K_LEFT:
                            i, playing = max(0, i - 1), False
                        if k == pygame.K_HOME:
                            i, playing = 0, False
                        if k == pygame.K_END:
                            i, playing = n - 1, False
                        if k == pygame.K_SPACE:
                            if i == n - 1:
                                i = 0
                            playing, last_advance = not playing, time.perf_counter()
                    if event.type == pygame.MOUSEWHEEL:
                        i, playing = min(n - 1, max(0, i + (1 if event.y < 0 else -1))), False
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if v.scrub_bar_rect().inflate(0, 14).collidepoint(event.pos):
                            scrubbing, playing, i = True, False, seek(event.pos[0])
                    if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        scrubbing = False
                    if event.type == pygame.MOUSEMOTION and scrubbing:
                        i = seek(event.pos[0])

                if playing and time.perf_counter() - last_advance >= self.period / self.speed:
                    last_advance = time.perf_counter()
                    if i < n - 1:
                        i += 1
                    else:
                        playing = False
                if self.on_idle is not None:
                    self.on_idle()
        except KeyboardInterrupt:
            return "idle"
        finally:
            pygame.key.set_repeat()

    # ---------------------------------------------------------------- shared

    def handle_common(self, event):
        """Resize, HUD text size and playback speed - same in live and replay."""
        if event.type == pygame.VIDEORESIZE:
            self.vis.handle_resize(event.w, event.h)
            return True
        if event.type != pygame.KEYDOWN:
            return False
        k = event.key
        if k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.vis.set_hud_font_size(self.vis.hud_font_size + 2)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.vis.set_hud_font_size(self.vis.hud_font_size - 2)
        elif k in (pygame.K_UP, pygame.K_RIGHTBRACKET):
            self.speed_idx = min(len(SPEEDS) - 1, self.speed_idx + 1)
        elif k in (pygame.K_DOWN, pygame.K_LEFTBRACKET):
            self.speed_idx = max(0, self.speed_idx - 1)
        elif k in (pygame.K_0, pygame.K_KP0):
            self.speed_idx = SPEEDS.index(1.0)
        else:
            return False
        return True
