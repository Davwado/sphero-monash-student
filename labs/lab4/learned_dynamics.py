"""Lab 4 - learned dynamics model for the Sphero.

The learned component of lab 4. It replaces the hand-tuned dynamics() used by
the simulator and the EKF with a model fitted to real robot data, behind the
SAME interface, so nothing else in the stack has to change:

    from learned_dynamics import load_dynamics

    dynamics = load_dynamics("weights/dynamics.pt")   # or load_dynamics() - see below
    env = SpheroEnv(dt=dynamics.dt, dynamics=dynamics, ...)
    ekf = EKF(dt=dynamics.dt, dynamics_fn=dynamics)

    next_state = dynamics(state, action)
        state      [x (m), y (m), heading (rad), speed]
        action     [speed_cmd, heading_cmd (rad)]
        next_state [x, y, heading, speed]
    Heading convention as everywhere else: 0 rad -> +y, pi/2 -> +x.

Give SpheroEnv and the EKF the same dt as the model (dynamics.dt). The model
advances one step of that length and has no way to know the caller's dt.

With no weights file, load_dynamics() returns the untrained model, which is
exactly the analytic base below. So integration can land first and the
trained weights can drop in later without code changes.

Model structure
---------------
    next_state = analytic(state, action; physical params) + residual(features)

The analytic part is the rate-limited unicycle this repo already uses. The
residual is a small MLP. It predicts corrections in the HEADING-ALIGNED frame
(along-track / cross-track) and rotates them into world x/y, so what it
learns about "the ball drifts right while turning" applies in every
direction instead of having to be relearned per heading. With a few hundred
samples that is the difference between generalising and memorising.

The final layer is zero-initialised, so an untrained residual contributes
exactly nothing.

Which channels can be learned - READ THIS
-----------------------------------------
On the real robot, the logged `heading` and `speed` are NOT measurements.
robot.py fills them from api.get_heading() / api.get_speed(), which echo back
the last command. Only x and y are measured. A model trained to predict the
logged heading or speed would just learn "next heading = the command", which
looks like a perfect fit and means nothing.

So `learn_channels` defaults to ("x", "y"): the residual and the loss are
masked to position. The heading and speed dynamics stay analytic, using
assumed parameters (see DynamicsConfig). Measured yaw and encoder speed are
now logged raw (meas_yaw_deg, meas_vel_{x,y}_cms - robot runs only). Once
dataset.py builds the state from those instead of the echoes, add "heading" /
"speed" to learn_channels and the same code trains them.

Of the physical parameters, only `speed_scale` (real ground speed per unit of
commanded speed) is identifiable from x/y, so it is the only one fitted by
default.

The echo also hides momentum. Because the logged speed drops to zero the
instant the command does, the state carries no memory of how fast the ball is
REALLY going - and it coasts. In the Phase 1 teleop log, 35 of 107 steps with
a zero command still moved more than 1 cm. A one-step model cannot predict that
from echoed state, whatever its size: the information is not in the input.
This is the main reason the position fit is weak today, and it will not be
fixed by training harder. Put measured speed (meas_vel_{x,y}_cms) in the
state and it becomes learnable.

Why float64
-----------
The EKF takes its Jacobian by finite differences with eps = 1e-6. In float32
a network's output moves in steps of ~1e-7 relative, so those differences
would be mostly rounding noise and the filter covariance would be garbage.
Everything here runs in float64, with a smooth activation (SiLU) so the
Jacobian is continuous.
"""
import argparse
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

DTYPE = torch.float64

# Measured on the real robot, Phase 1, 2026-09-16 (timing_report.py over
# labs/lab4/logs/timing_phase1_whiteboard_20260916.csv): 0.300 s median
# control period. The same as the EKF should use - lab3.py's 2.95 is ~10x too
# large.
DEFAULT_DT = 0.300

CHANNELS = ("x", "y", "heading", "speed")


def wrap_angle(a):
    """Wrap to [-pi, pi). Works on floats, numpy arrays and torch tensors."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class DynamicsConfig:
    dt: float = DEFAULT_DT

    # Analytic base. Turn rate and accel/decel are NOT measured - see the
    # module docstring. These are the lab-3 simulator's values: the only set
    # this repo has shown to give working closed-loop behaviour with the lab-3
    # controller at vel_limit 0.15. The lab-1 constants (0.3 rad/s,
    # 0.003 m/s^2) were tuned alongside a dt the code no longer uses, and at
    # 0.3 s they make the ball take ~50 s to reach cruising speed.
    max_turn_rate: float = 3.0     # rad/s
    max_accel: float = 0.3         # m/s^2
    max_decel: float = 0.5         # m/s^2

    # Ground speed = speed_scale * speed. Fitted by default (see fit_params).
    speed_scale: float = 1.0

    # Which physical params the optimiser may change. Only speed_scale is
    # identifiable from position-only data.
    fit_params: tuple = ("speed_scale",)

    # Which state channels the residual is allowed to correct and the loss is
    # computed on. Position only until yaw/encoder speed are logged.
    learn_channels: tuple = ("x", "y")

    hidden: int = 32

    # Fixed feature/output scales, so the network sees O(1) numbers without
    # having to save normalisation statistics alongside the weights.
    speed_ref: float = 0.15        # m/s, the lab vel_limit
    residual_scale: tuple = (0.05, 0.05, 0.5, 0.05)  # along, cross (m), heading (rad), speed

    # Free-form record of what the weights were trained on. Filled by
    # train_dynamics.py and printed by --check.
    meta: dict = field(default_factory=dict)


class ResidualDynamics(torch.nn.Module):
    """Analytic rate-limited unicycle plus a learned residual. Batched, float64."""

    def __init__(self, config: DynamicsConfig = None):
        super().__init__()
        self.config = config or DynamicsConfig()
        c = self.config

        # Physical parameters live in log space, so they stay positive
        # whatever the optimiser does. Frozen unless named in fit_params.
        for name in ("max_turn_rate", "max_accel", "max_decel", "speed_scale"):
            p = torch.nn.Parameter(torch.tensor(math.log(getattr(c, name)), dtype=DTYPE),
                                   requires_grad=name in c.fit_params)
            setattr(self, f"log_{name}", p)

        # features: speed, speed_cmd, sin(err), cos(err), dt
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, c.hidden, dtype=DTYPE), torch.nn.SiLU(),
            torch.nn.Linear(c.hidden, c.hidden, dtype=DTYPE), torch.nn.SiLU(),
            torch.nn.Linear(c.hidden, 4, dtype=DTYPE),
        )
        # Zero-init the output layer: an untrained residual is exactly zero,
        # so the untrained model IS the analytic model.
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

        unknown = set(c.learn_channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown learn_channels {unknown}; valid: {CHANNELS}")
        # Residual outputs are [along, cross, heading, speed]; along and cross
        # together make up x/y, so either of "x" or "y" enables both.
        pos = float("x" in c.learn_channels or "y" in c.learn_channels)
        mask = [pos, pos, float("heading" in c.learn_channels), float("speed" in c.learn_channels)]
        self.register_buffer("residual_mask", torch.tensor(mask, dtype=DTYPE))
        self.register_buffer("residual_scale", torch.tensor(c.residual_scale, dtype=DTYPE))

    def physical(self, name):
        return torch.exp(getattr(self, f"log_{name}"))

    def analytic(self, state, action, dt):
        """The hand-written part. Same form as labs/lab3/EKF.py's dynamics():
        heading and speed slew toward the command at limited rates, speed
        drops off with cos(heading error), and position moves with the speed
        carried INTO the step."""
        x, y, heading, speed = state.unbind(-1)
        speed_cmd, heading_cmd = action.unbind(-1)

        err = wrap_angle(heading_cmd - heading)
        max_turn = self.physical("max_turn_rate") * dt
        heading_new = wrap_angle(heading + torch.maximum(torch.minimum(err, max_turn), -max_turn))

        speed_target = speed_cmd * torch.clamp(torch.cos(err), min=0.0)
        speed_err = speed_target - speed
        max_step = torch.where(speed_err > 0,
                               self.physical("max_accel") * dt,
                               self.physical("max_decel") * dt)
        speed_new = torch.clamp(
            speed + torch.maximum(torch.minimum(speed_err, max_step), -max_step), 0.0, 1.0)

        ground = self.physical("speed_scale") * speed * dt
        x_new = x + ground * torch.sin(heading_new)
        y_new = y + ground * torch.cos(heading_new)
        return torch.stack([x_new, y_new, heading_new, speed_new], dim=-1)

    def features(self, state, action, dt):
        c = self.config
        speed, heading = state[..., 3], state[..., 2]
        speed_cmd, heading_cmd = action[..., 0], action[..., 1]
        err = wrap_angle(heading_cmd - heading)
        return torch.stack([
            speed / c.speed_ref,
            speed_cmd / c.speed_ref,
            torch.sin(err),
            torch.cos(err),
            dt / DEFAULT_DT,
        ], dim=-1)

    def forward(self, state, action, dt=None):
        """state [..., 4], action [..., 2], dt scalar or [...] -> next_state [..., 4]."""
        state = torch.as_tensor(state, dtype=DTYPE)
        action = torch.as_tensor(action, dtype=DTYPE)

        # Reverse, the way the real robot does it (robot.py:420): a negative
        # speed command is sent as the same positive speed at heading + pi.
        # The logged heading of a reversing step is that flipped heading, so
        # without this the model sees a 180 deg heading error on every reverse
        # step - and the analytic speed clamp would stop it reversing at all,
        # including lab 3's collision back-off of [-0.08, heading].
        rev = action[..., 0] < 0
        action = torch.stack([
            action[..., 0].abs(),
            torch.where(rev, wrap_angle(action[..., 1] + math.pi), action[..., 1]),
        ], dim=-1)

        if dt is None:
            dt = self.config.dt
        dt = torch.as_tensor(dt, dtype=DTYPE).expand(state.shape[:-1])

        base = self.analytic(state, action, dt)
        r = self.net(self.features(state, action, dt)) * self.residual_scale * self.residual_mask

        # Rotate [along, cross] from the heading-aligned frame into world x/y.
        # along is the direction of travel (sin h, cos h); cross is 90 deg to
        # its right (cos h, -sin h).
        h = base[..., 2]
        along, cross = r[..., 0], r[..., 1]
        dx = along * torch.sin(h) + cross * torch.cos(h)
        dy = along * torch.cos(h) - cross * torch.sin(h)

        return torch.stack([
            base[..., 0] + dx,
            base[..., 1] + dy,
            wrap_angle(base[..., 2] + r[..., 2]),
            torch.clamp(base[..., 3] + r[..., 3], 0.0, 1.0),
        ], dim=-1)


class DynamicsFn:
    """Plain-numpy wrapper with the dynamics(state, action) signature that
    SpheroEnv and the EKF expect. Returns float64, never tracks gradients."""

    def __init__(self, model: ResidualDynamics):
        self.model = model.eval()
        self.config = model.config
        self.dt = model.config.dt

    def __call__(self, state, action):
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(state, dtype=np.float64)[:4])
            a = torch.as_tensor(np.asarray(action, dtype=np.float64)[:2])
            return self.model(s, a).numpy()

    def __repr__(self):
        c = self.config
        trained = c.meta.get("trained_on") or "untrained (analytic base only)"
        scale = float(self.model.physical("speed_scale").detach())
        return (f"DynamicsFn(dt={c.dt}, speed_scale={scale:.3f}, "
                f"learn_channels={c.learn_channels}, weights={trained})")


def save(model: ResidualDynamics, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_model(path=None) -> ResidualDynamics:
    if path is None:
        return ResidualDynamics()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    for k in ("fit_params", "learn_channels", "residual_scale"):
        cfg[k] = tuple(cfg[k])
    model = ResidualDynamics(DynamicsConfig(**cfg))
    model.load_state_dict(ckpt["state_dict"])
    return model


def load_dynamics(path=None) -> DynamicsFn:
    """The drop-in. Pass a weights file from train_dynamics.py, or nothing for
    the untrained model (identical to the analytic base)."""
    return DynamicsFn(load_model(path))


# --------------------------------------------------------------------- check --

def check(path=None):
    """Contract test: does this drop into SpheroEnv and the EKF unchanged?"""
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}{('  - ' + detail) if detail else ''}")

    f = load_dynamics(path)
    print(f)

    # 1. Signature and types.
    out = f([0.0, 0.0, 0.3, 0.1], np.array([0.1, 0.5], dtype=np.float32))
    report("returns float64 array of shape (4,)",
           isinstance(out, np.ndarray) and out.shape == (4,) and out.dtype == np.float64,
           f"got {type(out).__name__} {getattr(out, 'shape', None)} {getattr(out, 'dtype', None)}")
    report("output finite", bool(np.all(np.isfinite(out))))

    # 2. An untrained model must be exactly the analytic base.
    if path is None:
        m = f.model
        s = torch.tensor([[0.1, -0.2, 0.4, 0.08]], dtype=DTYPE)
        a = torch.tensor([[0.12, -1.0]], dtype=DTYPE)
        dt = torch.tensor([m.config.dt], dtype=DTYPE)
        with torch.no_grad():
            diff = (m(s, a) - m.analytic(s, a, dt)).abs().max().item()
        report("untrained model == analytic base", diff == 0.0, f"max diff {diff:g}")

    # 2b. Reverse follows robot.py: -v at heading h is +v at h + pi. A ball
    #     already turned around (heading pi, facing -y) given [-v, 0] should
    #     keep going -y, not stop dead.
    rev = f([0.0, 0.0, math.pi, 0.1], [-0.1, 0.0])
    report("reverse command drives backwards (robot.py convention)",
           rev[1] < -1e-3 and rev[3] > 0, f"y {rev[1]:+.4f}, speed {rev[3]:.3f}")

    # 3. Jacobian quality at the EKF's eps. Compare against a coarser eps: if
    #    they disagree, finite differences are resolving noise, not slope.
    state0 = np.array([0.2, -0.1, 0.5, 0.08])
    action0 = np.array([0.1, 0.9])

    def jac(eps):
        J = np.zeros((4, 4))
        f0 = f(state0, action0)
        for k in range(4):
            p = state0.copy()
            p[k] += eps
            J[:, k] = (f(p, action0) - f0) / eps
        return J

    j_fine, j_coarse = jac(1e-6), jac(1e-4)
    jd = float(np.abs(j_fine - j_coarse).max())
    report("Jacobian stable at the EKF's eps=1e-6", jd < 1e-3, f"max |J(1e-6) - J(1e-4)| = {jd:.2e}")

    # 4. Drops into the simulator.
    try:
        from sphero_env.envs import SpheroEnv
        env = SpheroEnv(dt=f.dt, dynamics=f, render_mode=None, world_width=2.0, world_height=2.0)
        env.reset(seed=0)
        for _ in range(20):
            obs, *_ = env.step(np.array([0.1, 0.4], dtype=np.float32))
        env.close()
        report("SpheroEnv(dynamics=...) runs 20 steps", bool(np.all(np.isfinite(obs))))
    except Exception as e:
        report("SpheroEnv(dynamics=...) runs 20 steps", False, repr(e))

    # 5. Drops into the lab-3 EKF.
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lab3"))
        from EKF import EKF
        ekf = EKF(dt=f.dt, dynamics_fn=f)
        ekf.state_est = state0.copy()
        ekf.P = np.eye(4) * 1e-3
        est, P = ekf.predict(action0)
        report("EKF(dynamics_fn=...).predict()",
               bool(np.all(np.isfinite(est)) and np.all(np.isfinite(P))))
    except Exception as e:
        report("EKF(dynamics_fn=...).predict()", False, repr(e))

    # 6. Cost per call. The EKF calls dynamics 5 times per predict() for its
    #    Jacobian; at a 0.3 s control period there's a lot of headroom.
    n = 500
    t0 = time.perf_counter()
    for _ in range(n):
        f(state0, action0)
    per = (time.perf_counter() - t0) / n
    report("per-call cost", per < 0.01,
           f"{per * 1e6:.0f} us/call, ~{per * 5e3:.1f} ms per EKF predict")

    print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="run the integration contract test")
    ap.add_argument("--weights", default=None, help="weights file to check (default: untrained)")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(0 if check(args.weights) else 1)
    print(load_dynamics(args.weights))
