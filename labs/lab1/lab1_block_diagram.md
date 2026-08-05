# Lab 1 — code flow block diagram

How `lab1.py` runs, end to end, and where every piece of data lives.
Run with `python lab1.py --sim` (simulation) or `python lab1.py` (real Sphero).

## Key data formats

| Data | Shape / meaning |
|---|---|
| `state` | `[x, y, heading, speed]` — metres, radians, m/s. Heading `0` points **+y**, positive rotates toward **+x** (matches Sphero heading degrees) |
| `action` | `[speed_cmd, heading_cmd]` — clipped by the env to speed ±0.15 m/s, heading ±π rad |
| `obs` | `[x, y, heading, speed, collision_flag]` — odometry state + measurement noise (noise std = 0 in lab 1 sim) |
| `info` dict | `state_true`, `state_odom`, `collision`, commands; the real robot also adds raw accel/gyro/velocity sensor dicts |
| CSV logs | `logs/lab1_sim.csv` or `logs/lab1_real.csv` — one row per `step()` (ground truth, odometry, action, estimate, setpoint) |

## Overall flow

```mermaid
flowchart TD
    subgraph LAB1["lab1.py"]
        MAIN["main(argv)<br/>parses --sim flag"] --> ME["managed_env(sim)<br/>context manager: builds env,<br/>starts CSV logging, guarantees cleanup"]
    end

    ME -- "sim = True" --> MSE["make_sim_env() → SpheroEnv<br/>5×5 m world, dt=0.1 s, max 5000 steps<br/>goal (0.5, 0.5) ± 0.1 m, all noise = 0,<br/>no obstacles, pygame 800×800"]
    ME -- "sim = False" --> SAC["scan_and_connect()<br/>(sphero_env/robot/connect.py)<br/>BLE scan, user picks a toy,<br/>returns (toy, SpheroEduAPI)"]

    MSE --> DYN["dynamics.py — dynamics(state, action)<br/>unicycle model: heading set instantly,<br/>speed = clip(speed + speed_cmd·0.1, 0, 1),<br/>x += v·sin(θ)·0.1, y += v·cos(θ)·0.1"]
    SAC --> MRE["make_real_env(api) → Robot<br/>(sphero_env/robot/robot.py)<br/>same Gym interface as SpheroEnv,<br/>state comes from robot odometry"]

    DYN --> CL
    MRE --> CL

    subgraph CL["control_loop(env) — lab1.py"]
        RESET["env.reset(seed=0)<br/>→ first obs; RNG seeded with 0"] --> ACT["action = rng.uniform(−1, 1, size=2)<br/>⚠ placeholder — replace with your controller"]
        ACT --> STEP["env.step(action)<br/>→ obs, reward, terminated, truncated, info"]
        STEP --> REN["env.render()<br/>pygame window"]
        REN -- "repeat ×50" --> ACT
        REN --> ESTOP["env.emergency_stop()<br/>speed → 0"]
    end

    ESTOP --> FIN["managed_env finally block<br/>stop_logging(), close()<br/>(real robot also gets set_speed(0))"]
```

## Inside `env.step(action)` — sim vs real

```mermaid
flowchart TD
    subgraph SIM["SpheroEnv.step()  (simulation)"]
        S1["clip action to action_space<br/>speed ±0.15, heading ±π"] --> S2["run dynamics() twice<br/>state_true ← clean action<br/>state_odom ← action + process noise<br/>(noise std = 0 in lab 1, so they match)"]
        S2 --> S3["collision check vs occupancy grid (None here),<br/>out-of-bounds ⇒ terminated,<br/>step ≥ max_steps ⇒ truncated,<br/>reward = −distance to goal"]
        S3 --> S4["obs = odom + measurement noise<br/>[x, y, θ, v, collision_flag]"]
        S4 --> S5["vis.record() → CSV row<br/>logs/lab1_sim.csv"]
    end

    subgraph REAL["Robot.step()  (real Sphero)"]
        R1["clip action, convert:<br/>heading rad → deg (negative speed flips 180°),<br/>speed → raw 0–255 (|v|/0.15 · 255)"] --> R2["send over BLE:<br/>api.set_heading(), api.set_speed()"]
        R2 --> R3["read sensors back:<br/>location (cm → m ÷100), heading (deg → rad),<br/>speed (÷255 · 0.15) ⇒ state_odom"]
        R3 --> R4["_sense_collision()<br/>accel jerk + speed drop + gyro spikes<br/>vs tunable thresholds"]
        R4 --> R5["obs = odom + noise, reward = None,<br/>info carries raw sensor dicts,<br/>vis.record() → logs/lab1_real.csv"]
    end

    S5 --> VIS["Visualiser (envs/visualiser.py) — shared by both<br/>pygame drawing: true pose green, odom blue,<br/>goal marker, belief ellipse if set;<br/>CSV logger: one row per control step"]
    R5 --> VIS
```

## Points worth knowing before you edit

- **Where to put your controller:** the `rng.uniform(-1, 1, 2)` line in `control_loop()` is the only thing meant to be replaced in lab 1. Compute `[speed_cmd, heading_cmd]` from `obs` instead.
- **Actions get clipped:** the env clips speed to ±0.15 m/s, so the random ±1.0 samples mostly saturate the speed command.
- **Heading convention:** 0 rad = +y ("north"), positive angles rotate toward +x. `wrap_angle()` keeps everything in [−π, π).
- **`dynamics.py` is yours to improve:** the sim uses it for *both* ground truth and odometry. Note it snaps heading instantly, unlike `SpheroEnv._base_dynamics` which rate-limits turns and only translates when aligned.
- **Sim and real are interchangeable:** `SpheroEnv` and `Robot` expose the same `reset / step / render / emergency_stop` interface, so `control_loop` doesn't care which one it gets — that's the whole point of `managed_env`.
- **Reward differs:** sim returns `−distance_to_goal` (minus collision penalty); the real robot returns `None` (no ground truth available).
- **Logs:** every step appends a CSV row via the shared `Visualiser`; cleanup in `managed_env` closes the file, so let the program exit normally (Ctrl+C mid-run can lose the log tail).
