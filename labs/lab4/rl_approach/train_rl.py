"""Train a local waypoint-tracking policy with PPO + domain randomization,
reusing Lab 3's A* Planner + EKF (see maze_rl_env.py for the full env/reward/
randomization design). The policy's input is NOT the raw environment obs -
it's an 8-dim feature vector (relative position to the next two waypoints,
sin/cos heading, speed, collision flag) built from the EKF's filtered state
estimate, matching exactly what labs/lab4/lab4.py builds at deployment time.

After training, extracts the PPO actor's mean-action network into
labs/lab4/Policy.py's existing Policy class and saves labs/lab4/weight.pth.
The Policy class is just an MLP shape here (Linear(d_in,64) -> ReLU ->
Linear(64,2)) - its docstring's literal obs=[x,y,heading,speed] convention is
for the plain skeleton task; we feed it our own waypoint-relative features
consistently between training and deployment instead, which is what actually
matters (grading is CSV-based, not weight-loading-based - see
labs/lab4/instructions.md). policy_kwargs below shapes PPO's policy net to
match Policy.py's architecture exactly, so mlp_extractor.policy_net +
action_net can be copied straight across after training.

Usage:
    python train_rl.py --timesteps 500000
    python train_rl.py --timesteps 5000 --n-envs 2   # quick smoke test
"""
import argparse
import os
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from maze_rl_env import MazeRLEnv, OBS_DIM, ACTION_DIM

LAB4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LAB4_DIR)
from Policy import Policy  # noqa: E402  (labs/lab4/Policy.py)


def make_env(seed, randomize=True):
    def _init():
        env = MazeRLEnv(render_mode=None, seed=seed, randomize=randomize)
        return Monitor(env, info_keywords=("reached_final",))
    return _init


def extract_policy(model: PPO) -> Policy:
    """Copy the trained mean-action network into labs/lab4/Policy.py's class."""
    sb3_policy = model.policy
    policy_net = sb3_policy.mlp_extractor.policy_net  # Sequential(Linear, ReLU)
    action_net = sb3_policy.action_net                # Linear, no activation

    deployed = Policy(d_in=OBS_DIM, hidden=64, d_out=ACTION_DIM)
    with torch.no_grad():
        deployed.model[0].weight.copy_(policy_net[0].weight)
        deployed.model[0].bias.copy_(policy_net[0].bias)
        deployed.model[2].weight.copy_(action_net.weight)
        deployed.model[2].bias.copy_(action_net.bias)
    return deployed


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    parser.add_argument("--out", default=os.path.join(LAB4_DIR, "weight.pth"))
    parser.add_argument("--tb-log", default=os.path.join(LAB4_DIR, "..", "..", "logs", "lab4_rl_tb"))
    parser.add_argument("--checkpoint-dir", default=os.path.join(LAB4_DIR, "rl_approach", "checkpoints"))
    parser.add_argument("--no-dr", action="store_true",
                        help="Disable domain randomization (fixed nominal dynamics, low noise) - "
                             "for isolating whether DR range is the cause of a training problem.")
    args = parser.parse_args(argv)

    vec_env = DummyVecEnv([make_env(args.seed + i, randomize=not args.no_dr) for i in range(args.n_envs)])

    # log_std_init above SB3's default (0.0 -> std=1) widens initial action
    # noise, and ent_coef is raised from SB3's usual ~0 default - both push
    # back against a specific failure mode found while developing this: with
    # speed_cmd's raw output near zero at initialization (small random
    # weights) and heading essentially random, an early rollout's "moved
    # somewhere random in a narrow maze corridor" is worse in expectation
    # than "stood still" (which reliably costs only the small per-step
    # penalty) - so a low-exploration policy converges to never moving
    # before it ever experiences what a WELL-aimed motion earns. More noise
    # keeps sampling real (if bad) motion for longer, and the action_net
    # bias nudge below directly biases initial rollouts toward attempting
    # movement instead of leaving it to chance.
    policy_kwargs = dict(net_arch=dict(pi=[64], vf=[64]), activation_fn=torch.nn.ReLU,
                         log_std_init=0.3)

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=min(2048, args.n_steps * args.n_envs),
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        clip_range=0.2,
        n_epochs=10,
        seed=args.seed,
        tensorboard_log=args.tb_log,
        verbose=1,
    )

    # Bias the speed output's raw pre-clip mean upward so early rollouts
    # actually attempt forward motion (at a random heading) rather than
    # mostly sampling the "stand still" half of speed_cmd's range - see the
    # policy_kwargs comment above for why this matters.
    with torch.no_grad():
        model.policy.action_net.bias[0] += 1.0

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.n_steps, 20_000 // args.n_envs),
        save_path=args.checkpoint_dir,
        name_prefix="ppo_maze",
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb, progress_bar=False)

    deployed = extract_policy(model)
    torch.save(deployed.state_dict(), args.out)
    print(f"\nSaved deployable weights to {args.out}")

    model.save(os.path.join(args.checkpoint_dir, "ppo_maze_final"))
    print(f"Saved full PPO checkpoint to {args.checkpoint_dir}/ppo_maze_final.zip")


if __name__ == "__main__":
    main()
