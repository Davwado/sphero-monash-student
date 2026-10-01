"""Quick headless rollout of a trained policy against MazeRLEnv, for diagnosing
training problems without going through the full lab4.py deployment loop.

Usage:
    python eval_policy.py weight_nodr_test.pth --no-dr --episodes 5
"""
import argparse
import os
import sys

import numpy as np
import torch

from maze_rl_env import MazeRLEnv, OBS_DIM, ACTION_DIM

LAB4_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(LAB4_DIR))
from Policy import Policy  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args(argv)

    policy = Policy(d_in=OBS_DIM, hidden=64, d_out=ACTION_DIM)
    policy.load_state_dict(torch.load(args.weights, map_location="cpu"))
    policy.eval()

    env = MazeRLEnv(render_mode=None, seed=args.seed, randomize=not args.no_dr)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        n_waypoints = len(env.local_waypoints)
        total_reward = 0.0
        speeds, wp_reached = [], 0
        for t in range(250):
            with torch.no_grad():
                action = policy(torch.tensor(obs, dtype=torch.float32)).numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            speeds.append(env.ekf.state_est[3])
            wp_reached = info["wp_index"]
            if terminated or truncated:
                break
        print(f"ep{ep}: steps={t+1} waypoints {wp_reached}/{n_waypoints} "
              f"reached_final={info['reached_final']} reward={total_reward:.1f} "
              f"mean_speed={np.mean(speeds):.4f} max_speed={np.max(speeds):.4f}")


if __name__ == "__main__":
    main()
