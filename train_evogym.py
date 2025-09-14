#!/usr/bin/env python3
"""
Train a quick PPO policy in Evolution Gym (Walker-v0) using Gymnasium + SB3 v2.
Usage:
  python train_evogym_v2.py --timesteps 50000
"""
import argparse
import gymnasium as gym
import evogym.envs  # registers evogym envs
from evogym import sample_robot
from stable_baselines3 import PPO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50_000)
    args = ap.parse_args()

    # Build a small 5x5 soft body
    body, connections = sample_robot((5, 5))
    env = gym.make("Walker-v0", body=body, render_mode=None)

    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64)
    model.learn(total_timesteps=args.timesteps)
    model.save("ppo_evogym_walker")
    env.close()
    print("Saved model: ppo_evogym_walker.zip")

if __name__ == "__main__":
    main()