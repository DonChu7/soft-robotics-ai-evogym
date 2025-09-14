#!/usr/bin/env python3
# Better PPO training for Evolution Gym Walker-v0
# - Longer rollouts, larger net, normalized obs/rewards
# - Saves robot.npz, model, and vecnormalize stats

import argparse, numpy as np
import gymnasium as gym
import evogym.envs
from evogym import sample_robot
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env(body):
    # Wrap single env so we can use VecNormalize easily
    return lambda: gym.make("Walker-v0", body=body, render_mode=None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", type=str, default="ppo_evogym_walker")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--size", type=int, default=5, help="robot grid size (e.g., 5 or 6)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    # Build + save the robot morphology (bigger body can help learning)
    body, connections = sample_robot((args.size, args.size))
    np.savez("robot.npz", body=body, connections=connections)

    # Vec env + normalization
    venv = DummyVecEnv([make_env(body)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # Larger network + longer rollouts help with credit assignment
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=policy_kwargs,
        n_steps=8192,        # ↑ from 2048
        batch_size=256,      # ↑ from 64
        learning_rate=3e-4,  # good default; you can try 1e-4 if unstable
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps)

    # Save policy and normalization stats
    model.save(args.out)
    venv.save("vecnormalize.pkl")
    venv.close()
    print(f"Saved: {args.out}.zip, vecnormalize.pkl, robot.npz")

if __name__ == "__main__":
    main()