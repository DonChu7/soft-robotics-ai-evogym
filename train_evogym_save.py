#!/usr/bin/env python3
# Train PPO in EvoGym and save BOTH the model and the exact robot body.
import argparse, numpy as np
import gymnasium as gym
import evogym.envs
from evogym import sample_robot
from stable_baselines3 import PPO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--out", type=str, default="ppo_evogym_walker")
    args = ap.parse_args()

    # (Optional) make it reproducible
    # np.random.seed(123); import random; random.seed(123)

    # Generate a soft body robot
    body, connections = sample_robot((5, 5))
    # Save the robot for later playback
    np.savez("robot.npz", body=body, connections=connections)

    # Train PPO
    env = gym.make("Walker-v0", body=body, render_mode=None)
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    env.close()
    print(f"Saved model: {args.out}.zip and robot.npz")

if __name__ == "__main__":
    main()