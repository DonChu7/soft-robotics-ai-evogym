#!/usr/bin/env python3
# Replay with the EXACT robot used during training.
import numpy as np
import gymnasium as gym
import evogym.envs
from stable_baselines3 import PPO

def main():
    # Load the saved robot
    data = np.load("robot.npz", allow_pickle=True)
    body = data["body"]
    connections = data["connections"]

    env = gym.make("Walker-v0", body=body, render_mode="human")
    model = PPO.load("ppo_evogym_walker", env=env)

    obs, info = env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

if __name__ == "__main__":
    main()
    