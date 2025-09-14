#!/usr/bin/env python3
# Playback with VecNormalize stats + saved robot
import numpy as np
import gymnasium as gym
import evogym.envs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env(body):
    return lambda: gym.make("Walker-v0", body=body, render_mode="human")

def main():
    data = np.load("robot.npz", allow_pickle=True)
    body = data["body"]
    connections = data["connections"]

    # Restore normalized env
    venv = DummyVecEnv([make_env(body)])
    venv = VecNormalize.load("vecnormalize.pkl", venv)
    venv.training = False       # important: use eval mode
    venv.norm_reward = False    # optional: show raw rewards

    model = PPO.load("ppo_evogym_walker", env=venv)

    obs = venv.reset()
    for _ in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = venv.step(action)
        if (terminated | truncated).any():
            obs = venv.reset()

    venv.close()

if __name__ == "__main__":
    main()