#!/usr/bin/env python3
import gymnasium as gym
import evogym.envs
from evogym import sample_robot

body, connections = sample_robot((5,5))
env = gym.make('Walker-v0', body=body, render_mode='human')  # a window should appear
obs, info = env.reset()
for _ in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()