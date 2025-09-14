#!/usr/bin/env python3
# Replay a trained PPO policy in Evolution Gym (Gymnasium API).
import gymnasium as gym
import evogym.envs
from evogym import sample_robot
from stable_baselines3 import PPO

def main():
    # same body used during training (a small 5x5)
    body, connections = sample_robot((5, 5))

    # show a live window
    env = gym.make("Walker-v0", body=body, render_mode="human")

    # load the model saved by your training run
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