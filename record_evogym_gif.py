#!/usr/bin/env python3
"""
Record a short GIF/MP4 of a trained PPO policy in Evolution Gym (Gymnasium API).

Prereqs:
  pip install imageio imageio-ffmpeg gymnasium evogym stable-baselines3

Assumes the following exist in the current directory:
  - robot.npz              (saved by train_evogym_save.py)
  - ppo_evogym_walker.zip  (trained PPO model)

Usage examples:
  python record_evogym_gif.py
  python record_evogym_gif.py --steps 400 --fps 30 --gif evogym_walk.gif --mp4 evogym_walk.mp4
"""

import argparse
import numpy as np
import imageio
import gymnasium as gym
import evogym.envs  # registers envs
from stable_baselines3 import PPO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="ppo_evogym_walker", help="Model basename (.zip implied)")
    ap.add_argument("--robot", type=str, default="robot.npz", help="NPZ file with 'body' and 'connections'")
    ap.add_argument("--steps", type=int, default=400, help="Number of steps to record")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second for output")
    ap.add_argument("--gif", type=str, default="evogym_demo.gif", help="Output GIF filename (set empty to skip)")
    ap.add_argument("--mp4", type=str, default="", help="Optional MP4 filename (set empty to skip)")
    args = ap.parse_args()

    # Load robot morphology
    data = np.load(args.robot, allow_pickle=True)
    body = data["body"]
    connections = data["connections"]

    # RGB-array render mode for offscreen capture
    env = gym.make("Walker-v0", body=body, render_mode="rgb_array")
    model = PPO.load(args.model, env=env)

    frames = []
    obs, info = env.reset()
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()  # returns an RGB array
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

    if not frames:
        print("No frames captured. Ensure render_mode='rgb_array' is supported by the env.")
        return

    # Save GIF
    if args.gif:
        imageio.mimsave(args.gif, frames, fps=args.fps, loop=0)
        print(f"Saved GIF: {args.gif}  ({len(frames)} frames @ {args.fps} fps)")

    # Save MP4 (optional)
    if args.mp4:
        imageio.mimsave(args.mp4, frames, fps=args.fps, codec='libx264', quality=8)
        print(f"Saved MP4: {args.mp4}")

if __name__ == "__main__":
    main()
