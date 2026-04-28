from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo


VIDEO_DIR = Path(__file__).resolve().parent.parent / "videos"


def record_random_reacher_video(steps: int = 200) -> Path:
    """Record a short random-policy Reacher-v5 rollout."""
    VIDEO_DIR.mkdir(exist_ok=True)
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=str(VIDEO_DIR), episode_trigger=lambda _: True)

    try:
        observation, info = env.reset()
        for _ in range(steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()
    finally:
        env.close()

    print(f"Video saved to {VIDEO_DIR}")
    return VIDEO_DIR


if __name__ == "__main__":
    record_random_reacher_video()
