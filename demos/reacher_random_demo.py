from __future__ import annotations

import gymnasium as gym


def run_random_reacher(steps: int = 1000) -> None:
    """Run a simple random-policy rollout in Reacher-v5."""
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    try:
        observation, info = env.reset()
        for _ in range(steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    run_random_reacher()
