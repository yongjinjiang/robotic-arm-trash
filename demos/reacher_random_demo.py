from __future__ import annotations

import gymnasium as gym

from robotic_arm_trash.rollout import random_policy, rollout


def run_random_reacher(steps: int = 1000) -> None:
    """Run a simple random-policy rollout in Reacher-v5."""
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    try:
        rollout(env, random_policy(env), steps=steps)
    finally:
        env.close()


if __name__ == "__main__":
    run_random_reacher()
