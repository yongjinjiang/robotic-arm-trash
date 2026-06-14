"""Shared rollout utility.

The core abstraction for the whole project: *a policy is a callable, a rollout is a
function*. A policy maps an observation to an action; ``rollout`` drives an environment
with that policy for a fixed number of steps, auto-resetting at episode boundaries, and
returns summary statistics. Random, stable-baselines3, and from-scratch agents all plug
into the same harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

# A policy maps a single observation to a single action.
Policy = Callable[[np.ndarray], np.ndarray]


def random_policy(env: gym.Env) -> Policy:
    """Return a policy that samples uniformly from ``env``'s action space.

    Sampling uses the action space's own RNG, so seeding it (which :func:`rollout` does
    when given a ``seed``) makes the resulting trajectory reproducible.
    """

    def policy(observation: np.ndarray) -> np.ndarray:
        return env.action_space.sample()

    return policy


@dataclass
class RolloutStats:
    """Summary of a rollout. Only *completed* episodes are counted in the returns."""

    total_steps: int
    episode_returns: list[float]
    episode_lengths: list[int]

    @property
    def num_episodes(self) -> int:
        return len(self.episode_returns)

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.episode_returns)) if self.episode_returns else float("nan")


def rollout(
    env: gym.Env,
    policy: Policy,
    steps: int,
    seed: Optional[int] = None,
) -> RolloutStats:
    """Run ``policy`` in ``env`` for ``steps`` steps, auto-resetting on episode end.

    When ``seed`` is given, both the environment and its action space are seeded so the
    run is reproducible (a random policy included).
    """
    if seed is not None:
        env.action_space.seed(seed)

    observation, _info = env.reset(seed=seed)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    current_return = 0.0
    current_length = 0

    for _ in range(steps):
        action = policy(observation)
        observation, reward, terminated, truncated, _info = env.step(action)
        current_return += float(reward)
        current_length += 1

        if terminated or truncated:
            episode_returns.append(current_return)
            episode_lengths.append(current_length)
            current_return = 0.0
            current_length = 0
            observation, _info = env.reset()

    return RolloutStats(
        total_steps=steps,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
    )
