"""Evaluation harness.

Where :func:`robotic_arm_trash.rollout.rollout` runs a *fixed number of steps*,
:func:`evaluate` runs a *fixed number of complete episodes* — the standard way to score a
policy. It reports mean ± std return and a task success rate.

Success metric, in priority order:
1. if the env reports ``info["is_success"]``, an episode counts as a success when any step
   set it true (the convention used by goal-conditioned envs, e.g. Fetch — relevant for
   Phase 4 / HER);
2. else if ``success_threshold`` is given, an episode succeeds when its return ≥ threshold;
3. else the success rate is ``None`` (undefined for this env).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np

from robotic_arm_trash.rollout import Policy


@dataclass
class EvalReport:
    num_episodes: int
    episode_returns: list[float]
    episode_lengths: list[int]
    success_rate: Optional[float]

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.episode_returns))

    @property
    def std_return(self) -> float:
        return float(np.std(self.episode_returns))

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.episode_lengths))

    def summary(self) -> str:
        text = (
            f"{self.num_episodes} episodes: return "
            f"{self.mean_return:.3f} ± {self.std_return:.3f}, "
            f"mean length {self.mean_length:.1f}"
        )
        if self.success_rate is not None:
            text += f", success {self.success_rate:.0%}"
        return text


def evaluate(
    env: gym.Env,
    policy: Policy,
    episodes: int = 10,
    seed: Optional[int] = None,
    success_threshold: Optional[float] = None,
) -> EvalReport:
    """Run ``policy`` for ``episodes`` complete episodes and summarize performance."""
    if seed is not None:
        env.action_space.seed(seed)

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []
    saw_success_key = False

    for episode in range(episodes):
        # Seed only the first reset; later resets continue the same RNG stream, so a given
        # seed reproduces the whole evaluation.
        observation, _info = env.reset(seed=seed if episode == 0 else None)
        episode_return = 0.0
        episode_length = 0
        episode_success = False

        while True:
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            if "is_success" in info:
                saw_success_key = True
                if info["is_success"]:
                    episode_success = True
            if terminated or truncated:
                break

        returns.append(episode_return)
        lengths.append(episode_length)
        successes.append(episode_success)

    if saw_success_key:
        success_rate: Optional[float] = float(np.mean(successes))
    elif success_threshold is not None:
        success_rate = float(np.mean([r >= success_threshold for r in returns]))
    else:
        success_rate = None

    return EvalReport(
        num_episodes=episodes,
        episode_returns=returns,
        episode_lengths=lengths,
        success_rate=success_rate,
    )
