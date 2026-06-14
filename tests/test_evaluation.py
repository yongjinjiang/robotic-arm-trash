from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from robotic_arm_trash.evaluation import EvalReport, evaluate
from robotic_arm_trash.rollout import random_policy

ENV_ID = "Pendulum-v1"  # continuous, truncates at 200 steps, no MuJoCo/render


def test_runs_requested_number_of_complete_episodes() -> None:
    env = gym.make(ENV_ID)
    try:
        report = evaluate(env, random_policy(env), episodes=3, seed=0)
    finally:
        env.close()

    assert isinstance(report, EvalReport)
    assert report.num_episodes == 3
    assert len(report.episode_returns) == 3
    assert report.episode_lengths == [200, 200, 200]


def test_is_deterministic_given_a_seed() -> None:
    def run() -> list[float]:
        env = gym.make(ENV_ID)
        try:
            return evaluate(env, random_policy(env), episodes=2, seed=7).episode_returns
        finally:
            env.close()

    assert run() == run()


def test_success_rate_is_none_without_a_criterion() -> None:
    env = gym.make(ENV_ID)
    try:
        report = evaluate(env, random_policy(env), episodes=2, seed=0)
    finally:
        env.close()
    assert report.success_rate is None


def test_success_threshold_drives_success_rate() -> None:
    env = gym.make(ENV_ID)
    try:
        policy = random_policy(env)
        # Pendulum returns are strongly negative; success = return >= threshold, so a very
        # negative threshold passes every episode and a 0.0 threshold passes none.
        all_pass = evaluate(env, policy, episodes=3, seed=0, success_threshold=-1e9)
        none_pass = evaluate(env, policy, episodes=3, seed=0, success_threshold=0.0)
    finally:
        env.close()

    assert all_pass.success_rate == 1.0
    assert none_pass.success_rate == 0.0


class _ToyGoalEnv(gym.Env):
    """Minimal env that reports info['is_success'] at episode end."""

    def __init__(self, succeed: bool) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._succeed = succeed
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= 3
        info = {"is_success": self._succeed} if terminated else {}
        return np.zeros(1, dtype=np.float32), 1.0, terminated, False, info


def test_is_success_info_takes_priority() -> None:
    succeeding = evaluate(_ToyGoalEnv(succeed=True), random_policy(_ToyGoalEnv(True)), episodes=2)
    failing = evaluate(_ToyGoalEnv(succeed=False), random_policy(_ToyGoalEnv(False)), episodes=2)
    assert succeeding.success_rate == 1.0
    assert failing.success_rate == 0.0
