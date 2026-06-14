from __future__ import annotations

import gymnasium as gym
import pytest

from robotic_arm_trash.rollout import RolloutStats, random_policy, rollout


# Pendulum-v1 is continuous-action and ships with core gymnasium (no MuJoCo / no
# rendering), so these tests are fast and dependency-light. It truncates at 200 steps.
ENV_ID = "Pendulum-v1"


def test_rollout_runs_requested_number_of_steps() -> None:
    env = gym.make(ENV_ID)
    try:
        stats = rollout(env, random_policy(env), steps=50, seed=0)
    finally:
        env.close()

    assert isinstance(stats, RolloutStats)
    assert stats.total_steps == 50
    # Pendulum never terminates and truncates only at 200, so 50 steps = no full episode.
    assert stats.num_episodes == 0
    assert stats.episode_returns == []


def test_rollout_counts_completed_episodes() -> None:
    env = gym.make(ENV_ID)
    try:
        # 250 steps spans one full 200-step episode plus a partial second one.
        stats = rollout(env, random_policy(env), steps=250, seed=0)
    finally:
        env.close()

    assert stats.num_episodes == 1
    assert stats.episode_lengths == [200]


def test_rollout_is_deterministic_given_a_seed() -> None:
    def run() -> RolloutStats:
        env = gym.make(ENV_ID)
        try:
            return rollout(env, random_policy(env), steps=250, seed=123)
        finally:
            env.close()

    first, second = run(), run()
    assert first.episode_returns == second.episode_returns
    assert first.episode_lengths == second.episode_lengths


def test_random_policy_samples_valid_actions() -> None:
    env = gym.make(ENV_ID)
    try:
        policy = random_policy(env)
        observation, _info = env.reset(seed=0)
        action = policy(observation)
        assert env.action_space.contains(action)
    finally:
        env.close()


def test_reacher_env_constructs_and_rolls_out() -> None:
    """Smoke test on the real target env; skip if MuJoCo isn't available here."""
    try:
        env = gym.make("Reacher-v5")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Reacher-v5 unavailable: {exc}")

    try:
        stats = rollout(env, random_policy(env), steps=20, seed=0)
    finally:
        env.close()

    assert stats.total_steps == 20
