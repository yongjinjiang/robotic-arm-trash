from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest

from robotic_arm_trash.config import ExperimentConfig
from robotic_arm_trash.evaluation import evaluate

sb3 = pytest.importorskip("stable_baselines3")

from robotic_arm_trash.sb3 import load_policy, make_model, sb3_policy, train  # noqa: E402

# Pendulum keeps these fast and MuJoCo-free; tiny timesteps just exercise the seam.
ENV_ID = "Pendulum-v1"


def test_unsupported_algo_raises() -> None:
    from robotic_arm_trash.sb3 import _algo_class

    with pytest.raises(ValueError):
        _algo_class("dqn")


def test_sb3_policy_wraps_predict_into_valid_actions() -> None:
    env = gym.make(ENV_ID)
    try:
        # No training needed to test the adapter seam — predict() works on a fresh model.
        model = make_model(ExperimentConfig(env_id=ENV_ID, algo="ppo"), env)
        policy = sb3_policy(model)
        observation, _info = env.reset(seed=0)
        action = policy(observation)
        assert env.action_space.contains(action)
    finally:
        env.close()


def test_wrapped_model_is_scorable_by_the_harness() -> None:
    env = gym.make(ENV_ID)
    try:
        model = make_model(ExperimentConfig(env_id=ENV_ID, algo="ppo"), env)
        report = evaluate(env, sb3_policy(model), episodes=1, seed=0)
        assert report.num_episodes == 1
        assert report.episode_lengths == [200]
    finally:
        env.close()


@pytest.mark.slow
def test_train_logs_learning_curve_to_csv(tmp_path: Path) -> None:
    import csv

    config = ExperimentConfig(env_id=ENV_ID, algo="ppo", total_timesteps=256, seed=0)
    train(config, tmp_path, eval_freq=128, eval_episodes=1)

    metrics_path = tmp_path / "metrics.csv"
    assert metrics_path.exists()
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "expected at least one eval point on the learning curve"
    assert {"step", "eval_return", "eval_return_std"} <= set(rows[0])


@pytest.mark.slow
def test_train_save_load_roundtrip(tmp_path: Path) -> None:
    config = ExperimentConfig(env_id=ENV_ID, algo="ppo", total_timesteps=256, seed=0)
    model_path = train(config, tmp_path)
    assert model_path.exists()

    # Loaded policy scores through the harness exactly like any other policy.
    env = gym.make(ENV_ID)
    try:
        report = evaluate(env, load_policy(config.algo, model_path), episodes=1, seed=0)
    finally:
        env.close()
    assert report.num_episodes == 1
