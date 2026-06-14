from __future__ import annotations

from pathlib import Path

from robotic_arm_trash.config import ExperimentConfig


def test_defaults_target_reacher_sac() -> None:
    config = ExperimentConfig()
    assert config.env_id == "Reacher-v5"
    assert config.algo == "sac"
    assert config.seed == 0


def test_save_load_round_trip(tmp_path: Path) -> None:
    config = ExperimentConfig(env_id="Pendulum-v1", algo="ppo", seed=7, total_timesteps=500)
    path = config.save(tmp_path / "run" / "config.json")

    assert path.exists()
    assert ExperimentConfig.load(path) == config


def test_from_dict_ignores_unknown_keys() -> None:
    data = ExperimentConfig().to_dict()
    data["unexpected_future_field"] = 123
    assert ExperimentConfig.from_dict(data) == ExperimentConfig()


def test_is_frozen() -> None:
    config = ExperimentConfig()
    try:
        config.seed = 99  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ExperimentConfig should be immutable")
