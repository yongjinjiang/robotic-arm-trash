"""stable-baselines3 integration.

The harness is the spine; SB3 plugs in through one thin seam. ``sb3_policy`` wraps a
trained SB3 model into our :data:`~robotic_arm_trash.rollout.Policy` callable, so an SB3
agent is scored by the *same* ``evaluate``/``rollout`` as the random policy and (later)
the from-scratch SAC — exact apples-to-apples comparison.

SB3 (and torch) are imported lazily inside functions so importing this module — and the
CLI — stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import gymnasium as gym

from robotic_arm_trash.config import ExperimentConfig
from robotic_arm_trash.rollout import Policy
from robotic_arm_trash.seeding import seed_everything

if TYPE_CHECKING:  # pragma: no cover - typing only
    from stable_baselines3.common.base_class import BaseAlgorithm

# Algorithm id -> stable-baselines3 class name.
SUPPORTED_ALGOS = ("sac", "ppo")


def _algo_class(algo: str):
    import stable_baselines3 as sb3

    try:
        return {"sac": sb3.SAC, "ppo": sb3.PPO}[algo]
    except KeyError:
        raise ValueError(f"Unsupported algo {algo!r}; expected one of {SUPPORTED_ALGOS}.")


def sb3_policy(model: "BaseAlgorithm") -> Policy:
    """Wrap a trained SB3 model into a deterministic ``Policy`` callable."""

    def policy(observation):
        action, _state = model.predict(observation, deterministic=True)
        return action

    return policy


def make_model(config: ExperimentConfig, env: gym.Env) -> "BaseAlgorithm":
    """Construct an (untrained) SB3 model from an :class:`ExperimentConfig`."""
    algo_cls = _algo_class(config.algo)
    return algo_cls(
        "MlpPolicy",
        env,
        seed=config.seed,
        gamma=config.gamma,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        verbose=0,
    )


def train(config: ExperimentConfig, run_dir: Union[str, Path]) -> Path:
    """Train an SB3 agent per ``config``; save ``model.zip`` under ``run_dir``.

    Returns the path to the saved model. The caller owns ``run_dir`` (and typically also
    writes ``config.json`` there) so a run is fully reproducible from its directory.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    env = gym.make(config.env_id)
    try:
        model = make_model(config, env)
        model.learn(total_timesteps=config.total_timesteps)
        model_path = run_dir / "model.zip"
        model.save(model_path)
    finally:
        env.close()
    return model_path


def load_policy(algo: str, model_path: Union[str, Path]) -> Policy:
    """Load a saved SB3 model and return it wrapped as a ``Policy``."""
    model = _algo_class(algo).load(str(model_path))
    return sb3_policy(model)
