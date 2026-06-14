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
from robotic_arm_trash.evaluation import evaluate
from robotic_arm_trash.metrics import MetricsLogger
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


def _build_eval_callback(eval_env, logger, eval_freq, episodes, seed):
    """A periodic-evaluation callback that logs the learning curve via our harness.

    Defined inside a function so subclassing SB3's ``BaseCallback`` (and thus importing
    SB3) stays lazy.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _EvalMetricsCallback(BaseCallback):
        def _on_step(self) -> bool:
            if self.n_calls % eval_freq == 0:
                report = evaluate(eval_env, sb3_policy(self.model),
                                  episodes=episodes, seed=seed)
                logger.log(self.num_timesteps,
                           eval_return=report.mean_return,
                           eval_return_std=report.std_return)
            return True

    return _EvalMetricsCallback()


def train(
    config: ExperimentConfig,
    run_dir: Union[str, Path],
    *,
    eval_freq: int = 2000,
    eval_episodes: int = 10,
    use_tensorboard: bool = False,
) -> Path:
    """Train an SB3 agent per ``config``; save ``model.zip`` under ``run_dir``.

    During training, the agent is evaluated through our own ``evaluate`` every
    ``eval_freq`` steps and the eval return is logged (CSV + optional TensorBoard) to
    ``run_dir/metrics.csv`` — the learning curve. Returns the saved model path; ``run_dir``
    holds everything needed to reproduce the run.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    env = gym.make(config.env_id)
    eval_env = gym.make(config.env_id)
    logger = MetricsLogger(run_dir, use_tensorboard=use_tensorboard)
    try:
        model = make_model(config, env)
        callback = _build_eval_callback(eval_env, logger, eval_freq, eval_episodes, config.seed)
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
        model_path = run_dir / "model.zip"
        model.save(model_path)
    finally:
        logger.close()
        eval_env.close()
        env.close()
    return model_path


def load_policy(algo: str, model_path: Union[str, Path]) -> Policy:
    """Load a saved SB3 model and return it wrapped as a ``Policy``."""
    model = _algo_class(algo).load(str(model_path))
    return sb3_policy(model)
