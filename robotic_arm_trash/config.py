"""Experiment configuration.

A single frozen dataclass capturing everything needed to reproduce a run. Serialize it
alongside every run's outputs so any result can be regenerated from its config + seed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters and run settings. Defaults target Reacher-v5 with SAC."""

    env_id: str = "Reacher-v5"
    algo: str = "sac"
    total_timesteps: int = 100_000
    seed: int = 0
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 256

    # SAC-specific (used by the from-scratch agent in ``sac.py``; ignored by SB3, which
    # carries its own defaults). Off-policy knobs that don't apply to PPO.
    tau: float = 0.005  # Polyak coefficient for soft target updates.
    buffer_size: int = 1_000_000  # Replay buffer capacity.
    learning_starts: int = 1000  # Random-action warmup before gradient updates begin.
    train_freq: int = 1  # Gradient updates per environment step (after warmup).
    hidden_sizes: tuple[int, ...] = (256, 256)  # Actor/critic MLP widths.

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: Union[str, Path]) -> Path:
        """Write the config as JSON, creating parent directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Build from a dict, ignoring unknown keys for forward compatibility."""
        fields = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in fields}
        # JSON has no tuples: a round-tripped ``hidden_sizes`` comes back as a list. Coerce
        # it so save/load is a true round-trip (and the field stays hashable/frozen-friendly).
        if "hidden_sizes" in kwargs:
            kwargs["hidden_sizes"] = tuple(kwargs["hidden_sizes"])
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
