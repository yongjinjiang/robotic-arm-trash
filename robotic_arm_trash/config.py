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
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
