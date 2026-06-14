"""Metrics logging.

A small scalar-metrics logger that always writes a tidy CSV and, when available and
requested, mirrors scalars to TensorBoard. Training (Phase 1) and evaluation both log
through this one interface so every run leaves a machine-readable trail.

TensorBoard is optional: if the ``tensorboard`` package isn't installed, logging silently
falls back to CSV only.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Union


class MetricsLogger:
    """Buffer ``log(step, **scalars)`` calls; write CSV (and optional TensorBoard) on close."""

    def __init__(
        self,
        log_dir: Union[str, Path],
        *,
        csv_name: str = "metrics.csv",
        use_tensorboard: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / csv_name
        self._records: list[dict[str, Any]] = []

        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:  # pragma: no cover - tensorboard is an optional extra
                self.tensorboard_available = False
            else:
                self._tb = SummaryWriter(log_dir=str(self.log_dir))
                self.tensorboard_available = True
        else:
            self.tensorboard_available = False

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def log(self, step: int, **metrics: float) -> None:
        record: dict[str, Any] = {"step": int(step)}
        for key, value in metrics.items():
            record[key] = float(value)
            if self._tb is not None:
                self._tb.add_scalar(key, float(value), step)
        self._records.append(record)

    def _fieldnames(self) -> list[str]:
        # "step" first, then every other key in first-seen order.
        ordered = ["step"]
        seen = {"step"}
        for record in self._records:
            for key in record:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        return ordered

    def flush(self) -> None:
        with self.csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames())
            writer.writeheader()
            writer.writerows(self._records)
        if self._tb is not None:
            self._tb.flush()

    def close(self) -> None:
        self.flush()
        if self._tb is not None:
            self._tb.close()
            self._tb = None

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
