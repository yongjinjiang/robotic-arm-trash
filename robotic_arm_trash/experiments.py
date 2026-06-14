"""Multi-seed aggregation and reporting.

Turns a set of per-seed runs (each a ``metrics.csv`` learning curve) into cross-seed
statistics and a Markdown results table — the quantified status the README wants. Reused
by Phase 3's scientific study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from robotic_arm_trash.plotting import load_curve


def final_eval_return(metrics_csv: Union[str, Path]) -> float:
    """Return the last logged ``eval_return`` from a run's metrics CSV."""
    _steps, returns, _stds = load_curve(metrics_csv)
    if not returns:
        raise ValueError(f"No eval_return rows in {metrics_csv}.")
    return returns[-1]


@dataclass
class SeedSummary:
    """Cross-seed summary of one configuration's final eval return."""

    label: str
    per_seed_returns: list[float]
    final_step: int

    @property
    def n_seeds(self) -> int:
        return len(self.per_seed_returns)

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.per_seed_returns))

    @property
    def std_return(self) -> float:
        # Sample std (ddof=1) across seeds; 0.0 for a single seed.
        if self.n_seeds < 2:
            return 0.0
        return float(np.std(self.per_seed_returns, ddof=1))


def summarize_seeds(label: str, run_dirs: Sequence[Union[str, Path]]) -> SeedSummary:
    """Summarize a configuration's final eval return across its per-seed run dirs."""
    per_seed: list[float] = []
    final_step = 0
    for run_dir in run_dirs:
        metrics_csv = Path(run_dir) / "metrics.csv"
        steps, returns, _stds = load_curve(metrics_csv)
        if not returns:
            raise ValueError(f"No eval_return rows in {metrics_csv}.")
        per_seed.append(returns[-1])
        final_step = max(final_step, steps[-1])
    return SeedSummary(label=label, per_seed_returns=per_seed, final_step=final_step)


def format_results_table(
    summaries: Iterable[SeedSummary],
    *,
    baseline: Optional[SeedSummary] = None,
) -> str:
    """Render summaries as a Markdown table (one row per configuration).

    If ``baseline`` is given, add an "vs random" improvement column.
    """
    header = ["Policy", "Seeds", "Timesteps", "Eval return (mean ± std)"]
    if baseline is not None:
        header.append("vs random")

    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]

    rows = list(summaries)
    if baseline is not None:
        rows = [baseline, *rows]

    for summary in rows:
        cells = [
            summary.label,
            str(summary.n_seeds),
            f"{summary.final_step:,}",
            f"{summary.mean_return:.2f} ± {summary.std_return:.2f}",
        ]
        if baseline is not None:
            if summary is baseline:
                cells.append("—")
            else:
                cells.append(f"{summary.mean_return - baseline.mean_return:+.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)
