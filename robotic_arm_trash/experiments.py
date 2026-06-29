"""Multi-seed aggregation and reporting.

Turns a set of per-seed runs (each a ``metrics.csv`` learning curve) into cross-seed
statistics and a Markdown results table — the quantified status the README wants. Reused
by Phase 3's scientific study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from robotic_arm_trash.plotting import load_curve

# Two-sided 95% Student-t critical values by degrees of freedom (n-1). Hardcoded so we get
# proper small-sample confidence intervals without a scipy dependency; dof>30 ≈ normal.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042,
}
_Z95 = 1.960  # normal approximation for dof > 30.


def t_ci(values: Sequence[float], confidence: float = 0.95) -> "Tuple[float, float]":
    """Two-sided Student-t confidence interval for the mean of ``values``.

    Returns ``(low, high)`` = mean ± t · (sample_std / sqrt(n)). For a single sample the CI
    is undefined, so we return ``(mean, mean)``. Only 95% is tabulated (the value we report).
    """
    if confidence != 0.95:
        raise ValueError("Only 95% confidence is tabulated.")
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    if n < 2:
        return (mean, mean)
    stderr = float(arr.std(ddof=1) / np.sqrt(n))
    crit = _T95.get(n - 1, _Z95)
    half = crit * stderr
    return (mean - half, mean + half)


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

    @property
    def stderr(self) -> float:
        if self.n_seeds < 2:
            return 0.0
        return self.std_return / float(np.sqrt(self.n_seeds))

    @property
    def ci95(self) -> "Tuple[float, float]":
        """Two-sided 95% Student-t confidence interval for the mean across seeds."""
        return t_ci(self.per_seed_returns, confidence=0.95)


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


def format_ci_table(
    summaries: Iterable[SeedSummary],
    *,
    arm_header: str = "Configuration",
) -> str:
    """Render summaries with a **95% confidence interval** column (one row per config).

    Used by the Phase 3 report, where the headline is the CI on the final eval return rather
    than a vs-random delta.
    """
    header = [arm_header, "Seeds", "Timesteps", "Final eval return (mean)", "95% CI"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for s in summaries:
        lo, hi = s.ci95
        lines.append("| " + " | ".join([
            s.label,
            str(s.n_seeds),
            f"{s.final_step:,}",
            f"{s.mean_return:.2f}",
            f"[{lo:.2f}, {hi:.2f}]",
        ]) + " |")
    return "\n".join(lines)
