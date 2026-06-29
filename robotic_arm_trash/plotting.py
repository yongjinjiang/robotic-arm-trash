"""Learning-curve plotting.

Turns the ``metrics.csv`` written during training (see
:func:`robotic_arm_trash.sb3.train`) into a committable PNG. Handles one curve or several
overlaid (e.g. multiple seeds / algorithms), shading ± the logged eval-return std.

matplotlib is imported lazily with the non-interactive ``Agg`` backend so plotting works
headlessly and never opens a window.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

# A curve to plot: a label and the metrics.csv it comes from.
CurveSpec = Tuple[str, Union[str, Path]]


def load_curve(csv_path: Union[str, Path]) -> Tuple[list[int], list[float], list[float]]:
    """Read ``(steps, eval_returns, eval_return_stds)`` from a metrics CSV.

    Rows without an ``eval_return`` value are skipped; a missing std column reads as 0.
    """
    steps: list[int] = []
    returns: list[float] = []
    stds: list[float] = []
    with Path(csv_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("eval_return")
            if value in (None, ""):
                continue
            steps.append(int(float(row["step"])))
            returns.append(float(value))
            stds.append(float(row.get("eval_return_std") or 0.0))
    return steps, returns, stds


def plot_aggregated_curves(
    groups: "dict[str, Sequence[Union[str, Path]]]",
    out_path: Union[str, Path],
    *,
    title: str | None = None,
    xlabel: str = "timesteps",
    ylabel: str = "eval return",
) -> Path:
    """Plot one mean±std curve per group, aggregating across seeds.

    ``groups`` maps a label (e.g. ``"SAC"``) to its per-seed ``metrics.csv`` paths. Within
    a group the curves are assumed to share the same step grid (same ``eval_freq`` /
    timesteps); the band is the across-seed sample std.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for label, csv_paths in groups.items():
        steps_ref = None
        matrix = []
        for csv_path in csv_paths:
            steps, returns, _stds = load_curve(csv_path)
            if steps_ref is None:
                steps_ref = steps
            matrix.append(returns)
        if not matrix:
            continue
        data = np.asarray(matrix)  # (n_seeds, n_points)
        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=1) if data.shape[0] > 1 else np.zeros_like(mean)
        line, = ax.plot(steps_ref, mean, marker="o", label=f"{label} (n={data.shape[0]})")
        ax.fill_between(steps_ref, mean - std, mean + std, alpha=0.15, color=line.get_color())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_final_returns(
    arms: "Sequence[Tuple[str, float, Tuple[float, float]]]",
    out_path: Union[str, Path],
    *,
    title: str | None = None,
    ylabel: str = "final eval return",
    baseline_label: str | None = None,
    points: "Sequence[Sequence[float]] | None" = None,
) -> Path:
    """Bar plot of final eval return per ablation arm, with asymmetric 95%-CI error bars.

    ``arms`` is a list of ``(label, mean, (ci_low, ci_high))``. If ``baseline_label`` matches
    one arm's label, it's highlighted and drawn as a horizontal reference line. ``points``,
    if given, is the per-seed values for each arm, scattered over the bars — with small n the
    individual seeds tell the story the wide t-CIs obscure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [a[0] for a in arms]
    means = np.array([a[1] for a in arms])
    lows = np.array([a[1] - a[2][0] for a in arms])
    highs = np.array([a[2][1] - a[1] for a in arms])
    x = np.arange(len(arms))
    colors = ["tab:orange" if lbl == baseline_label else "tab:blue" for lbl in labels]

    fig, ax = plt.subplots(figsize=(max(5.0, 1.2 * len(arms)), 4.5))
    ax.bar(x, means, yerr=[lows, highs], capsize=6, color=colors, alpha=0.85)
    if points is not None:
        for xi, pts in zip(x, points):
            ax.scatter([xi] * len(pts), pts, color="black", zorder=3, s=22, alpha=0.7)
    if baseline_label in labels:
        ax.axhline(means[labels.index(baseline_label)], color="tab:orange",
                   ls="--", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_learning_curves(
    curves: Iterable[CurveSpec],
    out_path: Union[str, Path],
    *,
    title: str | None = None,
    xlabel: str = "timesteps",
    ylabel: str = "eval return",
) -> Path:
    """Plot one or more learning curves to ``out_path`` (PNG). Returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    plotted = False
    for label, csv_path in curves:
        steps, returns, stds = load_curve(csv_path)
        if not steps:
            continue
        plotted = True
        line, = ax.plot(steps, returns, marker="o", label=label)
        r = np.asarray(returns)
        s = np.asarray(stds)
        ax.fill_between(steps, r - s, r + s, alpha=0.15, color=line.get_color())

    if not plotted:
        raise ValueError("No curve had any eval_return rows to plot.")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
