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
from typing import Iterable, Tuple, Union

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
