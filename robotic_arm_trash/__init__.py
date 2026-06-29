"""robotic_arm_trash package.

Only torch-free modules are re-exported here so ``import robotic_arm_trash`` (and the CLI)
stays cheap. The from-scratch SAC and value-landscape viz pull in torch, so import them
explicitly when needed: ``from robotic_arm_trash.sac import SAC, train, load_agent`` /
``from robotic_arm_trash.viz import q_action_grid, value_vs_target``. The SB3 seam
(``robotic_arm_trash.sb3``) is likewise import-on-demand.
"""

from .cli import get_project_summary, main
from .config import ExperimentConfig
from .evaluation import EvalReport, evaluate
from .experiments import (
    SeedSummary,
    format_ci_table,
    format_results_table,
    summarize_seeds,
    t_ci,
)
from .metrics import MetricsLogger
from .plotting import (
    load_curve,
    plot_aggregated_curves,
    plot_final_returns,
    plot_learning_curves,
)
from .rollout import Policy, RolloutStats, random_policy, rollout
from .seeding import seed_everything

__all__ = [
    "get_project_summary",
    "main",
    "ExperimentConfig",
    "EvalReport",
    "evaluate",
    "SeedSummary",
    "summarize_seeds",
    "format_results_table",
    "format_ci_table",
    "t_ci",
    "MetricsLogger",
    "load_curve",
    "plot_learning_curves",
    "plot_aggregated_curves",
    "plot_final_returns",
    "Policy",
    "RolloutStats",
    "random_policy",
    "rollout",
    "seed_everything",
]
