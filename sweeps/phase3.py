"""Phase 3 scientific-study sweep driver.

Declares the whole study as a grid of :class:`ExperimentConfig`s and runs each cell through
the *same* trainers as the CLI (``sac.train`` / ``sb3.train``), writing each run to its own
``runs/phase3/<arm>/<name>`` dir. Idempotent: a cell whose ``metrics.csv`` already exists is
skipped, so the sweep is resumable and re-running it is a no-op — the reproducibility
guarantee Phase 3 wants.

Usage::

    python sweeps/phase3.py --list      # show the grid (no training)
    python sweeps/phase3.py --smoke     # tiny timesteps, exercises the whole path
    python sweeps/phase3.py             # run the full study (hours; resumable)
    python sweeps/phase3.py --only ablation_lr variance   # run selected arms

Aggregation into tables/figures lives in ``sweeps/phase3_report.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from robotic_arm_trash.config import ExperimentConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE3_DIR = PROJECT_ROOT / "runs" / "phase3"

SEEDS_VARIANCE = (0, 1, 2, 3, 4)  # s0–s2 are reused from runs/val (see report aggregator).
SEEDS_ABLATION = (0, 1, 2)
ABLATION_STEPS = 30_000
FULL_STEPS = 50_000

# Baseline config the from-scratch SAC was validated with (Phase 2).
BASE = ExperimentConfig(env_id="Reacher-v5", algo="sac_scratch", total_timesteps=FULL_STEPS)


@dataclass(frozen=True)
class Run:
    arm: str          # logical group, e.g. "ablation_lr"
    name: str         # cell name within the arm, e.g. "lr1e-04-s0"
    config: ExperimentConfig

    @property
    def run_dir(self) -> Path:
        return PHASE3_DIR / self.arm / self.name

    @property
    def done(self) -> bool:
        return (self.run_dir / "metrics.csv").exists()


def _grid() -> "list[Run]":
    runs: list[Run] = []

    # --- Variance study: 5 seeds @ 50k. s0–s2 already live in runs/val; only s3,s4 here. ---
    for seed in SEEDS_VARIANCE:
        if seed <= 2:
            continue  # reused from runs/val/sac_scratch-s{seed}
        cfg = replace(BASE, total_timesteps=FULL_STEPS, seed=seed)
        runs.append(Run("variance", f"sac_scratch-s{seed}", cfg))

    # --- Shared 30k baseline triple (lr 3e-4 / width 256 / replay 1e6). ---
    for seed in SEEDS_ABLATION:
        cfg = replace(BASE, total_timesteps=ABLATION_STEPS, seed=seed)
        runs.append(Run("baseline30k", f"base-s{seed}", cfg))

    # --- Ablation: learning rate (baseline 3e-4 reused from baseline30k). ---
    for lr in (1e-4, 1e-3):
        for seed in SEEDS_ABLATION:
            cfg = replace(BASE, total_timesteps=ABLATION_STEPS, seed=seed, learning_rate=lr)
            runs.append(Run("ablation_lr", f"lr{lr:.0e}-s{seed}", cfg))

    # --- Ablation: network width (baseline (256,256) reused from baseline30k). ---
    for width in ((64, 64),):
        for seed in SEEDS_ABLATION:
            cfg = replace(BASE, total_timesteps=ABLATION_STEPS, seed=seed, hidden_sizes=width)
            runs.append(Run("ablation_width", f"w{width[0]}-s{seed}", cfg))

    # --- Ablation: replay buffer size (baseline 1e6 reused from baseline30k). ---
    for buf in (10_000,):
        for seed in SEEDS_ABLATION:
            cfg = replace(BASE, total_timesteps=ABLATION_STEPS, seed=seed, buffer_size=buf)
            runs.append(Run("ablation_replay", f"buf{buf}-s{seed}", cfg))

    # --- Sample efficiency: SB3 PPO, 50k, 3 seeds. ---
    for seed in SEEDS_ABLATION:
        cfg = replace(BASE, algo="ppo", total_timesteps=FULL_STEPS, seed=seed)
        runs.append(Run("ppo", f"ppo-s{seed}", cfg))

    return runs


def _trainer(algo: str) -> Callable:
    if algo == "sac_scratch":
        from robotic_arm_trash.sac import train

        return lambda cfg, rd: train(cfg, rd, eval_freq=2000, eval_episodes=20)
    from robotic_arm_trash.sb3 import train

    return lambda cfg, rd: train(cfg, rd, eval_freq=2000, eval_episodes=20)


def run_sweep(runs: "list[Run]", *, smoke: bool = False) -> None:
    total = len(runs)
    for i, run in enumerate(runs, 1):
        if run.done and not smoke:
            print(f"[{i}/{total}] skip (done): {run.arm}/{run.name}")
            continue
        cfg = run.config
        out_dir = run.run_dir
        if smoke:  # tiny path just to exercise train/eval/log end-to-end, isolated from real runs
            cfg = replace(cfg, total_timesteps=400, learning_starts=100, batch_size=32)
            out_dir = PHASE3_DIR / "_smoke" / run.name
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(out_dir / "config.json")
        print(f"[{i}/{total}] train {run.arm}/{run.name}: {cfg.algo} "
              f"{cfg.total_timesteps} steps, seed {cfg.seed}", flush=True)
        _trainer(cfg.algo)(cfg, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 study sweep driver.")
    parser.add_argument("--list", action="store_true", help="Print the grid and exit.")
    parser.add_argument("--smoke", action="store_true", help="Tiny timesteps; exercise the path.")
    parser.add_argument("--only", nargs="+", default=None, help="Only run these arms.")
    args = parser.parse_args()

    runs = _grid()
    if args.only:
        runs = [r for r in runs if r.arm in set(args.only)]

    if args.list:
        for r in runs:
            flag = "done" if r.done else "todo"
            print(f"[{flag}] {r.arm:16s} {r.name:16s} "
                  f"{r.config.algo} steps={r.config.total_timesteps} seed={r.config.seed} "
                  f"lr={r.config.learning_rate:g} width={r.config.hidden_sizes} "
                  f"buf={r.config.buffer_size}")
        print(f"\n{len(runs)} cells "
              f"({sum(r.done for r in runs)} done, {sum(not r.done for r in runs)} todo)")
        return

    run_sweep(runs, smoke=args.smoke)
    print("=== sweep complete ===")


if __name__ == "__main__":
    main()
