from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import time

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from robotic_arm_trash.config import ExperimentConfig
from robotic_arm_trash.evaluation import evaluate
from robotic_arm_trash.metrics import MetricsLogger
from robotic_arm_trash.rollout import RolloutStats, random_policy, rollout
from robotic_arm_trash.seeding import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demos"
VIDEO_DIR = PROJECT_ROOT / "videos"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_ENV = "Reacher-v5"


def get_project_summary() -> str:
    return (
        "robotic-arm-trash: small Gymnasium/MuJoCo experiments for a "
        "Reacher-style robotic arm task.\n"
        f"Demo scripts: {DEMO_DIR}\n"
        f"Generated videos: {VIDEO_DIR}\n"
        "Run `robotic-arm-trash demo` for a random rollout, "
        "`robotic-arm-trash record` to save a video, or `--help` for all commands."
    )


def _format_stats(env_id: str, stats: RolloutStats) -> str:
    if stats.num_episodes:
        return (
            f"{env_id}: {stats.total_steps} steps, {stats.num_episodes} episode(s), "
            f"mean return {stats.mean_return:.3f}"
        )
    return f"{env_id}: {stats.total_steps} steps, no episode completed"


def _cmd_demo(args: argparse.Namespace) -> int:
    if args.seed is not None:
        seed_everything(args.seed)
    env = gym.make(args.env)
    try:
        stats = rollout(env, random_policy(env), steps=args.steps, seed=args.seed)
    finally:
        env.close()
    print(_format_stats(args.env, stats))
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    if args.seed is not None:
        seed_everything(args.seed)
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=str(video_dir), episode_trigger=lambda _: True)
    try:
        stats = rollout(env, random_policy(env), steps=args.steps, seed=args.seed)
    finally:
        env.close()

    print(f"Saved video(s) to {video_dir}")
    print(_format_stats(args.env, stats))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    # Imported here so the heavy stable-baselines3/torch import only happens on `train`.
    from robotic_arm_trash.sb3 import load_policy, train

    config = ExperimentConfig(
        env_id=args.env,
        algo=args.algo,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )
    run_dir = Path(args.run_dir) if args.run_dir else (
        RUNS_DIR / f"{config.env_id}-{config.algo}-s{config.seed}-{int(time.time())}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.json")

    print(f"Training {config.algo.upper()} on {config.env_id} for {config.total_timesteps} "
          f"timesteps (seed {config.seed}) → {run_dir}")
    model_path = train(
        config,
        run_dir,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        use_tensorboard=args.tensorboard,
    )

    # Score the trained policy through the SAME harness as the random baseline.
    trained = _evaluate_policy(config.env_id, load_policy(config.algo, model_path),
                               args.eval_episodes, config.seed)
    baseline = _evaluate_policy(config.env_id, None, args.eval_episodes, config.seed)

    print(f"Saved model → {model_path}")
    print(f"  trained {config.algo.upper()}: {trained.summary()}")
    print(f"  random baseline:    {baseline.summary()}")
    print(f"  improvement: {trained.mean_return - baseline.mean_return:+.3f} mean return")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from robotic_arm_trash.plotting import plot_learning_curves

    curves = [(Path(rd).name, Path(rd) / "metrics.csv") for rd in (args.run_dir or [])]
    curves += [(Path(c).stem, Path(c)) for c in (args.csv or [])]
    if not curves:
        print("Provide at least one --run-dir or --csv.")
        return 1

    if args.label:
        if len(args.label) != len(curves):
            print(f"Got {len(args.label)} --label(s) for {len(curves)} curve(s).")
            return 1
        curves = [(label, path) for label, (_default, path) in zip(args.label, curves)]

    missing = [str(path) for _label, path in curves if not Path(path).exists()]
    if missing:
        print(f"No metrics found: {', '.join(missing)}")
        return 1

    out = plot_learning_curves(curves, args.out, title=args.title)
    print(f"Wrote plot to {out}")
    return 0


def _evaluate_policy(env_id, policy, episodes, seed, success_threshold=None):
    """Build ``env_id``, evaluate ``policy`` (random when None), and close the env."""
    env = gym.make(env_id)
    try:
        chosen = policy if policy is not None else random_policy(env)
        return evaluate(env, chosen, episodes=episodes, seed=seed,
                        success_threshold=success_threshold)
    finally:
        env.close()


def _cmd_eval(args: argparse.Namespace) -> int:
    # No --model → score the random policy (the "vs random" baseline the results table
    # needs); --model → load a saved SB3 checkpoint and score it through the same harness.
    if args.seed is not None:
        seed_everything(args.seed)

    if args.model is not None:
        from robotic_arm_trash.sb3 import load_policy

        policy = load_policy(args.algo, args.model)
        label = f"{args.algo.upper()} {args.model}"
    else:
        policy = None
        label = "random policy"

    report = _evaluate_policy(args.env, policy, args.episodes, args.seed,
                              success_threshold=args.success_threshold)
    print(f"{args.env} ({label}): {report.summary()}")

    if args.log_dir is not None:
        with MetricsLogger(args.log_dir) as logger:
            for episode, (ep_return, ep_length) in enumerate(
                zip(report.episode_returns, report.episode_lengths)
            ):
                logger.log(episode, episode_return=ep_return, episode_length=ep_length)
        print(f"Wrote per-episode metrics to {logger.csv_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robotic-arm-trash",
        description="Gymnasium/MuJoCo experiments for a Reacher-style robotic arm.",
    )
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="Run a random-policy rollout and print stats.")
    demo.add_argument("--env", default=DEFAULT_ENV, help="Gymnasium env id.")
    demo.add_argument("--steps", type=int, default=1000, help="Number of steps to roll out.")
    demo.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    demo.set_defaults(func=_cmd_demo)

    record = subparsers.add_parser("record", help="Record a random-policy rollout to video.")
    record.add_argument("--env", default=DEFAULT_ENV, help="Gymnasium env id.")
    record.add_argument("--steps", type=int, default=200, help="Number of steps to record.")
    record.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    record.add_argument("--video-dir", default=str(VIDEO_DIR), help="Output directory.")
    record.set_defaults(func=_cmd_record)

    train = subparsers.add_parser("train", help="Train an SB3 agent and score it.")
    train.add_argument("--env", default=DEFAULT_ENV, help="Gymnasium env id.")
    train.add_argument("--algo", default="sac", choices=["sac", "ppo"], help="Algorithm.")
    train.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps.")
    train.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    train.add_argument("--eval-episodes", type=int, default=10,
                       help="Episodes per evaluation (learning curve + final score).")
    train.add_argument("--eval-freq", type=int, default=2000,
                       help="Evaluate (and log the learning curve) every N steps.")
    train.add_argument("--tensorboard", action="store_true",
                       help="Also mirror metrics to TensorBoard (if installed).")
    train.add_argument("--run-dir", default=None, help="Output dir (default: runs/<auto>).")
    train.set_defaults(func=_cmd_train)

    eval_p = subparsers.add_parser("eval", help="Evaluate a saved model or the random baseline.")
    eval_p.add_argument("--env", default=DEFAULT_ENV, help="Gymnasium env id.")
    eval_p.add_argument("--episodes", type=int, default=10, help="Evaluation episodes.")
    eval_p.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    eval_p.add_argument("--model", default=None, help="Path to a saved SB3 model (.zip).")
    eval_p.add_argument("--algo", default="sac", choices=["sac", "ppo"],
                        help="Algorithm of --model.")
    eval_p.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="Episode counts as success when its return >= this value.",
    )
    eval_p.add_argument(
        "--log-dir",
        default=None,
        help="If set, write per-episode metrics as CSV to this directory.",
    )
    eval_p.set_defaults(func=_cmd_eval)

    plot = subparsers.add_parser("plot", help="Plot learning curve(s) from run metrics.")
    plot.add_argument("--run-dir", action="append", help="Run dir with metrics.csv (repeatable).")
    plot.add_argument("--csv", action="append", help="Path to a metrics.csv (repeatable).")
    plot.add_argument("--label", action="append",
                      help="Override curve label(s), in input order (repeatable).")
    plot.add_argument("--out", default="docs/learning_curve.png", help="Output PNG path.")
    plot.add_argument("--title", default=None, help="Plot title.")
    plot.set_defaults(func=_cmd_plot)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "command", None) is None:
        print(get_project_summary())
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
