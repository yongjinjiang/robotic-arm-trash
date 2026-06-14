from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from robotic_arm_trash.evaluation import evaluate
from robotic_arm_trash.metrics import MetricsLogger
from robotic_arm_trash.rollout import RolloutStats, random_policy, rollout
from robotic_arm_trash.seeding import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demos"
VIDEO_DIR = PROJECT_ROOT / "videos"

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
    print(
        f"`train` is not implemented yet — planned for ROADMAP Phase 1 "
        f"({args.algo.upper()} on {args.env}, {args.timesteps} timesteps, seed {args.seed})."
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    # Until Phase 1 adds trained-model loading, eval scores the random policy — i.e. the
    # "vs random" baseline the Phase 1 results table needs.
    if args.seed is not None:
        seed_everything(args.seed)
    env = gym.make(args.env)
    try:
        report = evaluate(
            env,
            random_policy(env),
            episodes=args.episodes,
            seed=args.seed,
            success_threshold=args.success_threshold,
        )
    finally:
        env.close()
    print(f"{args.env} (random policy): {report.summary()}")

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

    train = subparsers.add_parser("train", help="Train an agent (Phase 1 — stub).")
    train.add_argument("--env", default=DEFAULT_ENV, help="Gymnasium env id.")
    train.add_argument("--algo", default="sac", choices=["sac", "ppo"], help="Algorithm.")
    train.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps.")
    train.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
    train.set_defaults(func=_cmd_train)

    eval_p = subparsers.add_parser("eval", help="Evaluate the random-policy baseline.")
    eval_p.add_argument("--env", default=DEFAULT_ENV, help="Gymnasium env id.")
    eval_p.add_argument("--episodes", type=int, default=10, help="Evaluation episodes.")
    eval_p.add_argument("--seed", type=int, default=0, help="Seed for reproducibility.")
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
