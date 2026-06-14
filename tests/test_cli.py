import pytest

from robotic_arm_trash.cli import build_parser, get_project_summary, main


def test_project_summary_mentions_demo_locations() -> None:
    summary = get_project_summary()
    assert "robotic-arm-trash" in summary
    assert "demos" in summary
    assert "videos" in summary


def test_no_command_prints_summary(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    assert exit_code == 0
    assert "robotic-arm-trash" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["demo", "record", "train", "eval"])
def test_each_subcommand_parses_and_binds_a_handler(command: str) -> None:
    args = build_parser().parse_args([command])
    assert args.command == command
    assert callable(args.func)


def test_demo_command_runs_and_reports_steps(capsys: pytest.CaptureFixture[str]) -> None:
    # Pendulum-v1 avoids MuJoCo/rendering, keeping the test fast and dependency-light.
    exit_code = main(["demo", "--env", "Pendulum-v1", "--steps", "10", "--seed", "0"])
    assert exit_code == 0
    assert "10 steps" in capsys.readouterr().out


@pytest.mark.slow
def test_train_command_trains_and_reports(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("stable_baselines3")
    exit_code = main([
        "train", "--algo", "ppo", "--env", "Pendulum-v1",
        "--timesteps", "256", "--eval-episodes", "1", "--seed", "0",
        "--run-dir", str(tmp_path),
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "trained PPO" in out
    assert "improvement" in out
    assert (tmp_path / "model.zip").exists()
    assert (tmp_path / "config.json").exists()


def test_eval_command_reports_random_baseline(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["eval", "--env", "Pendulum-v1", "--episodes", "2", "--seed", "0"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "random policy" in out
    assert "2 episodes" in out
