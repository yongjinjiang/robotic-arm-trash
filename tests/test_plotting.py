from __future__ import annotations

from pathlib import Path

import pytest

from robotic_arm_trash.cli import main
from robotic_arm_trash.plotting import (
    load_curve,
    plot_aggregated_curves,
    plot_learning_curves,
)


def _write_metrics(path: Path) -> Path:
    path.write_text(
        "step,eval_return,eval_return_std\n"
        "1000,-7.9,2.5\n"
        "2000,-6.2,3.1\n"
        "3000,-4.7,2.3\n"
    )
    return path


def test_load_curve_parses_rows(tmp_path: Path) -> None:
    steps, returns, stds = load_curve(_write_metrics(tmp_path / "metrics.csv"))
    assert steps == [1000, 2000, 3000]
    assert returns == [-7.9, -6.2, -4.7]
    assert stds == [2.5, 3.1, 2.3]


def test_load_curve_skips_rows_without_eval_return(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("step,eval_return,eval_return_std\n0,,\n1000,-5.0,1.0\n")
    steps, returns, _ = load_curve(path)
    assert steps == [1000]
    assert returns == [-5.0]


def test_plot_writes_png(tmp_path: Path) -> None:
    _write_metrics(tmp_path / "metrics.csv")
    out = plot_learning_curves(
        [("sac-seed0", tmp_path / "metrics.csv")],
        tmp_path / "out" / "curve.png",
        title="SAC on Reacher-v5",
    )
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic number


def test_plot_raises_when_no_data(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("step,eval_return,eval_return_std\n")
    with pytest.raises(ValueError):
        plot_learning_curves([("empty", path)], tmp_path / "out.png")


def test_plot_command_writes_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_metrics(run_dir / "metrics.csv")
    out = tmp_path / "curve.png"

    exit_code = main(["plot", "--run-dir", str(run_dir), "--out", str(out)])
    assert exit_code == 0
    assert out.exists()


def test_plot_command_errors_without_inputs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plot"]) == 1
    assert "at least one" in capsys.readouterr().out


def test_plot_aggregated_curves_writes_png(tmp_path: Path) -> None:
    sac = []
    for i, final in enumerate([-4.0, -6.0]):
        d = tmp_path / f"sac-s{i}"
        d.mkdir()
        (d / "metrics.csv").write_text(
            f"step,eval_return,eval_return_std\n1000,-9.0,1.0\n2000,{final},1.0\n"
        )
        sac.append(d / "metrics.csv")

    out = plot_aggregated_curves({"SAC": sac}, tmp_path / "agg.png", title="SAC")
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
