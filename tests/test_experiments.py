from __future__ import annotations

from pathlib import Path

import pytest

from robotic_arm_trash.experiments import (
    SeedSummary,
    final_eval_return,
    format_results_table,
    summarize_seeds,
)


def _make_run(dir_path: Path, final_return: float) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "metrics.csv").write_text(
        "step,eval_return,eval_return_std\n"
        "5000,-20.0,2.0\n"
        f"10000,{final_return},1.5\n"
    )
    return dir_path


def test_final_eval_return_reads_last_row(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "r", -5.0)
    assert final_eval_return(run / "metrics.csv") == -5.0


def test_summarize_seeds_computes_mean_and_sample_std(tmp_path: Path) -> None:
    runs = [_make_run(tmp_path / f"s{i}", value) for i, value in enumerate([-4.0, -6.0, -8.0])]
    summary = summarize_seeds("SAC", runs)

    assert summary.n_seeds == 3
    assert summary.mean_return == pytest.approx(-6.0)
    assert summary.std_return == pytest.approx(2.0)  # sample std of [-4,-6,-8]
    assert summary.final_step == 10000


def test_single_seed_std_is_zero(tmp_path: Path) -> None:
    summary = summarize_seeds("PPO", [_make_run(tmp_path / "only", -7.0)])
    assert summary.n_seeds == 1
    assert summary.std_return == 0.0


def test_summarize_raises_without_eval_rows(tmp_path: Path) -> None:
    run = tmp_path / "empty"
    run.mkdir()
    (run / "metrics.csv").write_text("step,eval_return,eval_return_std\n")
    with pytest.raises(ValueError):
        summarize_seeds("x", [run])


def test_format_results_table_with_baseline() -> None:
    sac = SeedSummary("SAC", [-4.0, -6.0, -8.0], 50000)
    ppo = SeedSummary("PPO", [-10.0, -12.0, -11.0], 50000)
    baseline = SeedSummary("Random", [-42.0, -43.0, -41.0], 50000)

    table = format_results_table([sac, ppo], baseline=baseline)

    assert "| Policy | Seeds | Timesteps | Eval return (mean ± std) | vs random |" in table
    assert "50,000" in table
    assert "-6.00 ± 2.00" in table  # SAC row
    assert "+36.00" in table  # SAC improvement vs random (-6 - -42)
    # Baseline appears first and has no improvement value.
    lines = table.splitlines()
    assert lines[2].startswith("| Random ")
