from __future__ import annotations

from pathlib import Path

import pytest

from robotic_arm_trash.experiments import (
    SeedSummary,
    final_eval_return,
    format_ci_table,
    format_results_table,
    summarize_seeds,
    t_ci,
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


def test_t_ci_brackets_the_mean_and_is_symmetric() -> None:
    lo, hi = t_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = 3.0
    assert lo < mean < hi
    assert abs((mean - lo) - (hi - mean)) < 1e-9  # symmetric about the mean


def test_t_ci_matches_known_t_value() -> None:
    # n=5 → dof=4, t_.975=2.776. values: mean 3, sample std sqrt(2.5), SE=sqrt(2.5/5)=sqrt(.5).
    import math

    lo, hi = t_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_half = 2.776 * math.sqrt(0.5)
    assert abs((hi - lo) / 2 - expected_half) < 1e-3


def test_t_ci_single_sample_is_degenerate() -> None:
    assert t_ci([7.0]) == (7.0, 7.0)


def test_t_ci_is_deterministic() -> None:
    vals = [-4.1, -4.2, -3.9, -4.0, -4.3]
    assert t_ci(vals) == t_ci(vals)


def test_seed_summary_ci_property_brackets_mean() -> None:
    s = SeedSummary("x", [-4.1, -4.2, -3.9, -4.0, -4.3], final_step=50000)
    lo, hi = s.ci95
    assert lo < s.mean_return < hi
    assert s.stderr > 0


def test_format_ci_table_has_ci_column(tmp_path: Path) -> None:
    s = SeedSummary("SAC", [-4.1, -4.2, -3.9], final_step=30000)
    table = format_ci_table([s], arm_header="Arm")
    assert "| Arm | Seeds | Timesteps | Final eval return (mean) | 95% CI |" in table
    assert "[" in table and "]" in table
