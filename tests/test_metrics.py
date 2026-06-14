from __future__ import annotations

import csv
from pathlib import Path

from robotic_arm_trash.metrics import MetricsLogger


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_writes_csv_with_step_and_metrics(tmp_path: Path) -> None:
    logger = MetricsLogger(tmp_path)
    logger.log(0, reward=1.0, length=200)
    logger.log(1, reward=2.5, length=180)
    logger.close()

    rows = _read_csv(logger.csv_path)
    assert [r["step"] for r in rows] == ["0", "1"]
    assert rows[0]["reward"] == "1.0"
    assert rows[1]["length"] == "180.0"


def test_records_property_exposes_logged_values(tmp_path: Path) -> None:
    logger = MetricsLogger(tmp_path)
    logger.log(0, a=1.0)
    assert logger.records == [{"step": 0, "a": 1.0}]


def test_union_of_keys_across_rows(tmp_path: Path) -> None:
    logger = MetricsLogger(tmp_path)
    logger.log(0, a=1.0)
    logger.log(1, a=2.0, b=3.0)
    logger.close()

    rows = _read_csv(logger.csv_path)
    assert reader_fieldnames(logger.csv_path) == ["step", "a", "b"]
    assert rows[0]["b"] == ""  # missing in the first row


def reader_fieldnames(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return csv.DictReader(handle).fieldnames or []


def test_context_manager_flushes_on_exit(tmp_path: Path) -> None:
    with MetricsLogger(tmp_path) as logger:
        logger.log(0, value=42.0)
    assert logger.csv_path.exists()
    assert _read_csv(logger.csv_path)[0]["value"] == "42.0"


def test_tensorboard_request_never_crashes_and_still_writes_csv(tmp_path: Path) -> None:
    # Whether or not tensorboard is installed, CSV logging must still work.
    with MetricsLogger(tmp_path, use_tensorboard=True) as logger:
        logger.log(0, value=1.0)
    assert logger.csv_path.exists()
    assert isinstance(logger.tensorboard_available, bool)
