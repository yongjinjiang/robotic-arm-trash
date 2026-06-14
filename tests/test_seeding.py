from __future__ import annotations

import numpy as np
import pytest

from robotic_arm_trash.seeding import seed_everything


def test_returns_the_seed() -> None:
    assert seed_everything(42) == 42


def test_numpy_is_reproducible_across_seeds() -> None:
    seed_everything(0)
    first = np.random.rand(5)
    seed_everything(0)
    second = np.random.rand(5)
    assert np.array_equal(first, second)


def test_different_seeds_differ() -> None:
    seed_everything(0)
    first = np.random.rand(5)
    seed_everything(1)
    second = np.random.rand(5)
    assert not np.array_equal(first, second)


def test_torch_is_reproducible_if_available() -> None:
    torch = pytest.importorskip("torch")
    seed_everything(0)
    first = torch.randn(5)
    seed_everything(0)
    second = torch.randn(5)
    assert torch.equal(first, second)
