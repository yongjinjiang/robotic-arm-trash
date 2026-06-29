from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robotic_arm_trash.sac import SAC  # noqa: E402
from robotic_arm_trash.viz import q_action_grid, value_vs_target  # noqa: E402

OBS_DIM, ACT_DIM = 10, 2  # Reacher-v5 shapes.
LOW = np.array([-1.0, -1.0], dtype=np.float32)
HIGH = np.array([1.0, 1.0], dtype=np.float32)


def _agent() -> SAC:
    return SAC(OBS_DIM, ACT_DIM, LOW, HIGH, hidden_sizes=(16, 16),
               device=torch.device("cpu"))


def test_q_action_grid_shapes_and_finite() -> None:
    agent = _agent()
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    axis, q, policy_action = q_action_grid(agent, obs, n=11)
    assert axis.shape == (11,)
    assert q.shape == (11, 11)
    assert np.isfinite(q).all()
    assert policy_action.shape == (ACT_DIM,)
    assert (policy_action >= -1).all() and (policy_action <= 1).all()


def test_value_vs_target_grid_shape() -> None:
    import gymnasium as gym

    agent = _agent()
    env = gym.make("Reacher-v5")
    try:
        value_grid, counts, extent = value_vs_target(agent, env, n_states=80, bins=6)
    finally:
        env.close()
    assert value_grid.shape == (6, 6)
    assert counts.sum() == 80
    assert extent.shape == (4,)
    # Cells with samples must be finite; empty cells are NaN by construction.
    assert np.isfinite(value_grid[counts > 0]).all()
