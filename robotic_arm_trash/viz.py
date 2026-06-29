"""Value/policy landscape visualization — the Phase 3 "physicist's edge".

Given a trained from-scratch SAC checkpoint (loaded with
:func:`robotic_arm_trash.sac.load_agent`, which restores the twin critics), this module
peeks *inside* the learned functions rather than just scoring them:

1. :func:`q_action_grid` — the critic's action-value surface ``min(Q1, Q2)(s, ·)`` over the
   2-D action square at a fixed state, with the policy's chosen action marked. Shows what the
   critic thinks of every torque pair and that the policy sits near the maximum.
2. :func:`value_vs_target` — the learned value ``V(s) = min(Q1,Q2)(s, π(s))`` as a function
   of *where the target is*, sampled over real environment resets and binned to a grid.
   Reveals the geometry of the task: how the agent's expected return depends on target
   location across the reachable workspace.

Compute functions return plain numpy arrays (easy to test); ``plot_*`` render them to PNG
with a headless matplotlib backend.

Reacher-v5 observation layout (10-D): ``[cos θ0, cos θ1, sin θ0, sin θ1, target_x,
target_y, θ̇0, θ̇1, fingertip_x − target_x, fingertip_y − target_y]``; action is a 2-D torque
in ``[-1, 1]²``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    import gymnasium as gym

    from robotic_arm_trash.sac import SAC

TARGET_X, TARGET_Y = 4, 5  # indices of the target position in the observation.


def _min_q(agent: "SAC", obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Twin-critic value ``min(Q1, Q2)(obs, actions)`` for batched numpy inputs."""
    o = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=agent.device)
    a = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=agent.device)
    with torch.no_grad():
        q = torch.min(agent.q1(o, a), agent.q2(o, a))
    return q.squeeze(-1).cpu().numpy()


def q_action_grid(
    agent: "SAC", obs: np.ndarray, *, n: int = 41,
) -> "Tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Critic value over the 2-D action grid at a fixed ``obs``.

    Returns ``(grid_axis, q_matrix, policy_action)``: ``grid_axis`` is the shared per-axis
    sample of ``[-1, 1]`` (length ``n``); ``q_matrix`` is ``(n, n)`` with
    ``q_matrix[i, j] = min Q(obs, [grid[j], grid[i]])``; ``policy_action`` is ``π(obs)``.
    Assumes a 2-D action space (Reacher).
    """
    axis = np.linspace(-1.0, 1.0, n)
    a0, a1 = np.meshgrid(axis, axis)  # a0 varies along columns, a1 along rows
    actions = np.stack([a0.ravel(), a1.ravel()], axis=1)
    obs_batch = np.repeat(np.asarray(obs, dtype=np.float32)[None, :], actions.shape[0], axis=0)
    q = _min_q(agent, obs_batch, actions).reshape(n, n)

    o = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=agent.device).unsqueeze(0)
    with torch.no_grad():
        policy_action, _ = agent.actor.sample(o, deterministic=True)
    return axis, q, policy_action.squeeze(0).cpu().numpy()


def value_vs_target(
    agent: "SAC", env: "gym.Env", *, n_states: int = 3000, bins: int = 20, seed: int = 0,
) -> "Tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Mean learned value ``V(s)=min Q(s, π(s))`` binned by target position over resets.

    Resets ``env`` ``n_states`` times (each randomizes the target and arm pose), records the
    target ``(x, y)`` and ``V(s)`` at that initial state, then bins into a ``bins × bins``
    grid. Returns ``(value_grid, counts, extent)`` where ``value_grid[i, j]`` is the mean
    value in that target cell (NaN where empty) and ``extent = (xmin, xmax, ymin, ymax)``.
    """
    env.reset(seed=seed)
    targets = np.zeros((n_states, 2), dtype=np.float32)
    obs_all = np.zeros((n_states, agent.actor.net[0].in_features), dtype=np.float32)
    for i in range(n_states):
        obs, _ = env.reset()
        obs_all[i] = obs
        targets[i] = obs[[TARGET_X, TARGET_Y]]

    o = torch.as_tensor(obs_all, device=agent.device)
    with torch.no_grad():
        actions, _ = agent.actor.sample(o, deterministic=True)
    values = _min_q(agent, obs_all, actions.cpu().numpy())

    xmin, xmax = float(targets[:, 0].min()), float(targets[:, 0].max())
    ymin, ymax = float(targets[:, 1].min()), float(targets[:, 1].max())
    sums = np.zeros((bins, bins))
    counts = np.zeros((bins, bins))
    xi = np.clip(((targets[:, 0] - xmin) / (xmax - xmin + 1e-9) * bins).astype(int), 0, bins - 1)
    yi = np.clip(((targets[:, 1] - ymin) / (ymax - ymin + 1e-9) * bins).astype(int), 0, bins - 1)
    for k in range(n_states):
        sums[yi[k], xi[k]] += values[k]
        counts[yi[k], xi[k]] += 1
    with np.errstate(invalid="ignore"):
        value_grid = np.where(counts > 0, sums / counts, np.nan)
    return value_grid, counts, np.array([xmin, xmax, ymin, ymax])


# --- plotting -------------------------------------------------------------------------

def plot_q_action_grid(agent: "SAC", obs: np.ndarray, out_path: Union[str, Path],
                       *, n: int = 41, title: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis, q, policy_action = q_action_grid(agent, obs, n=n)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(q, origin="lower", extent=(-1, 1, -1, 1), aspect="equal", cmap="viridis")
    ax.plot(policy_action[0], policy_action[1], "r*", markersize=16, label="π(s)")
    fig.colorbar(im, ax=ax, label="min(Q₁, Q₂)(s, a)")
    ax.set_xlabel("action[0] (joint-0 torque)")
    ax.set_ylabel("action[1] (joint-1 torque)")
    ax.set_title(title or "Critic action-value surface at a fixed state")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_value_vs_target(agent: "SAC", env: "gym.Env", out_path: Union[str, Path],
                         *, n_states: int = 3000, bins: int = 20,
                         title: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    value_grid, _counts, extent = value_vs_target(agent, env, n_states=n_states, bins=bins)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(value_grid, origin="lower", extent=tuple(extent), aspect="equal", cmap="magma")
    fig.colorbar(im, ax=ax, label="V(s) = min Q(s, π(s))")
    ax.set_xlabel("target x")
    ax.set_ylabel("target y")
    ax.set_title(title or "Learned value vs. target position")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
