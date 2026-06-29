from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from robotic_arm_trash.config import ExperimentConfig
from robotic_arm_trash.evaluation import evaluate

torch = pytest.importorskip("torch")

from robotic_arm_trash.sac import (  # noqa: E402
    SAC,
    GaussianPolicy,
    QNetwork,
    ReplayBuffer,
    load_policy,
    soft_update,
    train,
)

# Pendulum keeps the component tests fast and MuJoCo-free: a 3-D obs, 1-D bounded action.
ENV_ID = "Pendulum-v1"
OBS_DIM, ACT_DIM = 3, 1
LOW, HIGH = np.array([-2.0], dtype=np.float32), np.array([2.0], dtype=np.float32)


def _agent(**kwargs) -> SAC:
    # Force CPU so the tests are deterministic and device-independent.
    return SAC(OBS_DIM, ACT_DIM, LOW, HIGH, hidden_sizes=(32, 32),
               device=torch.device("cpu"), **kwargs)


# --- ReplayBuffer ---------------------------------------------------------------------

def test_replay_buffer_add_and_sample_shapes() -> None:
    buf = ReplayBuffer(capacity=100, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    for _ in range(10):
        buf.add(np.zeros(OBS_DIM), np.zeros(ACT_DIM), 1.0, np.ones(OBS_DIM), False)
    assert len(buf) == 10

    batch = buf.sample(4, device=torch.device("cpu"))
    assert batch["obs"].shape == (4, OBS_DIM)
    assert batch["actions"].shape == (4, ACT_DIM)
    assert batch["rewards"].shape == (4, 1)
    assert batch["dones"].shape == (4, 1)
    assert batch["obs"].dtype == torch.float32


def test_replay_buffer_wraparound_overwrites_oldest() -> None:
    buf = ReplayBuffer(capacity=3, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    for r in range(5):  # 5 adds into capacity 3 → only the last 3 survive.
        buf.add(np.full(OBS_DIM, r), np.zeros(ACT_DIM), float(r), np.zeros(OBS_DIM), False)
    assert len(buf) == 3
    stored = {float(v) for v in buf.rewards[:, 0]}
    assert stored == {2.0, 3.0, 4.0}


# --- soft_update ----------------------------------------------------------------------

def test_soft_update_tau_one_copies_source() -> None:
    target, source = QNetwork(OBS_DIM, ACT_DIM, (16,)), QNetwork(OBS_DIM, ACT_DIM, (16,))
    soft_update(target, source, tau=1.0)
    for t, s in zip(target.parameters(), source.parameters()):
        assert torch.allclose(t, s)


def test_soft_update_tau_zero_is_noop() -> None:
    target, source = QNetwork(OBS_DIM, ACT_DIM, (16,)), QNetwork(OBS_DIM, ACT_DIM, (16,))
    before = [p.clone() for p in target.parameters()]
    soft_update(target, source, tau=0.0)
    for p, b in zip(target.parameters(), before):
        assert torch.allclose(p, b)


def test_soft_update_interpolates() -> None:
    target, source = QNetwork(OBS_DIM, ACT_DIM, (16,)), QNetwork(OBS_DIM, ACT_DIM, (16,))
    t0 = [p.clone() for p in target.parameters()]
    s0 = [p.clone() for p in source.parameters()]
    soft_update(target, source, tau=0.25)
    for p, t, s in zip(target.parameters(), t0, s0):
        assert torch.allclose(p, 0.75 * t + 0.25 * s, atol=1e-6)


# --- GaussianPolicy -------------------------------------------------------------------

def test_policy_actions_within_bounds() -> None:
    actor = GaussianPolicy(OBS_DIM, ACT_DIM, (32, 32), LOW, HIGH)
    obs = torch.randn(64, OBS_DIM)
    action, log_prob = actor.sample(obs)
    assert action.shape == (64, ACT_DIM)
    assert log_prob.shape == (64, 1)
    assert torch.isfinite(log_prob).all()
    assert (action >= torch.as_tensor(LOW)).all() and (action <= torch.as_tensor(HIGH)).all()


def test_policy_deterministic_differs_from_stochastic() -> None:
    actor = GaussianPolicy(OBS_DIM, ACT_DIM, (32, 32), LOW, HIGH)
    obs = torch.randn(8, OBS_DIM)
    det1, _ = actor.sample(obs, deterministic=True)
    det2, _ = actor.sample(obs, deterministic=True)
    sto, _ = actor.sample(obs, deterministic=False)
    assert torch.allclose(det1, det2)  # deterministic mode is repeatable
    assert not torch.allclose(det1, sto)  # and differs from a stochastic draw


# --- QNetwork -------------------------------------------------------------------------

def test_qnetwork_output_shape() -> None:
    q = QNetwork(OBS_DIM, ACT_DIM, (32, 32))
    out = q(torch.randn(16, OBS_DIM), torch.randn(16, ACT_DIM))
    assert out.shape == (16, 1)


# --- SAC.update -----------------------------------------------------------------------

def test_update_returns_finite_losses_and_changes_params() -> None:
    agent = _agent()
    buf = ReplayBuffer(capacity=200, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    rng = np.random.default_rng(0)
    for _ in range(64):
        buf.add(rng.standard_normal(OBS_DIM), rng.uniform(-2, 2, ACT_DIM),
                rng.standard_normal(), rng.standard_normal(OBS_DIM), False)

    before = [p.clone() for p in agent.actor.parameters()]
    losses = agent.update(buf.sample(32, agent.device))

    assert {"q_loss", "actor_loss", "alpha_loss", "alpha"} <= set(losses)
    assert all(np.isfinite(v) for v in losses.values())
    after = list(agent.actor.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(after, before))


def test_act_produces_valid_env_action() -> None:
    env = gym.make(ENV_ID)
    try:
        agent = _agent()
        obs, _ = env.reset(seed=0)
        action = agent.act(obs, deterministic=True)
        assert env.action_space.contains(action.astype(np.float32))
    finally:
        env.close()


# --- End-to-end smoke (slow) ----------------------------------------------------------

@pytest.mark.slow
def test_train_logs_curve_and_roundtrips(tmp_path: Path) -> None:
    import csv

    config = ExperimentConfig(env_id=ENV_ID, algo="sac_scratch", total_timesteps=400,
                              seed=0, learning_starts=100, batch_size=32, hidden_sizes=(64, 64))
    model_path = train(config, tmp_path, eval_freq=200, eval_episodes=1)
    assert model_path.exists()

    with (tmp_path / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {"step", "eval_return", "eval_return_std"} <= set(rows[0])

    env = gym.make(ENV_ID)
    try:
        report = evaluate(env, load_policy(model_path), episodes=1, seed=0)
    finally:
        env.close()
    assert report.num_episodes == 1
