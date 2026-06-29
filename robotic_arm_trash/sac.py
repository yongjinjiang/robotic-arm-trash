"""Soft Actor-Critic, implemented from scratch in PyTorch.

The depth showcase (ROADMAP Phase 2): SAC re-implemented from the papers — twin Q-critics
with clipped double-Q targets, a reparameterized tanh-squashed Gaussian policy, soft target
networks (Polyak), a replay buffer, and automatic entropy-temperature tuning — rather than
called from a library. It plugs into the *same* harness as everything else: training scores
the agent through :func:`robotic_arm_trash.evaluation.evaluate` and logs the learning curve
via :class:`robotic_arm_trash.metrics.MetricsLogger`, exactly like the SB3 seam
(:mod:`robotic_arm_trash.sb3`), so the two are compared apples-to-apples.

See ``docs/sac.md`` for the derivation tying each loss below to the objective.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robotic_arm_trash.config import ExperimentConfig
from robotic_arm_trash.evaluation import evaluate
from robotic_arm_trash.metrics import MetricsLogger
from robotic_arm_trash.rollout import Policy
from robotic_arm_trash.seeding import seed_everything

# Numerical guards.
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0
EPS = 1e-6


def pick_device() -> torch.device:
    """Pick a torch device. ``ROBOTIC_ARM_DEVICE`` overrides; else MPS → CUDA → CPU.

    The override exists because for these small MLPs the MPS↔CPU transfer overhead can make
    CPU faster — handy when sweeping many runs (see ``sweeps/phase3.py``).
    """
    import os

    override = os.environ.get("ROBOTIC_ARM_DEVICE")
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mlp(in_dim: int, hidden_sizes: "tuple[int, ...]", out_dim: int) -> nn.Sequential:
    """A plain ReLU MLP: ``in_dim -> *hidden_sizes -> out_dim`` (no output activation)."""
    layers: list[nn.Module] = []
    last = in_dim
    for width in hidden_sizes:
        layers += [nn.Linear(last, width), nn.ReLU()]
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Tanh-squashed diagonal-Gaussian actor with the reparameterization trick.

    The network outputs a mean and (clamped) log-std; an action is ``tanh(mean + std*eps)``
    rescaled to the env's bounds. ``sample`` returns the log-prob *with the tanh
    change-of-variables correction* — the term that makes the entropy bonus correct for a
    bounded action space.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes, action_low, action_high):
        super().__init__()
        self.net = mlp(obs_dim, hidden_sizes, 2 * act_dim)
        self.act_dim = act_dim
        # Affine map from tanh's [-1, 1] to the env's [low, high]. Registered as buffers so
        # they move with .to(device) and are saved/restored with the module.
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        """Return ``(action, log_prob)``. ``log_prob`` sums over action dims → shape (B, 1)."""
        mean, log_std = self.forward(obs)
        if deterministic:
            u = mean
            squashed = torch.tanh(u)
            action = squashed * self.action_scale + self.action_bias
            # Deterministic action has no meaningful density; return a placeholder.
            log_prob = torch.zeros(action.shape[0], 1, device=obs.device)
            return action, log_prob

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()  # reparameterized: gradients flow through the sample.
        squashed = torch.tanh(u)
        action = squashed * self.action_scale + self.action_bias

        # log π(a|s) = log N(u|s) - Σ log(1 - tanh(u)^2), the tanh Jacobian correction.
        log_prob = normal.log_prob(u) - torch.log(self.action_scale * (1 - squashed.pow(2)) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    """State-action critic Q(s, a) → scalar."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class ReplayBuffer:
    """Fixed-capacity replay buffer over preallocated numpy arrays (FIFO overwrite)."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self._idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> "dict[str, torch.Tensor]":
        idx = np.random.randint(0, self._size, size=batch_size)
        as_t = lambda a: torch.as_tensor(a[idx], device=device)
        return {
            "obs": as_t(self.obs),
            "actions": as_t(self.actions),
            "rewards": as_t(self.rewards),
            "next_obs": as_t(self.next_obs),
            "dones": as_t(self.dones),
        }


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: ``target ← (1-τ)·target + τ·source``, in place."""
    with torch.no_grad():
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.mul_(1.0 - tau).add_(s_param, alpha=tau)


class SAC:
    """Soft Actor-Critic agent: twin critics, target critics, auto temperature."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low,
        action_high,
        *,
        hidden_sizes=(256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: "float | None" = None,
        device: "torch.device | None" = None,
    ):
        self.device = device or pick_device()
        self.gamma = gamma
        self.tau = tau

        self.actor = GaussianPolicy(obs_dim, act_dim, hidden_sizes, action_low, action_high).to(self.device)
        self.q1 = QNetwork(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate
        )

        # Automatic temperature: optimize log_alpha against a fixed target entropy. Default
        # target entropy −dim(A) is the SAC-paper heuristic.
        self.target_entropy = float(-act_dim if target_entropy is None else target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=learning_rate)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: "dict[str, torch.Tensor]") -> "dict[str, float]":
        """One SAC gradient step on a sampled batch. Returns scalar losses (for logging)."""
        obs, actions = batch["obs"], batch["actions"]
        rewards, next_obs, dones = batch["rewards"], batch["next_obs"], batch["dones"]

        # --- Critic update: regress twin Qs onto the clipped double-Q soft target. ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            target_q = torch.min(
                self.q1_target(next_obs, next_action), self.q2_target(next_obs, next_action)
            )
            target_v = target_q - self.alpha * next_log_prob
            target = rewards + (1.0 - dones) * self.gamma * target_v

        q1_loss = F.mse_loss(self.q1(obs, actions), target)
        q2_loss = F.mse_loss(self.q2(obs, actions), target)
        q_loss = q1_loss + q2_loss

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # --- Actor update: maximize Q − α·logπ (entropy-regularized). ---
        new_action, log_prob = self.actor.sample(obs)
        q_new = torch.min(self.q1(obs, new_action), self.q2(obs, new_action))
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- Temperature update: drive policy entropy toward the target. ---
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- Soft target networks. ---
        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)

        return {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
        }


def policy_from_actor(agent: SAC) -> Policy:
    """Wrap a trained agent's actor into a deterministic :data:`Policy` for eval/rollout."""

    def policy(observation: np.ndarray) -> np.ndarray:
        return agent.act(observation, deterministic=True)

    return policy


def train(
    config: ExperimentConfig,
    run_dir: Union[str, Path],
    *,
    eval_freq: int = 2000,
    eval_episodes: int = 10,
    use_tensorboard: bool = False,
) -> Path:
    """Train from-scratch SAC per ``config``; save ``model.pt`` under ``run_dir``.

    Mirrors :func:`robotic_arm_trash.sb3.train`'s contract: every ``eval_freq`` steps the
    agent is scored through our ``evaluate`` and the eval return logged to
    ``run_dir/metrics.csv`` (the learning curve). Returns the saved checkpoint path.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    env = gym.make(config.env_id)
    eval_env = gym.make(config.env_id)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    agent = SAC(
        obs_dim,
        act_dim,
        env.action_space.low,
        env.action_space.high,
        hidden_sizes=tuple(config.hidden_sizes),
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        tau=config.tau,
    )
    buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim)
    logger = MetricsLogger(run_dir, use_tensorboard=use_tensorboard)
    policy = policy_from_actor(agent)

    try:
        obs, _ = env.reset(seed=config.seed)
        for step in range(1, config.total_timesteps + 1):
            # Warm up the buffer with random actions for unbiased early data.
            if step <= config.learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            # Bootstrap on time-limit truncation but not on true termination.
            buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs if not (terminated or truncated) else env.reset()[0]

            if step > config.learning_starts and len(buffer) >= config.batch_size:
                for _ in range(config.train_freq):
                    agent.update(buffer.sample(config.batch_size, agent.device))

            if step % eval_freq == 0:
                report = evaluate(eval_env, policy, episodes=eval_episodes, seed=config.seed)
                logger.log(step, eval_return=report.mean_return, eval_return_std=report.std_return)

        model_path = run_dir / "model.pt"
        torch.save(
            {
                "actor": agent.actor.state_dict(),
                # Critics are saved too so the value-landscape analysis (viz.py) can query
                # Q after the fact; load_policy ignores them.
                "q1": agent.q1.state_dict(),
                "q2": agent.q2.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "hidden_sizes": tuple(config.hidden_sizes),
                "action_low": env.action_space.low,
                "action_high": env.action_space.high,
            },
            model_path,
        )
    finally:
        logger.close()
        eval_env.close()
        env.close()
    return model_path


def load_policy(model_path: Union[str, Path]) -> Policy:
    """Load a saved ``model.pt`` actor and return it wrapped as a deterministic ``Policy``."""
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    device = pick_device()
    actor = GaussianPolicy(
        ckpt["obs_dim"],
        ckpt["act_dim"],
        tuple(ckpt["hidden_sizes"]),
        ckpt["action_low"],
        ckpt["action_high"],
    ).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    def policy(observation: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=device).unsqueeze(0)
        with torch.no_grad():
            action, _ = actor.sample(obs_t, deterministic=True)
        return action.squeeze(0).cpu().numpy()

    return policy


def load_agent(model_path: Union[str, Path]) -> SAC:
    """Rebuild a full :class:`SAC` (actor **and** critics) from a checkpoint for analysis.

    Unlike :func:`load_policy` (which returns just the deterministic policy), this restores
    the twin critics so the value-landscape tools in :mod:`robotic_arm_trash.viz` can query
    Q. Requires a checkpoint that saved critics (Phase 3+); raises otherwise.
    """
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if "q1" not in ckpt:
        raise ValueError(
            f"{model_path} has no saved critics; retrain with the current sac.train "
            "to enable value-landscape analysis."
        )
    agent = SAC(
        ckpt["obs_dim"],
        ckpt["act_dim"],
        ckpt["action_low"],
        ckpt["action_high"],
        hidden_sizes=tuple(ckpt["hidden_sizes"]),
    )
    agent.actor.load_state_dict(ckpt["actor"])
    agent.q1.load_state_dict(ckpt["q1"])
    agent.q2.load_state_dict(ckpt["q2"])
    agent.actor.eval()
    agent.q1.eval()
    agent.q2.eval()
    return agent
