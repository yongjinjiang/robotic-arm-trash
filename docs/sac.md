# SAC, from objective to code

A short derivation tying the from-scratch implementation in
[`robotic_arm_trash/sac.py`](../robotic_arm_trash/sac.py) to the Soft Actor-Critic papers
(Haarnoja et al., 2018a/b). Each section points at the function that realizes it.

## 1. The maximum-entropy objective

Standard RL maximizes expected return. SAC adds an entropy bonus, rewarding the policy for
staying as stochastic as it can while still earning return:

$$
J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}\Big[\, r(s_t,a_t) + \alpha\, \mathcal{H}(\pi(\cdot\mid s_t)) \,\Big].
$$

The temperature $\alpha$ trades off reward against entropy. Higher entropy means broader
exploration and more robust policies; $\alpha$ is tuned automatically (§5).

## 2. Soft value functions and the soft Bellman backup

The entropy term folds into the value functions. The **soft Q-target** for a transition
$(s, a, r, s')$ is

$$
y = r + \gamma\,(1-d)\,\big[\, \min_{i=1,2} Q_{\bar\theta_i}(s', a') - \alpha \log \pi(a'\mid s') \,\big],
\qquad a' \sim \pi(\cdot\mid s').
$$

The $-\alpha\log\pi$ term is the entropy bonus baked into bootstrapping. This is computed
under `torch.no_grad()` in `SAC.update` (the `target_v` / `target` lines), with $(1-d)$
zeroing the bootstrap on true termination.

## 3. Twin critics (clipped double-Q)

Single Q-learning over-estimates values because the max operator is biased. SAC keeps **two
independent critics** and uses their **minimum** in the target — `torch.min(q1_target,
q2_target)` above. Both critics regress onto the same target by MSE:

$$
L_Q = \big(Q_{\theta_1}(s,a) - y\big)^2 + \big(Q_{\theta_2}(s,a) - y\big)^2.
$$

Code: `QNetwork` (the critic), the twin `q1`/`q2` plus their targets in `SAC.__init__`, and
`q1_loss + q2_loss` in `SAC.update`.

## 4. The reparameterized, tanh-squashed policy

The actor is a diagonal Gaussian in a pre-squash variable $u$, then squashed through
$\tanh$ and rescaled to the action bounds:

$$
u = \mu_\phi(s) + \sigma_\phi(s)\odot\epsilon,\quad \epsilon\sim\mathcal N(0, I),
\qquad a = \text{scale}\cdot\tanh(u) + \text{bias}.
$$

Sampling $u$ via `rsample()` (the **reparameterization trick**) keeps the sample
differentiable w.r.t. $\phi$, so the actor loss can backprop through the critic. The
$\tanh$ change of variables requires a log-prob correction:

$$
\log\pi(a\mid s) = \log\mathcal N(u\mid s) - \sum_i \log\!\big(\text{scale}_i\,(1-\tanh^2 u_i) + \varepsilon\big).
$$

Code: `GaussianPolicy.sample` — the `rsample`, the squash/rescale, and the correction term.
The actor then maximizes entropy-regularized value (minimizes its negation):

$$
L_\pi = \mathbb{E}_{s}\big[\, \alpha\log\pi(a\mid s) - \min_{i} Q_{\theta_i}(s, a) \,\big],
\qquad a \sim \pi_\phi(\cdot\mid s).
$$

(the `actor_loss` line in `SAC.update`; $\alpha$ is detached here so the temperature is
tuned separately).

## 5. Automatic temperature tuning

Rather than hand-pick $\alpha$, SAC treats it as the dual variable of a constrained problem
("keep entropy at least at a target $\bar{\mathcal H}$") and optimizes

$$
L_\alpha = \mathbb{E}_{a\sim\pi}\big[\, -\alpha\,(\log\pi(a\mid s) + \bar{\mathcal H}) \,\big],
$$

with target entropy $\bar{\mathcal H} = -\dim(\mathcal A)$ (the paper's heuristic). When the
policy is more deterministic than the target, $\alpha$ grows, pushing exploration back up;
when it is too random, $\alpha$ shrinks. We optimize $\log\alpha$ for positivity/stability.
Code: `log_alpha`, `target_entropy`, and the `alpha_loss` line in `SAC.update`.

## 6. Soft target networks

The target critics track the online critics by Polyak averaging after every update,
$\bar\theta \leftarrow (1-\tau)\bar\theta + \tau\theta$, which stabilizes the moving
regression target. Code: `soft_update`, called on both targets at the end of `SAC.update`.

## Putting it together

`SAC.update` performs, in order: critic step (§2–3) → actor step (§4) → temperature step
(§5) → soft target update (§6). The training loop in `train` fills a `ReplayBuffer` with
environment interaction, warms up with random actions (`learning_starts`), then runs one
update per step. The actor is wrapped by `policy_from_actor` into the project's `Policy`
callable, so it is scored by the exact same `evaluate`/`rollout` harness as the random
baseline and the SB3 reference — the apples-to-apples comparison this phase is about.
