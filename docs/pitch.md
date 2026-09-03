# Project pitch — Deep RL for Robotic Arm Control

Three lengths of the same story: a resume entry, a one-paragraph summary, and a
three-minute talk outline for an AI engineer / researcher audience. Every number comes
from [`README.md`](../README.md) and [`REPORT.md`](../REPORT.md).

---

## 1. Resume entry

**Deep RL for Robotic Arm Control: From-Scratch Soft Actor-Critic** — Python, PyTorch,
Gymnasium/MuJoCo, Stable-Baselines3 ·
[github.com/yongjinjiang/robotic-arm-trash](https://github.com/yongjinjiang/robotic-arm-trash)

- Built a reproducible deep RL experiment harness for a MuJoCo Reacher arm: a single
  rollout/eval abstraction ("a policy is a callable") so random, library, and hand-written
  agents are scored identically, with seeded configs, CSV/TensorBoard metrics,
  checkpointing, a CLI, and a 70-test suite.
- Implemented Soft Actor-Critic from scratch in PyTorch (twin clipped critics,
  tanh-squashed reparameterized Gaussian policy, automatic entropy-temperature tuning,
  Polyak target updates, replay buffer) and validated it against Stable-Baselines3:
  −4.12 ± 0.06 vs −4.16 ± 0.26 return over 3 seeds, matching within seed variance; wrote a
  derivation note mapping each loss term to the code.
- Ran a controlled scientific study: 5-seed variance with Student-t confidence intervals,
  learning-rate / network-width / replay-size ablations, and a SAC-vs-PPO
  sample-efficiency comparison (SAC −4.12 vs PPO −9.40 at 50k steps). Found replay
  freshness and step size dominate learning speed on this task while capacity is
  irrelevant.
- Visualized the learned critic's action-value surface and value field over target
  positions, showing actor/critic agreement and that the value function recovers the
  task's spatial geometry, including the asymmetry from the arm's start pose.

## 2. One-paragraph description

An end-to-end deep reinforcement learning project on a 2-joint MuJoCo robotic arm
(Gymnasium `Reacher-v5`), built to demonstrate algorithmic depth and experimental rigor
rather than just a working demo. I first built a reproducible experiment harness (seeded
rollouts, a shared evaluation protocol, config serialization, metrics logging,
checkpointing, a CLI, and a 70-test pytest suite) and established SAC and PPO baselines
with Stable-Baselines3 over 3 seeds. I then re-implemented Soft Actor-Critic from scratch
in PyTorch, including twin clipped double-Q critics, a reparameterized tanh-squashed
Gaussian actor with the change-of-variables log-prob correction, automatic temperature
tuning against a target entropy, and soft target updates. Scored through the same
harness, the from-scratch agent matches the library implementation within seed variance
(−4.12 ± 0.06 vs −4.16 ± 0.26). Finally I treated the agent as an object of study: a
5-seed confidence-interval analysis confirms the result is robust to seeding, ablations at
a fixed pre-convergence budget show that a higher learning rate and a smaller replay
buffer both speed learning while network width barely matters, SAC is far more
sample-efficient than PPO, and probing the trained critic shows a smooth single-peaked
Q-surface with the policy at its maximum and a value field that reproduces the workspace
geometry. Every figure and table regenerates from seeded configs via an idempotent sweep
script.

## 3. Three-minute pitch outline

**Hook (20s).** Most RL portfolio projects stop at "the arm reaches the target." I wanted
to answer three harder questions: can I implement the algorithm myself and prove it is
correct, is the result real or a lucky seed, and what did the network actually learn?

**Setup (30s).** MuJoCo Reacher, 10-D observation, 2-D torque action, reward is negative
fingertip distance minus control cost. The design principle is one abstraction: a policy
is a callable and a rollout is a function. Random, Stable-Baselines3, and from-scratch
agents all plug into the same harness, so every number in the project is measured the
same way. Seeds, frozen configs, CSV metrics, checkpoints, 70 tests.

**Depth (60s).** I wrote Soft Actor-Critic from scratch in PyTorch. The key pieces: soft
Bellman target with the entropy bonus inside the bootstrap, two critics with the min taken
to fight overestimation, a Gaussian actor reparameterized so the actor loss backprops
through the critic, the tanh change-of-variables correction to the log-probability, and
automatic temperature tuning by gradient descent on a target-entropy constraint. I
validated it head-to-head against SB3's SAC at 3 seeds by 50k steps. The learning curves
track each other throughout and converge to the same band, −4.12 versus −4.16, inside seed
noise. The derivation note ([`sac.md`](sac.md)) ties each equation to the specific
function.

**Science (50s).** With 5 seeds the final return has a 95% t-interval of [−4.33, −4.00],
so the result is robust. Ablations at a deliberately short 30k budget probe learning
speed: a higher learning rate helps monotonically, a smaller replay buffer helps because a
million-transition buffer never recycles at that horizon and stays diluted with stale
exploration data, and width 64 versus 256 makes almost no difference on a 10-D task. SAC
beats PPO on sample efficiency by a wide margin, the expected off-policy advantage.

**Interpretability, the physicist's angle (30s).** I opened up the trained critic. At a
fixed state the min-Q surface over the torque square is smooth and single-peaked, with the
policy's action sitting at the maximum, exactly what a converged actor-critic should show.
Binning the state value by target position gives a smooth spatial gradient: highest where
the target is near the fingertip's rest position, lowest on the far side the arm must
swing across. The critic recovered the task geometry, including the asymmetry from the
arm's initial pose, purely from learned values.

**Close (10s).** Everything regenerates from seeded configs with two scripts. Next step on
the roadmap is sparse-reward goal-conditioned control with hindsight experience replay.
