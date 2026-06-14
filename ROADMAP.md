# RL Roadmap — robotic-arm-trash

Turning a Reacher scaffold into a **mature deep RL + robotics** portfolio project that
demonstrates depth and creativity, not just a working demo.

**Design principles**
- *Reproducible science, not just a demo.* Seeds, configs, multi-seed stats, committed
  learning curves. Every claim has a number behind it.
- *Depth and breadth.* Use a trusted library to establish baselines, then re-implement
  the core algorithm from scratch to prove understanding.
- *One clean abstraction.* A policy is a callable; a rollout is a function. Random,
  SB3, and from-scratch agents all plug into the same harness.
- *Quantified status tables* (the `ai_pdf` bar) live in this file and the README.

---

## Status at a glance

| Phase | Theme | Status |
|------|-------|--------|
| 0 | Foundations & refactor (harness, CLI, seeding, eval, tests) | 🟨 In progress |
| 1 |  | ⬜ Not started |
| 2 | Algorithm from scratch (PyTorch SAC, validated vs SB3) | ⬜ Not started |
| 3 | Scientific study (seed variance, ablations, report) | ⬜ Not started |
| 4 | Harder robotics (sparse reward, goal-conditioned + HER) | ⬜ Not started |
| 5 | Presentation & portfolio finish | ⬜ Not started |

Legend: ⬜ not started · 🟨 in progress · ✅ done

**Center of gravity: Phase 2 (algorithmic depth).** The from-scratch PyTorch SAC is the
portfolio centerpiece; Phases 3–4 support and stress-test it. Phases 0→1 remain the
prerequisite spine.

---

## Phase 0 — Foundations & Refactor

*Goal: turn the scaffold into a real experiment harness. No learning yet, but everything
after this becomes cheap.*

- ✅ **Shared rollout utility** — `rollout(env, policy, steps, seed)` in
  `robotic_arm_trash/rollout.py`; `policy` is a callable `obs -> action`, random policy is
  now just one policy. Both demos route through it; returns `RolloutStats`
  (returns/lengths per completed episode); seeds env + action space for reproducibility.
- ✅ **CLI surface** — `robotic-arm-trash {demo,record,train,eval}` (argparse) in
  `cli.py`. `demo`/`record` drive the rollout harness with `--env/--steps/--seed`
  (nothing hardcoded); `train`/`eval` are wired stubs pointing at their phases. No-arg
  prints the project summary.
- **Config** — a frozen dataclass (env id, algo, timesteps, seed, hyperparameters);
  serialize alongside every run for reproducibility.
- **Deterministic seeding** — unified seeding across `gymnasium`, `numpy`, `torch`.
- **Eval harness** — run N episodes, report mean ± std return + a task success metric.
- **Metrics logging** — CSV + TensorBoard.
- **Tests** — env-construction smoke test; rollout shape/length test; seed-determinism
  test (same seed → same trajectory).

**Exit criteria:** demos run through the shared rollout; `train`/`eval` stubs wired;
`pytest -q` green with ≥4 meaningful tests.

## Phase 1 — Baseline Training (library)

*Goal: the first agent that actually learns. A credible, reproducible baseline.*

- Train **SAC** (off-policy, sample-efficient — natural fit for continuous control) and
  **PPO** via stable-baselines3. This finally uses the declared deps.
- Checkpoint save/load; record video of the *trained* policy reaching the target.
- **Results table** in README: algo · timesteps · eval return (mean ± std over ≥3 seeds)
  · vs random baseline · vs a published reference.
- Commit learning-curve plots.

**Exit criteria:** SAC reaches the near-optimal Reacher return band; README results table
populated; trained-arm video committed/linked.

## Phase 2 — Algorithm from Scratch (depth showcase)

*Goal: prove deep understanding — re-implement the core algorithm in PyTorch and validate
it against the library.*

- Implement **SAC from scratch**: twin Q-critics, entropy-temperature auto-tuning, target
  networks, replay buffer, the reparameterized policy. (Rich enough to showcase real
  understanding; the math ties to Sutton & Barto + the SAC papers.)
- Match SB3-SAC within seed variance.
- Component unit tests (replay buffer, soft target update, loss shapes).
- Short derivation note (`docs/sac.md`) connecting the objective to the code.

**Exit criteria:** scratch-SAC matches SB3-SAC within variance; component tests green;
derivation note written.

## Phase 3 — Scientific Study (the physicist's edge)

*Goal: treat RL as experimental science.*

- Seed-variance study (≥5 seeds) with proper confidence intervals.
- Hyperparameter ablations (entropy temp, learning rate, net width, replay size).
- Sample-efficiency comparison: PPO vs SAC.
- Creative angle (pick one): reward-shaping analysis, or visualize the learned
  value/policy landscape and interpret it.

**Exit criteria:** `REPORT.md` with plots + conclusions, fully reproducible from seeded
configs.

## Phase 4 — Harder Robotics (ambition / breadth)

*Goal: move beyond toy Reacher toward real manipulation.*

- Step up the task: `Pusher-v5`, a **sparse-reward** Reacher variant, or multi-goal Fetch.
- **Goal-conditioned RL + Hindsight Experience Replay (HER)**; HER vs no-HER comparison.
- Conceptual + small experiment on **domain randomization / sim-to-real**.

**Exit criteria:** agent solving a harder (ideally sparse-reward) task; HER ablation shown.

## Phase 5 — Presentation (portfolio finish)

*Goal: employer-facing polish.*

- README narrative: results tables, learning curves, embedded/linked videos.
- `ARCHITECTURE.md`; one-command reproduction (`uv run robotic-arm-trash train ...`).
- Housekeeping: gitignore `dist/` and `*.egg-info/`; version bump; CHANGELOG.
- Optional: a blog-style retrospective tying physics intuition to RL.

---

## Cross-cutting threads

- **Testing discipline** grows every phase (target: a real suite, not one assertion).
- **Reproducibility**: every result regenerable from a committed seed + config.
- **JOURNAL.md** records decisions and 💡/✨ moments as we go.

## Immediate next step

Phase 0, item 1: extract the shared `rollout(env, policy, steps, seed)` utility and route
both demos through it — the single change that unblocks every later phase.
