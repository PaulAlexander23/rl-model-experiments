# Part 2: how much memory does each kind of partial observability need?

Part 1 ([background.md](background.md), [findings.md](findings.md)) compared architectures on two
velocity-masked control tasks with SB3. Here the question is quantitative: **as partial
observability increases, how wide, how deep, and how far back must each memory architecture
reach?** Answering it needs grids of size × depth × difficulty × seed, which in SB3 (0.4–4k steps/s
for recurrent policies) would take weeks. So this part runs on a JAX PPO in which environment,
rollout and update are one compiled program, `vmap`-ped over seeds and difficulty levels.

Results: [memory-findings.md](memory-findings.md). Generated tables and figures: `docs/results/m*.md|png`.

## 1. Kinds of partial observability

"More partial observability" is not one dial. A history-dependent policy needs to compute a
*sufficient statistic* of the history (the belief state, or anything that determines the optimal
action), and tasks differ in what that statistic is and how hard it is to maintain:

| Kind | What the agent must do | Memory length | Memory size | Update complexity |
|------|------------------------|---------------|-------------|-------------------|
| Missing derivatives | difference recent frames | a few steps | small | linear |
| Observation noise | filter (average) many readings | medium | small (mean + uncertainty) | near-linear |
| Delayed recall | hold one item, untouched, for L steps | **long** | small | identity |
| Capacity | store K items, retrieve by key | medium | **K × bits per item** | write / lookup |
| Computation over history | maintain a state under a nonlinear update (counting, parity) | whole episode | small | **nonlinear** |
| Hidden switching state | track a belief over a latent regime | geometric (switch rate) | number of regimes | Bayes filter |

A confounder runs through all of these: **credit-assignment length**, the delay between an action
and its consequence. Ni et al. (2023, *When do Transformers Shine in RL? Decoupling Memory from
Credit Assignment*) showed that transformers extend memory length but not credit assignment; the
tasks here keep credit assignment immediate (the rewarded action is the one that needs the memory)
so that only the memory demand varies.

## 2. Where each architecture should shine (hypotheses)

| Architecture | Memory mechanism | Expected strengths | Expected weaknesses |
|---|---|---|---|
| MLP | none | control: measures how much a task really needs memory | anything with hidden state |
| Frame stack | last k inputs, verbatim | derivatives, short noise averaging | nothing beyond k; input grows with k |
| LSTM / GRU | gated state, sequential | filtering, counting, holding a value | long gaps (gradients through many steps), sequential compute |
| LRU (linear recurrent unit) | diagonal complex linear recurrence, nonlinear between layers | long memory (eigenvalues near 1), parallel training | state tracking that needs a nonlinear *recurrent* update (parity, mod-m counting): a time-invariant linear recurrence cannot implement it exactly (Merrill et al. 2024; Grazzi et al. 2024) |
| Transformer (gated, TrXL-style memory) | attention over the last C steps | exact retrieval of items within the context (recall) | anything older than C; integrating many noisy readings; RL instability (Part 1 E2/E5) |

**Sizing hypotheses.** Width should scale with the size of the sufficient statistic (capacity,
belief dimension); depth should matter only where the update is nonlinear and compositional
(modular counting); the **truncated-BPTT length** (and, for the transformer, the context), rather than
width, should limit delayed recall.

## 3. Tasks

All in `src/rl_experiments/jaxrl/tasks.py`; each has one scalar dial (`level`) so that every level
trains inside the same compiled program. Episode returns are scaled to at most 1 (CartPole 5); scores
are normalised so that random play = 0 and optimal play = 1.

| Task | Kind | Dial (levels) | Random / optimal |
|------|------|---------------|------------------|
| `noisy_cartpole` | derivatives + noise | noise std on (x, θ) in units of the failure thresholds: 0, 0.1, 0.2, 0.4 | ~22 steps / 500 steps (the MDP ceiling; may be unreachable with noise) |
| `delayed_cue` | memory length | steps between cue and query: 4, 16, 64, 256 | 1/4 / 1 |
| `associative_recall` | capacity | key-value pairs (2 bits each): 2, 4, 8, 16 | 1/4 / 1 |
| `modular_sum` | computation | modulus of the running sum: 2 (parity), 3, 5, 8 | 1/m / 1 |
| `hidden_regime` | belief tracking | probability an observation is truthful: 1.0, 0.5, 0.3, 0.2 | 1/4 / HMM forward filter, simulated |

* **`delayed_cue`** is the passive T-maze of Ni et al.: the cue (1 of 4) appears only at t=0, random
  distractors follow, and only the action at the query step is rewarded.
* **`associative_recall`** shows K (key, value) pairs one per step, then 4 blank steps, then 8
  queries; keys are distinct draws from 16, values from 4. A transformer can retrieve a pair by
  attending to the step whose key matches the query; a recurrent net must hold all K pairs in its state.
* **`modular_sum`** asks every step for the running sum mod m of the digits seen so far: little to
  store (log2 m bits) but a nonlinear update.
* **`hidden_regime`**: one of 4 regimes, switching with probability 0.05 per step; the observation is
  the true regime with probability q, otherwise uniform. The Bayes-optimal policy acts on the HMM
  filter's belief; its simulated return is the optimum used for normalisation.

Each task also has an `oracle=True` variant that adds the hidden sufficient statistic to the
observation (an MDP version), used to check that a task is learnable at all.

## 4. Implementation

### JAX stack (`src/rl_experiments/jaxrl/`)

* **Why JAX, not Warp.** Warp is for writing fast simulators; these tasks are a few lines of
  `jnp` each. What matters is fusing env, policy and update into one XLA program (PureJaxRL style)
  and batching whole training runs with `vmap`. The PPO is ported from the user's
  `transit-dispatch-bench` (`algos/ppo.py`, itself matched to SB3), with Flax `linen` for the networks.
* **One sequence API for every core**: `carry, y = core(carry, x[T,B,F], starts[T,B])`. Rollouts call
  it with T=1, training with T=`seq_len`. Tests check, for every core, that one call over a sequence
  equals T single-step calls with episode resets in between, and that nothing before a reset leaks
  through.
* **Cores** (`models.py`), all with a `width` and a `depth`:
  * `RNNCore`: `depth` stacked `OptimizedLSTMCell` / `GRUCell`, scanned with `nn.scan`.
  * `LRUCore`: embedding, then `depth` residual blocks LayerNorm → LRU → GELU → GLU (Orvieto et al.
    2023, eigenvalue ring r ∈ [0.9, 0.999]). The recurrence runs as a parallel
    `lax.associative_scan`; an episode start sets that step's λ to 0, so resets need no special case.
  * `TransformerCore`: pre-LN, GTrXL GRU gates in place of residual additions, learned per-head
    distance bias (no absolute positions, so rollout and training see identical inputs). Its carry is a
    Transformer-XL **per-layer memory**: each layer's inputs for the last C steps, plus which belong to
    the current episode. During training a chunk attends to that stored memory (no gradient into it),
    so retrieval works across chunk boundaries even though gradients do not.
  * `FrameStackCore` (window of k inputs, zeroed before the episode start) and `MLPCore`.
* **ActorCritic**: shared core, separate 64-unit tanh policy and value heads (0.01 / 1.0 orthogonal
  gains). Part 1's E4 found that sharing is safe when the value scale is small, and every return here
  is at most 1 (CartPole 5).
* **Recurrent PPO** (`ppo.py`): the rollout of `n_steps` per env is collected as chunks of `seq_len`
  steps, storing the carry at each chunk start, so every training sequence is replayed from exactly
  the state the policy acted from. Minibatches are sets of whole sequences. The entire training
  run (`lax.scan` over updates) plus a greedy evaluation of 256 fresh episodes is one function of
  `(key, level)`; `run.py` jits `vmap(vmap(train))` over 3 seeds × 4 levels.
* **Entry point**: `uv run rl-train-jax task=<task> model=<model> model.width=32`, configs in
  `conf_jax/`. Hydra's basic launcher runs a sweep's jobs one after another in one process, so the GPU
  only ever holds one job.

### Pilot findings that shaped the setup

PILOT_PLACEHOLDER

## 5. Plan

Every run reports a normalised score (random 0, optimal 1) per level and seed. The headline
quantity is the **smallest width reaching a score of 0.9** at each level (mean over 3 seeds).

| ID | Question | Sweep | Budget |
|----|----------|-------|--------|
| M0 | Does the stack work and how fast is it? | GRU-64 on every task; oracle MLPs; PPO settings | pilot |
| M1 | Width needed per task, level and architecture | 5 tasks × 6 models × widths 8–128, depth 1 | about 1 h per task |
| M2 | Does depth substitute for width? | depth 1–3 at small widths, where M1 shows a capacity limit | about 1 h |
| M3 | Delayed recall: BPTT length vs context vs width | `delayed_cue`: `seq_len` 16–256 for LSTM/GRU/LRU, context 16–256 for the transformer | about 1 h |
