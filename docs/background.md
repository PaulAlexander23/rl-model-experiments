# Background and investigation plan

This page explains each topic, the questions it raises, and the experiment that answers them.
Results and their interpretation are in [findings.md](findings.md).

## 1. PPO in one paragraph

PPO alternates two phases. **Rollout**: run the current policy in `n_envs` parallel environments for
`n_steps` steps and store observations, actions, log-probs, values and rewards. **Update**: compute
advantages with GAE, then run `n_epochs` passes of minibatch SGD (`batch_size`) on the clipped surrogate
objective `L = L_clip + vf_coef · L_value − ent_coef · H`. The two phases have very different compute
profiles, which is the key to the device question below:

| Phase   | Work                                                  | Batch size seen by the network |
|---------|-------------------------------------------------------|--------------------------------|
| Rollout | `n_steps` sequential forward passes + env steps (CPU) | `n_envs` (e.g. 8–16)           |
| Update  | `n_epochs × (n_steps·n_envs / batch_size)` fwd+bwd    | `batch_size` (e.g. 64–1024)    |

## 2. Shared trunk vs split actor-critic (MLP)

* **Split** (SB3's default): the actor `π(a|s)` and the critic `V(s)` are separate networks. Their losses
  can't interfere, and each can have its own capacity. The cost is twice the parameters and compute.
* **Shared**: one trunk produces features, with two small linear heads. It has half the parameters, and
  the critic's signal (dense, every step) can shape features the actor uses. But the two gradients
  share parameters, so their *relative scale* matters. `vf_coef` and reward scale become important: large
  returns produce large value gradients that can swamp the policy gradient. The PPO paper
  (Schulman et al., 2017) used shared networks for Atari and split networks for MuJoCo. "What Matters
  in On-Policy RL" (Andrychowicz et al., 2021) recommends separate networks for continuous control.
  Phasic Policy Gradient (Cobbe et al., 2021) is a hybrid that gets the representation benefit without
  the interference.

**How SB3 expresses this.** Since SB3 1.8, `net_arch` *always* builds separate `pi`/`vf` MLPs. The only
sharable part is the *features extractor* (`share_features_extractor=True`), which for vector
observations is just `Flatten`. A genuinely shared trunk is therefore:

```python
policy_kwargs = dict(
    features_extractor_class=MlpTrunkExtractor,  # rl_experiments.policies.extractors
    features_extractor_kwargs=dict(hidden_sizes=(64, 64)),
    share_features_extractor=True,
    net_arch=[],  # heads = single Linear layers
)
```

**Experiment E1.** `shared_mlp` vs `split_mlp` (same 64×64 hidden sizes) on CartPole, Acrobot,
Pendulum (continuous) and LunarLander, 5 seeds each, using rl-zoo hyperparameters. Metrics: final
deterministic eval return, the area under the training curve (sample efficiency) and parameter count.

## 3. Memory: frame stacking, LSTM, GRU, transformer

Fully observed classic-control tasks don't need memory. To test memory we remove the velocity
components from the observation (`MaskVelocityWrapper`). This gives **CartPole-NoVel** and
**Pendulum-NoVel**: the agent sees positions and angles only, and has to infer velocities from history.

| Model            | Memory mechanism                                   | Algorithm         |
|------------------|----------------------------------------------------|-------------------|
| `split_mlp`      | none (lower bound)                                 | PPO               |
| `framestack_mlp` | last 8 observations concatenated                   | PPO               |
| `transformer`    | self-attention over the last 8 observations        | PPO               |
| `lstm`           | LSTM hidden state, separate actor/critic LSTMs     | RecurrentPPO      |
| `gru`            | GRU hidden state, separate actor/critic GRUs       | RecurrentPPO      |
| `lstm_shared`    | one LSTM for both; value loss trains it            | RecurrentPPO      |
| `lstm_shared_detached` | one LSTM; critic gradient detached (sb3-contrib's `shared_lstm`) | RecurrentPPO |
| `gru_shared`     | one GRU for both; value loss trains it             | RecurrentPPO      |

**Implementation notes.**

* sb3-contrib's `RecurrentPPO` only supports LSTMs. It hard-codes `(h, c)` state tuples into the
  rollout buffer. `GRUAsLSTM` wraps `nn.GRU` with the LSTM call signature and passes `c` through
  unchanged, so GRUs plug in without touching the buffer or the algorithm.
* sb3-contrib's `shared_lstm=True` **detaches** the critic from the LSTM
  (`latent_vf = latent_pi.detach()`), so in its "shared" LSTM only the policy loss trains the recurrent
  layer. `shared_rnn_grad=True` removes the detach. E2 runs both, which turns shared vs split for RNNs
  into a three-way comparison: separate / shared / shared-detached.
* **Fast sequence path.** For training, `RecurrentRolloutBuffer` cuts the data into sequences *at episode
  boundaries*, so a hidden-state reset can only occur at `t=0` of each padded sequence. sb3-contrib
  still falls back to a Python loop over timesteps whenever any sequence starts a new episode, which is
  almost always. `RecurrentActorCriticPolicyV2._process_sequence` applies the `t=0` reset to the initial
  state and makes one cuDNN call over the whole sequence. It is numerically identical
  (`test_fast_sequence_matches_loop`) and should matter a lot on the GPU, where each loop iteration is a
  kernel launch.
* **Transformer.** A recurrent transformer (GTrXL-style memory) would need its own buffer and algorithm.
  The practical alternative within SB3 is attention over a **fixed window**: `VecFrameStack(n=8)` plus
  `TransformerWindowExtractor` (a pre-LN encoder with a learned positional embedding, which returns the newest
  token). This works with vanilla PPO, but the context is limited to the window, so the fair baseline is
  `framestack_mlp` with the same window.

**Experiment E2.** All of the models above on both POMDP tasks, with 3 seeds and identical
PPO hyperparameters per task.

## 4. CPU vs GPU, and hybrid CPU rollouts + GPU updates

Why the GPU is often *slower* for SB3 PPO: during rollouts the network sees a batch of `n_envs`
observations. Each step costs a host→device copy, a few tiny kernel launches and a device→host copy,
all around 10 µs each, while a 64×64 MLP forward on CPU takes about 10 µs in total. The environments
run on the CPU regardless. SB3 itself warns that PPO with an MLP policy is "primarily intended to run on
the CPU".

The update phase is the opposite: `n_epochs × n_minibatches` forward/backward passes at `batch_size`.
That is where a GPU can win, but only if the minibatch is large enough to amortise launch overhead.

**Hybrid device callback.** In SB3's on-policy loop:

1. `collect_rollouts` converts observations with `obs_as_tensor(obs, model.device)`. The rollout buffer
   stores numpy arrays, so it is device-agnostic while filling.
2. `callback.on_rollout_end()` fires *after* returns and advantages are computed and *before* `train()`.
3. `train()` reads minibatches through `rollout_buffer.get()`, which converts them to `rollout_buffer.device`.

So a callback can switch devices around the update:
`rl_experiments.callbacks.device.HybridDeviceCallback` (your hunch about using callbacks is right).
Two strategies:

* `move`: on rollout start, `policy.to("cpu")` and move the Adam moment tensors. On rollout end, move
  both back to `cuda`. Set `model.device` and `rollout_buffer.device` to match. Adam's `step` counter
  stays on the CPU, as `Optimizer.load_state_dict` does for non-capturable Adam. Each iteration moves
  about 3× the parameters in each direction.
* `mirror`: keep a CPU *copy* of the policy that is only used to act. Before each rollout,
  `load_state_dict` the GPU weights into it (parameters only, one direction), and swap
  `model.policy` between the two. The optimizer never leaves the GPU.

Both also move `RecurrentPPO._last_lstm_states`. Tests check that acting happens on the CPU and
updates on the GPU, and that a hybrid run ends with the same weights as a pure-CPU run with the same seed.

**Experiment E3.**
* E3a (MLP): LunarLander, 100k steps, widths 64/256/1024 × batch size 64/1024 × {cpu, cuda,
  hybrid_move, hybrid_mirror}. Each run reports rollout time, update time and transfer time.
* E3b (memory): CartPole-NoVel, 50k steps, LSTM/GRU/transformer × {cpu, cuda, hybrid_mirror}, plus the
  fast-sequence path on vs off.

Timing runs execute sequentially with `torch_threads=4`, so they don't compete for cores. The
learning experiments (E1, E2) run 14 at a time with `torch_threads=1`, on whichever device E3 shows is
fastest.

## 5. Beyond SB3: where GPUs actually pay off

When environments are cheap, the real speed-ups come from moving the *environment* onto the GPU and
avoiding host↔device traffic altogether. Examples are JAX (gymnax, Brax, PureJaxRL), Isaac
Gym/Lab, MuJoCo Playground or EnvPool (fast C++ CPU envs). With thousands of environments
stepping on the device, the rollout batch is large and the GPU is fully used. SB3's numpy-based `VecEnv`
interface rules this out by design, so this is listed as a follow-up, not part of this repo.
