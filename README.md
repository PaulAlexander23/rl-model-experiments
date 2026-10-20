# RL model experiments

PPO experiments with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) and sb3-contrib:

1. **Shared trunk vs split actor-critic** MLPs.
2. **Memory architectures** (frame stack, LSTM, GRU, windowed transformer; shared vs separate RNNs) on
   velocity-masked POMDP versions of classic-control tasks.
3. **CPU vs GPU vs hybrid** (collect rollouts on CPU, train on GPU) via an SB3 callback.

* Background, rationale and experiment design: [docs/background.md](docs/background.md)
* Results and conclusions: [docs/findings.md](docs/findings.md)
* Generated tables/figures: [docs/results/](docs/results/)

## Setup

Requires [uv](https://docs.astral.sh/uv/). PyTorch comes from the CUDA 12.6 wheel index, the newest
that still includes Pascal (GTX 10xx, sm_61) kernels. On a newer GPU you can switch the index in
`pyproject.toml` to a newer CUDA build.

```bash
make install          # uv sync + pre-commit install
make check            # ruff + pyright + pytest (with coverage)
```

## Running

Everything is configured with [Hydra](https://hydra.cc) from `conf/`:

```
conf/
  config.yaml          # seed, threads, eval episodes, output dirs
  env/                 # cartpole, acrobot, pendulum, lunarlander, cartpole_novel, pendulum_novel
  model/               # split_mlp(_256/_1024), shared_mlp, framestack_mlp, transformer,
                       # lstm, gru, lstm_shared, lstm_shared_detached, gru_shared
  device/              # cpu, cuda, hybrid_move, hybrid_mirror
  experiment/          # sweep presets: e1_shared_vs_split, e2_memory, e3_device_mlp, e3_device_recurrent
```

```bash
# single run -> runs/adhoc/<timestamp>_.../{metrics.json,progress.csv}
uv run rl-train env=lunarlander model=shared_mlp device=hybrid_mirror seed=3

# a full sweep -> multirun/<experiment>/<env>/<model>/<device>/s<seed>/
uv run rl-train -m experiment=e1_shared_vs_split

# tables + figures -> docs/results/<experiment>.md
uv run python scripts/aggregate.py e1_shared_vs_split
```

Rollouts on CPU with updates on GPU, in your own SB3 code:

```python
from rl_experiments.callbacks.device import HybridDeviceCallback

model = PPO("MlpPolicy", env, device="cuda")
model.learn(1_000_000, callback=HybridDeviceCallback("cpu", "cuda", strategy="mirror"))
```

## Code layout

| Path | Contents |
|------|----------|
| `src/rl_experiments/envs/` | `MaskVelocityWrapper` (POMDP), vec-env factory |
| `src/rl_experiments/policies/extractors.py` | `MlpTrunkExtractor` (shared trunk), `TransformerWindowExtractor` |
| `src/rl_experiments/policies/recurrent.py` | `RecurrentActorCriticPolicyV2`: GRU support, truly shared RNN, fast sequence path |
| `src/rl_experiments/callbacks/device.py` | `HybridDeviceCallback` (move / mirror), `PhaseTimerCallback` |
| `src/rl_experiments/builder.py` | config → SB3 algorithm |
| `src/rl_experiments/train.py` | Hydra entry point (`rl-train`) |
| `scripts/aggregate.py` | sweep results → markdown tables + figures |

## Tooling

| Tool | Role |
|------|------|
| uv | environment + lockfile (`uv.lock`) |
| ruff | lint (pyflakes, bugbear, pyupgrade, isort, pytest-style, …) + formatting |
| pyright | static type checking (`standard` mode) |
| pytest + pytest-cov | tests; coverage gate at 85 % (`pyproject.toml`) |
| pre-commit | runs ruff, pyright, codespell, uv-lock check and file hygiene hooks on every commit |
| GitHub Actions | `.github/workflows/ci.yml` runs the same checks (GPU tests skip without CUDA) |
| Hydra | configuration + sweeps (joblib launcher for parallel runs) |
