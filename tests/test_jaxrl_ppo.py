import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from rl_experiments.jaxrl.models import ActorCritic, make_core
from rl_experiments.jaxrl.ppo import PPOConfig, count_params, make_train
from rl_experiments.jaxrl.run import CONFIG_DIR, run
from rl_experiments.jaxrl.tasks import make_task

TINY = PPOConfig(
    total_timesteps=60_000,
    n_envs=16,
    n_steps=32,
    seq_len=16,
    n_minibatches=4,
    learning_rate=3e-3,
    ent_coef=0.0,
    eval_episodes=32,
)


def test_ppo_learns_fully_observed_regime() -> None:
    """At reliability 1 the hidden regime is observed: even an MLP should learn to name it."""
    task = make_task("hidden_regime", length=16)
    network = ActorCritic(make_core("mlp", task.obs_size, width=16), task.n_actions)
    out = jax.jit(make_train(task, network, TINY))(jax.random.key(0), jnp.float32(1.0))
    assert out["train_return"].shape == (TINY.n_updates,)
    assert np.isfinite(np.asarray(out["policy_loss"])).all()
    assert float(out["eval_return"]) > 0.9
    assert float(out["optimal_return"]) == pytest.approx(1.0)


@pytest.mark.parametrize("core", ["lru", "transformer"])
def test_ppo_vmaps_over_seeds_and_levels(core: str) -> None:
    task = make_task("delayed_cue", max_delay=4)
    kwargs = {"context": 4} if core == "transformer" else {}
    network = ActorCritic(make_core(core, task.obs_size, width=8, **kwargs), task.n_actions)
    config = TINY._replace(total_timesteps=2 * 16 * 32)
    train = jax.jit(jax.vmap(jax.vmap(make_train(task, network, config), (0, None)), (None, 0)))
    out = train(jax.random.split(jax.random.key(0), 2), jnp.array([1.0, 3.0]))
    assert out["eval_return"].shape == (2, 2)
    assert np.isfinite(np.asarray(out["value_loss"])).all()


def test_invalid_chunking() -> None:
    task = make_task("modular_sum")
    network = ActorCritic(make_core("gru", task.obs_size, width=8), task.n_actions)
    with pytest.raises(ValueError, match="must divide"):
        make_train(task, network, TINY._replace(seq_len=10))
    with pytest.raises(ValueError, match="minibatches"):
        make_train(task, network, TINY._replace(n_minibatches=3))


def test_count_params() -> None:
    task = make_task("modular_sum")
    network = ActorCritic(make_core("gru", task.obs_size, width=8), task.n_actions)
    counts = count_params(network, task)
    assert 0 < counts["n_core_params"] < counts["n_params"]


def test_run_writes_metrics(tmp_path: Path) -> None:
    overrides = [
        "task=delayed_cue",
        "task.kwargs.max_delay=4",
        "task.levels=[1,2]",
        "task.total_timesteps=1024",
        "model=transformer",
        "model.width=8",
        "model.memory=4",
        "n_seeds=2",
        "ppo.n_envs=8",
        "ppo.n_steps=32",
        "ppo.seq_len=16",
        "ppo.n_minibatches=2",
        "ppo.eval_episodes=8",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=overrides)
    metrics = run(cfg, tmp_path)
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["score"] == metrics["score"]
    assert np.asarray(saved["score"]).shape == (2, 2)
    assert np.asarray(saved["train_return"]).shape[:2] == (2, 2)
    assert saved["model_kwargs"] == {"context": 4}
