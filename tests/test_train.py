import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from rl_experiments.train import CONFIG_DIR, run

MODELS = ["split_mlp", "shared_mlp", "framestack_mlp", "transformer", "lstm", "gru", "lstm_shared", "gru_shared"]


@pytest.mark.parametrize("model", MODELS)
def test_every_model_config_trains(model: str, tmp_path: Path) -> None:
    env = "cartpole" if model in {"split_mlp", "shared_mlp"} else "cartpole_novel"
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        cfg = compose(
            "config",
            overrides=[f"model={model}", f"env={env}", "env.total_timesteps=256", "env.n_envs=2", "eval_episodes=2"],
        )
    metrics = run(cfg, tmp_path)
    assert metrics["model"] == model
    assert metrics["total_timesteps"] >= 256
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["eval_mean"] == metrics["eval_mean"]
    assert (tmp_path / "progress.csv").exists()


@pytest.mark.parametrize("env", ["cartpole", "acrobot", "pendulum", "lunarlander", "cartpole_novel", "pendulum_novel"])
def test_every_env_config_composes(env: str) -> None:
    with initialize_config_dir(str(CONFIG_DIR), version_base=None):
        cfg = compose("config", overrides=[f"env={env}"])
    assert cfg.env.total_timesteps > 0
    assert "n_steps" in cfg.env.ppo
