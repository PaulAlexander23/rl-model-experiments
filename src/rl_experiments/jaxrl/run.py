"""Hydra entry point: train one (task, model) configuration at every level and seed in one launch.

    uv run rl-train-jax task=delayed_cue model=gru model.width=32
    uv run rl-train-jax -m experiment=m1_width_associative_recall

Seeds and levels are `vmap`-ped, so one job is one compilation. Hydra's basic launcher runs the
jobs of a sweep one after another in this process, which keeps the GPU to one job at a time.
Writes `metrics.json`: per level and seed the greedy evaluation return, the score normalised to
random = 0 and optimal = 1, and the training-return curve.
"""

from __future__ import annotations

import os

# Allocate GPU memory on demand rather than 75% up front, so a test run or a second process can coexist.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import time
from pathlib import Path
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from rl_experiments.jaxrl.models import ActorCritic, make_core
from rl_experiments.jaxrl.ppo import PPOConfig, count_params, make_train
from rl_experiments.jaxrl.tasks import make_task

CONFIG_DIR = Path(__file__).resolve().parents[3] / "conf_jax"
CURVE_POINTS = 100


def _container(node: Any) -> dict[str, Any]:
    return OmegaConf.to_container(node, resolve=True) if node is not None else {}  # type: ignore[return-value]


def build(cfg: DictConfig) -> tuple[Any, ActorCritic, PPOConfig]:
    task = make_task(cfg.task.name, **_container(cfg.task.kwargs))
    core_kwargs = {"width": cfg.model.width, "depth": cfg.model.depth, **_container(cfg.model.kwargs)}
    network = ActorCritic(make_core(cfg.model.core, task.obs_size, **core_kwargs), task.n_actions)
    return task, network, PPOConfig(**_container(cfg.ppo))


def run(cfg: DictConfig, output_dir: Path) -> dict[str, Any]:
    task, network, ppo = build(cfg)
    levels = jnp.asarray(list(cfg.task.levels), jnp.float32)
    keys = jax.random.split(jax.random.key(cfg.seed), cfg.n_seeds)
    train = jax.jit(jax.vmap(jax.vmap(make_train(task, network, ppo), (0, None)), (None, 0)))

    start = time.perf_counter()
    compiled = train.lower(keys, levels).compile()
    compile_time = time.perf_counter() - start
    start = time.perf_counter()
    out = jax.tree.map(np.asarray, jax.block_until_ready(compiled(keys, levels)))
    run_time = time.perf_counter() - start

    score = (out["eval_return"] - out["random_return"]) / (out["optimal_return"] - out["random_return"])
    stride = max(ppo.n_updates // CURVE_POINTS, 1)
    steps_per_update = ppo.n_envs * ppo.n_steps
    n_runs = len(levels) * cfg.n_seeds
    metrics = {
        "experiment": cfg.experiment,
        "task": cfg.task.name,
        "model": cfg.model.name,
        "core": cfg.model.core,
        "width": cfg.model.width,
        "depth": cfg.model.depth,
        "model_kwargs": _container(cfg.model.kwargs),
        "ppo": ppo._asdict(),
        **count_params(network, task),
        "levels": [float(x) for x in levels],
        "n_seeds": cfg.n_seeds,
        "eval_return": out["eval_return"].tolist(),  # [level][seed]
        "score": score.tolist(),
        "random_return": out["random_return"][:, 0].tolist(),
        "optimal_return": out["optimal_return"][:, 0].tolist(),
        "curve_steps": [(i + 1) * steps_per_update for i in range(0, ppo.n_updates, stride)],
        "train_return": out["train_return"][:, :, ::stride].tolist(),  # [level][seed][point]
        "compile_time": compile_time,
        "run_time": run_time,
        "steps_per_second": ppo.n_updates * steps_per_update * n_runs / run_time,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> float:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    metrics = run(cfg, output_dir)
    mean_scores = np.mean(metrics["score"], axis=1)
    print(
        f"{cfg.task.name} {cfg.model.name} w{cfg.model.width} d{cfg.model.depth}: "
        f"scores by level {np.round(mean_scores, 2).tolist()} "
        f"({metrics['run_time']:.0f}s, {metrics['steps_per_second'] / 1e3:.0f}k steps/s)",
        flush=True,
    )
    return float(np.mean(mean_scores))


if __name__ == "__main__":
    main()
