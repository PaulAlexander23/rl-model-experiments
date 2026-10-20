"""Turn a (Hydra) config into an SB3 algorithm instance."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

from rl_experiments.policies.extractors import ACTIVATIONS, MlpTrunkExtractor, TransformerWindowExtractor
from rl_experiments.policies.recurrent import RecurrentActorCriticPolicyV2

ALGOS: dict[str, type[OnPolicyAlgorithm]] = {"ppo": PPO, "recurrent_ppo": RecurrentPPO}
POLICIES: dict[str, str | type] = {
    "MlpPolicy": "MlpPolicy",
    "RecurrentPolicy": RecurrentActorCriticPolicyV2,
}
EXTRACTORS: dict[str, type[nn.Module]] = {
    "mlp_trunk": MlpTrunkExtractor,
    "transformer": TransformerWindowExtractor,
}


def linear_schedule(initial: float) -> Callable[[float], float]:
    """Linearly anneal from `initial` to 0 (SB3 passes progress_remaining: 1 -> 0)."""

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial

    return schedule


def parse_schedule(value: float | str) -> float | Callable[[float], float]:
    """Accept rl-zoo style values: a float, or `"lin_<float>"` for linear annealing."""
    if isinstance(value, str):
        if not value.startswith("lin_"):
            raise ValueError(f"Unknown schedule {value!r}")
        return linear_schedule(float(value.removeprefix("lin_")))
    return float(value)


def build_policy_kwargs(policy_kwargs: Mapping[str, Any], frame_stack: int) -> dict[str, Any]:
    kwargs = dict(policy_kwargs)
    if "activation_fn" in kwargs:
        kwargs["activation_fn"] = ACTIVATIONS[kwargs["activation_fn"]]
    if "net_arch" in kwargs and isinstance(kwargs["net_arch"], Mapping):
        kwargs["net_arch"] = {k: list(v) for k, v in kwargs["net_arch"].items()}
    extractor = kwargs.get("features_extractor_class")
    if extractor is not None:
        kwargs["features_extractor_class"] = EXTRACTORS[extractor]
        fe_kwargs = dict(kwargs.get("features_extractor_kwargs") or {})
        if "hidden_sizes" in fe_kwargs:
            fe_kwargs["hidden_sizes"] = tuple(fe_kwargs["hidden_sizes"])
        if extractor == "transformer":
            assert frame_stack > 1, "the transformer extractor needs model.frame_stack > 1"
            fe_kwargs["n_stack"] = frame_stack
        kwargs["features_extractor_kwargs"] = fe_kwargs
    return kwargs


def build_model(
    algo: str,
    policy: str,
    env: VecEnv,
    *,
    hyperparams: Mapping[str, Any],
    policy_kwargs: Mapping[str, Any],
    frame_stack: int = 0,
    device: str = "cpu",
    seed: int | None = None,
) -> OnPolicyAlgorithm:
    params = dict(hyperparams)
    for key in ("learning_rate", "clip_range"):
        if key in params:
            params[key] = parse_schedule(params[key])
    return ALGOS[algo](
        POLICIES[policy],  # type: ignore[arg-type]
        env,
        policy_kwargs=build_policy_kwargs(policy_kwargs, frame_stack),
        device=device,
        seed=seed,
        verbose=0,
        **params,
    )
