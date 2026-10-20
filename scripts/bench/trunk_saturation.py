"""Mechanism check for E1/E4: do unnormalised returns saturate the shared tanh trunk?

Trains shared/split MLPs on Acrobot for a short budget, with and without reward normalisation,
then reports the fraction of hidden tanh units with |activation| > 0.99 on states visited by random actions.

    uv run python scripts/bench/trunk_saturation.py
"""

import numpy as np
import torch as th
from torch import nn

from rl_experiments.builder import build_model
from rl_experiments.envs.factory import make_env

ACROBOT_PPO = {"n_steps": 256, "batch_size": 256, "gae_lambda": 0.94, "gamma": 0.99, "n_epochs": 4, "ent_coef": 0.0}
SHARED = {
    "features_extractor_class": "mlp_trunk",
    "features_extractor_kwargs": {"hidden_sizes": [64, 64]},
    "share_features_extractor": True,
    "net_arch": [],
}
SPLIT = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}


def saturation(policy: nn.Module, obs: th.Tensor) -> list[float]:
    """Fraction of saturated units per tanh layer (all tanh layers in the policy, in module order)."""
    fractions: list[float] = []

    def hook(_module: nn.Module, _inp: object, out: th.Tensor) -> None:
        fractions.append(float((out.abs() > 0.99).float().mean()))

    handles = [m.register_forward_hook(hook) for m in policy.modules() if isinstance(m, nn.Tanh)]
    with th.no_grad():
        policy.get_distribution(obs)  # type: ignore[operator]
        policy.predict_values(obs)  # type: ignore[operator]
    for handle in handles:
        handle.remove()
    return fractions


def main() -> None:
    th.set_num_threads(2)
    for name, policy_kwargs in (("shared", SHARED), ("split", SPLIT)):
        for norm_reward in (False, True):
            env = make_env("Acrobot-v1", 16, 0, norm_reward=norm_reward)
            model = build_model("ppo", "MlpPolicy", env, hyperparams=ACROBOT_PPO, policy_kwargs=policy_kwargs, seed=0)
            model.learn(60_000)
            batches = []
            for _ in range(100):
                obs, *_ = env.step(np.array([env.action_space.sample()] * 16))
                assert isinstance(obs, np.ndarray)
                batches.append(obs)
            fractions = saturation(model.policy, th.as_tensor(np.concatenate(batches), dtype=th.float32))
            assert model.ep_info_buffer is not None
            returns = [ep["r"] for ep in model.ep_info_buffer]
            print(
                f"{name:6s} norm_reward={norm_reward!s:5s} train return {np.mean(returns):7.1f}  "
                f"saturated units per tanh layer: {', '.join(f'{f:.0%}' for f in fractions)}"
            )


if __name__ == "__main__":
    main()
