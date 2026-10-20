"""Feature extractors.

In SB3 the *features extractor* is the only part of `ActorCriticPolicy` that can be shared between
actor and critic (`share_features_extractor=True`). `net_arch` always builds *separate* pi/vf MLPs.
So a genuinely shared trunk is: `MlpTrunkExtractor` + `share_features_extractor=True` + `net_arch=[]`
(the heads become single linear layers on top of the shared trunk).
"""

from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}


def _flat_dim(observation_space: gym.Space) -> int:
    assert isinstance(observation_space, gym.spaces.Box)
    return int(th.tensor(observation_space.shape).prod())


class MlpTrunkExtractor(BaseFeaturesExtractor):
    """Plain MLP trunk: Flatten -> [Linear -> act] * n."""

    def __init__(
        self, observation_space: gym.Space, hidden_sizes: tuple[int, ...] = (64, 64), activation: str = "tanh"
    ) -> None:
        assert hidden_sizes, "MlpTrunkExtractor needs at least one hidden layer"
        super().__init__(observation_space, features_dim=hidden_sizes[-1])
        layers: list[nn.Module] = [nn.Flatten()]
        in_dim = _flat_dim(observation_space)
        for size in hidden_sizes:
            layers += [nn.Linear(in_dim, size), ACTIVATIONS[activation]()]
            in_dim = size
        self.net = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


class TransformerWindowExtractor(BaseFeaturesExtractor):
    """Transformer encoder over a sliding window of the last `n_stack` observations.

    Expects observations from `VecFrameStack` (frames concatenated on the last axis, oldest first).
    This gives the agent memory without recurrent state, so it works with vanilla PPO; the price is a
    fixed context length. The representation of the newest frame is returned.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        n_stack: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim=d_model)
        total = _flat_dim(observation_space)
        assert total % n_stack == 0, f"obs dim {total} not divisible by n_stack={n_stack}"
        self.n_stack = n_stack
        self.frame_dim = total // n_stack
        self.embed = nn.Linear(self.frame_dim, d_model)
        self.pos = nn.Parameter(th.zeros(1, n_stack, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=dim_feedforward, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        frames = observations.reshape(-1, self.n_stack, self.frame_dim)
        tokens = self.embed(frames) + self.pos
        return self.norm(self.encoder(tokens)[:, -1])
