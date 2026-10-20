import gymnasium as gym
import numpy as np
import pytest

from rl_experiments.envs.factory import make_env
from rl_experiments.envs.wrappers import MaskVelocityWrapper


@pytest.mark.parametrize(("env_id", "full_dim", "masked_dim"), [("CartPole-v1", 4, 2), ("Pendulum-v1", 3, 2)])
def test_mask_velocity_removes_components(env_id: str, full_dim: int, masked_dim: int) -> None:
    env = gym.make(env_id)
    masked = MaskVelocityWrapper(gym.make(env_id))
    assert env.observation_space.shape == (full_dim,)
    assert masked.observation_space.shape == (masked_dim,)

    full_obs, _ = env.reset(seed=3)
    masked_obs, _ = masked.reset(seed=3)
    assert masked_obs.shape == (masked_dim,)
    assert masked.observation_space.contains(masked_obs)
    keep = [i for i in range(full_dim) if i not in {"CartPole-v1": (1, 3), "Pendulum-v1": (2,)}[env_id]]
    np.testing.assert_allclose(masked_obs, full_obs[keep])


def test_mask_velocity_rejects_unknown_env() -> None:
    with pytest.raises(ValueError, match="No velocity indices"):
        MaskVelocityWrapper(gym.make("FrozenLake-v1"))


def test_make_env_frame_stack_shape() -> None:
    venv = make_env("CartPole-v1", n_envs=3, seed=0, mask_velocity=True, frame_stack=4)
    obs = venv.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (3, 2 * 4)
    venv.close()
