"""Observation wrappers used to turn standard benchmarks into partially observable (POMDP) tasks."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

# Indices of velocity-like components in each environment's observation vector.
# Removing them forces the agent to infer velocities from history, i.e. it needs memory.
VELOCITY_INDICES: dict[str, tuple[int, ...]] = {
    "CartPole-v1": (1, 3),  # cart velocity, pole angular velocity
    "Pendulum-v1": (2,),  # angular velocity
    "Acrobot-v1": (4, 5),  # joint angular velocities
    "MountainCar-v0": (1,),
    "MountainCarContinuous-v0": (1,),
    "LunarLander-v3": (2, 3, 5),  # linear velocities, angular velocity
}


class MaskVelocityWrapper(gym.ObservationWrapper[np.ndarray, Any, np.ndarray]):
    """Drop the velocity components of a Box observation.

    The components are removed (not zeroed) so the observation space honestly reflects what the agent sees.
    """

    def __init__(self, env: gym.Env[np.ndarray, Any]) -> None:
        super().__init__(env)
        assert env.spec is not None, "MaskVelocityWrapper needs an env created with gym.make"
        env_id = env.spec.id
        if env_id not in VELOCITY_INDICES:
            raise ValueError(f"No velocity indices registered for {env_id!r}")
        space = env.observation_space
        assert isinstance(space, gym.spaces.Box), "MaskVelocityWrapper only supports Box observations"

        self._keep = np.array([i for i in range(space.shape[0]) if i not in VELOCITY_INDICES[env_id]])
        self.observation_space = gym.spaces.Box(
            low=space.low[self._keep], high=space.high[self._keep], dtype=space.dtype
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[self._keep]
