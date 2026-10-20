"""Vectorised environment construction."""

from __future__ import annotations

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack

from rl_experiments.envs.wrappers import MaskVelocityWrapper


def make_env(
    env_id: str,
    n_envs: int,
    seed: int,
    *,
    mask_velocity: bool = False,
    frame_stack: int = 0,
    subproc: bool = False,
) -> VecEnv:
    """Create a (optionally partially observable and frame-stacked) vectorised env.

    `DummyVecEnv` is the default because classic-control steps are microseconds; process
    pipes would dominate the step time.
    """
    venv = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=MaskVelocityWrapper if mask_velocity else None,
        vec_env_cls=SubprocVecEnv if subproc else DummyVecEnv,
    )
    if frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)
    return venv
