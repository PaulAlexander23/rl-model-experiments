"""Vectorised environment construction."""

from __future__ import annotations

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize

from rl_experiments.envs.wrappers import MaskVelocityWrapper


def make_env(
    env_id: str,
    n_envs: int,
    seed: int,
    *,
    mask_velocity: bool = False,
    frame_stack: int = 0,
    norm_obs: bool = False,
    norm_reward: bool = False,
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
    if norm_obs or norm_reward:
        venv = VecNormalize(venv, norm_obs=norm_obs, norm_reward=norm_reward)
    return venv


def make_eval_env(env_id: str, seed: int, train_env: VecEnv, **kwargs: bool | int) -> VecEnv:
    """Single-env copy of `train_env` for evaluation: raw rewards, frozen observation statistics (if any)."""
    norm_obs = isinstance(train_env, VecNormalize) and train_env.norm_obs
    eval_env = make_env(env_id, 1, seed, **{**kwargs, "norm_obs": norm_obs, "norm_reward": False})  # type: ignore[arg-type]
    if norm_obs:
        assert isinstance(train_env, VecNormalize)
        assert isinstance(eval_env, VecNormalize)
        eval_env.training = False
        eval_env.obs_rms = train_env.obs_rms
    return eval_env
