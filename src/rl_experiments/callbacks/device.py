"""Collect rollouts on one device (CPU) and run the PPO update on another (GPU).

How SB3 uses devices in an on-policy iteration:

1. `collect_rollouts` converts observations with `obs_as_tensor(obs, model.device)` and runs the policy.
   Everything stored in the rollout buffer is numpy, so the buffer itself is device-agnostic while filling.
2. `callback.on_rollout_end()` fires *after* GAE has been computed, right before `train()`.
3. `train()` iterates `rollout_buffer.get()`, which converts minibatches with `rollout_buffer.device`.

So switching `policy`, its optimizer state, `model.device` and `rollout_buffer.device` in
`on_rollout_start` / `on_rollout_end` is enough. Two strategies are provided:

* ``move``:   move the single policy + Adam moments back and forth every iteration.
* ``mirror``: keep a CPU copy of the policy used only for acting; before each rollout copy the GPU
              weights into it (one-way, params only). The optimizer never leaves the GPU.
"""

from __future__ import annotations

import time
from typing import Literal

import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy

Strategy = Literal["move", "mirror"]


def move_optimizer_state(optimizer: th.optim.Optimizer, device: th.device) -> None:
    """Move optimizer state tensors (e.g. Adam moments) to `device`.

    `step` is left alone: non-capturable Adam expects it on the CPU (mirrors `Optimizer.load_state_dict`).
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, th.Tensor) and key != "step":
                state[key] = value.to(device)


class HybridDeviceCallback(BaseCallback):
    def __init__(
        self,
        rollout_device: str | th.device = "cpu",
        train_device: str | th.device = "cuda",
        strategy: Strategy = "move",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.rollout_device = th.device(rollout_device)
        self.train_device = th.device(train_device)
        self.strategy = strategy
        self._train_policy: BasePolicy | None = None
        self._rollout_policy: BasePolicy | None = None
        self.transfer_time = 0.0

    def _on_training_start(self) -> None:
        if self.strategy == "mirror":
            self._train_policy = self.model.policy.to(self.train_device)
            move_optimizer_state(self._train_policy.optimizer, self.train_device)
            # Build a second policy exactly as the algorithm's _setup_model does. (deepcopy fails with gSDE,
            # whose rsampled exploration matrix is a non-leaf tensor.)
            model = self.model
            self._rollout_policy = model.policy_class(  # type: ignore[call-arg]
                model.observation_space,
                model.action_space,
                model.lr_schedule,
                use_sde=model.use_sde,
                **model.policy_kwargs,
            ).to(self.rollout_device)

    def _on_rollout_start(self) -> None:
        start = time.perf_counter()
        if self.strategy == "move":
            self._move_policy(self.rollout_device)
        else:
            assert self._train_policy is not None
            assert self._rollout_policy is not None
            self._rollout_policy.load_state_dict(self._train_policy.state_dict())
            self.model.policy = self._rollout_policy
        self._set_device(self.rollout_device)
        # SB3 samples the gSDE exploration matrix *before* on_rollout_start, i.e. on the old device/policy.
        if getattr(self.model, "use_sde", False):
            self.model.policy.reset_noise(self.training_env.num_envs)  # type: ignore[operator]
        self.transfer_time += time.perf_counter() - start

    def _on_rollout_end(self) -> None:
        start = time.perf_counter()
        if self.strategy == "move":
            self._move_policy(self.train_device)
        else:
            assert self._train_policy is not None
            self.model.policy = self._train_policy
        self._set_device(self.train_device)
        self.transfer_time += time.perf_counter() - start

    def _on_step(self) -> bool:
        return True

    def _move_policy(self, device: th.device) -> None:
        policy = self.model.policy.to(device)
        move_optimizer_state(policy.optimizer, device)

    def _set_device(self, device: th.device) -> None:
        model = self.model
        model.device = device
        rollout_buffer = getattr(model, "rollout_buffer", None)
        if rollout_buffer is not None:
            rollout_buffer.device = device
        # RecurrentPPO keeps the last hidden states as tensors between rollouts.
        states = getattr(model, "_last_lstm_states", None)
        if states is not None:
            model._last_lstm_states = type(states)(  # type: ignore[attr-defined]
                *(tuple(t.to(device) for t in pair) for pair in states)
            )


class PhaseTimerCallback(BaseCallback):
    """Accumulate wall-clock time spent collecting rollouts vs. in the gradient update.

    Place it *first* in a `CallbackList` so device transfers done by later callbacks are attributed
    to the phase that follows them.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.rollout_time = 0.0
        self.train_time = 0.0
        self._mark: float | None = None

    def _now(self) -> float:
        if th.cuda.is_available() and th.cuda.is_initialized():
            th.cuda.synchronize()
        return time.perf_counter()

    def _on_rollout_start(self) -> None:
        now = self._now()
        if self._mark is not None:
            self.train_time += now - self._mark
        self._mark = now

    def _on_rollout_end(self) -> None:
        now = self._now()
        assert self._mark is not None
        self.rollout_time += now - self._mark
        self._mark = now

    def _on_training_end(self) -> None:
        if self._mark is not None:
            self.train_time += self._now() - self._mark
            self._mark = None

    def _on_step(self) -> bool:
        return True
