"""Synthetic partially observable tasks, one per kind of partial observability.

Each task has one difficulty dial, passed as a traced scalar `level` so that a single compiled
training program can be `vmap`-ped over levels. Its static settings fix the array shapes for every
level in a sweep (e.g. `max_delay` for `DelayedCue`).

| Task | Kind of partial observability | Dial (`level`) |
|------|-------------------------------|----------------|
| `NoisyCartPole` | missing derivatives + observation noise (filtering) | noise std on (x, theta) |
| `DelayedCue` | memory length (passive T-maze) | steps between cue and query |
| `AssociativeRecall` | memory capacity (key-value store) | number of key-value pairs |
| `ModularSum` | computation over history (state tracking) | modulus of the running sum |
| `HiddenRegime` | belief over a hidden switching state | observation reliability |

All tasks share the interface

    state, obs = task.reset(key, level)
    state, obs, reward, done = task.step(key, state, action, level)

and `task.oracle=True` adds the hidden sufficient statistic to the observation (an MDP upper bound).
`bounds(key, level)` gives (random, optimal) expected returns for normalising scores to [0, 1].
Rewards are scaled so that an episode's return is at most 1 (CartPole: 5).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp

Array = jax.Array


class Task(Protocol):
    # Read-only members, so dataclass fields and properties both satisfy the protocol.
    @property
    def name(self) -> str: ...
    @property
    def obs_size(self) -> int: ...
    @property
    def n_actions(self) -> int: ...
    @property
    def max_steps(self) -> int: ...  # longest episode at any level of the sweep: sizes the evaluation loop
    @property
    def oracle(self) -> bool: ...

    def reset(self, key: Array, level: Array) -> tuple[Any, Array]: ...

    def step(self, key: Array, state: Any, action: Array, level: Array) -> tuple[Any, Array, Array, Array]: ...

    def bounds(self, key: Array, level: Array) -> tuple[Array, Array]: ...


def _one_hot(index: Array, n: int) -> Array:
    return jax.nn.one_hot(index, n, dtype=jnp.float32)


# -- 1+2. Missing derivatives and noise ---------------------------------------------------------------


class CartPoleState(NamedTuple):
    physics: Array  # (4,) x, x_dot, theta, theta_dot
    t: Array


@dataclass(frozen=True)
class NoisyCartPole:
    """CartPole-v1 observing only (x, theta), each with Gaussian noise of std `level` (in units of the
    failure thresholds). At level 0 this is velocity-masked CartPole: a short window of frames suffices.
    With noise the agent must also filter: estimate position and velocity from many noisy readings."""

    oracle: bool = False
    name: str = "noisy_cartpole"
    n_actions: int = 2
    max_steps: int = 500
    reward_scale: float = 0.01

    # Gymnasium's CartPole-v1 constants.
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_mag: float = 10.0
    tau: float = 0.02
    theta_threshold: float = 12 * 2 * math.pi / 360
    x_threshold: float = 2.4

    @property
    def obs_size(self) -> int:
        return 4 if self.oracle else 2

    def _observe(self, key: Array, physics: Array, level: Array) -> Array:
        if self.oracle:
            return physics / jnp.array([self.x_threshold, 2.0, self.theta_threshold, 2.0])
        clean = physics[jnp.array([0, 2])] / jnp.array([self.x_threshold, self.theta_threshold])
        return clean + level * jax.random.normal(key, (2,))

    def reset(self, key: Array, level: Array) -> tuple[CartPoleState, Array]:
        k_state, k_obs = jax.random.split(key)
        physics = jax.random.uniform(k_state, (4,), minval=-0.05, maxval=0.05)
        return CartPoleState(physics, jnp.int32(0)), self._observe(k_obs, physics, level)

    def step(
        self, key: Array, state: CartPoleState, action: Array, level: Array
    ) -> tuple[CartPoleState, Array, Array, Array]:
        x, x_dot, theta, theta_dot = state.physics
        force = jnp.where(action == 1, self.force_mag, -self.force_mag)
        cos, sin = jnp.cos(theta), jnp.sin(theta)
        total_mass = self.masscart + self.masspole
        polemass_length = self.masspole * self.length
        temp = (force + polemass_length * theta_dot**2 * sin) / total_mass
        theta_acc = (self.gravity * sin - cos * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos**2 / total_mass)
        )
        x_acc = temp - polemass_length * theta_acc * cos / total_mass
        physics = jnp.array(
            [
                x + self.tau * x_dot,
                x_dot + self.tau * x_acc,
                theta + self.tau * theta_dot,
                theta_dot + self.tau * theta_acc,
            ]
        )
        t = state.t + 1
        fallen = (jnp.abs(physics[0]) > self.x_threshold) | (jnp.abs(physics[2]) > self.theta_threshold)
        # The time limit counts as termination (no bootstrap), as in gymnax/PureJaxRL.
        done = fallen | (t >= self.max_steps)
        obs = self._observe(key, physics, level)
        return CartPoleState(physics, t), obs, jnp.float32(self.reward_scale), done

    def bounds(self, key: Array, level: Array) -> tuple[Array, Array]:
        # Uniform random play lasts ~22 steps; the optimum is the time limit.
        return jnp.float32(22.0 * self.reward_scale), jnp.float32(self.max_steps * self.reward_scale)


# -- 3. Memory length ---------------------------------------------------------------------------------


class CueState(NamedTuple):
    cue: Array
    delay: Array
    t: Array


@dataclass(frozen=True)
class DelayedCue:
    """Passive T-maze (Ni et al. 2023): a cue (one of `n_cues`) is shown at t=0, then `level` steps of
    random distractors, then a query; the reward is 1 if the action at the query names the cue.

    Credit assignment is immediate (the rewarded action is the last), so only memory *length* varies.
    Observation: cue one-hot (zero after t=0), query flag, distractor one-hot."""

    max_delay: int = 512
    n_cues: int = 4
    n_distractors: int = 4
    oracle: bool = False
    name: str = "delayed_cue"

    @property
    def obs_size(self) -> int:
        return self.n_cues + 1 + self.n_distractors

    @property
    def n_actions(self) -> int:
        return self.n_cues

    @property
    def max_steps(self) -> int:
        return self.max_delay + 1

    def _observe(self, key: Array, state: CueState) -> Array:
        show = (state.t == 0) | self.oracle
        cue = _one_hot(state.cue, self.n_cues) * show
        query = (state.t == state.delay).astype(jnp.float32)[None]
        distractor = _one_hot(jax.random.randint(key, (), 0, self.n_distractors), self.n_distractors)
        return jnp.concatenate([cue, query, distractor])

    def reset(self, key: Array, level: Array) -> tuple[CueState, Array]:
        k_cue, k_obs = jax.random.split(key)
        delay = jnp.round(level).astype(jnp.int32)
        state = CueState(jax.random.randint(k_cue, (), 0, self.n_cues), delay, jnp.int32(0))
        return state, self._observe(k_obs, state)

    def step(self, key: Array, state: CueState, action: Array, level: Array) -> tuple[CueState, Array, Array, Array]:
        done = state.t >= state.delay
        reward = (done & (action == state.cue)).astype(jnp.float32)
        state = state._replace(t=state.t + 1)
        return state, self._observe(key, state), reward, done

    def bounds(self, key: Array, level: Array) -> tuple[Array, Array]:
        return jnp.float32(1.0 / self.n_cues), jnp.float32(1.0)


# -- 4. Memory capacity -------------------------------------------------------------------------------


class RecallState(NamedTuple):
    keys: Array  # (max_items,) a permutation of the key vocabulary; the first n_items are used
    values: Array  # (max_items,) value of each key in `keys`
    queries: Array  # (n_queries,) indices into keys[:n_items]
    n_items: Array
    t: Array


@dataclass(frozen=True)
class AssociativeRecall:
    """Key-value recall: `level` (key, value) pairs are shown one per step, then `delay` blank steps,
    then `n_queries` steps each showing a key; reward 1/n_queries for answering its value.

    Keys are drawn without replacement from `max_items`, values from `n_values`, so the agent must
    store `level * log2(n_values)` bits. Observation: key one-hot, value one-hot, query flag."""

    max_items: int = 16
    n_values: int = 4
    delay: int = 4
    n_queries: int = 8
    oracle: bool = False
    name: str = "associative_recall"

    @property
    def obs_size(self) -> int:
        return self.max_items + self.n_values + 1

    @property
    def n_actions(self) -> int:
        return self.n_values

    @property
    def max_steps(self) -> int:
        return self.max_items + self.delay + self.n_queries

    def _phase(self, state: RecallState) -> tuple[Array, Array]:
        """(is_query, index into keys/values shown now)."""
        first_query = state.n_items + self.delay
        is_query = state.t >= first_query
        query = state.queries[jnp.clip(state.t - first_query, 0, self.n_queries - 1)]
        return is_query, jnp.where(is_query, query, jnp.clip(state.t, 0, self.max_items - 1))

    def _observe(self, state: RecallState) -> Array:
        is_query, index = self._phase(state)
        presenting = state.t < state.n_items
        key = _one_hot(state.keys[index], self.max_items) * (presenting | is_query)
        value = _one_hot(state.values[index], self.n_values) * (presenting | (is_query & self.oracle))
        return jnp.concatenate([key, value, is_query.astype(jnp.float32)[None]])

    def reset(self, key: Array, level: Array) -> tuple[RecallState, Array]:
        k_keys, k_values, k_queries = jax.random.split(key, 3)
        n_items = jnp.round(level).astype(jnp.int32)
        state = RecallState(
            keys=jax.random.permutation(k_keys, self.max_items),
            values=jax.random.randint(k_values, (self.max_items,), 0, self.n_values),
            queries=jax.random.randint(k_queries, (self.n_queries,), 0, n_items),
            n_items=n_items,
            t=jnp.int32(0),
        )
        return state, self._observe(state)

    def step(
        self, key: Array, state: RecallState, action: Array, level: Array
    ) -> tuple[RecallState, Array, Array, Array]:
        is_query, index = self._phase(state)
        reward = (is_query & (action == state.values[index])).astype(jnp.float32) / self.n_queries
        done = state.t >= state.n_items + self.delay + self.n_queries - 1
        state = state._replace(t=state.t + 1)
        return state, self._observe(state), reward, done

    def bounds(self, key: Array, level: Array) -> tuple[Array, Array]:
        return jnp.float32(1.0 / self.n_values), jnp.float32(1.0)


# -- 5. Computation over history ----------------------------------------------------------------------


class SumState(NamedTuple):
    total: Array  # running sum mod m, including the digit on show
    digit: Array
    modulus: Array
    t: Array


@dataclass(frozen=True)
class ModularSum:
    """State tracking: each step shows a digit in [0, m); the action must be the running sum mod m
    (including that digit), for reward 1/length. m = `level` (2 is running parity).

    Little to store (log2 m bits) but a nonlinear update: a finite automaton with m states and m inputs.
    Time-invariant linear recurrences (e.g. LRU) cannot represent this update exactly over long
    sequences; gated RNNs can."""

    max_modulus: int = 8
    length: int = 32
    oracle: bool = False
    name: str = "modular_sum"

    @property
    def obs_size(self) -> int:
        return 2 * self.max_modulus if self.oracle else self.max_modulus

    @property
    def n_actions(self) -> int:
        return self.max_modulus

    @property
    def max_steps(self) -> int:
        return self.length

    def _observe(self, state: SumState) -> Array:
        digit = _one_hot(state.digit, self.max_modulus)
        if not self.oracle:
            return digit
        previous = (state.total - state.digit) % state.modulus
        return jnp.concatenate([digit, _one_hot(previous, self.max_modulus)])

    def reset(self, key: Array, level: Array) -> tuple[SumState, Array]:
        modulus = jnp.round(level).astype(jnp.int32)
        digit = jax.random.randint(key, (), 0, modulus)
        state = SumState(digit, digit, modulus, jnp.int32(0))
        return state, self._observe(state)

    def step(self, key: Array, state: SumState, action: Array, level: Array) -> tuple[SumState, Array, Array, Array]:
        reward = (action == state.total).astype(jnp.float32) / self.length
        done = state.t >= self.length - 1
        digit = jax.random.randint(key, (), 0, state.modulus)
        state = SumState((state.total + digit) % state.modulus, digit, state.modulus, state.t + 1)
        return state, self._observe(state), reward, done

    def bounds(self, key: Array, level: Array) -> tuple[Array, Array]:
        return 1.0 / jnp.round(level), jnp.float32(1.0)


# -- 6. Belief over a hidden state --------------------------------------------------------------------


class RegimeState(NamedTuple):
    regime: Array
    t: Array


@dataclass(frozen=True)
class HiddenRegime:
    """A hidden regime z in [0, n) switches to a uniformly random other regime with probability
    `hazard` per step. Each step shows one-hot(z) with probability q = `level`, else a uniformly random
    one-hot. Reward 1/length for naming z. The optimal policy acts on the HMM forward filter's belief,
    whose return is the `optimal` bound (computed by simulation)."""

    n_regimes: int = 4
    hazard: float = 0.05
    length: int = 64
    oracle: bool = False
    name: str = "hidden_regime"

    @property
    def obs_size(self) -> int:
        return 2 * self.n_regimes if self.oracle else self.n_regimes

    @property
    def n_actions(self) -> int:
        return self.n_regimes

    @property
    def max_steps(self) -> int:
        return self.length

    def _observe(self, key: Array, state: RegimeState, level: Array) -> Array:
        k_reliable, k_noise = jax.random.split(key)
        reliable = jax.random.uniform(k_reliable) < level
        shown = jnp.where(reliable, state.regime, jax.random.randint(k_noise, (), 0, self.n_regimes))
        obs = _one_hot(shown, self.n_regimes)
        if self.oracle:
            obs = jnp.concatenate([obs, _one_hot(state.regime, self.n_regimes)])
        return obs

    def reset(self, key: Array, level: Array) -> tuple[RegimeState, Array]:
        k_regime, k_obs = jax.random.split(key)
        state = RegimeState(jax.random.randint(k_regime, (), 0, self.n_regimes), jnp.int32(0))
        return state, self._observe(k_obs, state, level)

    def _transition(self, key: Array, regime: Array) -> Array:
        k_switch, k_to = jax.random.split(key)
        switch = jax.random.uniform(k_switch) < self.hazard
        other = (regime + 1 + jax.random.randint(k_to, (), 0, self.n_regimes - 1)) % self.n_regimes
        return jnp.where(switch, other, regime)

    def step(
        self, key: Array, state: RegimeState, action: Array, level: Array
    ) -> tuple[RegimeState, Array, Array, Array]:
        k_transition, k_obs = jax.random.split(key)
        reward = (action == state.regime).astype(jnp.float32) / self.length
        done = state.t >= self.length - 1
        state = RegimeState(self._transition(k_transition, state.regime), state.t + 1)
        return state, self._observe(k_obs, state, level), reward, done

    def filter_update(self, belief: Array, obs: Array, level: Array) -> Array:
        """One HMM forward step: predict with the switching kernel, then weight by P(obs | z)."""
        n = self.n_regimes
        stay = 1.0 - self.hazard
        predicted = stay * belief + self.hazard * (1.0 - belief) / (n - 1)
        likelihood = level * obs[:n] + (1.0 - level) / n
        posterior = predicted * likelihood
        return posterior / posterior.sum()

    def bounds(self, key: Array, level: Array, n_episodes: int = 512) -> tuple[Array, Array]:
        """(random, Bayes-optimal) returns; the latter by simulating the forward filter."""
        uniform = jnp.full(self.n_regimes, 1.0 / self.n_regimes)

        def episode(key: Array) -> Array:
            k_reset, k_run = jax.random.split(key)
            state, obs = self.reset(k_reset, level)
            # The reset belief is uniform; filter_update's predict step leaves it uniform.
            belief = self.filter_update(uniform, obs, level)

            def body(carry: tuple[RegimeState, Array, Array], key: Array) -> tuple[Any, Array]:
                state, obs, belief = carry
                state, obs, reward, _ = self.step(key, state, belief.argmax(), level)
                return (state, obs, self.filter_update(belief, obs, level)), reward

            _, rewards = jax.lax.scan(body, (state, obs, belief), jax.random.split(k_run, self.length))
            return rewards.sum()

        optimal = jax.vmap(episode)(jax.random.split(key, n_episodes)).mean()
        return jnp.float32(1.0 / self.n_regimes), optimal


TASKS: dict[str, Callable[..., Task]] = {
    "noisy_cartpole": NoisyCartPole,
    "delayed_cue": DelayedCue,
    "associative_recall": AssociativeRecall,
    "modular_sum": ModularSum,
    "hidden_regime": HiddenRegime,
}


def make_task(name: str, **kwargs: Any) -> Task:
    return TASKS[name](**kwargs)


# -- vectorisation ------------------------------------------------------------------------------------


def make_vec(task: Task, n_envs: int) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Batched reset/step over `n_envs` copies; a finished env is reset within the same step, so the
    returned observation is the first of its next episode (and `done` marks the boundary)."""

    def reset(key: Array, level: Array) -> tuple[Any, Array]:
        return jax.vmap(task.reset, (0, None))(jax.random.split(key, n_envs), level)

    def step_one(key: Array, state: Any, action: Array, level: Array) -> tuple[Any, Array, Array, Array]:
        k_step, k_reset = jax.random.split(key)
        state, obs, reward, done = task.step(k_step, state, action, level)
        reset_state, reset_obs = task.reset(k_reset, level)
        state = jax.tree.map(lambda fresh, old: jnp.where(done, fresh, old), reset_state, state)
        return state, jnp.where(done, reset_obs, obs), reward, done

    def step(key: Array, state: Any, action: Array, level: Array) -> tuple[Any, Array, Array, Array]:
        return jax.vmap(step_one, (0, 0, 0, None))(jax.random.split(key, n_envs), state, action, level)

    return reset, step
