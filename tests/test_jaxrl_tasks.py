from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_experiments.jaxrl.tasks import (
    TASKS,
    AssociativeRecall,
    DelayedCue,
    HiddenRegime,
    ModularSum,
    NoisyCartPole,
    Task,
    make_task,
    make_vec,
)

LEVELS = {
    "noisy_cartpole": 0.1,
    "delayed_cue": 5.0,
    "associative_recall": 3.0,
    "modular_sum": 3.0,
    "hidden_regime": 0.5,
}


def play(task: Task, level: float, act: Any, key: jax.Array) -> tuple[float, int]:
    """Run one episode with `act(state, obs) -> action`; return (return, length)."""
    level_arr = jnp.float32(level)
    k_reset, key = jax.random.split(key)
    state, obs = task.reset(k_reset, level_arr)
    total, length = 0.0, 0
    for _ in range(task.max_steps):
        key, k_step = jax.random.split(key)
        state, obs, reward, done = task.step(k_step, state, jnp.int32(act(state, obs)), level_arr)
        total += float(reward)
        length += 1
        if done:
            break
    return total, length


@pytest.mark.parametrize("name", sorted(TASKS))
@pytest.mark.parametrize("oracle", [False, True])
def test_shapes_and_episode_ends(name: str, oracle: bool) -> None:
    task = make_task(name, oracle=oracle)
    _, obs = task.reset(jax.random.key(0), jnp.float32(LEVELS[name]))
    assert obs.shape == (task.obs_size,)
    assert obs.dtype == jnp.float32
    _, length = play(task, LEVELS[name], lambda s, o: 0, jax.random.key(1))
    assert 1 <= length <= task.max_steps


def test_delayed_cue_shows_cue_once_and_rewards_recall() -> None:
    task = DelayedCue(max_delay=8)
    state, obs = task.reset(jax.random.key(0), jnp.float32(5))
    cue = int(state.cue)
    assert obs[cue] == 1
    assert play(task, 5, lambda s, o: s.cue, jax.random.key(0)) == (1.0, 6)
    assert play(task, 5, lambda s, o: (s.cue + 1) % 4, jax.random.key(0))[0] == 0.0

    state, obs = task.step(jax.random.key(1), state, jnp.int32(0), jnp.float32(5))[:2]
    assert obs[: task.n_cues].sum() == 0  # hidden after t=0
    oracle = DelayedCue(max_delay=8, oracle=True)
    state, _ = oracle.reset(jax.random.key(0), jnp.float32(5))
    _, obs, _, _ = oracle.step(jax.random.key(1), state, jnp.int32(0), jnp.float32(5))
    assert obs[int(state.cue)] == 1


def test_associative_recall_queries_shown_keys() -> None:
    task = AssociativeRecall()
    total, length = play(task, 3, lambda s, o: s.values[task._phase(s)[1]], jax.random.key(0))
    assert total == pytest.approx(1.0)
    assert length == 3 + task.delay + task.n_queries

    state, obs = task.reset(jax.random.key(2), jnp.float32(3))
    assert np.all(np.asarray(state.queries) < 3)
    # During queries the value is hidden unless oracle.
    for _ in range(3 + task.delay):
        state, obs, _, _ = task.step(jax.random.key(3), state, jnp.int32(0), jnp.float32(3))
    assert obs[-1] == 1
    assert obs[task.max_items : task.max_items + task.n_values].sum() == 0
    assert obs[: task.max_items].sum() == 1


def test_modular_sum_oracle_policy_is_perfect() -> None:
    task = ModularSum(oracle=True)
    m = 5

    def act(state: Any, obs: jax.Array) -> int:
        return (int(obs[: task.max_modulus].argmax()) + int(obs[task.max_modulus :].argmax())) % m

    total, length = play(task, m, act, jax.random.key(0))
    assert total == pytest.approx(1.0)
    assert length == task.length
    assert ModularSum().bounds(jax.random.key(0), jnp.float32(4))[0] == pytest.approx(0.25)


def test_hidden_regime_bounds() -> None:
    task = HiddenRegime()
    random, optimal = task.bounds(jax.random.key(0), jnp.float32(1.0))
    assert float(random) == pytest.approx(0.25)
    assert float(optimal) == pytest.approx(1.0)
    _, noisy = task.bounds(jax.random.key(0), jnp.float32(0.3))
    # Filtering beats acting on the latest observation alone (q + (1 - q) / n), but not the oracle.
    assert 0.3 + 0.7 / 4 < float(noisy) < 1.0


def test_hidden_regime_switches() -> None:
    task = HiddenRegime(hazard=1.0)
    state, _ = task.reset(jax.random.key(0), jnp.float32(1.0))
    next_state, _, _, _ = task.step(jax.random.key(1), state, jnp.int32(0), jnp.float32(1.0))
    assert int(next_state.regime) != int(state.regime)


def test_cartpole_noise_and_oracle() -> None:
    clean = NoisyCartPole()
    state, obs = clean.reset(jax.random.key(0), jnp.float32(0.0))
    expected = state.physics[jnp.array([0, 2])] / jnp.array([clean.x_threshold, clean.theta_threshold])
    np.testing.assert_allclose(obs, expected, rtol=1e-6)
    _, noisy_obs = clean.reset(jax.random.key(0), jnp.float32(1.0))
    assert not np.allclose(noisy_obs, expected)
    assert NoisyCartPole(oracle=True).obs_size == 4
    # Pushing one way only falls over quickly.
    _, length = play(clean, 0.0, lambda s, o: 1, jax.random.key(0))
    assert length < 30


def test_vec_auto_reset() -> None:
    task = DelayedCue(max_delay=4)
    reset, step = make_vec(task, 3)
    level = jnp.float32(2)
    state, obs = reset(jax.random.key(0), level)
    assert obs.shape == (3, task.obs_size)
    done = jnp.zeros(3, bool)
    for i in range(3):
        state, obs, _, done = step(jax.random.key(i), state, jnp.zeros(3, jnp.int32), level)
    assert bool(done.all())
    assert np.all(np.asarray(state.t) == 0)
    assert np.all(np.asarray(obs[:, : task.n_cues]).sum(1) == 1)  # the next episode's cue
