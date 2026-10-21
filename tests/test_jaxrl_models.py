from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_experiments.jaxrl.models import ActorCritic, TransformerCore, _linear_recurrence, make_core

T, B, F = 10, 3, 5
CORE_KWARGS: dict[str, dict[str, Any]] = {
    "mlp": {},
    "framestack": {"n_frames": 3},
    "lstm": {"depth": 2},
    "gru": {"depth": 2},
    "lru": {"depth": 2},
    "transformer": {"depth": 2, "context": 3},
}


def setup(name: str) -> tuple[Any, Any, Any]:
    core = make_core(name, F, width=16, **CORE_KWARGS[name])
    carry = core.initialize_carry(B)
    x = jnp.zeros((1, B, F))
    params = core.init(jax.random.key(0), carry, x, jnp.zeros((1, B), bool))
    # Warm the carry up so the tests start from a non-trivial state.
    warm = jax.random.normal(jax.random.key(1), (4, B, F))
    carry, _ = run(core, params, carry, warm, jnp.zeros((4, B), bool))
    return core, params, carry


def run(module: Any, params: Any, *args: Any) -> tuple[Any, ...]:
    return cast(tuple[Any, ...], module.apply(params, *args))


def max_diff(a: Any, b: Any) -> float:
    pairs = zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    return max(float(jnp.abs(jnp.asarray(x, jnp.complex64) - jnp.asarray(y, jnp.complex64)).max()) for x, y in pairs)


@pytest.mark.parametrize("name", sorted(CORE_KWARGS))
def test_sequence_matches_stepwise(name: str) -> None:
    """Training replays chunks in one call; it must reproduce the rollout's one-step calls exactly."""
    core, params, carry = setup(name)
    x = jax.random.normal(jax.random.key(2), (T, B, F))
    starts = jnp.zeros((T, B), bool).at[0, 0].set(True).at[4, 1].set(True).at[8, 2].set(True)
    seq_carry, seq_y = run(core, params, carry, x, starts)
    step_carry, ys = carry, []
    for t in range(T):
        step_carry, y = run(core, params, step_carry, x[t : t + 1], starts[t : t + 1])
        ys.append(y[0])
    assert max_diff(seq_y, jnp.stack(ys)) < 1e-4
    assert max_diff(seq_carry, step_carry) < 1e-4


@pytest.mark.parametrize("name", sorted(CORE_KWARGS))
def test_episode_start_forgets_the_past(name: str) -> None:
    """After a start, outputs do not depend on the carry or on inputs before it."""
    core, params, carry = setup(name)
    x = jax.random.normal(jax.random.key(3), (T, B, F))
    starts = jnp.zeros((T, B), bool).at[5].set(True)
    _, y = run(core, params, carry, x, starts)
    _, y_fresh = run(core, params, core.initialize_carry(B), x.at[:5].set(7.0), starts)
    np.testing.assert_allclose(y[5:], y_fresh[5:], atol=1e-5)


def test_memoryless_and_windowed_cores_have_bounded_memory() -> None:
    x = jax.random.normal(jax.random.key(4), (T, B, F))
    starts = jnp.zeros((T, B), bool)
    for name, horizon in (("mlp", 1), ("framestack", 3)):
        core, params, carry = setup(name)
        _, y = run(core, params, carry, x, starts)
        _, y2 = run(core, params, carry, x.at[: T - horizon].set(-3.0), starts)
        np.testing.assert_allclose(y[-1], y2[-1], atol=1e-6)
        assert not np.allclose(y[-1], run(core, params, carry, x.at[-1].set(-3.0), starts)[1][-1])


def test_transformer_window() -> None:
    """With context C, step t sees steps t-C..t, and changing anything older has no effect."""
    core = TransformerCore(width=16, depth=1, context=3)
    carry = core.initialize_carry(B)
    x = jax.random.normal(jax.random.key(5), (T, B, F))
    starts = jnp.zeros((T, B), bool).at[0].set(True)
    params = core.init(jax.random.key(0), carry, x, starts)
    _, y = run(core, params, carry, x, starts)
    _, y_old = run(core, params, carry, x.at[:6].set(9.0), starts)
    np.testing.assert_allclose(y[-1], y_old[-1], atol=1e-5)  # step 9 sees 6..9
    _, y_in = run(core, params, carry, x.at[6].set(9.0), starts)
    assert not np.allclose(y[-1], y_in[-1])


def test_linear_recurrence_matches_loop() -> None:
    key_a, key_b, key_h = jax.random.split(jax.random.key(6), 3)
    a = jax.random.uniform(key_a, (T, B, 4)) * jnp.exp(1j * 0.3)
    b = jax.random.normal(key_b, (T, B, 4)).astype(jnp.complex64)
    h0 = jax.random.normal(key_h, (B, 4)).astype(jnp.complex64)
    h, expected = h0, []
    for t in range(T):
        h = a[t] * h + b[t]
        expected.append(h)
    np.testing.assert_allclose(_linear_recurrence(a, b, h0), jnp.stack(expected), atol=1e-5)


def test_actor_critic_heads() -> None:
    net = ActorCritic(make_core("gru", F, width=8), n_actions=4)
    carry = net.initialize_carry(B)
    x = jnp.ones((T, B, F))
    params = net.init(jax.random.key(0), carry, x, jnp.zeros((T, B), bool))
    _, logits, value = run(net, params, carry, x, jnp.zeros((T, B), bool))
    assert logits.shape == (T, B, 4)
    assert value.shape == (T, B)
    # The 0.01-gain policy head starts close to uniform.
    assert float(jnp.abs(logits).max()) < 0.1
