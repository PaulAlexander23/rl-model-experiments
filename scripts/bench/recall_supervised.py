"""Capacity or optimisation? Train memory cores on associative recall with supervised labels.

In M1 every recurrent core plateaued at ~1.5-3 stored pairs whatever its width. Here the same cores
(same task, same observation stream, same ActorCritic policy head) are trained by cross-entropy on
the correct answer at every query step, with full backpropagation through each episode. If a narrow
core solves large K this way, the width was never the limit: PPO's learning signal was.

    JAX_PLATFORMS=cpu uv run python scripts/bench/recall_supervised.py
"""

from __future__ import annotations

import argparse
import time
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax

from rl_experiments.jaxrl.models import ActorCritic, make_core
from rl_experiments.jaxrl.tasks import AssociativeRecall


def episodes(task: AssociativeRecall, key: jax.Array, n: int, k: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(obs (T, n, F), answer (T, n), is_query (T, n)) for n episodes of K = k pairs."""
    length = k + task.delay + task.n_queries
    level = jnp.float32(k)

    def one(key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        state, _ = task.reset(key, level)

        def body(state: Any, _: Any) -> tuple[Any, tuple[jax.Array, jax.Array, jax.Array]]:
            is_query, index = task._phase(state)
            answer = state.values[index]
            next_state, _, _, _ = task.step(key, state, answer, level)
            return next_state, (task._observe(state), answer, is_query)

        _, out = jax.lax.scan(body, state, None, length=length)
        return out

    obs, answer, is_query = jax.vmap(one)(jax.random.split(key, n))
    return obs.swapaxes(0, 1), answer.swapaxes(0, 1), is_query.swapaxes(0, 1)


def train(core_name: str, width: int, k: int, steps: int, seed: int = 0) -> float:
    task = AssociativeRecall()
    network = ActorCritic(make_core(core_name, task.obs_size, width=width), task.n_actions)
    batch = 256
    obs, _, _ = episodes(task, jax.random.key(0), batch, k)
    starts = jnp.zeros(obs.shape[:2], bool).at[0].set(True)
    carry = network.initialize_carry(batch)
    params = network.init(jax.random.key(seed), carry, obs, starts)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    def loss_fn(params: Any, obs: jax.Array, answer: jax.Array, is_query: jax.Array) -> tuple[jax.Array, Any]:
        _, logits, _ = cast(tuple[Any, jax.Array, Any], network.apply(params, carry, obs, starts))
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, answer)
        correct = (logits.argmax(-1) == answer) & is_query
        return (ce * is_query).sum() / is_query.sum(), correct.sum() / is_query.sum()

    @jax.jit
    def step(params: Any, opt_state: Any, key: jax.Array) -> tuple[Any, Any, jax.Array]:
        obs, answer, is_query = episodes(task, key, batch, k)
        (_, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, obs, answer, is_query)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, acc

    key = jax.random.key(seed + 1)
    accs = []
    for i in range(steps):
        params, opt_state, acc = step(params, opt_state, jax.random.fold_in(key, i))
        accs.append(float(acc))
    final = sum(accs[-50:]) / 50
    return (final - 1 / task.n_values) / (1 - 1 / task.n_values)  # same normalisation as M1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--cores", nargs="+", default=["gru", "lstm", "lru"])
    parser.add_argument("--widths", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--pairs", nargs="+", type=int, default=[2, 4, 8, 16])
    args = parser.parse_args()
    print(f"| core | width | {' | '.join(f'K={k}' for k in args.pairs)} |")
    print(f"|---|---|{'---|' * len(args.pairs)}")
    for core in args.cores:
        for width in args.widths:
            start = time.perf_counter()
            scores = [train(core, width, k, args.steps) for k in args.pairs]
            cells = " | ".join(f"{s:.2f}" for s in scores)
            print(f"| {core} | {width} | {cells} |  <!-- {time.perf_counter() - start:.0f}s -->", flush=True)


if __name__ == "__main__":
    main()
