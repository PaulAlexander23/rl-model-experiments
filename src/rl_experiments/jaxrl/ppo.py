"""Recurrent PPO in JAX: a whole training run is one compiled function of (key, level).

Ported from transit-dispatch-bench's `algos.ppo` (itself SB3-faithful: clipped surrogate, GAE,
per-minibatch advantage normalisation, Adam eps 1e-5, gradient norm 0.5) and extended with memory:

* The rollout is `n_steps` per env, collected as `n_steps // seq_len` chunks. The core's carry is
  stored at each chunk start, so training replays every chunk from exactly the state the policy
  acted from. Gradients flow within a chunk only (truncated BPTT of length `seq_len`); information
  can still be carried further forward, but learning to store it must happen inside one chunk.
* Minibatches are sets of whole chunks (sequences), never individual steps.

`make_train(task, network, config)` returns `train(key, level) -> results`, which runs every
update inside `lax.scan`; `jax.vmap` it over seeds and levels and `jax.jit` the lot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from rl_experiments.jaxrl.tasks import Task, make_vec

Array = jax.Array


class PPOConfig(NamedTuple):
    total_timesteps: int = 5_000_000
    n_envs: int = 256
    n_steps: int = 128
    seq_len: int = 128  # truncated-BPTT length; divides n_steps
    n_minibatches: int = 4
    n_epochs: int = 4
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_episodes: int = 256

    @property
    def n_updates(self) -> int:
        return max(self.total_timesteps // (self.n_envs * self.n_steps), 1)


class Transition(NamedTuple):
    obs: Array
    start: Array
    action: Array
    log_prob: Array
    value: Array
    reward: Array
    done: Array


class Runner(NamedTuple):
    params: Any
    opt_state: Any
    env_state: Any
    obs: Array
    start: Array  # whether `obs` is the first of an episode
    carry: Any
    episode_return: Array
    key: Array


def _categorical_log_prob(logits: Array, action: Array) -> Array:
    return jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1)[..., 0]


def _entropy(logits: Array) -> Array:
    log_p = jax.nn.log_softmax(logits)
    return -(jnp.exp(log_p) * log_p).sum(-1)


# One closure per phase, all sharing the compiled scope: long by design (PureJaxRL style).
def make_train(task: Task, network: nn.Module, config: PPOConfig) -> Callable[[Array, Array], dict[str, Array]]:  # noqa: PLR0915
    if config.n_steps % config.seq_len:
        raise ValueError(f"seq_len {config.seq_len} must divide n_steps {config.n_steps}")
    n_chunks = config.n_steps // config.seq_len
    n_sequences = n_chunks * config.n_envs
    if n_sequences % config.n_minibatches:
        raise ValueError(f"{n_sequences} sequences do not split into {config.n_minibatches} minibatches")
    vec_reset, vec_step = make_vec(task, config.n_envs)
    initialize_carry: Callable[[int], Any] = network.initialize_carry  # type: ignore[attr-defined]

    def schedule(count: jax.typing.ArrayLike) -> Array:
        updates_done = jnp.asarray(count) // (config.n_minibatches * config.n_epochs)
        return config.learning_rate * (1.0 - updates_done / config.n_updates)

    learning_rate: optax.ScalarOrSchedule = schedule if config.anneal_lr else config.learning_rate
    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate, eps=1e-5))

    def apply(params: Any, carry: Any, x: Array, starts: Array) -> tuple[Any, Array, Array]:
        """(carry, logits (T, B, A), value (T, B)) for a sequence."""
        return cast(tuple[Any, Array, Array], network.apply(params, carry, x, starts))

    def policy(params: Any, carry: Any, obs: Array, start: Array) -> tuple[Any, Array, Array]:
        """One step for a batch: (carry, logits (B, A), value (B,))."""
        carry, logits, value = apply(params, carry, obs[None], start[None])
        return carry, logits[0], value[0]

    def evaluate(params: Any, key: Array, level: Array) -> Array:
        """Greedy return of each of `eval_episodes` fresh episodes (one per env)."""
        e_reset, e_step = make_vec(task, config.eval_episodes)
        k_reset, k_run = jax.random.split(key)
        state, obs = e_reset(k_reset, level)
        n = config.eval_episodes

        def body(carry: tuple[Any, ...], key: Array) -> tuple[tuple[Any, ...], None]:
            state, obs, start, core_carry, total, finished = carry
            core_carry, logits, _ = policy(params, core_carry, obs, start)
            state, obs, reward, done = e_step(key, state, logits.argmax(-1), level)
            total = total + jnp.where(finished, 0.0, reward)
            return (state, obs, done, core_carry, total, finished | done), None

        init = (state, obs, jnp.ones(n, bool), initialize_carry(n), jnp.zeros(n), jnp.zeros(n, bool))
        (*_, total, _), _ = jax.lax.scan(body, init, jax.random.split(k_run, task.max_steps))
        return total

    def train(key: Array, level: Array) -> dict[str, Array]:  # noqa: PLR0915
        k_params, k_env, k_eval, k_bounds, key = jax.random.split(key, 5)
        carry = initialize_carry(config.n_envs)
        env_state, obs = vec_reset(k_env, level)
        start = jnp.ones(config.n_envs, bool)
        params = network.init(k_params, carry, obs[None], start[None])
        runner = Runner(params, optimizer.init(params), env_state, obs, start, carry, jnp.zeros(config.n_envs), key)

        def env_step(runner: Runner, _: Any) -> tuple[Runner, tuple[Transition, Array, Array]]:
            key, k_act, k_env = jax.random.split(runner.key, 3)
            carry, logits, value = policy(runner.params, runner.carry, runner.obs, runner.start)
            action = jax.random.categorical(k_act, logits)
            env_state, obs, reward, done = vec_step(k_env, runner.env_state, action, level)
            transition = Transition(
                runner.obs, runner.start, action, _categorical_log_prob(logits, action), value, reward, done
            )
            episode_return = runner.episode_return + reward
            runner = runner._replace(
                env_state=env_state,
                obs=obs,
                start=done,
                carry=carry,
                episode_return=jnp.where(done, 0.0, episode_return),
                key=key,
            )
            return runner, (transition, episode_return, done)

        def chunk(runner: Runner, _: Any) -> tuple[Runner, tuple[Any, ...]]:
            carry0 = runner.carry
            runner, steps = jax.lax.scan(env_step, runner, None, length=config.seq_len)
            return runner, (carry0, *steps)

        def gae(traj: Transition, last_value: Array) -> tuple[Array, Array]:
            def back(carry: tuple[Array, Array], step: tuple[Array, Array, Array]) -> tuple[tuple[Array, Array], Array]:
                advantage, next_value = carry
                reward, value, done = step
                not_done = 1.0 - done
                delta = reward + config.gamma * next_value * not_done - value
                advantage = delta + config.gamma * config.gae_lambda * not_done * advantage
                return (advantage, value), advantage

            _, advantages = jax.lax.scan(
                back,
                (jnp.zeros_like(last_value), last_value),
                (traj.reward, traj.value, traj.done.astype(jnp.float32)),
                reverse=True,
            )
            return advantages, advantages + traj.value

        def loss_fn(params: Any, batch: tuple[Any, ...]) -> tuple[Array, tuple[Array, ...]]:
            carry0, traj, advantages, returns = batch
            _, logits, value = apply(params, carry0, traj.obs, traj.start)
            log_prob = _categorical_log_prob(logits, traj.action)
            ratio = jnp.exp(log_prob - traj.log_prob)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            clipped = jnp.clip(ratio, 1 - config.clip_range, 1 + config.clip_range)
            policy_loss = -jnp.minimum(ratio * advantages, clipped * advantages).mean()
            value_loss = ((returns - value) ** 2).mean()
            entropy = _entropy(logits).mean()
            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy
            return loss, (policy_loss, value_loss, entropy)

        def update(runner: Runner, _: Any) -> tuple[Runner, dict[str, Array]]:
            runner, (carries, traj, returns_seen, dones) = jax.lax.scan(chunk, runner, None, length=n_chunks)
            # traj leaves: (n_chunks, seq_len, n_envs, ...) -> time-major (n_steps, n_envs, ...) for GAE.
            flat_time = jax.tree.map(lambda x: x.reshape(config.n_steps, *x.shape[2:]), traj)
            _, _, last_value = policy(runner.params, runner.carry, runner.obs, runner.start)
            advantages, returns = gae(flat_time, last_value)

            def to_sequences(x: Array) -> Array:
                """(n_steps, n_envs, ...) -> (seq_len, n_chunks * n_envs, ...)."""
                x = x.reshape(n_chunks, config.seq_len, config.n_envs, *x.shape[2:])
                return jnp.swapaxes(x, 0, 1).reshape(config.seq_len, n_sequences, *x.shape[3:])

            sequences = (
                jax.tree.map(lambda c: c.reshape(n_sequences, *c.shape[2:]), carries),
                jax.tree.map(to_sequences, flat_time),
                to_sequences(advantages),
                to_sequences(returns),
            )

            def epoch(state: tuple[Any, Any, Array], _: Any) -> tuple[tuple[Any, Any, Array], tuple[Array, ...]]:
                params, opt_state, key = state
                key, k_perm = jax.random.split(key)
                order = jax.random.permutation(k_perm, n_sequences)
                size = n_sequences // config.n_minibatches

                def take(batch_axis: int) -> Callable[[Array], Array]:
                    def split(x: Array) -> Array:
                        x = jnp.take(x, order, axis=batch_axis)
                        x = x.reshape(*x.shape[:batch_axis], config.n_minibatches, size, *x.shape[batch_axis + 1 :])
                        return jnp.moveaxis(x, batch_axis, 0)

                    return split

                carry0, traj_seq, adv_seq, ret_seq = sequences
                minibatches = (
                    jax.tree.map(take(0), carry0),
                    jax.tree.map(take(1), traj_seq),
                    take(1)(adv_seq),
                    take(1)(ret_seq),
                )

                def step(state: tuple[Any, Any], batch: tuple[Any, ...]) -> tuple[tuple[Any, Any], tuple[Array, ...]]:
                    params, opt_state = state
                    grads, aux = jax.grad(loss_fn, has_aux=True)(params, batch)
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    return (optax.apply_updates(params, updates), opt_state), aux

                (params, opt_state), aux = jax.lax.scan(step, (params, opt_state), minibatches)
                return (params, opt_state, key), aux

            key, k_epochs = jax.random.split(runner.key)
            (params, opt_state, _), (policy_loss, value_loss, entropy) = jax.lax.scan(
                epoch, (runner.params, runner.opt_state, k_epochs), None, length=config.n_epochs
            )
            n_done = dones.sum()
            metrics = {
                "train_return": jnp.where(n_done > 0, jnp.where(dones, returns_seen, 0.0).sum() / n_done, jnp.nan),
                "policy_loss": policy_loss.mean(),
                "value_loss": value_loss.mean(),
                "entropy": entropy.mean(),
            }
            return runner._replace(params=params, opt_state=opt_state, key=key), metrics

        runner, metrics = jax.lax.scan(update, runner, None, length=config.n_updates)
        random_return, optimal_return = task.bounds(k_bounds, level)
        eval_returns = evaluate(runner.params, k_eval, level)
        return {
            **metrics,
            "eval_return": eval_returns.mean(),
            "eval_std": eval_returns.std(),
            "random_return": random_return,
            "optimal_return": optimal_return,
        }

    return train


def count_params(network: nn.Module, task: Task) -> dict[str, int]:
    """Parameter counts of the whole network and of its memory core."""
    carry = network.initialize_carry(1)  # type: ignore[attr-defined]
    shapes = jax.eval_shape(
        network.init, jax.random.key(0), carry, jnp.zeros((1, 1, task.obs_size)), jnp.zeros((1, 1), bool)
    )
    total = sum(x.size for x in jax.tree.leaves(shapes))
    core = sum(x.size for x in jax.tree.leaves(shapes["params"]["core"]))
    return {"n_params": int(total), "n_core_params": int(core)}
