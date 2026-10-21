"""Memory cores behind one sequence API, and the actor-critic that wraps them.

Every core maps

    carry, y = core(carry, x, starts)      x: (T, B, F), starts: (T, B) bool, y: (T, B, width)

where `starts[t]` marks the first step of an episode: the carry is cleared before that step is
processed. Rollouts call it with T=1; PPO calls it on stored chunks of `seq_len` steps starting from
the carry recorded at the chunk start. Both must agree exactly (tests check this for every core).
Every carry leaf has the batch axis first, so PPO can slice minibatches out of stored carries.

| Core | Memory | Training over a chunk |
|------|--------|-----------------------|
| `MLPCore` | none | parallel |
| `FrameStackCore` | last `n_frames` inputs | parallel |
| `RNNCore` (LSTM/GRU) | gated hidden state, `depth` stacked cells | sequential scan |
| `LRUCore` | diagonal complex linear recurrence (Orvieto et al. 2023) | parallel associative scan |
| `TransformerCore` | per-layer cache of the last `context` hidden states (TrXL), GRU-gated | parallel attention |
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
Carry = Any


def _clear(carry: Carry, start: Array) -> Carry:
    """Zero each batch row of every carry leaf where `start` (B,) is set."""
    return jax.tree.map(lambda c: jnp.where(start.reshape(-1, *([1] * (c.ndim - 1))), 0, c), carry)


def _mlp(x: Array, width: int, depth: int) -> Array:
    for _ in range(depth):
        x = nn.tanh(nn.Dense(width, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x))
    return x


class MLPCore(nn.Module):
    """No memory: `depth` tanh layers applied per step (the memoryless baseline)."""

    width: int
    depth: int = 1

    def initialize_carry(self, batch: int) -> Carry:
        return jnp.zeros((batch, 1))

    @nn.compact
    def __call__(self, carry: Carry, x: Array, starts: Array) -> tuple[Carry, Array]:
        return carry, _mlp(x, self.width, self.depth)


class FrameStackCore(nn.Module):
    """An MLP over the last `n_frames` inputs; frames from before the episode start are zero."""

    width: int
    depth: int = 1
    n_frames: int = 8
    input_size: int = 0  # needed to shape the carry before seeing any input

    def initialize_carry(self, batch: int) -> Carry:
        return jnp.zeros((batch, self.n_frames - 1, self.input_size))

    @nn.compact
    def __call__(self, carry: Carry, x: Array, starts: Array) -> tuple[Carry, Array]:
        def push(window: Array, step: tuple[Array, Array]) -> tuple[Array, Array]:
            frame, start = step
            window = _clear(window, start)
            stacked = jnp.concatenate([window, frame[:, None]], axis=1)  # (B, n_frames, F)
            return stacked[:, 1:], stacked.reshape(stacked.shape[0], -1)

        carry, windows = jax.lax.scan(push, carry, (x, starts))
        return carry, _mlp(windows, self.width, self.depth)


class _RNNStep(nn.Module):
    cell: str
    width: int
    depth: int

    @nn.compact
    def __call__(self, carry: Carry, inputs: tuple[Array, Array]) -> tuple[Carry, Array]:
        x, start = inputs
        carry = _clear(carry, start)
        new_carry = []
        for layer in range(self.depth):
            if self.cell == "lstm":
                cell: nn.RNNCellBase = nn.OptimizedLSTMCell(self.width, name=f"lstm_{layer}")
            else:
                cell = nn.GRUCell(self.width, name=f"gru_{layer}")
            layer_carry, x = cell(carry[layer], x)
            new_carry.append(layer_carry)
        return tuple(new_carry), x


class RNNCore(nn.Module):
    """`depth` stacked LSTM or GRU cells of `width` units, scanned over time."""

    width: int
    depth: int = 1
    cell: str = "gru"

    def initialize_carry(self, batch: int) -> Carry:
        zeros = jnp.zeros((batch, self.width))
        return tuple((zeros, zeros) if self.cell == "lstm" else zeros for _ in range(self.depth))

    @nn.compact
    def __call__(self, carry: Carry, x: Array, starts: Array) -> tuple[Carry, Array]:
        scan = nn.scan(_RNNStep, variable_broadcast="params", split_rngs={"params": False}, in_axes=0, out_axes=0)
        return scan(self.cell, self.width, self.depth, name="rnn")(carry, (x, starts))


def _linear_recurrence(a: Array, b: Array, h0: Array) -> Array:
    """h_t = a_t * h_{t-1} + b_t for all t, via a parallel associative scan (time is axis 0)."""
    b = b.at[0].add(a[0] * h0)

    def combine(left: tuple[Array, Array], right: tuple[Array, Array]) -> tuple[Array, Array]:
        a_l, b_l = left
        a_r, b_r = right
        return a_r * a_l, a_r * b_l + b_r

    _, h = jax.lax.associative_scan(combine, (a, b), axis=0)
    return h


class LRULayer(nn.Module):
    """Linear Recurrent Unit: h_t = lambda * h_{t-1} + gamma * B x_t, y_t = Re(C h_t) + D x_t with diagonal
    complex lambda initialised on a ring r in [r_min, r_max] (Orvieto et al. 2023). An episode start
    sets that step's lambda to zero, which clears the state inside the parallel scan."""

    width: int
    r_min: float = 0.9
    r_max: float = 0.999
    max_phase: float = 2 * np.pi

    @nn.compact
    def __call__(self, h0: Array, x: Array, starts: Array) -> tuple[Array, Array]:
        n = h = self.width

        def nu_init(key: Array, shape: tuple[int, ...]) -> Array:
            u = jax.random.uniform(key, shape)
            return jnp.log(-0.5 * jnp.log(u * (self.r_max**2 - self.r_min**2) + self.r_min**2))

        def theta_init(key: Array, shape: tuple[int, ...]) -> Array:
            return jnp.log(self.max_phase * jax.random.uniform(key, shape))

        nu_log = self.param("nu_log", nu_init, (n,))
        theta_log = self.param("theta_log", theta_init, (n,))
        lam = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        gamma_log = self.param("gamma_log", lambda _key: jnp.log(jnp.sqrt(1 - jnp.abs(lam) ** 2)))
        b_re = self.param("b_re", nn.initializers.normal(1 / np.sqrt(2 * h)), (n, h))
        b_im = self.param("b_im", nn.initializers.normal(1 / np.sqrt(2 * h)), (n, h))
        c_re = self.param("c_re", nn.initializers.normal(1 / np.sqrt(n)), (h, n))
        c_im = self.param("c_im", nn.initializers.normal(1 / np.sqrt(n)), (h, n))
        d = self.param("d", nn.initializers.normal(1.0), (h,))

        b_x = jnp.einsum("tbh,nh->tbn", x.astype(jnp.complex64), b_re + 1j * b_im) * jnp.exp(gamma_log)
        a = jnp.where(starts[..., None], 0, jnp.broadcast_to(lam, b_x.shape))
        states = _linear_recurrence(a, b_x, h0)
        y = jnp.einsum("tbn,hn->tbh", states, c_re + 1j * c_im).real + d * x
        return states[-1], y


class LRUCore(nn.Module):
    """`depth` residual blocks of LayerNorm -> LRU -> GELU -> GLU, after a linear embedding."""

    width: int
    depth: int = 1

    def initialize_carry(self, batch: int) -> Carry:
        return tuple(jnp.zeros((batch, self.width), jnp.complex64) for _ in range(self.depth))

    @nn.compact
    def __call__(self, carry: Carry, x: Array, starts: Array) -> tuple[Carry, Array]:
        x = nn.Dense(self.width)(x)
        new_carry = []
        for layer in range(self.depth):
            h, y = LRULayer(self.width, name=f"lru_{layer}")(carry[layer], nn.LayerNorm()(x), starts)
            y = nn.Dense(2 * self.width)(nn.gelu(y))
            x = x + y[..., : self.width] * nn.sigmoid(y[..., self.width :])
            new_carry.append(h)
        return tuple(new_carry), nn.LayerNorm()(x)


class GRUGate(nn.Module):
    """GTrXL's gated residual: x is the residual stream, y the sublayer output. The bias starts the
    gate near the identity, so a fresh layer passes its input through (Parisotto et al. 2020)."""

    width: int
    bias: float = 2.0

    @nn.compact
    def __call__(self, x: Array, y: Array) -> Array:
        def dense(name: str) -> nn.Dense:
            return nn.Dense(self.width, use_bias=False, name=name)

        r = nn.sigmoid(dense("w_r")(y) + dense("u_r")(x))
        bias = self.param("b_g", nn.initializers.constant(self.bias), (self.width,))
        z = nn.sigmoid(dense("w_z")(y) + dense("u_z")(x) - bias)
        h = nn.tanh(dense("w_g")(y) + dense("u_g")(r * x))
        return (1 - z) * x + z * h


class TransformerCore(nn.Module):
    """Pre-LN causal transformer whose query at step t sees itself and the previous `context` steps
    of the same episode, at every layer (Transformer-XL's per-layer memory, with stop-gradient).

    The carry holds, per layer, that layer's inputs for the last `context` steps, plus which of those
    belong to the current episode. Positions enter as a learned per-head bias on the distance
    (0..context), so rollout and chunked training see identical inputs. `gated` swaps the residual
    additions for GTrXL's GRU gates."""

    width: int
    depth: int = 1
    context: int = 64
    head_dim: int = 16
    gated: bool = True

    @property
    def n_heads(self) -> int:
        return max(1, self.width // self.head_dim)

    def initialize_carry(self, batch: int) -> Carry:
        return {
            "memory": jnp.zeros((batch, self.depth, self.context, self.width)),
            "valid": jnp.zeros((batch, self.context), bool),
        }

    def _visibility(self, valid: Array, starts: Array) -> tuple[Array, Array, Array]:
        """(mask (B, T, C+T), distance (T, C+T), valid for the next carry (B, C))."""
        t_len, context = starts.shape[0], self.context
        key_pos = jnp.concatenate([jnp.arange(-context, 0), jnp.arange(t_len)])
        distance = jnp.arange(t_len)[:, None] - key_pos[None, :]
        window = (distance >= 0) & (distance <= context)
        # Index of the latest episode start at or before each query step (-1: none in this chunk).
        last_start = jax.lax.cummax(jnp.where(starts.T, jnp.arange(t_len), -1), axis=1)  # (B, T)
        carried = jnp.concatenate([valid, jnp.ones((valid.shape[0], t_len), bool)], axis=1)  # (B, C+T)
        same_episode = jnp.where(
            last_start[..., None] >= 0, key_pos[None, None, :] >= last_start[..., None], carried[:, None, :]
        )
        next_valid = jnp.where(last_start[:, -1:] >= 0, key_pos[None, :] >= last_start[:, -1:], carried)[:, -context:]
        return window[None] & same_episode, distance, next_valid

    @nn.compact
    def __call__(self, carry: Carry, x: Array, starts: Array) -> tuple[Carry, Array]:
        heads, head_dim = self.n_heads, self.width // self.n_heads
        mask, distance, next_valid = self._visibility(carry["valid"], starts)
        h = nn.Dense(self.width)(x).transpose(1, 0, 2)  # (B, T, d)
        new_memory = []
        for layer in range(self.depth):
            keys_in = jnp.concatenate([carry["memory"][:, layer], h], axis=1)  # (B, C+T, d)
            new_memory.append(keys_in[:, -self.context :])
            norm = nn.LayerNorm(name=f"ln_attn_{layer}")
            q = nn.DenseGeneral((heads, head_dim), name=f"q_{layer}")(norm(h))
            k = nn.DenseGeneral((heads, head_dim), name=f"k_{layer}")(norm(keys_in))
            v = nn.DenseGeneral((heads, head_dim), name=f"v_{layer}")(norm(keys_in))
            bias = self.param(f"distance_bias_{layer}", nn.initializers.zeros, (heads, self.context + 1))
            logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(head_dim)
            logits = logits + bias[:, jnp.clip(distance, 0, self.context)][None]
            logits = jnp.where(mask[:, None], logits, -1e9)
            attended = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, axis=-1), v)
            y = nn.DenseGeneral(self.width, axis=(-2, -1), name=f"out_{layer}")(attended)
            h = GRUGate(self.width, name=f"gate_attn_{layer}")(h, nn.relu(y)) if self.gated else h + y
            y = nn.Dense(self.width, name=f"ff_out_{layer}")(
                nn.gelu(nn.Dense(4 * self.width, name=f"ff_in_{layer}")(nn.LayerNorm(name=f"ln_ff_{layer}")(h)))
            )
            h = GRUGate(self.width, name=f"gate_ff_{layer}")(h, nn.relu(y)) if self.gated else h + y
        carry = {"memory": jnp.stack(new_memory, axis=1), "valid": next_valid}
        return carry, nn.LayerNorm()(h).transpose(1, 0, 2)


CORES: dict[str, Callable[..., nn.Module]] = {
    "mlp": MLPCore,
    "framestack": FrameStackCore,
    "lstm": lambda **kw: RNNCore(cell="lstm", **kw),
    "gru": lambda **kw: RNNCore(cell="gru", **kw),
    "lru": LRUCore,
    "transformer": TransformerCore,
}


def make_core(name: str, input_size: int, **kwargs: Any) -> nn.Module:
    if name == "framestack":
        kwargs["input_size"] = input_size
    return CORES[name](**kwargs)


class ActorCritic(nn.Module):
    """A shared memory core with separate memoryless policy and value heads."""

    core: nn.Module
    n_actions: int
    head_width: int = 64

    def initialize_carry(self, batch: int) -> Carry:
        return self.core.initialize_carry(batch)  # type: ignore[attr-defined]

    @nn.compact
    def __call__(self, carry: Carry, x: Array, starts: Array) -> tuple[Carry, Array, Array]:
        carry, features = self.core(carry, x, starts)
        orthogonal = nn.initializers.orthogonal
        pi = nn.tanh(nn.Dense(self.head_width, kernel_init=orthogonal(np.sqrt(2)))(features))
        logits = nn.Dense(self.n_actions, kernel_init=orthogonal(0.01))(pi)
        v = nn.tanh(nn.Dense(self.head_width, kernel_init=orthogonal(np.sqrt(2)))(features))
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(v)[..., 0]
        return carry, logits, value
