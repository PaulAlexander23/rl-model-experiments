"""JAX PPO for memory-architecture sizing on partially observable tasks.

Rollout and update are compiled into one XLA program (PureJaxRL style), and a whole training run is
a pure function of `(key, level)`, so seeds and difficulty levels are `vmap`-ped into one launch.

* `tasks`: synthetic POMDPs, each with one difficulty dial (the `level`).
* `models`: memory cores (MLP, frame stack, LSTM, GRU, LRU, gated transformer) behind one sequence API.
* `ppo`: recurrent PPO over chunks of `seq_len` steps (truncated BPTT).
* `run`: Hydra entry point (`rl-train-jax`).
"""
