"""Why is GRU ~5x slower than LSTM on CPU?  PyTorch's CPU LSTM has a fused oneDNN (mkldnn) kernel; GRU does not.

uv run python scripts/bench/rnn_cpu_kernels.py
"""

import time

import torch as th
from torch import nn


def bench(
    cls: type[nn.LSTM] | type[nn.GRU], mkldnn: bool, steps: int = 128, batch: int = 16, hidden: int = 64
) -> float:
    """Mean milliseconds for one forward+backward pass over a (steps, batch) sequence."""
    rnn = cls(2, hidden)
    x = th.randn(steps, batch, 2)
    h = th.zeros(1, batch, hidden)
    state = (h, h.clone()) if cls is nn.LSTM else h
    reps = 50
    with th.backends.mkldnn.flags(enabled=mkldnn):
        for _ in range(5):
            rnn(x, state)[0].sum().backward()
        start = time.perf_counter()
        for _ in range(reps):
            rnn(x, state)[0].sum().backward()
    return (time.perf_counter() - start) / reps * 1e3


if __name__ == "__main__":
    th.set_num_threads(4)
    for cls in (nn.LSTM, nn.GRU):
        for mkldnn in (True, False):
            print(f"{cls.__name__:5s} mkldnn={mkldnn!s:5s} fwd+bwd {bench(cls, mkldnn):6.2f} ms")
