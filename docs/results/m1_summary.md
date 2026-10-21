# M1 summary: smallest width with mean score ≥ 0.9

Depth 1, widths 8-128, 3 seeds; '>max' = no width reached it.

### noisy_cartpole

|             | noise std 0   | noise std 0.1   | noise std 0.2   | noise std 0.4   |
|:------------|:--------------|:----------------|:----------------|:----------------|
| mlp         | >max          | >max            | >max            | >max            |
| framestack  | 8             | 8               | >max            | >max            |
| transformer | 8             | 8               | 32              | >max            |
| lstm        | 8             | 8               | 32              | >max            |
| gru         | 8             | 8               | 32              | >max            |
| lru         | 8             | 8               | 32              | >max            |

### delayed_cue

|             | delay L 4   | delay L 16   | delay L 64   | delay L 256   |
|:------------|:------------|:-------------|:-------------|:--------------|
| mlp         | >max        | >max         | >max         | >max          |
| framestack  | 8           | >max         | >max         | >max          |
| transformer | 8           | 8            | 32           | >max          |
| lstm        | 8           | 8            | >max         | >max          |
| gru         | 8           | 8            | >max         | >max          |
| lru         | 8           | 8            | >max         | >max          |

### associative_recall

|             | pairs K 2   | pairs K 4   | pairs K 8   | pairs K 16   |
|:------------|:------------|:------------|:------------|:-------------|
| mlp         | >max        | >max        | >max        | >max         |
| framestack  | >max        | >max        | >max        | >max         |
| transformer | 8           | 8           | 8           | 32           |
| lstm        | 128         | >max        | >max        | >max         |
| gru         | 128         | >max        | >max        | >max         |
| lru         | 128         | >max        | >max        | >max         |

### modular_sum

|             | modulus m 2   | modulus m 3   | modulus m 5   | modulus m 8   |
|:------------|:--------------|:--------------|:--------------|:--------------|
| mlp         | >max          | >max          | >max          | >max          |
| framestack  | >max          | >max          | >max          | >max          |
| transformer | >max          | 128           | >max          | >max          |
| lstm        | 32            | >max          | >max          | >max          |
| gru         | 32            | >max          | >max          | >max          |
| lru         | 32            | >max          | >max          | >max          |

### hidden_regime

|             | reliability q 0.2   | reliability q 0.3   | reliability q 0.5   |   reliability q 1 |
|:------------|:--------------------|:--------------------|:--------------------|------------------:|
| mlp         | >max                | >max                | >max                |                 8 |
| framestack  | >max                | >max                | 8                   |                 8 |
| transformer | >max                | >max                | 32                  |                 8 |
| lstm        | 8                   | 8                   | 8                   |                 8 |
| gru         | 8                   | 8                   | 8                   |                 8 |
| lru         | 32                  | 32                  | 32                  |                 8 |

## Core parameters by width (associative-recall input size)

| model       |    8 |    32 |    128 |
|:------------|-----:|------:|-------:|
| mlp         |   24 |    96 |    384 |
| framestack  |  136 |   544 |   2176 |
| transformer | 1761 | 25346 | 396296 |
| lstm        |  352 |  4480 |  67072 |
| gru         |  272 |  3392 |  50432 |
| lru         |  488 |  6560 |  99968 |
