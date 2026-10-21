"""Aggregate the JAX memory-sizing sweeps (Part 2) into markdown tables and figures under docs/results/.

uv run python scripts/aggregate_memory.py m1            # every multirun/m1_width_<task> + the summary
uv run python scripts/aggregate_memory.py m2_depth
uv run python scripts/aggregate_memory.py m3_delay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aggregate import GRID, INK_2, NEUTRAL, RESULTS, ROOT, SLOTS

# Fixed slots, consistent with Part 1 (frame stack, transformer, LSTM, GRU keep their colours).
MODELS = ["mlp", "framestack", "transformer", "lstm", "gru", "lru"]
COLORS = {
    "mlp": NEUTRAL,
    "framestack": SLOTS[0],
    "transformer": SLOTS[1],
    "lstm": SLOTS[2],
    "gru": SLOTS[3],
    "lru": SLOTS[4],
}
TASKS = ["noisy_cartpole", "delayed_cue", "associative_recall", "modular_sum", "hidden_regime"]
DIALS = {
    "noisy_cartpole": "noise std",
    "delayed_cue": "delay L",
    "associative_recall": "pairs K",
    "modular_sum": "modulus m",
    "hidden_regime": "reliability q",
}
THRESHOLD = 0.9


def load(experiment: str) -> pd.DataFrame:
    """One row per (run, level, seed)."""
    rows = []
    for path in sorted((ROOT / "multirun" / experiment).rglob("metrics.json")):
        m = json.loads(path.read_text())
        for i, level in enumerate(m["levels"]):
            for seed, score in enumerate(m["score"][i]):
                rows.append(
                    {
                        "task": m["task"],
                        "model": m["model"],
                        "width": m["width"],
                        "depth": m["depth"],
                        "memory": m["model_kwargs"].get("context", m["model_kwargs"].get("n_frames", 0)),
                        "seq_len": m["ppo"]["seq_len"],
                        "level": level,
                        "seed": seed,
                        "score": score,
                        "n_core_params": m["n_core_params"],
                        "steps_per_second": m["steps_per_second"],
                        "run_time": m["run_time"],
                        "curve_steps": m["curve_steps"],
                        "curve": m["train_return"][i][seed],
                    }
                )
    if not rows:
        raise SystemExit(f"no metrics.json under multirun/{experiment}")
    return pd.DataFrame(rows)


def ordered(models: set[str]) -> list[str]:
    return [m for m in MODELS if m in models]


def level_label(level: float) -> str:
    return f"{level:g}"


def score_table(df: pd.DataFrame, row: str) -> str:
    """Mean ± std score over seeds, one row per (model, `row`), one column per level."""
    stats = df.groupby(["model", row, "level"]).score.agg(["mean", "std"]).reset_index()
    stats["cell"] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(stats["mean"], stats["std"], strict=True)]
    table = stats.pivot_table(index=["model", row], columns="level", values="cell", aggfunc="first")
    table = table.reindex(ordered(set(df.model)), level=0)
    table.columns = [level_label(c) for c in table.columns]
    return table.to_markdown()


def min_width_table(df: pd.DataFrame) -> pd.DataFrame:
    """Smallest width whose mean score reaches THRESHOLD, per model and level ('>max' if none)."""
    means = df.groupby(["model", "width", "level"]).score.mean().reset_index()
    out = {}
    for (model, level), group in means.groupby(["model", "level"]):
        passing = group[group.score >= THRESHOLD].width
        out[(model, level)] = str(int(passing.min())) if len(passing) else ">max"
    table = pd.Series(out).unstack()
    table = table.reindex(ordered(set(df.model)))
    table.columns = [level_label(c) for c in table.columns]
    return table


def plot_score_vs(df: pd.DataFrame, x: str, path: Path, title: str, *, log2: bool = True) -> None:
    """Small multiples, one per level: mean score vs `x` per model, seeds as faint dots."""
    levels = sorted(set(df.level))
    fig, axes = plt.subplots(1, len(levels), figsize=(2.6 * len(levels) + 0.6, 2.9), sharey=True, squeeze=False)
    for ax, level in zip(axes[0], levels, strict=True):
        sub = df[df.level == level]
        for model in ordered(set(sub.model)):
            rows = sub.loc[sub.model == model]
            mean = rows.groupby(x).score.mean()
            color = COLORS[model]
            ax.scatter(rows[x], rows.score, s=8, color=color, alpha=0.25, linewidths=0, zorder=2)
            ax.plot(mean.index, mean.values, color=color, marker="o", markersize=4, label=model, zorder=3)
        ax.axhline(THRESHOLD, color=INK_2, linewidth=0.8, linestyle=":", zorder=1)
        if log2:
            ax.set_xscale("log", base=2)
        xs = sorted(set(sub[x]))
        ax.set_xticks(xs, [str(v) for v in xs])
        ax.minorticks_off()
        ax.set_ylim(-0.15, 1.08)
        ax.set_title(f"{DIALS.get(df.task.iloc[0], 'level')} = {level_label(level)}")
        ax.set_xlabel(x)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("score (random 0, optimal 1)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.1, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def throughput_table(df: pd.DataFrame) -> str:
    runs = df.drop_duplicates(["task", "model", "width", "depth", "memory", "seq_len"])
    table = runs.pivot_table(index="model", columns="width", values="steps_per_second", aggfunc="mean") / 1e3
    table = table.reindex(ordered(set(runs.model)))
    return table.round(0).astype("Int64").to_markdown()


def params_table(df: pd.DataFrame) -> str:
    runs = df.drop_duplicates(["model", "width", "depth"])
    table = runs.pivot_table(index="model", columns="width", values="n_core_params", aggfunc="first")
    return table.reindex(ordered(set(runs.model))).astype("Int64").to_markdown()


def m1_task(task: str) -> tuple[str, pd.DataFrame]:
    df = load(f"m1_width_{task}")
    name = f"m1_width_{task}"
    plot_score_vs(df, "width", RESULTS / f"{name}.png", f"{task}: score vs width (depth 1, 3 seeds)")
    minimum = min_width_table(df)
    text = (
        f"# M1 width sweep: {task}\n\nDial: {DIALS[task]}. Score: random 0, optimal 1; mean ± std over 3 seeds.\n\n"
        f"![](./{name}.png)\n\n## Smallest width with mean score ≥ {THRESHOLD}\n\n{minimum.to_markdown()}\n\n"
        f"## Scores\n\n{score_table(df, 'width')}\n\n"
        f"## Throughput (k env steps/s, 12 runs vmapped)\n\n{throughput_table(df)}\n"
    )
    (RESULTS / f"{name}.md").write_text(text)
    return text, minimum


def m1() -> None:
    sections, params = [], None
    for task in TASKS:
        if not any((ROOT / "multirun" / f"m1_width_{task}").rglob("metrics.json")):
            continue
        _, minimum = m1_task(task)
        minimum.columns = [f"{DIALS[task]} {c}" for c in minimum.columns]
        sections.append(f"### {task}\n\n{minimum.to_markdown()}\n")
        if params is None:
            params = params_table(load(f"m1_width_{task}"))
    text = (
        f"# M1 summary: smallest width with mean score ≥ {THRESHOLD}\n\n"
        "Depth 1, widths 8-128, 3 seeds; '>max' = no width reached it.\n\n" + "\n".join(sections)
    )
    if params is not None:
        text += f"\n## Core parameters by width (associative-recall input size)\n\n{params}\n"
    (RESULTS / "m1_summary.md").write_text(text)


def m2() -> None:
    """Depth 2-3 from m2_depth, plus the matching depth-1 runs from M1."""
    df = load("m2_depth")
    for task in set(df.task):
        base = load(f"m1_width_{task}")
        keep = base.width.isin(set(df.width)) & base.model.isin(set(df.loc[df.task == task].model))
        df = pd.concat([df, base[keep]], ignore_index=True)
    parts = ["# M2: depth vs width\n\nScore: mean ± std over 3 seeds.\n"]
    for task in [t for t in TASKS if t in set(df.task)]:
        sub = df.loc[df.task == task]
        sub = sub.assign(config=[f"w{w} d{d}" for w, d in zip(sub.width, sub.depth, strict=True)])
        for width in sorted(set(sub.width)):
            name = f"m2_depth_{task}_w{width}"
            plot_score_vs(
                sub[sub.width == width], "depth", RESULTS / f"{name}.png", f"{task}: width {width}", log2=False
            )
            parts.append(f"## {task}, width {width}\n\n![](./{name}.png)\n")
        parts.append(f"{score_table(sub, 'config')}\n")
    (RESULTS / "m2_depth.md").write_text("\n".join(parts))


def m3() -> None:
    df = load("m3_delay")
    df = df.assign(reach=np.where(df.model == "transformer", df.memory, df.seq_len))
    path = RESULTS / "m3_delay.png"
    models = ordered(set(df.model))
    fig, axes = plt.subplots(1, len(models), figsize=(2.7 * len(models) + 0.6, 2.9), sharey=True, squeeze=False)
    reaches = sorted(set(df.reach))
    shades = plt.get_cmap("Greys")(np.linspace(0.35, 0.95, len(reaches)))
    for ax, model in zip(axes[0], models, strict=True):
        sub = df[df.model == model]
        for reach, shade in zip(reaches, shades, strict=True):
            mean = sub.loc[sub.reach == reach].groupby("level").score.mean()
            if len(mean):
                ax.plot(mean.index, mean.values, marker="o", markersize=4, color=shade, label=str(reach))
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(set(df.level)), [level_label(v) for v in sorted(set(df.level))])
        ax.minorticks_off()
        ax.set_ylim(-0.15, 1.08)
        ax.set_title(model + (" (context)" if model == "transformer" else " (BPTT length)"))
        ax.set_xlabel("delay L")
        ax.axhline(THRESHOLD, color=INK_2, linewidth=0.8, linestyle=":")
    axes[0][0].set_ylabel("score")
    axes[0][-1].legend(title="reach", loc="lower left")
    for ax in axes[0]:
        ax.grid(color=GRID)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    table = (
        df.groupby(["model", "reach", "level"])
        .score.mean()
        .unstack()
        .rename(columns=level_label)
        .round(2)
        .reindex(models, level=0)
    )
    text = (
        "# M3: delayed recall vs gradient reach\n\n"
        "Reach = truncated-BPTT length (`seq_len`) for recurrent cores, attention context for the transformer "
        "(its BPTT length is fixed at the default). Width 64, depth 1, mean of 3 seeds.\n\n"
        f"![](./m3_delay.png)\n\n{table.to_markdown()}\n"
    )
    (RESULTS / "m3_delay.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["m1", "m2_depth", "m3_delay"])
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    {"m1": m1, "m2_depth": m2, "m3_delay": m3}[args.experiment]()


if __name__ == "__main__":
    main()
