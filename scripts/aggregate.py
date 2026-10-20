"""Aggregate `multirun/<experiment>/**/metrics.json` into markdown tables and figures under docs/results/.

uv run python scripts/aggregate.py e1_shared_vs_split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"

# Categorical slots in fixed order (validated reference palette); entities keep their slot across charts.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e6e5e0", "#8a8984"
COLORS = {
    "shared_mlp": SLOTS[0],
    "split_mlp": SLOTS[1],
    "framestack_mlp": SLOTS[0],
    "transformer": SLOTS[1],
    "lstm": SLOTS[2],
    "gru": SLOTS[3],
    "cpu": SLOTS[0],
    "cuda": SLOTS[1],
    "hybrid_move": SLOTS[2],
    "hybrid_mirror": SLOTS[3],
    "rollout": SLOTS[0],
    "train": SLOTS[1],
}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def load_runs(experiment: str) -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "multirun" / experiment).rglob("metrics.json")):
        row = json.loads(path.read_text())
        row["run_dir"] = str(path.parent)
        row["batch_size"] = next(
            (int(o.split("=")[1]) for o in row.get("overrides", []) if o.startswith("env.ppo.batch_size=")), None
        )
        rows.append(row)
    if not rows:
        raise SystemExit(f"no metrics.json under multirun/{experiment}")
    return pd.DataFrame(rows)


def learning_curve(run_dir: str, grid: np.ndarray) -> np.ndarray:
    """Training-episode return (SB3's 100-episode rolling mean) interpolated onto `grid` timesteps."""
    progress = pd.read_csv(Path(run_dir) / "progress.csv")
    progress = progress.dropna(subset=["rollout/ep_rew_mean"])
    return np.interp(grid, progress["time/total_timesteps"], progress["rollout/ep_rew_mean"])


def add_auc(df: pd.DataFrame) -> pd.DataFrame:
    """Mean training return over the whole run: a sample-efficiency score (higher = learned sooner)."""
    df = df.copy()
    df["auc"] = [
        float(learning_curve(run_dir, np.linspace(0, steps, 200)).mean())
        for run_dir, steps in zip(df["run_dir"], df["total_timesteps"], strict=True)
    ]
    return df


def fmt(mean: float, std: float, digits: int = 1) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summary_table(df: pd.DataFrame, keys: list[str], metrics: dict[str, str], digits: int = 1) -> str:
    grouped = df.groupby(keys)
    out = pd.DataFrame({"seeds": grouped.size()})
    for column, label in metrics.items():
        stats = grouped[column].agg(["mean", "std"]).fillna(0.0)
        out[label] = [fmt(m, s, digits) for m, s in zip(stats["mean"], stats["std"], strict=True)]
    return out.reset_index().to_markdown(index=False)


def plot_curves(df: pd.DataFrame, envs: list[str], models: list[str], path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(envs), figsize=(3.4 * len(envs), 2.8), squeeze=False)
    for ax, env in zip(axes[0], envs, strict=True):
        sub = df[df["env"] == env]
        grid = np.linspace(0, sub["total_timesteps"].max(), 200)
        for model in models:
            run_dirs = sub.loc[sub["model"] == model, "run_dir"].tolist()
            if not run_dirs:
                continue
            curves = np.stack([learning_curve(r, grid) for r in run_dirs])
            mean, std = curves.mean(0), curves.std(0)
            color = COLORS.get(model, NEUTRAL)
            ax.plot(grid, mean, color=color, label=model)
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.set_title(env)
        ax.set_xlabel("timesteps")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0][0].set_ylabel("training return (100-ep mean)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.1, color=INK)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_dots(df: pd.DataFrame, envs: list[str], models: list[str], path: Path, title: str) -> None:
    """Final deterministic eval per model: one dot per seed, a bar for the mean (single hue: identity is the row)."""
    fig, axes = plt.subplots(1, len(envs), figsize=(3.6 * len(envs), 0.35 * len(models) + 1), squeeze=False)
    for ax, env in zip(axes[0], envs, strict=True):
        sub = df[df["env"] == env]
        for i, model in enumerate(models):
            vals = sub.loc[sub["model"] == model, "eval_mean"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            ax.scatter(vals, np.full_like(vals, i), s=18, color=SLOTS[0], alpha=0.6, linewidths=0)
            ax.plot([vals.mean()] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=2)
        ax.set_yticks(range(len(models)), models)
        ax.invert_yaxis()
        ax.set_title(env)
        ax.set_xlabel("eval return (20 episodes, deterministic)")
        ax.grid(axis="y", visible=False)
    for ax in axes[0][1:]:
        ax.tick_params(labelleft=False)
    fig.suptitle(title, y=1.02, color=INK)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_time_split(df: pd.DataFrame, facet: str, path: Path, title: str) -> None:
    """Stacked rollout/train seconds per device, one panel per facet value."""
    devices = [d for d in ["cpu", "cuda", "hybrid_move", "hybrid_mirror"] if d in set(df.device)]
    facets = list(dict.fromkeys(df[facet]))
    fig, axes = plt.subplots(1, len(facets), figsize=(2.6 * len(facets), 2.6), squeeze=False, sharey=False)
    for ax, value in zip(axes[0], facets, strict=True):
        sub = df[df[facet] == value].set_index("device").reindex(devices)
        x = np.arange(len(devices))
        ax.bar(x, sub.rollout_time, color=COLORS["rollout"], width=0.6, label="rollout", edgecolor="white", lw=1)
        ax.bar(
            x,
            sub.train_time,
            bottom=sub.rollout_time,
            color=COLORS["train"],
            width=0.6,
            label="update",
            edgecolor="white",
            lw=1,
        )
        ax.set_xticks(x, [d.replace("hybrid_", "hyb.\n") for d in devices])
        ax.set_title(str(value))
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("wall-clock seconds")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(title, y=1.12, color=INK)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def e1(df: pd.DataFrame) -> str:
    df = add_auc(df)
    envs = [e for e in ["cartpole", "acrobot", "pendulum", "lunarlander"] if e in set(df.env)]
    plot_curves(df, envs, ["shared_mlp", "split_mlp"], RESULTS / "e1_curves.png", "Shared trunk vs split actor-critic")
    table = summary_table(
        df, ["env", "model"], {"eval_mean": "final eval", "auc": "mean train return (AUC)", "n_params": "params"}
    )
    return f"![E1 learning curves](e1_curves.png)\n\n{table}\n"


def e2(df: pd.DataFrame) -> str:
    df = add_auc(df)
    envs = [e for e in ["cartpole_novel", "pendulum_novel"] if e in set(df.env)]
    models = [m for m in COLORS if m in set(df.model)] + sorted(set(df.model) - set(COLORS))
    models = list(dict.fromkeys(["split_mlp", "framestack_mlp", "transformer", "lstm", "gru", *models]))
    models = [m for m in models if m in set(df.model)]
    plot_dots(df, envs, models, RESULTS / "e2_eval.png", "Memory architectures on velocity-masked tasks")
    plot_curves(
        df,
        envs,
        ["framestack_mlp", "transformer", "lstm", "gru"],
        RESULTS / "e2_curves.png",
        "Memory architectures (split actor/critic variants)",
    )
    table = summary_table(
        df,
        ["env", "model"],
        {"eval_mean": "final eval", "auc": "mean train return (AUC)", "wall_time": "wall s", "n_params": "params"},
    )
    return f"![E2 eval](e2_eval.png)\n\n![E2 curves](e2_curves.png)\n\n{table}\n"


def e3(df: pd.DataFrame, experiment: str) -> str:
    df = df.copy()
    facet = "case"
    if df.batch_size.notna().any():
        df[facet] = (
            df.model.str.replace("split_mlp", "mlp64").str.replace("mlp64_", "mlp")
            + " / bs"
            + (df.batch_size.astype("Int64").astype(str))
        )
    else:
        df[facet] = df.model
    plot_time_split(df, facet, RESULTS / f"{experiment}.png", "Where the time goes: rollout vs PPO update")
    cols = ["wall_time", "rollout_time", "train_time", "transfer_time", "fps", "eval_mean"]
    table = df.sort_values([facet, "device"])[[facet, "device", *cols]].round(1).to_markdown(index=False)
    return f"![{experiment}]({experiment}.png)\n\n{table}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_runs(args.experiment)
    if args.experiment.startswith("e1"):
        body = e1(df)
    elif args.experiment.startswith("e2"):
        body = e2(df)
    else:
        body = e3(df, args.experiment)
    out = RESULTS / f"{args.experiment}.md"
    out.write_text(f"<!-- generated by scripts/aggregate.py; do not edit -->\n# {args.experiment}\n\n{body}")
    print(out.read_text())


if __name__ == "__main__":
    main()
