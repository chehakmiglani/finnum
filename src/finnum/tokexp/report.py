"""Aggregate week-3 runs across seeds, chart them, and write a plain-text summary."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .numtok import SCHEMES

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SPLIT_COLORS = {"test_id": "#2a78d6", "test_ood": "#eb6834"}          # 2-series categorical
SCHEME_COLORS = {"l2r3": "#2a78d6", "r2l3": "#eb6834", "indian": "#1baf7a", "digit": "#eda100"}
SPLIT_LABELS = {"test_id": "same length as training", "test_ood": "longer numbers"}


def load_results(out_dir: str | Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("*_seed*.json"))]


def tidy(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for split in ("test_id", "test_ood"):
            rows.append({"scheme": r["scheme"], "seed": r["seed"], "split": split, "task": "ALL",
                         "accuracy": r[split]["accuracy"]})
            for task, acc in r[split]["by_task"].items():
                rows.append({"scheme": r["scheme"], "seed": r["seed"], "split": split, "task": task, "accuracy": acc})
    return pd.DataFrame(rows)


def summary_table(results: list[dict]) -> pd.DataFrame:
    """Mean and std of accuracy across seeds, per scheme x split x task."""
    df = tidy(results)
    g = df.groupby(["scheme", "split", "task"])["accuracy"].agg(mean="mean", std="std", seeds="count").reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def cost_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{"scheme": r["scheme"], "tokens_per_example": r["mean_tokens_per_example"],
                        "answer_tokens": r["mean_answer_tokens"], "params": r["params"],
                        "vocab": r["vocab_size"], "train_minutes": r["train_minutes"]} for r in results])
    return df.groupby("scheme").mean().reindex([s for s in SCHEMES if s in set(df["scheme"])]).round(2)


def length_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for split in ("test_id", "test_ood"):
            for row in r[split]["by_task_len"]:
                rows.append({"scheme": r["scheme"], "seed": r["seed"], "split": split, **row})
    df = pd.DataFrame(rows)
    return df.groupby(["scheme", "task", "n_digits"])["accuracy"].mean().reset_index()


def _axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_accuracy(results: list[dict]):
    """One panel per task: accuracy per tokenizer scheme, same-length vs longer numbers."""
    s = summary_table(results)
    tasks = ["ALL"] + sorted(t for t in s["task"].unique() if t != "ALL")
    schemes = [x for x in SCHEMES if x in set(s["scheme"])]
    fig, axes = plt.subplots(1, len(tasks), figsize=(2.7 * len(tasks), 3.3), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    w = 0.38
    for ax, task in zip(axes[0], tasks):
        for j, split in enumerate(("test_id", "test_ood")):
            d = s[(s["task"] == task) & (s["split"] == split)].set_index("scheme").reindex(schemes)
            xs = [i + (j - 0.5) * w for i in range(len(schemes))]
            ax.bar(xs, d["mean"], width=w, color=SPLIT_COLORS[split], edgecolor=SURFACE, linewidth=2,
                   label=SPLIT_LABELS[split])
            ax.errorbar(xs, d["mean"], yerr=d["std"], fmt="none", ecolor=INK_2, elinewidth=1, capsize=0)
        ax.set_xticks(range(len(schemes)), schemes)
        ax.set_title("all tasks" if task == "ALL" else task, fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, 1.05)
        _axes(ax)
    axes[0][0].set_ylabel("exact-match accuracy (mean ± sd over seeds)", color=INK_2, fontsize=8)
    axes[0][-1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("Tiny GPTs trained from scratch: does number tokenization change what they learn?",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def plot_length(results: list[dict], tasks=("add", "sub", "cmp")):
    """Accuracy vs number of digits, one line per scheme (dashed line = longest training length)."""
    from .tasks import LENGTHS
    lt = length_table(results)
    tasks = [t for t in tasks if t in set(lt["task"])]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.2), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, task in zip(axes[0], tasks):
        for scheme, color in SCHEME_COLORS.items():
            d = lt[(lt["task"] == task) & (lt["scheme"] == scheme)].sort_values("n_digits")
            if d.empty:
                continue
            ax.plot(d["n_digits"], d["accuracy"], color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=scheme)
        ax.axvline(LENGTHS[task][0][1] + 0.5, color=INK_2, linewidth=1, linestyle="--")
        ax.text(LENGTHS[task][0][1] + 0.6, 1.0, "longer than training →", fontsize=7, color=INK_2, va="top")
        ax.set_title(task, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("digits in the numbers", color=INK_2, fontsize=8)
        ax.set_ylim(0, 1.05)
        _axes(ax)
    axes[0][0].set_ylabel("accuracy", color=INK_2, fontsize=8)
    axes[0][-1].legend(fontsize=8, frameon=False, loc="lower left", title="tokenizer", title_fontsize=8)
    fig.suptitle("Accuracy by number length", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def text_summary(results: list[dict]) -> str:
    s = summary_table(results)
    lines = ["ACCURACY, mean (sd) over seeds"]
    for split in ("test_id", "test_ood"):
        lines.append(f"\n[{SPLIT_LABELS[split]}]")
        piv = s[s["split"] == split].assign(cell=lambda d: d["mean"].map("{:.1%}".format) + " (" +
                                            d["std"].map("{:.1%}".format) + ")")
        lines.append(piv.pivot(index="task", columns="scheme", values="cell")
                     .reindex(columns=[x for x in SCHEMES if x in set(piv["scheme"])]).to_string())
    lines += ["\nCOST", cost_table(results).to_string()]
    allrow = s[s["task"] == "ALL"]
    for split in ("test_id", "test_ood"):
        d = allrow[allrow["split"] == split].sort_values("mean", ascending=False)
        if len(d):
            best = d.iloc[0]
            lines.append(f"best on {SPLIT_LABELS[split]}: {best['scheme']} ({best['mean']:.1%})")
    lines.append(f"seeds per scheme: {int(s['seeds'].min())}" + ("  (use >= 3 seeds before drawing conclusions)"
                                                                 if s["seeds"].min() < 3 else ""))
    return "\n".join(lines)
