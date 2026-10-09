"""Charts for FinNum results (matplotlib, works in Colab).

Design choices: one panel per model (small multiples, shared y axis), a single
hue for bars since the x-axis already names each style, Wilson 95% intervals as
thin whiskers, recessive grid, and direct value labels only on bar ends.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from finnum.numbers import STYLES

from .metrics import accuracy_table

INK, INK_2, GRID, SURFACE, BAR = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
# table formats are a 3-series categorical: first three validated slots (blue, orange, aqua)
FORMAT_COLORS = {"csv": "#2a78d6", "markdown": "#eb6834", "json": "#1baf7a"}
STYLE_ORDER = list(STYLES)


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def accuracy_by_style(df: pd.DataFrame, table_format: str = "csv"):
    """Accuracy per number style, one panel per model, for one table format."""
    acc = accuracy_table(df[df["table_format"] == table_format], by=["model", "style"])
    models = sorted(acc["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.4), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, model in zip(axes[0], models):
        a = acc[acc["model"] == model].set_index("style").reindex(STYLE_ORDER).dropna(subset=["accuracy"])
        x = range(len(a))
        ax.bar(x, a["accuracy"], width=0.7, color=BAR, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(x, a["accuracy"], yerr=[a["accuracy"] - a["ci_low"], a["ci_high"] - a["accuracy"]],
                    fmt="none", ecolor=INK_2, elinewidth=1, capsize=0)
        for xi, v, top in zip(x, a["accuracy"], a["ci_high"]):
            ax.text(xi, top + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(list(x), [s.replace("_", "\n") for s in a.index])
        ax.set_title(model.split("/")[-1], fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, 1.12)
        _style_axes(ax)
    axes[0][0].set_ylabel("accuracy (95% CI)", color=INK_2, fontsize=9)
    fig.suptitle(f"FinNum accuracy by number style ({table_format} tables)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def cost_vs_accuracy(df: pd.DataFrame):
    """Mean prompt tokens vs accuracy for every (style, format) condition, one panel per model."""
    acc = accuracy_table(df)
    acc = acc.dropna(subset=["mean_prompt_tokens"]) if "mean_prompt_tokens" in acc else acc.iloc[0:0]
    models = sorted(acc["model"].unique())
    if not models:
        raise ValueError("no prompt token counts recorded - run_eval needs count_tokens")
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, model in zip(axes[0], models):
        a = acc[acc["model"] == model]
        for fmt, color in FORMAT_COLORS.items():
            f = a[a["table_format"] == fmt]
            ax.scatter(f["mean_prompt_tokens"], f["accuracy"], s=40, color=color, edgecolor=SURFACE,
                       linewidth=2, zorder=3, label=fmt)
        for r in a.itertuples():
            ax.annotate(r.style.replace("scaled_", "sc_"), (r.mean_prompt_tokens, r.accuracy), fontsize=7,
                        color=INK_2, xytext=(5, 2), textcoords="offset points")
        ax.set_title(model.split("/")[-1], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("mean prompt tokens (lower = cheaper)", color=INK_2, fontsize=9)
        _style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    axes[0][0].set_ylabel("accuracy", color=INK_2, fontsize=9)
    axes[0][-1].legend(title="table format", fontsize=8, title_fontsize=8, frameon=False, loc="best")
    fig.suptitle("Cost vs accuracy (top-left is best; sc_ = scaled)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig
