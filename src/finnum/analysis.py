"""Week 7: cross-experiment analysis on FinNum predictions (any mix of models / methods).

Works on the prediction tables written by finnum.eval.runner.run_eval (week 2 zero-shot
models and week 4-6 retrofit methods share that format).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from finnum.eval.metrics import accuracy_table, holm, mcnemar_exact
from finnum.eval.runner import load_predictions
from finnum.numbers import STYLES

INK, INK_2, GRID, SURFACE, MARK = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"


def load_dir(path: str | Path, experiment: str) -> pd.DataFrame:
    files = sorted(Path(path).glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no prediction files in {path}")
    return load_predictions(*files).assign(experiment=experiment)


def paired_models(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Every model vs ``baseline`` on the SAME prompts (paired on prompt id), McNemar exact + Holm."""
    wide = df.pivot_table(index="id", columns="model", values="correct", aggfunc="first")
    if baseline not in wide.columns:
        raise ValueError(f"baseline {baseline!r} not in {list(wide.columns)}")
    rows = []
    for m in wide.columns:
        if m == baseline:
            continue
        both = wide[[baseline, m]].dropna().astype(bool)
        a, c = both[baseline], both[m]
        b_only, c_only = int((a & ~c).sum()), int((~a & c).sum())
        rows.append({"model": m, "baseline": baseline, "n_pairs": len(both), "acc_baseline": a.mean(),
                     "acc_model": c.mean(), "diff_pp": (c.mean() - a.mean()) * 100,
                     "baseline_only_correct": b_only, "model_only_correct": c_only,
                     "p_value": mcnemar_exact(b_only, c_only)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(list(out["p_value"]))
        out["significant"] = out["p_holm"] < 0.05
    return out.sort_values("diff_pp", ascending=False).reset_index(drop=True)


def robustness(df: pd.DataFrame) -> pd.DataFrame:
    """How much does accuracy depend on how numbers are WRITTEN? Lower spread = more robust."""
    by_style = df.groupby(["model", "style"])["correct"].mean().unstack("style")
    by_style = by_style.reindex(columns=[s for s in STYLES if s in by_style.columns])
    out = pd.DataFrame({"accuracy": df.groupby("model")["correct"].mean(),
                        "worst_style": by_style.min(axis=1), "best_style": by_style.max(axis=1)})
    out["spread_pp"] = (out["best_style"] - out["worst_style"]) * 100
    return out.join(by_style).sort_values("accuracy", ascending=False)


def frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Mean prompt tokens vs accuracy per model; ``pareto`` marks models no other model beats on both."""
    g = df.groupby("model").agg(accuracy=("correct", "mean"), mean_prompt_tokens=("prompt_tokens", "mean")).reset_index()
    g["pareto"] = [not ((g["accuracy"] >= r.accuracy) & (g["mean_prompt_tokens"] <= r.mean_prompt_tokens) &
                        ((g["accuracy"] > r.accuracy) | (g["mean_prompt_tokens"] < r.mean_prompt_tokens))).any()
                   for r in g.itertuples()]
    return g.sort_values("mean_prompt_tokens").reset_index(drop=True)


def classify_error(row) -> str:
    """Why was a numeric answer wrong? Scale errors (x10, lakh/crore mix-ups) are the interesting ones."""
    if row["correct"]:
        return "correct"
    if row["pred"] is None or (isinstance(row["pred"], float) and math.isnan(row["pred"])):
        return "no answer"
    if not isinstance(row["gold"], (int, float)) or isinstance(row["gold"], bool):
        return "wrong choice"
    pred, gold = float(row["pred"]), float(row["gold"])
    if gold == 0 or pred == 0:
        return "far off"
    ratio = abs(pred / gold)
    k = round(math.log10(ratio))
    if k != 0 and abs(math.log10(ratio) - k) < 0.02:
        return f"scale x10^{k}"
    if abs(pred - gold) / abs(gold) < 0.01:
        return "close (<1%)"
    if (pred < 0) != (gold < 0) and abs(abs(pred) - abs(gold)) / abs(gold) < 0.01:
        return "sign flip"
    return "far off"


def error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    e = df.assign(error=df.apply(classify_error, axis=1))
    e = e[e["error"] != "correct"]
    return e.groupby(["model", "error"]).size().unstack("error", fill_value=0)


def _axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_frontier(df: pd.DataFrame, title: str = "Cost vs accuracy across all methods"):
    f = frontier(df)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    for r in f.itertuples():
        ax.scatter(r.mean_prompt_tokens, r.accuracy, s=60, zorder=3, linewidth=2,
                   color=MARK if r.pareto else SURFACE, edgecolor=MARK)
        ax.annotate(r.model.split("/")[-1], (r.mean_prompt_tokens, r.accuracy), xytext=(6, 3),
                    textcoords="offset points", fontsize=8, color=INK if r.pareto else INK_2)
    p = f[f["pareto"]].sort_values("mean_prompt_tokens")
    ax.plot(p["mean_prompt_tokens"], p["accuracy"], color=MARK, linewidth=1.5, zorder=2)
    ax.set_xlabel("mean prompt tokens (lower = cheaper)", color=INK_2, fontsize=9)
    ax.set_ylabel("accuracy", color=INK_2, fontsize=9)
    _axes(ax)
    ax.set_title(f"{title}\nfilled = best trade-off (no method is both cheaper and more accurate)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    return fig


def plot_robustness(df: pd.DataFrame):
    r = robustness(df)
    styles = [s for s in STYLES if s in r.columns]
    models = list(r.index)
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(models) + 1.4))
    fig.patch.set_facecolor(SURFACE)
    for y, m in enumerate(models):
        vals = r.loc[m, styles].astype(float)
        ax.plot([vals.min(), vals.max()], [y, y], color=GRID, linewidth=4, solid_capstyle="round", zorder=1)
        ax.scatter(vals, [y] * len(vals), s=36, color=MARK, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"spread {r.loc[m, 'spread_pp']:.0f} pp", (vals.max(), y), xytext=(8, -3),
                    textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(models)), [m.split("/")[-1] for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("accuracy in each of the 5 number styles (dots); bar = best-to-worst range", color=INK_2, fontsize=9)
    _axes(ax)
    ax.set_title("How much does the way numbers are written change accuracy?", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    return fig


def text_report(df: pd.DataFrame, baseline: str) -> str:
    lines = ["ACCURACY BY MODEL x TASK"]
    lines.append(accuracy_table(df, by=["model", "task"]).pivot(index="model", columns="task", values="accuracy")
                 .round(3).to_string())
    lines += ["\nROBUSTNESS TO NUMBER STYLE", robustness(df).round(3).to_string()]
    lines += ["\nCOST vs ACCURACY", frontier(df).round(3).to_string(index=False)]
    lines += [f"\nPAIRED vs {baseline}", paired_models(df, baseline).round(4).to_string(index=False)]
    lines += ["\nERROR TYPES", error_breakdown(df).to_string()]
    return "\n".join(lines)
