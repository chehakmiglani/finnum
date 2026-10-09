"""Accuracy, confidence intervals and paired significance tests for FinNum.

Every base question is asked in every (style, table format) condition, so two
conditions can be compared *paired*: McNemar's exact test only looks at the
questions where the two conditions disagree, which is far more sensitive than
comparing two independent accuracies.
"""
from __future__ import annotations

from math import comb, sqrt

import pandas as pd

KEYS = ["model", "style", "table_format"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def accuracy_table(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = by or KEYS
    g = df.groupby(by)["correct"].agg(n="size", k="sum").reset_index()
    g["accuracy"] = g["k"] / g["n"]
    ci = [wilson(int(k), int(n)) for k, n in zip(g["k"], g["n"])]
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]
    if "prompt_tokens" in df and df["prompt_tokens"].notna().any():
        g = g.merge(df.groupby(by)["prompt_tokens"].mean().rename("mean_prompt_tokens").reset_index(), on=by)
    return g


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired_vs_baseline(df: pd.DataFrame, baseline: tuple[str, str] = ("plain", "csv")) -> pd.DataFrame:
    """Each condition vs the baseline condition, per model, paired on base question."""
    rows = []
    for model, m in df.groupby("model"):
        wide = m.pivot_table(index="base_id", columns=["style", "table_format"], values="correct", aggfunc="first")
        if baseline not in wide.columns:
            continue
        base = wide[baseline]
        for cond in wide.columns:
            if cond == baseline:
                continue
            both = pd.concat([base, wide[cond]], axis=1).dropna().astype(bool)
            a, c = both.iloc[:, 0], both.iloc[:, 1]
            b_only, c_only = int((a & ~c).sum()), int((~a & c).sum())
            rows.append({"model": model, "style": cond[0], "table_format": cond[1], "n_pairs": len(both),
                         "acc_baseline": a.mean(), "acc_condition": c.mean(),
                         "diff_pp": (c.mean() - a.mean()) * 100,
                         "baseline_only_correct": b_only, "condition_only_correct": c_only,
                         "p_value": mcnemar_exact(b_only, c_only)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = out.groupby("model")["p_value"].transform(lambda s: holm(list(s)))
        out["significant"] = out["p_holm"] < 0.05
    return out


def summary_lines(df: pd.DataFrame, baseline: tuple[str, str] = ("plain", "csv")) -> list[str]:
    """Plain-English headline findings per model."""
    lines = []
    acc = accuracy_table(df)
    paired = paired_vs_baseline(df, baseline)
    for model, a in acc.groupby("model"):
        best = a.loc[a["accuracy"].idxmax()]
        worst = a.loc[a["accuracy"].idxmin()]
        lines.append(f"{model}: best = {best['style']}/{best['table_format']} {best['accuracy']:.1%}, "
                     f"worst = {worst['style']}/{worst['table_format']} {worst['accuracy']:.1%}")
        sig = paired[(paired["model"] == model) & paired["significant"]] if not paired.empty else paired
        for r in sig.itertuples():
            lines.append(f"   {r.style}/{r.table_format} vs {baseline[0]}/{baseline[1]}: "
                         f"{r.diff_pp:+.1f} pp (Holm p={r.p_holm:.3g})")
        if sig.empty:
            lines.append("   no condition differs significantly from the baseline")
    return lines
