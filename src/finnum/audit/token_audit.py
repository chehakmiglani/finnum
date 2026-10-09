"""Experiment 0: how many tokens do financial numbers cost, per style and tokenizer?

Outputs (in out_dir):
  number_tokens.csv   one row per (number, column kind, style, tokenizer)
  number_summary.csv  mean tokens per number, pivoted style x tokenizer
  prompt_tokens.csv   tokens per full benchmark prompt (real tokenizers only)
  prompt_summary.csv  mean prompt tokens per style x format x tokenizer, and % vs plain/csv
  examples.md         how each tokenizer splits a few example numbers
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from finnum.benchmark.render import COLS
from finnum.numbers import STYLES, format_number

from .tokenizers import RuleChunker, Tok

KINDS = {"close": "price", "volume": "volume", "turnover": "turnover"}


def sample_numbers(df: pd.DataFrame, n: int = 300, seed: int = 0) -> pd.DataFrame:
    rows = df.sample(min(n, len(df)), random_state=seed)
    recs = []
    for col, kind in KINDS.items():
        for v in rows[col]:
            recs.append({"kind": kind, "column": col, "value": v})
    return pd.DataFrame(recs)


def audit_numbers(nums: pd.DataFrame, toks: list[Tok]) -> pd.DataFrame:
    recs = []
    for r in nums.itertuples():
        spec = COLS[r.column]
        for style in STYLES:
            text = format_number(r.value, style, decimals=spec.decimals, scalable=spec.scalable)
            for t in toks:
                p = t.pieces(text)
                recs.append({"kind": r.kind, "value": r.value, "style": style, "text": text,
                             "tokenizer": t.name, "n_tokens": len(p), "pieces": "|".join(p)})
    return pd.DataFrame(recs)


def audit_prompts(variants: list[dict], toks: list[Tok]) -> pd.DataFrame:
    recs = []
    for t in toks:
        if isinstance(t, RuleChunker):
            continue  # rule chunkers are for numbers only
        for v in variants:
            recs.append({"id": v["id"], "base_id": v["base_id"], "task": v["task"], "style": v["style"],
                         "table_format": v["table_format"], "tokenizer": t.name,
                         "n_tokens": len(t.pieces(v["prompt"]))})
    return pd.DataFrame(recs)


def examples_md(toks: list[Tok]) -> str:
    samples = [("price", 2915.35, "close"), ("price", 130500.5, "close"),
               ("volume", 12345678, "volume"), ("turnover", 14912345678.55, "turnover")]
    lines = ["# How tokenizers split financial numbers", ""]
    for kind, value, col in samples:
        spec = COLS[col]
        lines += [f"## {kind}: {value}", "", "| style | text | " + " | ".join(t.name for t in toks) + " |",
                  "|---|---|" + "---|" * len(toks)]
        for style in STYLES:
            text = format_number(value, style, decimals=spec.decimals, scalable=spec.scalable)
            cells = [f"{len(p)}: `{'·'.join(p)}`" for p in (t.pieces(text) for t in toks)]
            lines.append(f"| {style} | `{text}` | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def run(df: pd.DataFrame, variants: list[dict], toks: list[Tok], out_dir: str | Path) -> dict[str, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    num = audit_numbers(sample_numbers(df), toks)
    num.to_csv(out / "number_tokens.csv", index=False)
    num_sum = num.pivot_table(index=["kind", "style"], columns="tokenizer", values="n_tokens", aggfunc="mean").round(2)
    num_sum.to_csv(out / "number_summary.csv")
    (out / "examples.md").write_text(examples_md(toks), encoding="utf-8")
    res = {"number_tokens": num, "number_summary": num_sum}

    pr = audit_prompts(variants, toks)
    if not pr.empty:
        pr.to_csv(out / "prompt_tokens.csv", index=False)
        s = pr.groupby(["tokenizer", "style", "table_format"])["n_tokens"].mean().rename("mean_tokens").reset_index()
        base = s[(s["style"] == "plain") & (s["table_format"] == "csv")].set_index("tokenizer")["mean_tokens"]
        s["vs_plain_csv_pct"] = ((s["mean_tokens"] / s["tokenizer"].map(base) - 1) * 100).round(1)
        s.to_csv(out / "prompt_summary.csv", index=False)
        res.update(prompt_tokens=pr, prompt_summary=s)
    return res
