"""Command line: python -m finnum <command> --help"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd


def _load_stocks(path: str) -> pd.DataFrame:
    from finnum.data import validate_stocks
    return validate_stocks(pd.read_csv(path))


def cmd_sample(a):
    from finnum.data import make_sample
    df = make_sample(start=a.start, end=a.end, seed=a.seed)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} synthetic rows -> {a.out}")


def cmd_download_nse(a):
    from finnum.data.nse import NSEClient
    syms = a.symbols.split(",") if a.symbols else None
    df = NSEClient(cache_dir=a.cache).fetch_range(date.fromisoformat(a.start), date.fromisoformat(a.end), syms)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} rows for {df['symbol'].nunique()} symbols -> {a.out}")


def cmd_download_amfi(a):
    from finnum.data.amfi import fetch_navall
    df = fetch_navall()
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} schemes -> {a.out}")


def cmd_build(a):
    from finnum.benchmark import Generator, write_jsonl
    variants = Generator(_load_stocks(a.data), seed=a.seed).build(a.n_per_task)
    write_jsonl(variants, a.out)
    print(f"wrote {len(variants):,} prompts ({len(variants) // 15:,} base questions x 5 styles x 3 formats) -> {a.out}")


def cmd_audit(a):
    from finnum.audit.token_audit import run
    from finnum.audit.tokenizers import DEFAULT_REAL, RULES, load_many
    from finnum.benchmark import read_jsonl
    specs = a.tokenizers.split(",") if a.tokenizers else RULES + DEFAULT_REAL
    toks = load_many(specs)
    res = run(_load_stocks(a.data), read_jsonl(a.bench), toks, a.out)
    print("\nMean tokens per number:\n", res["number_summary"].to_string())
    if "prompt_summary" in res:
        print("\nMean tokens per prompt:\n", res["prompt_summary"].to_string(index=False))
    print(f"\nresults -> {a.out}")


def main():
    p = argparse.ArgumentParser(prog="finnum")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("sample", help="write synthetic NSE-like data (offline dev)")
    s.add_argument("--start", default="2023-10-02"); s.add_argument("--end", default="2026-10-06")
    s.add_argument("--seed", type=int, default=7); s.add_argument("--out", default="data/sample/stocks_sample.csv")
    s.set_defaults(fn=cmd_sample)

    s = sub.add_parser("download-nse", help="download real NSE bhavcopies (run on your machine)")
    s.add_argument("--start", required=True); s.add_argument("--end", required=True)
    s.add_argument("--symbols", help="comma-separated; default all EQ symbols")
    s.add_argument("--cache", default="data/raw/nse"); s.add_argument("--out", default="data/raw/stocks.csv")
    s.set_defaults(fn=cmd_download_nse)

    s = sub.add_parser("download-amfi", help="download today's AMFI NAVs")
    s.add_argument("--out", default="data/raw/navall.csv")
    s.set_defaults(fn=cmd_download_amfi)

    s = sub.add_parser("build", help="generate the FinNum benchmark")
    s.add_argument("--data", required=True); s.add_argument("--n-per-task", type=int, default=200)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--out", default="data/benchmark/finnum_v0.jsonl")
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("audit", help="experiment 0: token cost per style/tokenizer")
    s.add_argument("--data", required=True); s.add_argument("--bench", required=True)
    s.add_argument("--tokenizers", help="comma-separated specs, e.g. rule:r2l3,tiktoken:o200k_base,hf:Qwen/Qwen2.5-0.5B")
    s.add_argument("--out", default="results/audit")
    s.set_defaults(fn=cmd_audit)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
