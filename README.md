# FinNum: how should LLMs read market numbers?

FinNum is a benchmark and three studies on **real NSE market data**. They ask how the way numbers
reach a language model affects both **accuracy** and **token cost**. That covers writing style
(1,234,567 vs 12,34,567 vs 12.35 lakh), tokenization (digits grouped from the left, from the right,
in lakh/crore pairs, or one at a time), and single-token number encodings (xVal, FoNE, BitTokens,
value prefix, plus our unit-aware variant).

> **Status:** all code is written and tested (100+ tests). The GPU experiments run on free Google
> Colab. See **[RUNNING.md](RUNNING.md)** for the run order. There are no results yet, so the numbers
> in this README are sanity checks on synthetic data, not findings. The paper draft is in
> [`paper/draft.md`](paper/draft.md).

## Structure

| Week | What | Code | Notebook |
|---|---|---|---|
| 1 | FinNum benchmark: 6 tasks × 5 number styles × 3 table formats, paired by question; NSE/AMFI loaders; token audit | `numbers/`, `data/`, `benchmark/`, `audit/` | (CLI: `python -m finnum`) |
| 2 | Zero-shot evaluation of 4 open LLMs, with paired McNemar tests, Wilson CIs and cost vs accuracy | `eval/` | `02_baseline_eval_colab.ipynb` |
| 3 | Tiny GPTs trained from scratch that differ only in digit grouping (`l2r3`, `r2l3`, `indian`, `digit`); longer-number generalisation; reversed-answer ablation | `tokexp/` | `03_tokenization_from_scratch_colab.ipynb` |
| 4–5 | xVal, FoNE, BitTokens and value-prefix retrofitted into Qwen2.5-0.5B with LoRA; date-based split | `retrofit/` | `04_retrofit_number_encodings_colab.ipynb` |
| 6 | **Ours, `fone_units`:** Fourier value features plus how the number was written (₹, %, lakh/crore, grouping) | `retrofit/features.py` | (same notebook) |
| 7 | Cross-study analysis: robustness to number style, cost vs accuracy, error types (×10ᵏ scale errors) | `analysis.py` | `05_final_analysis_colab.ipynb` |
| 8 | Paper draft, run guide | `paper/`, `RUNNING.md` | |

## Benchmark tasks

| Task | Example | Answer |
|---|---|---|
| lookup | Closing price of TCS on 2026-08-12? | number |
| pct_change | % change in close from date A to date B? | number (2 dp) |
| mean | Average close over these 7 days? | number (2 dp) |
| compare | Which moved more vs previous close: A or B? | symbol |
| magnitude | Which of these 5 had the highest turnover? (units differ across rows in scaled styles) | symbol |
| anomaly | Was today's volume > 3× the 20-day average? (balanced, ambiguous ratios excluded) | yes/no |

## Design decisions worth knowing

- **Paired design.** Every question exists in all 15 renderings with one gold answer, so styles are
  compared on the *same* questions (exact McNemar, Holm-corrected). A test re-derives every gold answer
  from the rendered prompt text alone.
- **No leakage.** In week 3, test operands come from values held out of training. In weeks 4–6,
  test questions come only from the last 120 trading days.
- **Fair length test (week 3).** Random position offsets during training mean longer test numbers
  don't land on untrained position embeddings.
- **Digit-exact FoNE features.** A unit test decodes them back to the original number for values up
  to 10¹².
- **Deviations from the original papers:** xVal is scaled by log-magnitude, because market values span
  10⁻² to 10¹². All encodings are input-side only: answers are generated as text, so one scorer grades
  every method.

## Quick start (laptop, no GPU)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,audit]"
pytest -q                                              # all non-GPU code

python -m finnum sample                                # synthetic data for offline work
python -m finnum build --data data/sample/stocks_sample.csv --n-per-task 50 --out data/benchmark/finnum_sample.jsonl
python -m finnum audit --data data/sample/stocks_sample.csv --bench data/benchmark/finnum_sample.jsonl

# real NSE data (NSE blocks most cloud IPs, so run this on your own machine)
python -m finnum download-nse --start 2023-10-02 --end 2026-10-08 \
    --symbols RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,SBIN,ITC,LT,MRF,YESBANK
```

To rebuild the Colab notebooks after a code change, run `cd scripts && python make_notebook.py &&
python make_notebook_week3.py && python make_notebook_week4.py && python make_notebook_week7.py`.
`python scripts/run_offline.py notebooks/*.ipynb` executes every notebook without a GPU, with fake
models, to check all the non-GPU cells.

## Sanity checks on synthetic data (not results)

- Indian grouping costs more tokens than international grouping: 12,345,678 is 5 pieces and
  1,23,45,678 is 7.
- Scale words cut large numbers from about 13 digit-level tokens to about 6, at the cost of precision.
- Splitting digits from the left or the right gives the same token count but different place-value
  alignment, so that choice can only affect accuracy (week 3 tests it).

## Data notes

- `DATA_SOURCE = "auto"` uses NSE when it's reachable, otherwise Yahoo Finance. With Yahoo, turnover
  is an approximation, so state the source used in any write-up.
- `data/sample/` is synthetic. Never report it as market data.
- `data/raw/` and `data/private/` are git-ignored.
