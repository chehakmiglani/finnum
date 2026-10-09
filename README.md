# FinNum

**How should LLMs read financial numbers?** FinNum is a benchmark and a set of
experiments on real NSE market data. It compares ways of feeding numbers to a
language model and measures both **token cost** and **accuracy**.

Methods compared:

- **Writing style.** Grouping (1,234,567 vs 12,34,567) and scale words (crore/lakh vs million).
- **Number tokenization.** Splitting digits from the left vs the right, or one token per digit.
- **Single-token number encodings** (later weeks). xVal, FoNE, BitTokens and value-aware
  prefix tokens, retrofitted into a pretrained small LLM. We add a unit-aware
  extension for ₹, %, lakh and crore.

## Status: weeks 1–3

| Component | What it does | Where |
|---|---|---|
| Number formats and parser | Renders and parses ₹, %, lakh/crore/million, Indian and international grouping, accounting negatives. Exact `Decimal` rounding. | `src/finnum/numbers/` |
| Data loaders | NSE bhavcopy (UDiFF and legacy formats), AMFI NAVAll, mfapi.in history | `src/finnum/data/` |
| Synthetic sample | NSE-shaped data with holidays, tick size and volume spikes, for offline development. **Not real data.** | `src/finnum/data/sample.py` |
| FinNum generator | 6 tasks with exact gold answers. Each question is rendered in 5 number styles x 3 table formats, so comparisons are paired. | `src/finnum/benchmark/` |
| Scorer | Pulls the answer out of free-text model output and checks it with a tolerance | `src/finnum/benchmark/scoring.py` |
| Token audit (exp. 0) | Tokens per number and per prompt, by style and tokenizer | `src/finnum/audit/` |
| Eval runner | Batched greedy decoding with any HF model; resumable JSONL output; strips `<think>` blocks | `src/finnum/eval/runner.py` |
| Statistics | Accuracy with Wilson 95% CIs; paired McNemar exact tests vs plain/csv with Holm correction | `src/finnum/eval/metrics.py` |
| Charts | Accuracy by style (small multiples), cost vs accuracy frontier | `src/finnum/eval/plots.py` |
| Colab notebook | Week 2 end to end: data → benchmark → token audit → 4 models → stats → charts → error analysis | `notebooks/02_baseline_eval_colab.ipynb` |
| Tests | 59 tests. Gold answers are re-derived independently from the prompt text, and the stats are checked against known values. | `tests/` |

### Tasks

| Task | Example | Answer |
|---|---|---|
| lookup | Closing price of TCS on 2026-08-12? | number |
| pct_change | % change in close from date A to date B? | number (2 dp) |
| mean | Average close over these 7 days? | number (2 dp) |
| compare | Which moved more vs previous close: A or B? | symbol |
| magnitude | Which of these 5 had the highest turnover? (units differ across rows in scaled styles) | symbol |
| anomaly | Was today's volume > 3x the 20-day average? (balanced, ambiguous ratios excluded) | yes/no |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,audit]"
pytest -q

# offline, synthetic data
python -m finnum sample
python -m finnum build --data data/sample/stocks_sample.csv --n-per-task 50 --out data/benchmark/finnum_sample.jsonl
python -m finnum audit --data data/sample/stocks_sample.csv --bench data/benchmark/finnum_sample.jsonl

# real data: run on your own machine, because NSE blocks most cloud IPs
python -m finnum download-nse --start 2023-10-02 --end 2026-10-06 \
    --symbols RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,SBIN,ITC,LT,MRF,YESBANK
python -m finnum build --data data/raw/stocks.csv --n-per-task 200
python -m finnum audit --data data/raw/stocks.csv --bench data/benchmark/finnum_v0.jsonl
```

`audit` loads the tokenizers it can and skips the rest with a message. Gated
models such as Llama and Gemma need `huggingface-cli login` first.

## First observations (synthetic sample, rule-based tokenizers only)

These are a sanity check, not results. Real tokenizer numbers come from running
`audit` on your machine.

- **Indian grouping costs more tokens than international grouping.** It adds more commas:
  12,345,678 is 5 pieces, 1,23,45,678 is 7.
- **Scale words cut large numbers sharply.** Turnover drops from about 13.4 digit-level tokens
  (plain) to about 5.7 (`12.35 million`). The cost is precision, which the accuracy experiments will measure.
- **Splitting from the left vs the right costs the same number of tokens but aligns
  differently.** `12345678` becomes `123·456·78` from the left and `12·345·678` from the right.
  Only the right-to-left split lines up with place value. That's an accuracy question, not
  a cost question, and experiment 2 tests it.

## Roadmap

1. **Week 1 (done):** benchmark, parser, loaders, token audit
2. **Week 2 (code done, run in Colab):** baseline accuracy of 4 open models on FinNum
3. **Week 3 (code done, run in Colab):** tiny GPTs trained from scratch, one per number tokenization
   (`l2r3`, `r2l3`, `indian`, `digit`). Five arithmetic tasks built from real NSE numbers, tested on
   numbers of the training length and on longer numbers. Notebook:
   `notebooks/03_tokenization_from_scratch_colab.ipynb`. Code: `src/finnum/tokexp/`.
4. **Weeks 4–5:** retrofit xVal / FoNE / BitTokens / value-prefix into Qwen2.5-0.5B with LoRA
5. **Week 6:** unit-aware extension (₹, %, lakh, crore)
6. **Week 7:** ablations, cost vs accuracy frontier, error analysis
7. **Week 8:** write-up and demo

## Running week 2 in Colab

1. Upload `finnum_week2.zip` to your Google Drive root, or upload it when the notebook asks.
2. Open `notebooks/02_baseline_eval_colab.ipynb` in Colab and pick a **T4 GPU** runtime.
3. Run every cell with `QUICK_TEST = 5`. This smoke test takes a few minutes.
4. Set `QUICK_TEST = 0` and run again for the full results. Progress is saved to Drive, so
   after a disconnect just run all the cells again.

Every question appears in all 15 conditions, so the conditions are compared **paired**
(McNemar's test on the questions where they disagree). A question whose numbers are too
small to change between styles is flagged `differs_from_plain = False`, and the notebook
reports headlines both with and without those questions.

## Data notes

- `data/sample/` is synthetic. Never report it as market data.
- `data/raw/` (real downloads) and `data/private/` are git-ignored.
