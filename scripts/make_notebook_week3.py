"""Builds notebooks/03_tokenization_from_scratch_colab.ipynb (self-contained). Run: python scripts/make_notebook_week3.py"""
import base64
import io
import zipfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# FinNum week 3: does number tokenization change what a model can learn?

We train **tiny GPTs from scratch** on arithmetic built from real NSE numbers. The four models are
identical except for **how digits are split into tokens**:

| scheme | `12345678` becomes | like |
|---|---|---|
| `l2r3` | `123·456·78` | GPT-4 / Llama-3 |
| `r2l3` | `12·345·678` | grouped from the right, so the pieces line up with place value |
| `indian` | `1·23·45·678` | lakh/crore grouping |
| `digit` | `1·2·3·4·5·6·7·8` | Llama-2 / Gemma |

**Tasks:** `add`, `sub` (daily volumes and turnovers), `cmp` (which number is bigger),
`pct` (% change between two prices) and `cr` (rupees → crore).

**Two test sets:**
- **Same length as training:** checks the model learned the task.
- **Longer numbers:** checks whether it learned an *algorithm* or just memorised patterns.

**Before you start:** go to *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.
Leave `QUICK_TEST = True` for the first run, which takes a few minutes. Then set it to `False`."""))

cells.append(code("""# ======================= SETTINGS =======================
QUICK_TEST = True        # True: short smoke test (1 seed, few steps). False: the real experiment.

DATA_SOURCE = "auto"     # "auto" (NSE, falls back to Yahoo), "yahoo", "nse", "sample" (synthetic), or a CSV path
SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "LT",
           "BHARTIARTL", "MARUTI", "MRF", "PAGEIND", "YESBANK", "SUZLON", "IRFC"]

SCHEMES = ["l2r3", "r2l3", "indian", "digit"]
SEEDS = [0, 1, 2]                 # 3 seeds -> mean +- sd; needed before claiming a difference
N_TRAIN_PER_TASK = 40_000
N_TEST_PER_TASK = 500
STEPS = 6000
BATCH_SIZE = 256
MODEL = dict(n_layer=4, n_head=4, n_embd=256)   # ~3M parameters
REVERSE_ANSWERS = False   # ablation: write add/sub answers lowest digit first (the order carries flow)

SAVE_TO_DRIVE = False     # True keeps results in Google Drive (asks for Drive access)

if QUICK_TEST:
    SEEDS, N_TRAIN_PER_TASK, N_TEST_PER_TASK, STEPS = [0], 5_000, 200, 300
"""))

cells.append(md("## 1. Setup"))
cells.append(code("""import os, sys, io, base64, zipfile, torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NONE - Runtime > Change runtime type > T4 GPU")

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS = "/content/drive/MyDrive/finnum_results"
else:
    RESULTS = "/content/finnum_results"
os.makedirs(RESULTS, exist_ok=True)

# The finnum package is embedded below, so nothing needs uploading
FINNUM_ZIP = "__FINNUM_ZIP_B64__"
zipfile.ZipFile(io.BytesIO(base64.b64decode(FINNUM_ZIP))).extractall("/content/finnum")
sys.path.insert(0, "/content/finnum/src")
!pip -q install yfinance
import finnum
print("finnum", finnum.__version__, "| results ->", RESULTS)"""))

cells.append(md("## 2. Data and tasks"))
cells.append(code("""from collections import Counter
import pandas as pd
from finnum.data.auto import load_stocks
from finnum.tokexp.tasks import TaskBuilder

stocks, used = load_stocks(DATA_SOURCE, SYMBOLS, cache_csv=f"{RESULTS}/stocks_{DATA_SOURCE}.csv"
                           if DATA_SOURCE in ("auto", "nse", "yahoo") else None, nse_cache_dir="/content/nse_cache")
print(f"data: {used} | {len(stocks):,} rows | {stocks.symbol.nunique()} symbols")

data = TaskBuilder(stocks, seed=0).build(N_TRAIN_PER_TASK, N_TEST_PER_TASK, reverse_answers=REVERSE_ANSWERS)
display(pd.DataFrame({split: Counter(e.task for e in exs) for split, exs in data.items()}))
for split in data:
    print(split, "->", [e.text for e in data[split][:3]])"""))

cells.append(code("""# What each tokenizer sees
from finnum.tokexp.numtok import NumTokenizer
toks = {s: NumTokenizer(s) for s in SCHEMES}
examples = [data["test_ood"][0].text, data["test_id"][0].text, "cr 14912345678=1491.23"]
pd.set_option("display.max_colwidth", 200)
display(pd.DataFrame([{"text": t, **{s: f"{len(toks[s].encode(t))}: " + "·".join(toks[s].pieces(t)) for s in SCHEMES}}
                      for t in examples]))"""))

cells.append(md("""## 3. Train

One model per (scheme, seed). Each finished model's results are saved straight away; if Colab
disconnects, run all the cells again and finished runs are skipped. The ETA is printed while it trains."""))
cells.append(code("""from finnum.tokexp.train import TrainConfig, run_all

cfg = TrainConfig(steps=STEPS, batch_size=BATCH_SIZE, **MODEL)
tag = ("quick" if QUICK_TEST else "full") + ("_rev" if REVERSE_ANSWERS else "") + f"_{used.split(':')[0]}"
OUT = f"{RESULTS}/week3_{tag}"
results = run_all(data, SCHEMES, SEEDS, cfg, OUT)
print("saved to", OUT)"""))

cells.append(md("## 4. Results"))
cells.append(code("""import matplotlib.pyplot as plt
from finnum.tokexp.report import load_results, plot_accuracy, plot_length, cost_table

results = load_results(OUT)
display(cost_table(results))
fig = plot_accuracy(results); fig.savefig(f"{OUT}/accuracy.png", dpi=150, bbox_inches="tight"); plt.show()
fig = plot_length(results); fig.savefig(f"{OUT}/accuracy_by_length.png", dpi=150, bbox_inches="tight"); plt.show()"""))

cells.append(code("""# Read some mistakes on longer numbers - they show HOW each tokenizer fails
for r in results:
    if r["seed"] != SEEDS[0]:
        continue
    errs = [e for e in r["test_ood"]["errors"] if e["task"] in ("add", "sub")][:5]
    print(f"\\n=== {r['scheme']} ===")
    for e in errs:
        print(f"  {e['prompt']:<34} gold {e['gold']:<14} model {e['pred']}")"""))

cells.append(md("## 5. Plain-text summary (copy everything below and paste it to Claude)"))
cells.append(code("""from finnum.tokexp.report import text_summary
print(f"DATA: {used} | quick={QUICK_TEST} | reverse_answers={REVERSE_ANSWERS} | steps={STEPS} | seeds={SEEDS}")
print(f"train examples: {len(data['train']):,} | model: {MODEL}")
print(text_summary(results))
print("\\nFINNUM_W3_DONE")"""))

cells.append(md("""## Next
- Run with `QUICK_TEST = False` for the real result (4 schemes × 3 seeds = 12 models).
- Optional ablation: set `REVERSE_ANSWERS = True` and run again. If reversing helps every scheme,
  the problem is generation order. If `r2l3` still wins, the tokenization itself matters.
- These results feed **weeks 4–5**: retrofitting number encodings into a pretrained model."""))

# embed the package
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted((ROOT / "src").rglob("*.py")):
        zf.write(f, f.relative_to(ROOT))
    zf.write(ROOT / "pyproject.toml", "pyproject.toml")
b64 = base64.b64encode(buf.getvalue()).decode()
for c in cells:
    if c.cell_type == "code" and "__FINNUM_ZIP_B64__" in c.source:
        c.source = c.source.replace("__FINNUM_ZIP_B64__", b64)

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}})
out = ROOT / "notebooks" / "03_tokenization_from_scratch_colab.ipynb"
nbf.write(nb, out)
print("wrote", out)
