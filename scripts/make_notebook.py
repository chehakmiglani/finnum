"""Builds notebooks/02_baseline_eval_colab.ipynb. Run: python scripts/make_notebook.py"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# FinNum week 2: how accurately do LLMs read financial numbers?

This notebook runs open-source LLMs on the **FinNum** benchmark. Every question is asked
in **5 number styles × 3 table formats**, and for each condition we measure:

- **accuracy**: does the model get the answer right?
- **cost**: how many prompt tokens did the style use?

**Before you start:** go to *Runtime → Change runtime type → T4 GPU*.

**How to use it:**
1. Run the cells top to bottom.
2. The finnum code is built into this notebook, so there's nothing to upload. With
   `DATA_SOURCE = "auto"` the data comes from NSE, or from Yahoo Finance if NSE blocks Colab.
3. With `SAVE_TO_DRIVE = True`, results are saved to Google Drive after every batch. If Colab
   disconnects, reconnect and run all the cells again. Finished work is skipped.

**Start small:** leave `QUICK_TEST = 5` for the first run, which takes a few minutes and
checks that everything works. Then set it to `0` for the full run."""))

cells.append(code("""# ======================= SETTINGS =======================
DATA_SOURCE = "auto"     # "auto":   try NSE, fall back to Yahoo Finance (real prices; turnover approximated)
                         # "upload": stocks.csv made on your laptop with `python -m finnum download-nse`
                         # "nse" / "yahoo": force one source
                         # "sample": synthetic data - only to test the pipeline, never report it
SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "LT",
           "BHARTIARTL", "MARUTI", "MRF", "PAGEIND", "YESBANK", "SUZLON", "IRFC"]   # used by "auto"/"nse"/"yahoo"

N_PER_TASK = 40     # 6 tasks x 40 = 240 questions x 15 renderings = 3,600 prompts per model
QUICK_TEST = 5      # >0: only this many questions per task (smoke test). Set 0 for the full run.

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",          # fastest; also our week 4-5 retrofit target
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",     # 3.8B, slowest on a T4
    # "meta-llama/Llama-3.2-3B-Instruct",  # gated: accept the licence on Hugging Face and set HF_TOKEN
]
BATCH_SIZE = 16     # lower to 8 if you see "CUDA out of memory"
SAVE_TO_DRIVE = False   # True = keep results in Google Drive (asks for Drive access)
HF_TOKEN = None     # only needed for gated models
"""))

cells.append(md("## 1. Setup: GPU, Google Drive, code"))
cells.append(code("""import os, sys, glob, zipfile, torch

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU! Runtime > Change runtime type > T4 GPU (CPU will be extremely slow)")

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS = "/content/drive/MyDrive/finnum_results"
else:
    RESULTS = "/content/finnum_results"
os.makedirs(RESULTS, exist_ok=True)
print("results folder:", RESULTS)"""))

cells.append(code("""# The finnum package is embedded below (base64 zip), so nothing needs uploading
import base64, io
FINNUM_ZIP = "__FINNUM_ZIP_B64__"
zipfile.ZipFile(io.BytesIO(base64.b64decode(FINNUM_ZIP))).extractall("/content/finnum")
sys.path.insert(0, "/content/finnum/src")
!pip -q install tiktoken yfinance
import finnum
print("finnum", finnum.__version__, "ready")"""))

cells.append(md("## 2. Data"))
cells.append(code("""import pandas as pd
from datetime import date
from finnum.data import validate_stocks, make_sample

if DATA_SOURCE == "upload":
    path = f"{RESULTS}/stocks.csv"
    if not os.path.exists(path):
        from google.colab import files
        print("Upload stocks.csv - on your laptop run: python -m finnum download-nse --start 2023-10-02 --end <today> --symbols ...")
        up = files.upload()
        with open(path, "wb") as fh:
            fh.write(up[next(iter(up))])
    stocks = validate_stocks(pd.read_csv(path))
elif DATA_SOURCE in ("auto", "nse", "yahoo"):
    path = f"{RESULTS}/stocks_{DATA_SOURCE}.csv"
    if os.path.exists(path):
        stocks = validate_stocks(pd.read_csv(path))
    else:
        stocks, source = None, DATA_SOURCE
        if DATA_SOURCE in ("auto", "nse"):
            from finnum.data.nse import NSEClient
            client = NSEClient(cache_dir="/content/nse_cache", retries=2)
            try:
                probe = next(d for d in pd.bdate_range(end=date.today(), periods=10)[::-1].date
                             if client.fetch_day(d) is not None)
                print("NSE reachable (probe day", probe, ") - downloading 3 years, ~15 min")
                stocks = client.fetch_range(date(2023, 10, 2), date.today(), SYMBOLS)
                source = "nse"
            except Exception as e:
                if DATA_SOURCE == "nse":
                    raise
                print("NSE not reachable from Colab:", type(e).__name__, "- using Yahoo Finance")
        if stocks is None:
            from finnum.data.yahoo import fetch_yahoo
            stocks = fetch_yahoo(SYMBOLS, "2023-10-01", str(date.today()))
            source = "yahoo"
        print("data source used:", source)
        stocks.to_csv(path, index=False)
elif DATA_SOURCE == "sample":
    stocks = make_sample()
    print("WARNING: synthetic data - pipeline check only, do not report these numbers")
else:
    raise ValueError(DATA_SOURCE)

print(f"{len(stocks):,} rows | {stocks.symbol.nunique()} symbols | {stocks.date.min().date()} -> {stocks.date.max().date()}")
stocks.head()"""))

cells.append(md("""## 3. Build the benchmark

The benchmark file is saved once and reused, so every model and every resumed session
answers exactly the same questions."""))
cells.append(code("""from finnum.benchmark import Generator, write_jsonl, read_jsonl

bench_path = f"{RESULTS}/finnum_{DATA_SOURCE}_n{N_PER_TASK}.jsonl"  # fixed questions for every model/session
if not os.path.exists(bench_path):
    write_jsonl(Generator(stocks, seed=0).build(N_PER_TASK), bench_path)
all_items = read_jsonl(bench_path)

def first_k_per_task(items, k):
    keep, seen = set(), {}
    for it in items:
        if it["base_id"] not in keep and seen.get(it["task"], 0) < k:
            keep.add(it["base_id"]); seen[it["task"]] = seen.get(it["task"], 0) + 1
    return [it for it in items if it["base_id"] in keep]

items = first_k_per_task(all_items, QUICK_TEST) if QUICK_TEST else all_items
print(f"{len(items):,} prompts to run per model ({len(all_items):,} in the full benchmark)")
print("\\n--- example prompt ---\\n" + next(it for it in items if it["style"] == "scaled_indian" and it["task"] == "magnitude")["prompt"])"""))

cells.append(md("""## 4. Experiment 0: token cost (no GPU needed)
How many tokens does each number style cost with real tokenizers?"""))
cells.append(code("""from finnum.audit.token_audit import run as run_audit
from finnum.audit.tokenizers import RULES, load_many

toks = load_many(RULES + ["tiktoken:cl100k_base", "tiktoken:o200k_base"] + [f"hf:{m}" for m in MODELS])
audit = run_audit(stocks, all_items, toks, f"{RESULTS}/audit")
print("Mean tokens per number:")
display(audit["number_summary"])
if "prompt_summary" in audit:
    print("Mean tokens per prompt, % vs plain/csv:")
    display(audit["prompt_summary"].pivot_table(index=["style", "table_format"], columns="tokenizer", values="vs_plain_csv_pct"))
else:
    print("No real tokenizer loaded (check the [skip] messages above), so prompt costs were not measured")"""))
cells.append(code("""from IPython.display import Markdown
Markdown(open(f"{RESULTS}/audit/examples.md").read())"""))

cells.append(md("""## 5. Run the models

Each model is downloaded, run on every prompt, then removed from GPU memory.
Answers are saved after each batch, so you can re-run this cell after a disconnect."""))
cells.append(code("""from finnum.eval.runner import HFGenerator, run_eval

if HF_TOKEN:
    from huggingface_hub import login
    login(HF_TOKEN)

tag = f"quick{QUICK_TEST}" if QUICK_TEST else "full"
for model_id in MODELS:
    out = f"{RESULTS}/preds_{DATA_SOURCE}_n{N_PER_TASK}_{tag}/{model_id.replace('/', '__')}.jsonl"
    if os.path.exists(out) and sum(1 for _ in open(out)) >= len(items):
        print("already done:", model_id)
        continue
    print("loading", model_id)
    gen = HFGenerator(model_id, batch_size=BATCH_SIZE)
    try:
        run_eval(items, gen, model_id, out, batch_size=BATCH_SIZE, count_tokens=gen.count_tokens)
    finally:
        gen.close()"""))

cells.append(md("## 6. Results"))
cells.append(code("""from finnum.eval import load_predictions, accuracy_table, paired_vs_baseline, summary_lines

pred_dir = f"{RESULTS}/preds_{DATA_SOURCE}_n{N_PER_TASK}_{tag}"
preds = load_predictions(*sorted(glob.glob(f"{pred_dir}/*.jsonl")))
print(f"{len(preds):,} scored answers from {preds.model.nunique()} models\\n")

print("Accuracy by model x number style x table format")
display(accuracy_table(preds).pivot_table(index=["model", "style"], columns="table_format",
                                          values="accuracy").style.format("{:.1%}").background_gradient(axis=None, cmap="Blues"))
print("Accuracy by task")
display(accuracy_table(preds, by=["model", "task"]).pivot(index="task", columns="model",
                                                          values="accuracy").style.format("{:.1%}"))"""))
cells.append(code("""# Paired significance: every condition vs plain/csv on the SAME questions (McNemar exact, Holm-corrected)
paired = paired_vs_baseline(preds)
display(paired.sort_values(["model", "diff_pp"])[["model", "style", "table_format", "n_pairs", "acc_baseline",
        "acc_condition", "diff_pp", "p_holm", "significant"]].style.format(
        {"acc_baseline": "{:.1%}", "acc_condition": "{:.1%}", "diff_pp": "{:+.1f}", "p_holm": "{:.3g}"}))

print("HEADLINES (all questions)")
print("\\n".join(summary_lines(preds)))
print("\\nHEADLINES (only questions where the style visibly changed the prompt)")
print("\\n".join(summary_lines(preds[preds.differs_from_plain])))
if QUICK_TEST:
    print("\\nQUICK_TEST is on: too few questions for reliable statistics. Set QUICK_TEST = 0 for real results.")"""))
cells.append(code("""import matplotlib.pyplot as plt
from finnum.eval.plots import accuracy_by_style, cost_vs_accuracy

for fmt in ("csv", "markdown", "json"):
    fig = accuracy_by_style(preds, fmt)
    fig.savefig(f"{pred_dir}/accuracy_by_style_{fmt}.png", dpi=150, bbox_inches="tight")
    plt.show()
fig = cost_vs_accuracy(preds)
fig.savefig(f"{pred_dir}/cost_vs_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()"""))

cells.append(md("""## 7. Error analysis
Read some wrong answers. Patterns here often become the most interesting part of the paper,
for example "the model reads 1,23,45,678 as 1.23 million"."""))
cells.append(code("""pd.set_option("display.max_colwidth", 120)
wrong = preds[~preds.correct]
print("Unparseable answers (format failures, not number failures):",
      f"{wrong.pred.isna().mean():.0%} of wrong answers")
for (model, style), g in wrong.groupby(["model", "style"]):
    print(f"\\n=== {model} | {style} | {len(g)} wrong ===")
    display(g.sample(min(4, len(g)), random_state=0)[["task", "table_format", "gold", "pred", "output"]])"""))

cells.append(md("## 8. Save and download"))
cells.append(code("""import shutil
accuracy_table(preds).to_csv(f"{pred_dir}/accuracy.csv", index=False)
accuracy_table(preds, by=["model", "task", "style"]).to_csv(f"{pred_dir}/accuracy_by_task.csv", index=False)
paired.to_csv(f"{pred_dir}/paired_tests.csv", index=False)
archive = shutil.make_archive(f"/content/finnum_week2_results_{tag}", "zip", RESULTS)
print("results saved in", RESULTS, "and zipped to", archive)
print("To download: Files panel (left) > right-click the zip > Download")"""))

cells.append(md("## 9. Plain-text summary (easy to copy and share)"))
cells.append(code("""acc_all = accuracy_table(preds)
print(f"DATA: {DATA_SOURCE} | questions per task: {QUICK_TEST or N_PER_TASK} | prompts per model: {len(items)}")
print("\\nACCURACY (rows: model/style, columns: table format)")
print(acc_all.pivot_table(index=["model", "style"], columns="table_format", values="accuracy").round(3).to_string())
print("\\nMEAN PROMPT TOKENS (model's own tokenizer)")
print(acc_all.pivot_table(index=["model", "style"], columns="table_format", values="mean_prompt_tokens").round(1).to_string())
print("\\nACCURACY BY TASK")
print(accuracy_table(preds, by=["model", "task"]).pivot(index="task", columns="model", values="accuracy").round(3).to_string())
print("\\nPAIRED TESTS vs plain/csv")
print(paired[["model", "style", "table_format", "diff_pp", "p_holm", "significant"]].round(4).to_string(index=False))
print("\\nHEADLINES")
print("\\n".join(summary_lines(preds)))
print("\\nHEADLINES (style visibly changed the prompt)")
print("\\n".join(summary_lines(preds[preds.differs_from_plain])))
print("\\nFINNUM_SUMMARY_DONE")"""))

cells.append(md("""## Next steps
- Run with `QUICK_TEST = 0` for the real numbers. Small models take minutes; the 3.8B model is much slower.
- Send back `accuracy.csv`, `paired_tests.csv` and the charts. We'll read the results together and
  decide which conditions to carry into **week 3**: a small GPT trained from scratch with
  digits split from the left vs from the right."""))

import base64, io, zipfile
root = Path(__file__).resolve().parents[1]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted((root / "src").rglob("*.py")):
        zf.write(f, f.relative_to(root))
    zf.write(root / "pyproject.toml", "pyproject.toml")
b64 = base64.b64encode(buf.getvalue()).decode()
for c in cells:
    if c.cell_type == "code" and "__FINNUM_ZIP_B64__" in c.source:
        c.source = c.source.replace("__FINNUM_ZIP_B64__", b64)

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_baseline_eval_colab.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
