"""Builds notebooks/04_retrofit_number_encodings_colab.ipynb (weeks 4-6). Run: python scripts/make_notebook_week4.py"""
from nbcommon import code, md, setup_cell, write

cells = [md("""# FinNum weeks 4–6: teaching a pretrained LLM to read numbers by their VALUE

We take **Qwen2.5-0.5B-Instruct** and fine-tune it with LoRA on FinNum questions, six times.
Every run uses the same data and the same LoRA settings. The only difference is how numbers
in the prompt reach the model:

| method | what happens to `2,915.35` | source |
|---|---|---|
| `text` | stays as text (`2`,`,`,`915`,`.`,`35`) | control: fine-tuning alone |
| `xval` | replaced by one `<num>` token, embedding scaled by the value | Golkar et al. 2023 |
| `fone` | one `<num>` token with Fourier digit features (digit-exact) | Zhou et al., ICLR 2026 |
| `bittokens` | one `<num>` token with the float64 bit pattern | Kreitner et al. 2025 |
| `value_prefix` | `<num>` token **added before** the digits (keeps the text) | Dutulescu et al. 2026 style |
| `fone_units` | **ours**: one `<num>` token with Fourier features **plus** how the number was written (₹, %, lakh/crore, Indian vs international grouping) | this project |

The test questions come from the **last 120 trading days**, and training uses only earlier dates, so
nothing leaks from training into the test. Every test question is asked in all 5 number styles
× 3 table formats.

**Before you start:** go to *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.
Start with `QUICK_TEST = True` (a few minutes), then set it to `False` (about 6 methods × 10–20 min)."""),
    code("""# ======================= SETTINGS =======================
QUICK_TEST = True

DATA_SOURCE = "auto"   # "auto" (NSE -> Yahoo fallback), "yahoo", "nse", "sample" (synthetic, testing only), or a CSV path
METHODS = ["text", "xval", "fone", "bittokens", "value_prefix", "fone_units"]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_TRAIN_PER_TASK = 300   # 6 tasks x 300 questions x 3 renderings = 5,400 training examples
N_TEST_PER_TASK = 30     # 6 tasks x 30 questions x 15 renderings = 2,700 test prompts per method
EPOCHS = 1
SAVE_TO_DRIVE = True     # keeps results across disconnects and lets the week 7 notebook find them

if QUICK_TEST:
    METHODS, N_TRAIN_PER_TASK, N_TEST_PER_TASK = ["text", "fone_units"], 20, 3
"""),
    md("## 1. Setup"),
    setup_cell("peft yfinance"),
    md("## 2. Data: train on earlier dates, test on the last 120 trading days"),
    code("""import pandas as pd
from collections import Counter
from finnum.data.auto import load_stocks
from finnum.retrofit.data import build_items

stocks, used = load_stocks(DATA_SOURCE, cache_csv=f"{RESULTS}/stocks_{DATA_SOURCE}.csv"
                           if DATA_SOURCE in ("auto", "nse", "yahoo") else None, nse_cache_dir="/content/nse_cache")
train_items, test_items = build_items(stocks, N_TRAIN_PER_TASK, N_TEST_PER_TASK)
print(f"data: {used} | train examples {len(train_items):,} | test prompts {len(test_items):,}")
display(pd.DataFrame({"train": Counter(i["task"] for i in train_items), "test": Counter(i["task"] for i in test_items)}))"""),
    code("""# What the model actually receives under each method
from finnum.retrofit.preprocess import mark
ex = next(i for i in test_items if i["task"] == "magnitude" and i["style"] == "scaled_indian" and i["table_format"] == "csv")
for m in METHODS:
    print(f"--- {m} ---"); print(mark(ex["prompt"], m)[0].split("Question:")[0][-260:])"""),
    md("## 3. Train and evaluate each method\nFinished methods are skipped on re-run. A method interrupted part-way restarts from scratch."),
    code("""from finnum.retrofit.lm import RetrofitConfig, run_method
cfg = RetrofitConfig(model_id=MODEL_ID, epochs=EPOCHS)
tag = ("quick" if QUICK_TEST else "full") + f"_{used.split(':')[0]}"
OUT = f"{RESULTS}/retrofit_{tag}"
for m in METHODS:
    print(f"\\n===== {m} =====")
    run_method(m, train_items, test_items, cfg, OUT)
print("saved to", OUT)"""),
    md("## 4. Results"),
    code("""import glob, json
import matplotlib.pyplot as plt
from finnum.analysis import load_dir, paired_models, robustness, frontier, error_breakdown, plot_frontier, plot_robustness
from finnum.eval import accuracy_table

preds = load_dir(OUT, "retrofit")
display(accuracy_table(preds, by=["model", "task"]).pivot(index="model", columns="task", values="accuracy").style.format("{:.1%}"))
print("Paired vs text-only LoRA (same prompts):")
display(paired_models(preds, "text"))
display(robustness(preds).style.format("{:.3f}"))
display(frontier(preds))
display(error_breakdown(preds))
fig = plot_frontier(preds, "Weeks 4-6: number encodings retrofitted into Qwen2.5-0.5B"); fig.savefig(f"{OUT}/frontier.png", dpi=150, bbox_inches="tight"); plt.show()
fig = plot_robustness(preds); fig.savefig(f"{OUT}/robustness.png", dpi=150, bbox_inches="tight"); plt.show()"""),
    code("""# Training curves and cost
for f in sorted(glob.glob(f"{OUT}/*_train.json")):
    log = json.load(open(f))
    print(f"{log['method']:<13} {log['train_minutes']:>6.1f} min  trainable params {log['trainable_params']:,}  "
          f"final loss {log['losses'][-1]['loss'] if log['losses'] else float('nan'):.4f}")"""),
    md("## 5. Plain-text summary (copy everything below and paste it to Claude)"),
    code("""from finnum.analysis import text_report
print(f"DATA: {used} | quick={QUICK_TEST} | model={MODEL_ID} | train={len(train_items)} test={len(test_items)}")
print(text_report(preds, "text"))
print("\\nFINNUM_W46_DONE")"""),
]
write(cells, "04_retrofit_number_encodings_colab.ipynb")
