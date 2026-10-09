"""Builds notebooks/05_final_analysis_colab.ipynb (week 7). Run: python scripts/make_notebook_week7.py"""
from nbcommon import code, md, setup_cell, write

cells = [md("""# FinNum week 7: the full picture

This notebook reads the saved results of weeks 2, 3 and 4–6 from Google Drive (`finnum_results/`)
and puts them together:

- zero-shot LLMs (week 2) and retrofitted methods (weeks 4–6) on the same cost vs accuracy chart
- how much each model depends on how numbers are written
- error types, for example scale mistakes such as mixing up lakh and crore
- the tiny-GPT tokenization results (week 3)

It needs no GPU. Run all, then copy the final cell's output to Claude."""),
    code("""SAVE_TO_DRIVE = True   # results of the earlier notebooks live in Drive
PREFER = "full"        # use full runs when present, else quick runs
"""),
    md("## 1. Setup"),
    setup_cell("yfinance"),
    md("## 2. Find the results"),
    code("""import glob
def newest(pattern):
    hits = sorted(glob.glob(f"{RESULTS}/{pattern}"))
    full = [h for h in hits if PREFER in h]
    return (full or hits or [None])[-1]

W2 = newest("preds_*")
W3 = newest("week3_*")
W46 = newest("retrofit_*")
print("week 2  :", W2); print("week 3  :", W3); print("weeks 4-6:", W46)
if not any([W2, W3, W46]):
    print("Nothing found - run the earlier notebooks with SAVE_TO_DRIVE = True first.")"""),
    md("## 3. Weeks 2 + 4–6 together"),
    code("""import pandas as pd, matplotlib.pyplot as plt
from finnum.analysis import load_dir, robustness, frontier, error_breakdown, plot_frontier, plot_robustness, paired_models
frames = []
if W2: frames.append(load_dir(W2, "zero-shot"))
if W46: frames.append(load_dir(W46, "retrofit"))
both = pd.concat(frames, ignore_index=True) if frames else None
if both is not None:
    display(robustness(both).style.format("{:.3f}"))
    display(frontier(both))
    display(error_breakdown(both))
    fig = plot_frontier(both); fig.savefig(f"{RESULTS}/final_frontier.png", dpi=150, bbox_inches="tight"); plt.show()
    fig = plot_robustness(both); fig.savefig(f"{RESULTS}/final_robustness.png", dpi=150, bbox_inches="tight"); plt.show()
    print("Note: week 2 and weeks 4-6 used different test questions, so compare them on accuracy and cost,")
    print("not with paired tests. Paired tests are only valid inside one experiment (shown below).")"""),
    md("## 4. Week 3: tokenization from scratch"),
    code("""if W3:
    from finnum.tokexp.report import load_results, plot_accuracy, plot_length
    w3 = load_results(W3)
    fig = plot_accuracy(w3); plt.show()
    fig = plot_length(w3); plt.show()"""),
    md("## 5. Plain-text summary (copy everything below and paste it to Claude)"),
    code("""from finnum.analysis import text_report
if W2:
    w2 = load_dir(W2, "zero-shot")
    print("=" * 30, "WEEK 2 (zero-shot)", "=" * 30)
    first = sorted(w2["model"].unique())[0]
    print(text_report(w2, first))
if W46:
    print("=" * 30, "WEEKS 4-6 (retrofit)", "=" * 30)
    print(text_report(load_dir(W46, "retrofit"), "text"))
if both is not None:
    print("=" * 30, "COMBINED COST vs ACCURACY", "=" * 30)
    print(frontier(both).round(3).to_string(index=False))
if W3:
    from finnum.tokexp.report import text_summary
    print("=" * 30, "WEEK 3 (from scratch)", "=" * 30)
    print(text_summary(w3))
print("\\nFINNUM_W7_DONE")"""),
]
write(cells, "05_final_analysis_colab.ipynb")
