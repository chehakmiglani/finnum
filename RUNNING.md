# Running FinNum: the Colab checklist

All GPU work runs on **free Google Colab (T4)**. Each notebook is self-contained, with the code built in.
To run one: upload it (*File → Upload notebook*), choose *Runtime → Change runtime type → T4 GPU*,
then choose *Runtime → Run all*.

Results are saved to **Google Drive → `finnum_results/`** (`SAVE_TO_DRIVE = True`), so:
- a disconnect loses at most the item in progress. Re-run all cells to resume.
- the week 7 notebook can find every earlier result.

Colab asks for permission to access Google Drive the first time. Click **Connect to Google Drive** and allow.

## Order

| # | Notebook | Quick test | Full run | Paste back to Claude |
|---|---|---|---|---|
| 1 | `02_baseline_eval_colab.ipynb` (week 2) | `QUICK_TEST = 5`, ~5–10 min | `QUICK_TEST = 0`, ~1–2 h (Phi-3.5 is the slow one) | last cell, ends with `FINNUM_SUMMARY_DONE` |
| 2 | `03_tokenization_from_scratch_colab.ipynb` (week 3) | `QUICK_TEST = True`, ~5 min | `QUICK_TEST = False`, 12 small models | last cell, ends with `FINNUM_W3_DONE` |
| 3 | `04_retrofit_number_encodings_colab.ipynb` (weeks 4–6) | `QUICK_TEST = True`, ~10–15 min | `QUICK_TEST = False`, 6 methods | last cell, ends with `FINNUM_W46_DONE` |
| 4 | `05_final_analysis_colab.ipynb` (week 7) | n/a (no GPU needed) | runs in minutes | last cell, ends with `FINNUM_W7_DONE` |

These times are estimates. Each notebook prints a live estimate of the time left while it runs.

**Do all the quick tests first.** They catch problems in minutes instead of hours. If a cell fails,
copy the red error text and paste it to Claude.

## Data

`DATA_SOURCE = "auto"` tries NSE first and falls back to Yahoo Finance, because NSE often blocks
Colab. The notebook prints which source it used. With Yahoo, prices and volumes are real, but turnover
is estimated as volume × (high+low+close)/3. Say so in the paper.

To use NSE data downloaded on your laptop (`python -m finnum download-nse ...`), upload
`stocks.csv` to Drive and set `DATA_SOURCE = "/content/drive/MyDrive/finnum_results/stocks.csv"`.

## Common problems

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Lower `BATCH_SIZE` (week 2) or set `cfg.batch_size = 4` (weeks 4–6) |
| Session disconnected | Reconnect and run all again. Finished parts are skipped. |
| `No GPU!` | *Runtime → Change runtime type → T4 GPU*. Free GPU time is limited per day, so try later or use Kaggle. |
| Gated model error (Llama/Gemma) | Accept the licence on huggingface.co and set `HF_TOKEN` in the settings cell |
