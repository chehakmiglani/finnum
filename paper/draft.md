# How Should Language Models Read Market Numbers? Tokenization, Number Encodings and Indian Notation on Real NSE Data

*Draft. The method sections are complete. Every `[RESULT]` placeholder gets filled from the Colab
summaries; nothing in this draft is a measured number yet.*

**Author:** Chehak Miglani

## Abstract

Large language models read numbers as text fragments. The same value can reach a model as
`12345678`, `12,345,678`, `1,23,45,678`, `1.23 crore` or `12.35 million`, and each form is split
into different tokens. We introduce **FinNum**, a benchmark built from real National Stock Exchange
of India data. Every question is asked in 5 number styles × 3 table formats, which allows paired
comparisons. We run three studies. (1) Zero-shot evaluation of open LLMs to measure how accuracy and
token cost depend on number style. (2) A controlled from-scratch study in which small GPTs differ only
in how digits are grouped into tokens: left-to-right, right-to-left, Indian lakh/crore grouping, or
single digits. (3) Retrofitting four published single-token number encodings (xVal, FoNE, BitTokens,
value prefix) into a pretrained 0.5B model, plus a unit-aware extension that also encodes how a number
was written (₹, %, lakh/crore, digit grouping). [RESULT: one sentence per study.]

## 1 Introduction

- Financial text and tables are mostly numbers, and LLM cost scales with tokens.
- Tokenizers split numbers inconsistently: GPT-4-style tokenizers group up to 3 digits from the left,
  so the pieces do not line up with place value. Prior work found that grouping from the right
  improves arithmetic (Singh & Strouse, 2024).
- Indian notation adds a further layer: lakh/crore grouping (12,34,567) and scale words.
  We know of no work that studies it.
- Recent single-token number encodings (xVal; FoNE, ICLR 2026; BitTokens; value-aware prefix tokens)
  are evaluated mostly on synthetic arithmetic with models trained from scratch.
- **Contributions:**
  1. The FinNum benchmark: real NSE data, 6 task types, 15 paired renderings per question.
  2. A controlled tokenization study that includes Indian grouping and longer-number generalisation.
  3. The first head-to-head retrofit of four number encodings into one pretrained LLM on real
     financial data, with a date-based train/test split.
  4. A unit-aware number encoding.

## 2 Related work

- **Number tokenization:** Singh & Strouse 2024 (Tokenization Counts); NumeroLogic 2024.
- **Continuous and single-token number encodings:** xVal (Golkar et al., 2023); FoNE (Zhou et al.,
  ICLR 2026); BitTokens (Kreitner et al., 2025); value-aware representations (Dutulescu et al.,
  2026); interleaved text and number embeddings (2024).
- **Units:** CONE (2026), which is unit-aware but encoder-only.
- **Table serialization and token cost:** TOON and JTON (2026); 11-format comparisons (2025).
- **LLMs on structured financial data:** FinFRE-RAG (2025).

## 3 FinNum benchmark

- **Data:** daily NSE equity bhavcopies (UDiFF and legacy formats), or Yahoo Finance as a fallback, for
  15 liquid and illiquid symbols spanning prices from ₹20 to ₹1.3 lakh. [RESULT: date range, rows, source used]
- **Tasks:**
  - lookup
  - % change
  - mean
  - compare
  - magnitude (highest turnover, with mixed units)
  - anomaly (volume > 3× the 20-day average, balanced classes, ambiguous ratios excluded)
- **Renderings:**
  - Number styles: `plain`, `intl`, `indian`, `scaled_indian` (lakh/crore), `scaled_intl` (million/billion).
  - Table formats: CSV, Markdown, JSON.
  - All 15 renderings share one exact gold answer, so every comparison is paired. A test checks each
    gold answer by recomputing it from the rendered text alone.
- **Scoring:**
  - Numbers: the last number in the output, within a tolerance.
  - Choices and booleans: pattern matching.
  - Prompts whose rendering doesn't change between styles are flagged (`differs_from_plain`).

## 4 Study 1: zero-shot LLMs (week 2)

- **Models:** Qwen2.5-0.5B/1.5B-Instruct, SmolLM2-1.7B-Instruct, Phi-3.5-mini-instruct. Greedy decoding.
- **Statistics:** Wilson 95% CIs; exact McNemar tests on paired questions vs plain/CSV; Holm correction.
- **Token cost:** GPT-4 (cl100k), GPT-4o (o200k) and each model's own tokenizer.

[RESULT: accuracy table; cost table; significant style effects; error types]

## 5 Study 2: tokenization from scratch (week 3)

- **Setup:**
  - 4-layer, 256-dim GPTs (~3M parameters), identical except for the digit-grouping scheme.
  - Chunk schemes share a 1,110-token digit vocabulary.
  - Loss only on answer tokens.
  - Randomized position offsets during training, so longer test sequences don't land on untrained
    positions.
- **Tasks:**
  - add / subtract real volumes and turnovers
  - compare
  - % change between prices
  - rupees → crore
- **Test sets:**
  - Same-length test: operands never used in training.
  - Longer-number test: 9–11 digits, against 4–8 in training.
- **Seeds:** 3 seeds, mean ± sd.
- **Ablation:** answers written least-significant digit first.

[RESULT: accuracy by scheme; by number length; reversed-answer ablation; failure examples]

## 6 Study 3: retrofitting number encodings (weeks 4–6)

- **Base model:** Qwen2.5-0.5B-Instruct with LoRA (r=16) on all attention and MLP projections.
  Identical data and settings across methods.
- **Methods:**
  - `text`: the control.
  - `xval`: a `<num>` embedding scaled by signed log-magnitude.
  - `fone`: Fourier features on periods 10⁻¹…10¹², digit-exact. A unit test decodes them back exactly.
  - `bittokens`: the float64 bit pattern.
  - `value_prefix`: FoNE features on a `<num>` token inserted before the digits.
  - `fone_units` (ours): FoNE plus 14 notation features (percent, currency, scale word, Indian/intl
    grouping, sign, magnitude, decimals).
- **Encoder:** a 3-layer MLP maps the features to the embedding space, replacing the `<num>` token's
  embedding.
- **Split:** by date. The last 120 trading days are test; earlier days are train.
- **Training data:** each training question is shown in 3 random renderings. Each test question is
  shown in all 15.

**Deviations from the original papers, stated for transparency:**
- xVal is scaled by log-magnitude instead of the raw value, because of the 10⁻² to 10¹² range.
- FoNE is used on the input side only. Answers are decoded as text by the LM head, so every method is
  scored identically.

[RESULT: accuracy and paired tests vs `text`; robustness spread across styles; cost vs accuracy
frontier; error types (scale ×10ᵏ, sign, close, far)]

## 7 Discussion

[RESULT-dependent. Questions to answer:
- Does place-value alignment matter more than token count?
- Do value encodings remove the dependence on number style?
- Does notation information help or hurt?
- How much do the cost savings from single-token numbers save at scale?]

## 8 Limitations

- Small models (≤3.8B zero-shot; 0.5B retrofit). The results may not transfer to frontier models.
- One market (NSE) and 15 symbols.
- Retrofit methods were trained with one LoRA configuration and one seed (unless re-run).
- The input-side encodings don't give the model a numeric output head.
- Yahoo-sourced turnover is approximate.

## 9 Reproducibility

- **Code and tests:** all in this repository, with 100+ automated tests. Every notebook runs on a free
  Colab T4.
- **Execution:** `RUNNING.md` gives the exact order to run the notebooks.
- **Seeds:** fixed throughout.

## References

1. Singh, A. K., & Strouse, D. J. (2024). *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs.* arXiv:2402.14903.
2. Golkar, S. et al. (2023). *xVal: A continuous number encoding for large language models.* arXiv:2310.02989.
3. Zhou, T. et al. (2026). *FoNE: Precise single-token number embeddings via Fourier features.* ICLR 2026. arXiv:2502.09741.
4. Kreitner, L. et al. (2025). *Efficient numeracy in language models through single-token number embeddings.* arXiv:2510.06824.
5. Dutulescu, A., Ruseti, S., & Dascalu, M. (2026). *Value-aware numerical representations for transformer language models.* arXiv:2601.09706.
6. Schwartz, E. et al. (2024). *NumeroLogic: Number encoding for enhanced LLMs' numerical reasoning.* arXiv:2404.00459.
7. *CONE: Embeddings for complex numerical data preserving unit and variable semantics.* (2026). arXiv:2603.04741.
8. *JTON: A token-efficient JSON superset with Zen Grid tabular encoding for LLMs.* (2026). arXiv:2604.05865.
9. Ruoss, A. et al. (2023). *Randomized positional encodings boost length generalization of transformers.* ACL 2023.
10. *Understanding structured financial data with LLMs: a case study on fraud detection.* (2025). arXiv:2512.13040.

*Check every reference against the paper itself before submission. Some were identified from search
summaries, and author lists marked "et al." need completing.*
