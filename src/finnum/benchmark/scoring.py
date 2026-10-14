"""Score a model's free-text answer against a FinNum item."""
from __future__ import annotations

import re

from finnum.numbers import find_numbers

_YES = re.compile(r"\b(yes|true)\b", re.I)
_NO = re.compile(r"\b(no|false)\b", re.I)
# dates (full or cut off mid-way, e.g. "2024-12-") and times must never be read as the answer
_DATE = re.compile(r"\b\d{4}-\d{1,2}(?:-\d{0,2})?|\b\d{1,2}[-/](?:\d{1,2}|[A-Za-z]{3})[-/]\d{2,4}\b|\b\d{1,2}:\d{2}\b")
_ANSWER_CUE = re.compile(r"(?:answer|result|average|mean|change)\s*(?:is|=|:|of)\s*", re.I)

# Secondary, lenient tolerances (reported next to the strict, pre-registered ones):
# mean -> within 1% of the true value, pct_change -> within 0.5 percentage points.
APPROX_REL = {"mean": 0.01, "lookup": 0.0}
APPROX_ABS = {"pct_change": 0.5}


def extract_answer(text: str, answer_type: str, options: list[str] | None = None):
    """Pull the final answer out of model output. Returns None if nothing usable."""
    if answer_type == "number":
        return _extract_number(text)
    if answer_type == "choice":
        hits = [(m.start(), o) for o in options or [] for m in re.finditer(rf"\b{re.escape(o)}\b", text)]
        return max(hits)[1] if hits else None
    if answer_type == "bool":
        y, n = _YES.search(text), _NO.search(text)
        if y and (not n or y.start() < n.start()):
            return True
        if n:
            return False
        return None
    raise ValueError(answer_type)


def _extract_number(text: str) -> float | None:
    """Pick the answer number from free text.

    1. If the output starts with a number, the model answered first ("46.39\\n\\nHere's the
       calculation: ...") -> take that number.
    2. Else if there is an explicit cue ("the answer is X") -> the number right after the last cue.
    3. Else (reasoning, then answer) -> the last number.
    Dates and times are masked first so "from 2024-12-..." never yields 12.
    """
    masked = _DATE.sub(lambda m: " " * len(m.group(0)), text.strip())
    first_line = masked.split("\n", 1)[0]
    if re.match(r"\s*(?:\*\*)?\s*(?:₹|rs\.?|inr)?\s*[-−+(]?\s*\d", first_line, re.I):
        nums = find_numbers(first_line)
        if nums:
            return nums[0].value
    cues = list(_ANSWER_CUE.finditer(masked))
    if cues:
        after = find_numbers(masked[cues[-1].end():])
        if after:
            return after[0].value
    nums = find_numbers(masked)
    return nums[-1].value if nums else None


def is_correct_approx(text: str, item: dict) -> bool:
    """Lenient score: number tasks use APPROX_* tolerances, other tasks are scored as usual."""
    if item["answer_type"] != "number":
        return is_correct(text, item)
    pred = extract_answer(text, "number")
    if pred is None:
        return False
    gold = float(item["gold"])
    task = item.get("task", "")
    if task in APPROX_ABS:
        return abs(pred - gold) <= APPROX_ABS[task] + 1e-9
    rel = APPROX_REL.get(task, 0.0)
    return abs(pred - gold) <= max(item.get("tolerance", 0.0), rel * abs(gold)) + 1e-9


TOLERANCE = {"lookup": 0.005, "pct_change": 0.011, "mean": 0.011}


def rescore(preds):
    """Re-score saved predictions from their raw ``output`` (no model needed).

    Number tasks are re-extracted with the current rules (``pred``/``correct`` updated);
    every row gets ``correct_approx`` (lenient tolerance for number tasks, same as strict otherwise).
    """
    import re as _re
    think = _re.compile(r"<think>.*?</think>", _re.S)
    df = preds.copy()
    pred, strict, approx = list(df["pred"]), list(df["correct"].astype(bool)), list(df["correct"].astype(bool))
    for i, r in enumerate(df.itertuples(index=False)):
        if r.task not in TOLERANCE:
            continue
        text = think.sub("", r.output or "").strip()
        item = {"answer_type": "number", "gold": r.gold, "task": r.task, "tolerance": TOLERANCE[r.task]}
        pred[i] = extract_answer(text, "number")
        strict[i] = is_correct(text, item)
        approx[i] = is_correct_approx(text, item)
    df["pred"], df["correct"], df["correct_approx"] = pred, strict, approx
    return df


def is_correct(text: str, item: dict) -> bool:
    pred = extract_answer(text, item["answer_type"], item.get("options"))
    if pred is None:
        return False
    if item["answer_type"] == "number":
        return abs(pred - float(item["gold"])) <= item["tolerance"] + 1e-9
    return pred == item["gold"]
