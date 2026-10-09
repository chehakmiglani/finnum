"""Score a model's free-text answer against a FinNum item."""
from __future__ import annotations

import re

from finnum.numbers import find_numbers

_YES = re.compile(r"\b(yes|true)\b", re.I)
_NO = re.compile(r"\b(no|false)\b", re.I)


def extract_answer(text: str, answer_type: str, options: list[str] | None = None):
    """Pull the final answer out of model output. Returns None if nothing usable."""
    if answer_type == "number":
        nums = find_numbers(text)
        return nums[-1].value if nums else None  # models often restate inputs first
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


def is_correct(text: str, item: dict) -> bool:
    pred = extract_answer(text, item["answer_type"], item.get("options"))
    if pred is None:
        return False
    if item["answer_type"] == "number":
        return abs(pred - float(item["gold"])) <= item["tolerance"] + 1e-9
    return pred == item["gold"]
