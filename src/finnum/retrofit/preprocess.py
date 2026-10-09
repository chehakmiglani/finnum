"""Turn FinNum prompts into (text with <num> markers, number features) - pure python.

Methods
-------
  text          no change; LoRA fine-tuning only. The control: any gain over this comes from the encoding.
  xval          each number REPLACED by one <num> token whose embedding is scaled by its value
  fone          each number REPLACED by one <num> token with Fourier digit features
  bittokens     each number REPLACED by one <num> token with float64 bit features
  value_prefix  a <num> token (fone features) is INSERTED BEFORE the number, digits are kept

Dates (YYYY-MM-DD) are left as text: they are identifiers here, not quantities.
"""
from __future__ import annotations

import re

import numpy as np

from finnum.numbers import find_numbers
from finnum.numbers.formats import ParsedNumber

from .features import FEATURE_DIMS, features

NUM_TOKEN = "<num>"
ANSWER_PREFIX = "\nAnswer: "

METHODS: dict[str, dict] = {
    "text": {"mode": None, "features": None},
    "xval": {"mode": "replace", "features": "xval"},
    "fone": {"mode": "replace", "features": "fone"},
    "bittokens": {"mode": "replace", "features": "bits"},
    "value_prefix": {"mode": "prefix", "features": "fone"},
}

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def feature_dim(method: str) -> int:
    kind = METHODS[method]["features"]
    return FEATURE_DIMS[kind] if kind else 0


def quantities(text: str) -> list[ParsedNumber]:
    """Numbers that are quantities.

    Dates are masked out BEFORE parsing (same length, so positions are unchanged); otherwise a CSV
    row like ``2025-05-12,147.50`` would be read as the grouped number ``12,147.50``.
    """
    masked = _DATE.sub(lambda m: "#" * len(m.group(0)), text)
    return find_numbers(masked)


def mark(text: str, method: str) -> tuple[str, np.ndarray | None]:
    """Return (text with <num> markers, features array [n_numbers, dim] or None)."""
    spec = METHODS[method]
    if spec["mode"] is None:
        return text, None
    nums = quantities(text)
    out, pos = [], 0
    for n in nums:
        out.append(text[pos:n.start])
        if spec["mode"] == "replace":
            # keep surrounding whitespace that the regex may have swallowed
            lead = text[n.start:n.end][: len(text[n.start:n.end]) - len(text[n.start:n.end].lstrip())]
            out.append(lead + NUM_TOKEN)
        else:
            out.append(NUM_TOKEN + text[n.start:n.end])
        pos = n.end
    out.append(text[pos:])
    feats = features(nums, spec["features"]) if nums else np.zeros((0, feature_dim(method)), np.float32)
    return "".join(out), feats


def answer_text(item: dict) -> str:
    g = item["gold"]
    if item["answer_type"] == "bool":
        return "yes" if g else "no"
    if item["answer_type"] == "choice":
        return str(g)
    return f"{float(g):.2f}"


def encode(item_or_prompt, tokenizer, method: str, with_answer: bool, num_token_id: int | None = None):
    """Token ids (+ labels with -100 on the prompt) and features for one example.

    ``tokenizer`` is any object with ``encode(text, add_special_tokens=False) -> list[int]``
    and an ``eos_token_id``. Raises if the number of <num> tokens != number of features.
    """
    prompt = item_or_prompt["prompt"] if isinstance(item_or_prompt, dict) else item_or_prompt
    marked, feats = mark(prompt, method)
    p_ids = tokenizer.encode(marked + ANSWER_PREFIX, add_special_tokens=False)
    ids, labels = list(p_ids), [-100] * len(p_ids)
    if with_answer:
        a_ids = tokenizer.encode(answer_text(item_or_prompt), add_special_tokens=False) + [tokenizer.eos_token_id]
        ids += a_ids
        labels += a_ids
    if feats is not None and num_token_id is not None:
        n_marks = sum(1 for i in ids if i == num_token_id)
        if n_marks != len(feats):
            raise ValueError(f"{n_marks} <num> tokens but {len(feats)} numbers - tokenizer splits <num>?")
    return {"input_ids": ids, "labels": labels, "features": feats, "prompt_tokens": len(p_ids)}
