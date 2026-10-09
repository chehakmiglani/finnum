"""Arithmetic tasks built from real market numbers, for training tiny GPTs.

Each example is a prompt and an exact answer, e.g.

  add 5123456 2087345=7210801        sum of two daily volumes
  sub 14912345678 9876543210=...     difference of two turnovers
  cmp 45120000 45210000=<            which is bigger
  pct 2889.40 2915.35=0.90           % change between two closes (2 dp)
  cr 14912345678=1491.23             rupees -> crore (2 dp)

Splits:
  train     in-distribution lengths, operands from the "train" half of the values
  test_id   same lengths, operands the model never saw in training
  test_ood  LONGER numbers than anything in training - tests whether the model
            learned an algorithm (helped by place-value-aligned tokens) or memorised
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

TASKS = ("add", "sub", "cmp", "pct", "cr")

# digit-length ranges (integer part)                    in-distribution   longer (OOD)
LENGTHS = {
    "add": ((4, 8), (9, 11)),
    "sub": ((4, 8), (9, 11)),
    "cmp": ((4, 8), (9, 11)),
    "pct": ((2, 4), (5, 6)),    # price integer digits: <10,000 vs >=10,000 (MRF, PAGEIND...)
    "cr": ((8, 10), (11, 12)),  # rupee amounts
}


@dataclass
class Example:
    task: str
    split: str
    prompt: str
    answer: str
    n_digits: int

    @property
    def text(self) -> str:
        return self.prompt + self.answer


def _is_test_value(v) -> bool:
    """Stable 20% of values are reserved for testing (never used in training)."""
    return zlib.crc32(str(v).encode()) % 5 == 0


def _pools(df: pd.DataFrame) -> dict[str, list]:
    ints = pd.concat([df["volume"].astype("int64"), df["turnover"].round().astype("int64")]).unique().tolist()
    prices = sorted({f"{p:.2f}" for p in df["close"]})
    return {"int": [int(v) for v in ints if v > 0], "price": prices}


def _by_len(values, lo: int, hi: int, key=len):
    return [v for v in values if lo <= key(v) <= hi]


def _round2(x: Decimal) -> str:
    return str(x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class TaskBuilder:
    def __init__(self, df: pd.DataFrame, seed: int = 0):
        self.rng = random.Random(seed)
        self.pools = _pools(df)
        self._cache: dict[tuple[str, str], tuple[list, dict[int, list]]] = {}

    def _pool(self, task: str, split: str) -> list:
        if (task, split) not in self._cache:
            vals = self._compute_pool(task, split)
            by_len: dict[int, list] = {}
            for v in vals:
                by_len.setdefault(len(str(v)), []).append(v)
            self._cache[(task, split)] = (vals, by_len)
        return self._cache[(task, split)][0]

    def _compute_pool(self, task: str, split: str) -> list:
        (lo, hi), (olo, ohi) = LENGTHS[task]
        if task == "pct":
            vals = self.pools["price"]
            key = lambda p: len(p.split(".")[0])  # noqa: E731
        else:
            vals = self.pools["int"]
            key = lambda v: len(str(v))  # noqa: E731
        if split == "test_ood":
            return _by_len(vals, olo, ohi, key)
        vals = _by_len(vals, lo, hi, key)
        want_test = split == "test_id"
        return [v for v in vals if _is_test_value(v) == want_test]

    def make(self, task: str, split: str) -> Example:
        pool = self._pool(task, split)
        if len(pool) < 2:
            raise ValueError(f"not enough numbers for {task}/{split} - use more symbols or dates")
        r = self.rng
        if task == "add":
            a, b = r.sample(pool, 2)
            return Example(task, split, f"add {a} {b}=", str(a + b), max(len(str(a)), len(str(b))))
        if task == "sub":
            a, b = sorted(r.sample(pool, 2), reverse=True)
            return Example(task, split, f"sub {a} {b}=", str(a - b), len(str(a)))
        if task == "cmp":
            a = r.choice(pool)
            same_len = self._cache[(task, split)][1][len(str(a))]
            b = a
            while b == a:  # 70% same length (hard), else any length
                b = r.choice(same_len) if len(same_len) > 1 and r.random() < 0.7 else r.choice(pool)
            return Example(task, split, f"cmp {a} {b}=", ">" if a > b else "<", max(len(str(a)), len(str(b))))
        if task == "pct":
            a = r.choice(pool)
            # a realistic second price: within +-10% of the first, on the 0.05 tick
            b = Decimal(a) * (1 + Decimal(r.uniform(-0.1, 0.1)).quantize(Decimal("0.0001")))
            b = (b * 20).quantize(Decimal(1), rounding=ROUND_HALF_UP) / 20
            b = b.quantize(Decimal("0.01"))
            pct = (b / Decimal(a) - 1) * 100
            return Example(task, split, f"pct {a} {b}=", _round2(pct), len(a.split(".")[0]))
        if task == "cr":
            a = r.choice(pool)
            # half the time use a combined amount (sum of two real values) for more variety
            lo, hi = LENGTHS[task][1 if split == "test_ood" else 0]
            if r.random() < 0.5:
                s2 = a + r.choice(pool)
                if lo <= len(str(s2)) <= hi:
                    a = s2
            return Example(task, split, f"cr {a}=", _round2(Decimal(a) / Decimal(10**7)), len(str(a)))
        raise ValueError(task)

    def build(self, n_train: int, n_test: int, tasks=TASKS, reverse_answers: bool = False) -> dict[str, list[Example]]:
        """Unique prompts per split; test prompts never appear in train.

        reverse_answers: write add/sub answers least-significant digit first
        ("7210801" -> "1080127"), so the model produces digits in the order the
        carries flow. A classic ablation: it separates "tokenization" effects
        from "generation order" effects.
        """
        out: dict[str, list[Example]] = {"train": [], "test_id": [], "test_ood": []}
        seen: set[str] = set()
        for split, n in (("test_id", n_test), ("test_ood", n_test), ("train", n_train)):
            for task in tasks:
                got, tries = 0, 0
                while got < n and tries < n * 20:
                    tries += 1
                    ex = self.make(task, split)
                    if ex.prompt in seen:
                        continue
                    seen.add(ex.prompt)
                    if reverse_answers and ex.task in ("add", "sub"):
                        ex.answer = ex.answer[::-1]
                    out[split].append(ex)
                    got += 1
        for split in out:
            self.rng.shuffle(out[split])
        return out
