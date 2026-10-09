"""Train/test FinNum items for the retrofit experiment, split by DATE (no leakage)."""
from __future__ import annotations

import random
from dataclasses import asdict

import pandas as pd

from finnum.benchmark import Generator


def split_by_date(stocks: pd.DataFrame, test_days: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last ``test_days`` trading days are test; everything before is train."""
    dates = sorted(stocks["date"].unique())
    if len(dates) <= test_days + 60:
        raise ValueError(f"need more than {test_days + 60} trading days, got {len(dates)}")
    cut = dates[-test_days]
    return stocks[stocks["date"] < cut].copy(), stocks[stocks["date"] >= cut].copy()


def build_items(stocks: pd.DataFrame, n_train_per_task: int, n_test_per_task: int,
                renderings_per_train_question: int = 3, test_days: int = 120, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Train: each base question in a few random (style, format) renderings, so the model sees every
    style but cannot memorise pairs. Test: every base question in all 15 renderings (paired analysis)."""
    train_df, test_df = split_by_date(stocks, test_days)
    rng = random.Random(seed)
    train_all = [asdict(v) for v in Generator(train_df, seed=seed + 1).build(n_train_per_task)]
    by_base: dict[str, list[dict]] = {}
    for it in train_all:
        by_base.setdefault(it["base_id"], []).append(it)
    train = [it for vs in by_base.values() for it in rng.sample(vs, min(renderings_per_train_question, len(vs)))]
    rng.shuffle(train)
    for it in train:  # make train ids distinct from test ids
        it["id"], it["base_id"] = "train-" + it["id"], "train-" + it["base_id"]
    test = [asdict(v) for v in Generator(test_df, seed=seed + 2).build(n_test_per_task)]
    return train, test
