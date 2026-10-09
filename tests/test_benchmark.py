import csv
import io
import re
from collections import defaultdict

import pytest

from finnum.benchmark import Generator, extract_answer, is_correct
from finnum.data import make_sample
from finnum.numbers import parse_number


@pytest.fixture(scope="module")
def variants():
    df = make_sample(start="2025-01-01", end="2026-06-30", seed=3)
    return Generator(df, seed=11).build(n_per_task=15)


def _table_rows(prompt: str) -> list[dict]:
    """Recover the CSV table from a prompt (independent of the generator's internals)."""
    block = prompt.split("\n\n")[1]
    return list(csv.DictReader(io.StringIO(block)))


def test_counts_and_pairing(variants):
    by_base = defaultdict(list)
    for v in variants:
        by_base[v.base_id].append(v)
    assert len(by_base) == 6 * 15
    for vs in by_base.values():
        assert len(vs) == 5 * 3                       # 5 styles x 3 table formats
        assert len({repr(v.gold) for v in vs}) == 1   # same gold across every rendering


def test_gold_answers_recomputed_from_rendered_text(variants):
    """Re-derive every gold answer from the plain-CSV prompt text alone."""
    for v in (v for v in variants if v.style == "plain" and v.table_format == "csv"):
        rows = _table_rows(v.prompt)
        q = v.prompt.split("Question: ")[1]
        if v.task == "lookup":
            date = q.split(" on ")[1].rstrip("?").split("?")[0]
            got = next(float(r["close"]) for r in rows if r["date"] == date)
            assert got == pytest.approx(v.gold)
        elif v.task == "mean":
            assert round(sum(float(r["close"]) for r in rows) / len(rows), 2) == pytest.approx(v.gold, abs=0.006)
        elif v.task == "pct_change":
            d1, d2 = re.findall(r"\d{4}-\d{2}-\d{2}", q)
            c = {r["date"]: float(r["close"]) for r in rows}
            assert round((c[d2] / c[d1] - 1) * 100, 2) == pytest.approx(v.gold, abs=0.006)
        elif v.task == "compare":
            pct = {r["symbol"]: float(r["close"]) / float(r["prev_close"]) - 1 for r in rows}
            assert max(pct, key=pct.get) == v.gold
        elif v.task == "magnitude":
            t = {r["symbol"]: float(r["turnover_inr"]) for r in rows}
            assert max(t, key=t.get) == v.gold
        elif v.task == "anomaly":
            vols = [float(r["volume"]) for r in rows]
            assert (vols[-1] > 3 * sum(vols[:-1]) / 20) == v.gold


def test_magnitude_winner_survives_unit_rendering(variants):
    for v in (v for v in variants if v.task == "magnitude" and v.table_format == "csv"):
        rows = _table_rows(v.prompt)
        t = {r["symbol"]: parse_number(r["turnover_inr"]) for r in rows}
        assert max(t, key=t.get) == v.gold, v.style


def test_anomaly_is_roughly_balanced(variants):
    labels = [v.gold for v in variants if v.task == "anomaly" and v.style == "plain" and v.table_format == "csv"]
    assert 3 <= sum(labels) <= 12


def test_styles_really_differ(variants):
    lookups = [v for v in variants if v.task == "lookup" and v.table_format == "csv"]
    base = lookups[0].base_id
    prompts = {v.style: v.prompt for v in lookups if v.base_id == base}
    assert len(set(prompts.values())) >= 4  # plain/intl/indian/scaled differ (scaled styles may coincide)


@pytest.mark.parametrize("text,atype,options,expected", [
    ("The closing price was ₹2,915.35", "number", None, 2915.35),
    ("From 100 to 103.5 is 3.50%", "number", None, 3.5),
    ("TCS", "choice", ["TCS", "INFY"], "TCS"),
    ("INFY moved less, so the answer is TCS", "choice", ["TCS", "INFY"], "TCS"),
    ("Yes.", "bool", None, True),
    ("no, it was only 2.1x", "bool", None, False),
    ("I am not sure", "number", None, None),
])
def test_extract(text, atype, options, expected):
    assert extract_answer(text, atype, options) == expected


def test_is_correct_tolerance():
    item = {"answer_type": "number", "gold": 3.45, "tolerance": 0.011}
    assert is_correct("3.46", item) and not is_correct("3.47", item)


def test_differs_from_plain_flag(variants):
    for v in variants:
        if v.style == "plain":
            assert v.differs_from_plain
    # small-number tables (e.g. mean of a sub-1000 price) render identically in every style
    assert any(not v.differs_from_plain for v in variants)
    assert any(v.differs_from_plain and v.style != "plain" for v in variants)
