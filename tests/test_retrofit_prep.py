import random
import re

import numpy as np
import pytest

from finnum.benchmark import Generator
from finnum.data import make_sample
from finnum.numbers import find_numbers
from finnum.retrofit.features import (FEATURE_DIMS, bit_features, decode_fone, features, fone_features,
                                      grouping_of)
from finnum.retrofit.preprocess import METHODS, NUM_TOKEN, answer_text, encode, feature_dim, mark, quantities


class FakeTok:
    """Whitespace-ish tokenizer that treats <num> as one special token (like an added special token)."""
    eos_token_id = 0

    def __init__(self):
        self.vocab = {"<eos>": 0, NUM_TOKEN: 1}

    def encode(self, text, add_special_tokens=False):
        out = []
        for piece in re.findall(r"<num>|\w+|[^\w\s]|\s", text):
            out.append(self.vocab.setdefault(piece, len(self.vocab)))
        return out


@pytest.mark.parametrize("x", [0.0, 7.0, 2915.35, 130500.5, 12345678.0, 14912345678.55, 0.05, -3.45, 999999999999.99])
def test_fone_is_digit_exact(x):
    assert decode_fone(fone_features(np.array([x]))[0]) == pytest.approx(x, abs=1e-6)


def test_fone_random_roundtrip():
    rng = random.Random(0)
    for _ in range(2000):
        x = round(rng.uniform(0, 1e12), 2)
        assert decode_fone(fone_features(np.array([x]))[0]) == pytest.approx(x, abs=1e-3)


def test_bits_are_exact_and_signed():
    b = bit_features(np.array([2915.35, -1.0]))
    assert b.shape == (2, 64) and set(np.unique(b)) <= {-1.0, 1.0}
    packed = np.packbits((b[0] > 0).astype(np.uint8)).view(">f8")[0]
    assert packed == 2915.35


@pytest.mark.parametrize("text,g", [("1,23,45,678", "indian"), ("12,345,678", "intl"), ("12345", "none"),
                                    ("₹1,234.50", "intl"), ("12,34,567.00", "indian")])
def test_grouping(text, g):
    assert grouping_of(text) == g


def test_feature_shapes_all_kinds():
    nums = find_numbers("₹12.35 crore, -3.45% and 1,23,45,678")
    for kind, dim in FEATURE_DIMS.items():
        f = features(nums, kind)
        assert f.shape == (3, dim) and f.dtype == np.float32 and np.isfinite(f).all()
    u = features(nums, "units")
    assert u[0, 0] == 0 and u[0, 1] == 1 and u[1, 0] == 1  # currency on first, percent on second


def test_dates_are_not_quantities():
    nums = quantities("On 2025-05-27, close was 2,915.35 and volume 1.2 lakh")
    assert [n.value for n in nums] == pytest.approx([2915.35, 1.2e5])


def test_mark_replace_and_prefix():
    t = "date,close\n2025-05-27,\"2,915.35\"\nWhat was the close on 2025-05-27? Round to 2 decimal places."
    r, f = mark(t, "fone")
    assert "2,915.35" not in r and r.count(NUM_TOKEN) == 2 and "2025-05-27" in r and f.shape == (2, FEATURE_DIMS["fone"])
    p, f2 = mark(t, "value_prefix")
    assert NUM_TOKEN + "2,915.35" in p and len(f2) == 2
    same, none = mark(t, "text")
    assert same == t and none is None


@pytest.fixture(scope="module")
def items():
    from dataclasses import asdict
    return [asdict(v) for v in Generator(make_sample(start="2025-01-01", end="2025-12-31"), seed=4).build(4)]


@pytest.mark.parametrize("method", list(METHODS))
def test_encode_aligns_markers_and_features(items, method):
    tok = FakeTok()
    for it in items:
        enc = encode(it, tok, method, with_answer=True, num_token_id=1)
        n_marks = enc["input_ids"].count(1)
        if METHODS[method]["features"]:
            assert n_marks == len(enc["features"]) > 0
            assert enc["features"].shape[1] == feature_dim(method)
        else:
            assert n_marks == 0 and enc["features"] is None
        # labels only on the answer (+eos)
        n_ans = sum(1 for x in enc["labels"] if x != -100)
        assert n_ans == len(tok.encode(answer_text(it))) + 1
        assert enc["input_ids"][-1] == tok.eos_token_id


def test_replace_methods_cut_prompt_tokens(items):
    tok = FakeTok()
    for it in items:
        if it["style"] != "plain":
            continue
        t = encode(it, tok, "text", False)["prompt_tokens"]
        f = encode(it, tok, "fone", False)["prompt_tokens"]
        p = encode(it, tok, "value_prefix", False)["prompt_tokens"]
        assert f < t < p


def test_answer_text():
    assert answer_text({"answer_type": "number", "gold": 3.456}) == "3.46"
    assert answer_text({"answer_type": "bool", "gold": False}) == "no"
    assert answer_text({"answer_type": "choice", "gold": "TCS"}) == "TCS"


def test_csv_row_after_date_is_not_merged():
    nums = quantities("2025-05-12,147.50,5123456\nRELIANCE,2915.35,\"1,23,456\"")
    assert [n.value for n in nums] == pytest.approx([147.50, 5123456, 2915.35, 123456])


def test_every_rendered_number_is_found(items):
    """Every numeric cell the renderer wrote must be marked exactly once."""
    for it in items:
        if it["table_format"] != "csv":
            continue
        import csv, io
        block = it["prompt"].split("\n\n")[1]
        rows = list(csv.reader(io.StringIO(block)))[1:]
        cells = [c for r in rows for c in r if any(ch.isdigit() for ch in c) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", c)]
        table_nums = quantities(block)
        assert len(table_nums) == len(cells), (cells, [n.text for n in table_nums])


def test_build_items_split_by_date():
    from finnum.retrofit.data import build_items, split_by_date
    df = make_sample()
    tr_df, te_df = split_by_date(df, 120)
    assert tr_df["date"].max() < te_df["date"].min()
    train, test = build_items(df, n_train_per_task=20, n_test_per_task=5, seed=0)
    assert len(test) == 6 * 5 * 15 and len(train) == 6 * 20 * 3
    assert not {t["id"] for t in train} & {t["id"] for t in test}
    test_dates = set(te_df["date"].dt.strftime("%Y-%m-%d"))
    for it in test:  # every date mentioned in a test prompt is inside the test period
        for d in re.findall(r"\d{4}-\d{2}-\d{2}", it["prompt"]):
            assert d in test_dates
    for it in train:
        for d in re.findall(r"\d{4}-\d{2}-\d{2}", it["prompt"]):
            assert d not in test_dates
