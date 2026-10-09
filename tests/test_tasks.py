from decimal import Decimal

import pytest

from finnum.data import make_sample
from finnum.tokexp.numtok import SCHEMES, NumTokenizer
from finnum.tokexp.tasks import LENGTHS, TASKS, TaskBuilder, _is_test_value


@pytest.fixture(scope="module")
def data():
    return TaskBuilder(make_sample(), seed=0).build(n_train=400, n_test=60)


def check_answer(ex):
    op, rest = ex.prompt[:-1].split(" ", 1)
    args = rest.split(" ")
    if op == "add":
        return int(ex.answer) == int(args[0]) + int(args[1])
    if op == "sub":
        return int(ex.answer) == int(args[0]) - int(args[1]) >= 0
    if op == "cmp":
        return ex.answer == (">" if int(args[0]) > int(args[1]) else "<")
    if op == "pct":
        exact = (Decimal(args[1]) / Decimal(args[0]) - 1) * 100
        return abs(Decimal(ex.answer) - exact) <= Decimal("0.005")
    if op == "cr":
        return abs(Decimal(ex.answer) - Decimal(args[0]) / 10**7) <= Decimal("0.005")
    raise AssertionError(op)


def test_answers_are_correct(data):
    for split in data.values():
        for ex in split:
            assert check_answer(ex), ex


def test_split_sizes_and_uniqueness(data):
    assert len(data["test_id"]) == len(TASKS) * 60
    assert len(data["test_ood"]) == len(TASKS) * 60
    train_prompts = {e.prompt for e in data["train"]}
    assert len(train_prompts) == len(data["train"])
    for split in ("test_id", "test_ood"):
        assert not train_prompts & {e.prompt for e in data[split]}


def test_lengths_respect_splits(data):
    for split, idx in (("train", 0), ("test_id", 0), ("test_ood", 1)):
        for ex in data[split]:
            lo, hi = LENGTHS[ex.task][idx]
            assert lo <= ex.n_digits <= hi, (split, ex)


def test_test_operands_never_seen_in_training(data):
    for ex in data["train"]:
        if ex.task in ("add", "sub"):  # cr may use sums of two train values
            for a in ex.prompt[:-1].split(" ")[1:]:
                assert not _is_test_value(int(a))


def test_every_example_tokenizes_in_every_scheme(data):
    toks = [NumTokenizer(s) for s in SCHEMES]
    for split in data.values():
        for ex in split:
            for t in toks:
                assert t.decode(t.encode(ex.text)) == ex.text


def test_cmp_is_not_trivially_one_sided(data):
    ans = [e.answer for e in data["train"] if e.task == "cmp"]
    assert 0.3 < ans.count(">") / len(ans) < 0.7


def test_reverse_answers_only_for_add_sub():
    d = TaskBuilder(make_sample(start="2025-01-01", end="2025-12-31"), seed=0).build(50, 10, reverse_answers=True)
    for ex in d["test_id"]:
        args = ex.prompt[:-1].split(" ")
        if ex.task == "add":
            assert ex.answer[::-1] == str(int(args[1]) + int(args[2]))
        if ex.task == "cmp":
            assert ex.answer in "<>"
