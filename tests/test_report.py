import json
import random

import matplotlib
import pytest

matplotlib.use("Agg")

from finnum.tokexp.report import (cost_table, load_results, plot_accuracy, plot_length,  # noqa: E402
                                  summary_table, text_summary)

TASKS = ["add", "sub", "cmp", "pct", "cr"]


def fake_result(scheme, seed, rng):
    base = {"l2r3": 0.6, "r2l3": 0.8, "indian": 0.7, "digit": 0.75}[scheme]
    res = {"scheme": scheme, "seed": seed, "params": 3_000_000, "vocab_size": 1147 if scheme != "digit" else 47,
           "train_minutes": 2.0, "mean_tokens_per_example": 20.0 if scheme != "digit" else 35.0,
           "mean_answer_tokens": 3.0, "losses": []}
    for split, drop in (("test_id", 0.0), ("test_ood", 0.3)):
        by_task = {t: max(0.0, min(1.0, base - drop + rng.uniform(-0.05, 0.05))) for t in TASKS}
        res[split] = {"accuracy": sum(by_task.values()) / len(TASKS), "by_task": by_task,
                      "by_task_len": [{"task": t, "n_digits": d, "n": 50, "accuracy": by_task[t]}
                                      for t in TASKS for d in ((4, 6, 8) if split == "test_id" else (9, 11))],
                      "errors": []}
    return res


@pytest.fixture
def results(tmp_path):
    rng = random.Random(0)
    for seed in (0, 1, 2):
        for s in ("l2r3", "r2l3", "indian", "digit"):
            (tmp_path / f"{s}_seed{seed}.json").write_text(json.dumps(fake_result(s, seed, rng)))
    return load_results(tmp_path)


def test_summary_and_cost(results):
    s = summary_table(results)
    assert set(s["seeds"]) == {3}
    allrow = s[(s["task"] == "ALL") & (s["split"] == "test_id")].set_index("scheme")["mean"]
    assert allrow.idxmax() == "r2l3"
    c = cost_table(results)
    assert list(c.index) == ["l2r3", "r2l3", "indian", "digit"]
    assert c.loc["digit", "tokens_per_example"] > c.loc["l2r3", "tokens_per_example"]


def test_text_summary(results):
    t = text_summary(results)
    assert "best on same length as training: r2l3" in t and "COST" in t


def test_plots(results, tmp_path):
    plot_accuracy(results).savefig(tmp_path / "a.png")
    plot_length(results).savefig(tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 10_000
