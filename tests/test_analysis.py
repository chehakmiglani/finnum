import random
from dataclasses import asdict

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from finnum.analysis import (classify_error, error_breakdown, frontier, paired_models,  # noqa: E402
                             plot_frontier, plot_robustness, robustness, text_report)
from finnum.benchmark import Generator  # noqa: E402
from finnum.data import make_sample  # noqa: E402
from finnum.eval import run_eval  # noqa: E402


@pytest.fixture(scope="module")
def preds(tmp_path_factory):
    items = [asdict(v) for v in Generator(make_sample(start="2025-01-01", end="2026-03-31"), seed=3).build(15)]
    by_prompt = {}
    for it in items:
        by_prompt.setdefault(it["prompt"], it)
    out = tmp_path_factory.mktemp("p")
    frames = []
    # "good" never errs on indian style; "bad" errs a lot on scaled styles with x10^5 scale mistakes
    for name, err, tokdiv in (("text", 0.3, 4), ("good", 0.1, 8), ("bad", 0.5, 2)):
        rng = random.Random(name)

        def gen(prompts, err=err):
            res = []
            for p in prompts:
                it = by_prompt[p]
                if rng.random() < err:
                    res.append(str(float(it["gold"]) * 1e5) if it["answer_type"] == "number" else "nonsense")
                else:
                    res.append(str(it["gold"]))
            return res
        frames.append(run_eval(items, gen, name, out / f"{name}.jsonl", count_tokens=lambda s, d=tokdiv: len(s) // d,
                               progress=False))
    return pd.concat(frames, ignore_index=True)


def test_paired_models(preds):
    p = paired_models(preds, "text").set_index("model")
    assert p.loc["good", "diff_pp"] > 0 and p.loc["good", "significant"]
    assert p.loc["bad", "diff_pp"] < 0 and p.loc["bad", "significant"]


def test_frontier_marks_dominating_model(preds):
    f = frontier(preds).set_index("model")
    assert bool(f.loc["good", "pareto"]) and not bool(f.loc["text", "pareto"])


def test_robustness_columns(preds):
    r = robustness(preds)
    assert {"accuracy", "spread_pp", "plain", "indian"} <= set(r.columns)
    assert r.index[0] == "good"


def test_error_classes(preds):
    e = error_breakdown(preds)
    assert "scale x10^5" in e.columns
    assert classify_error({"correct": False, "pred": 1.0e7, "gold": 1.0e5}) == "scale x10^2"
    assert classify_error({"correct": False, "pred": 100.5, "gold": 100.0}) == "close (<1%)"
    assert classify_error({"correct": False, "pred": None, "gold": 1.0}) == "no answer"
    assert classify_error({"correct": False, "pred": "TCS", "gold": "INFY"}) == "wrong choice"
    assert classify_error({"correct": False, "pred": -3.45, "gold": 3.45}) == "sign flip"


def test_report_and_plots(preds, tmp_path):
    t = text_report(preds, "text")
    assert "PAIRED vs text" in t and "ERROR TYPES" in t
    plot_frontier(preds).savefig(tmp_path / "f.png")
    plot_robustness(preds).savefig(tmp_path / "r.png")
