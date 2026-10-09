import json
import random
from dataclasses import asdict

import matplotlib
import pytest

matplotlib.use("Agg")

from finnum.benchmark import Generator  # noqa: E402
from finnum.data import make_sample  # noqa: E402
from finnum.eval import accuracy_table, paired_vs_baseline, run_eval, summary_lines  # noqa: E402
from finnum.eval.metrics import holm, mcnemar_exact, wilson  # noqa: E402
from finnum.eval.runner import clean_output  # noqa: E402


@pytest.fixture(scope="module")
def items():
    df = make_sample(start="2025-01-01", end="2026-03-31", seed=5)
    return [asdict(v) for v in Generator(df, seed=2).build(n_per_task=20)]


def fake_model(items, wrong_rate: dict[str, float], seed=0):
    """A fake LLM: answers correctly except at a per-style error rate."""
    rng = random.Random(seed)
    by_prompt = {}
    for it in items:  # identical prompts (style made no visible change) -> same answer, like a real model
        by_prompt.setdefault(it["prompt"], it)

    def answer(it):
        if rng.random() < wrong_rate.get(it["style"], 0):
            return {"number": "-999", "choice": "NOPE", "bool": "maybe"}[it["answer_type"]]
        g = it["gold"]
        if it["answer_type"] == "bool":
            return "<think>hmm</think>Yes" if g else "No."
        return f"The answer is {g}"

    return lambda prompts: [answer(by_prompt[p]) for p in prompts]


def test_run_eval_scores_and_resumes(items, tmp_path):
    out = tmp_path / "preds.jsonl"
    gen = fake_model(items, {})
    half = items[: len(items) // 2]
    run_eval(half, gen, "fake", out, batch_size=7, count_tokens=len, progress=False)
    df = run_eval(items, gen, "fake", out, batch_size=7, count_tokens=len, progress=False)
    assert len(df) == len(items) and df["id"].is_unique       # resumed, no duplicates
    assert df["correct"].all()                                # oracle is always right
    assert df["prompt_tokens"].gt(0).all()


def test_paired_test_detects_degraded_style(items, tmp_path):
    gen = fake_model(items, {"scaled_intl": 0.5, "indian": 0.02}, seed=1)
    df = run_eval(items, gen, "fake", tmp_path / "p.jsonl", count_tokens=len, progress=False)
    acc = accuracy_table(df)
    assert acc.set_index(["style", "table_format"]).loc[("plain", "csv"), "accuracy"] == 1.0
    paired = paired_vs_baseline(df)
    bad = paired[(paired["style"] == "scaled_intl") & (paired["table_format"] == "csv")].iloc[0]
    assert bad["significant"] and bad["diff_pp"] < -15  # diluted by style-identical prompts
    mild = paired[(paired["style"] == "indian") & (paired["table_format"] == "csv")].iloc[0]
    assert not mild["significant"]
    assert any("scaled_intl" in line for line in summary_lines(df))


def test_stats_reference_values():
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert mcnemar_exact(5, 5) == 1.0 and mcnemar_exact(0, 0) == 1.0
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


def test_clean_output():
    assert clean_output("<think>long\nreasoning</think> 42") == "42"


def test_plots_render(items, tmp_path):
    from finnum.eval.plots import accuracy_by_style, cost_vs_accuracy
    gen = fake_model(items, {"scaled_intl": 0.3}, seed=3)
    df = run_eval(items, gen, "org/fake-model", tmp_path / "p.jsonl", count_tokens=len, progress=False)
    df2 = df.assign(model="org/other-model")
    both = __import__("pandas").concat([df, df2])
    accuracy_by_style(both).savefig(tmp_path / "a.png")
    cost_vs_accuracy(both).savefig(tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 10_000


def test_predictions_are_json_roundtrippable(items, tmp_path):
    out = tmp_path / "p.jsonl"
    run_eval(items[:30], fake_model(items, {}), "fake", out, progress=False)
    for line in out.read_text().splitlines():
        json.loads(line)
