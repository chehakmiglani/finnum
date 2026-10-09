import pytest

from finnum.audit.token_audit import audit_numbers, run, sample_numbers
from finnum.audit.tokenizers import RuleChunker, load_many
from finnum.benchmark import Generator
from finnum.data import make_sample


@pytest.mark.parametrize("mode,text,expected", [
    ("l2r3", "1234567", ["123", "456", "7"]),
    ("r2l3", "1234567", ["1", "234", "567"]),
    ("r2l3", "123456", ["123", "456"]),
    ("digit", "2915.35", ["2", "9", "1", "5", ".", "3", "5"]),
    ("l2r3", "1,23,45,678.90", ["1", ",", "23", ",", "45", ",", "678", ".", "90"]),
    ("r2l3", "1.23 crore", ["1", ".", "23", " crore"]),
])
def test_rule_chunkers(mode, text, expected):
    assert RuleChunker(mode).pieces(text) == expected


def test_unavailable_tokenizers_are_skipped_not_fatal():
    toks = load_many(["rule:digit", "hf:definitely/not-a-real-model-xyz"])
    assert [t.name for t in toks] == ["rule:digit"]


def test_audit_end_to_end(tmp_path):
    df = make_sample(start="2025-01-01", end="2025-06-30")
    variants = [v.__dict__ for v in Generator(df, seed=1).build(n_per_task=2)]
    toks = [RuleChunker(m) for m in ("l2r3", "r2l3", "digit")]
    res = run(df, variants, toks, tmp_path)
    assert (tmp_path / "number_summary.csv").exists() and (tmp_path / "examples.md").exists()
    s = res["number_summary"]
    # digit-level always costs at least as much as 3-digit chunking
    assert (s["rule:digit"] >= s["rule:l2r3"]).all()


def test_scaled_styles_save_tokens_on_large_numbers():
    df = make_sample(start="2025-01-01", end="2025-03-31")
    out = audit_numbers(sample_numbers(df, n=100), [RuleChunker("digit")])
    t = out[out["kind"] == "turnover"].groupby("style")["n_tokens"].mean()
    assert t["scaled_indian"] < t["plain"] and t["scaled_intl"] < t["plain"]
