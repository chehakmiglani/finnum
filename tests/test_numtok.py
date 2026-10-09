import random

import pytest

from finnum.tokexp.numtok import SCHEMES, NumTokenizer, chunk_digits


@pytest.mark.parametrize("scheme,run,expected", [
    ("l2r3", "1234567", ["123", "456", "7"]),
    ("r2l3", "1234567", ["1", "234", "567"]),
    ("indian", "1234567", ["12", "34", "567"]),
    ("indian", "12345678", ["1", "23", "45", "678"]),
    ("indian", "123", ["123"]),
    ("indian", "1234", ["1", "234"]),
    ("digit", "1234", ["1", "2", "3", "4"]),
    ("r2l3", "1000045", ["1", "000", "045"]),
])
def test_chunking(scheme, run, expected):
    assert chunk_digits(run, scheme) == expected


def test_indian_chunks_match_indian_commas():
    from finnum.numbers.formats import group_indian
    rng = random.Random(0)
    for _ in range(500):
        run = str(rng.randrange(1, 10**12))
        assert ",".join(chunk_digits(run, "indian")) == group_indian(run)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_roundtrip(scheme):
    tok = NumTokenizer(scheme)
    rng = random.Random(1)
    for _ in range(300):
        a, b = rng.randrange(10**12), rng.randrange(10**6)
        text = f"add {a} {b}={a + b}"
        assert tok.decode(tok.encode(text)) == text
    assert tok.decode(tok.encode("pct 2915.35 3001.20=-2.94") + [tok.eos_id, 5, 6]) == "pct 2915.35 3001.20=-2.94"


def test_same_vocab_for_chunk_schemes():
    sizes = {s: NumTokenizer(s).vocab_size for s in SCHEMES}
    assert sizes["l2r3"] == sizes["r2l3"] == sizes["indian"] > sizes["digit"]


def test_token_counts_differ_as_expected():
    t = {s: NumTokenizer(s) for s in SCHEMES}
    n = "12345678"
    assert [len(t[s].encode(n)) for s in SCHEMES] == [3, 3, 4, 8]


def test_unknown_char_rejected():
    with pytest.raises(ValueError):
        NumTokenizer("digit").encode("₹100")
