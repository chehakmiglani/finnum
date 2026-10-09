"""Number-aware tokenizers for the from-scratch experiment (week 3).

Everything except runs of digits is tokenized one character at a time, so
the ONLY difference between schemes is how digits are grouped:

  l2r3    1234567 -> 123|456|7        groups of 3 from the left  (GPT-4 / Llama-3 style)
  r2l3    1234567 -> 1|234|567        groups of 3 from the right (aligned with place value)
  indian  1234567 -> 12|34|567        last 3 digits, then pairs  (lakh/crore grouping)
  digit   1234567 -> 1|2|3|4|5|6|7    one digit per token

The chunk schemes share one vocabulary (every 1-3 digit string, so "045" and
"45" are different tokens), which keeps the model size identical across them.
"""
from __future__ import annotations

import itertools
import re

SCHEMES = ("l2r3", "r2l3", "indian", "digit")
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
CHARS = list("abcdefghijklmnopqrstuvwxyz .-=+%<>")
_RUN = re.compile(r"\d+|.", re.S)


def chunk_digits(run: str, scheme: str) -> list[str]:
    if scheme == "digit":
        return list(run)
    if scheme == "l2r3":
        return [run[i:i + 3] for i in range(0, len(run), 3)]
    if scheme == "r2l3":
        head = len(run) % 3
        return ([run[:head]] if head else []) + [run[i:i + 3] for i in range(head, len(run), 3)]
    if scheme == "indian":
        if len(run) <= 3:
            return [run]
        head, tail = run[:-3], run[-3:]
        first = len(head) % 2
        groups = ([head[:first]] if first else []) + [head[i:i + 2] for i in range(first, len(head), 2)]
        return groups + [tail]
    raise ValueError(f"unknown scheme {scheme!r}")


class NumTokenizer:
    def __init__(self, scheme: str):
        if scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}")
        self.scheme = scheme
        if scheme == "digit":
            digit_tokens = [str(d) for d in range(10)]
        else:
            digit_tokens = ["".join(p) for n in (1, 2, 3) for p in itertools.product("0123456789", repeat=n)]
        self.itos = [PAD, BOS, EOS] + CHARS + digit_tokens
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def pieces(self, text: str) -> list[str]:
        out = []
        for m in _RUN.finditer(text):
            s = m.group(0)
            out.extend(chunk_digits(s, self.scheme) if s[0].isdigit() else [s])
        return out

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[p] for p in self.pieces(text)]
        except KeyError as e:
            raise ValueError(f"character {e} not in vocabulary") from None

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            out.append(self.itos[i])
        return "".join(out)
