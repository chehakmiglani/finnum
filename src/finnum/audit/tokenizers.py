"""Uniform interface over real tokenizers plus rule-based number baselines.

Rule-based chunkers model the three number-splitting strategies found in
practice and in the literature, and need no downloads:

  l2r3   groups of up to 3 digits from the LEFT   (GPT-4 / Llama-3 style)
         1234567 -> 123|456|7
  r2l3   groups of up to 3 digits from the RIGHT  (place-value aligned,
         "Tokenization Counts", Singh & Strouse 2024)
         1234567 -> 1|234|567
  digit  one token per digit                      (Llama-2 / Gemma style)

Digit runs are split by the rule; a word (with its leading space) counts as one
token, roughly what BPE does for common words like " crore"; any other
character is its own token. Use them for number strings, not whole prompts.
"""
from __future__ import annotations

import re
from typing import Protocol

_DIGITS = re.compile(r"\d+| ?[A-Za-z]+|\D")


class Tok(Protocol):
    name: str

    def pieces(self, text: str) -> list[str]: ...


class RuleChunker:
    def __init__(self, mode: str):
        if mode not in ("l2r3", "r2l3", "digit"):
            raise ValueError(mode)
        self.mode = mode
        self.name = f"rule:{mode}"

    def _split(self, run: str) -> list[str]:
        if self.mode == "digit":
            return list(run)
        if self.mode == "l2r3":
            return [run[i:i + 3] for i in range(0, len(run), 3)]
        head = len(run) % 3
        return ([run[:head]] if head else []) + [run[i:i + 3] for i in range(head, len(run), 3)]

    def pieces(self, text: str) -> list[str]:
        out = []
        for m in _DIGITS.finditer(text):
            s = m.group(0)
            out.extend(self._split(s) if s[0].isdigit() else [s])
        return out


class TiktokenTok:
    def __init__(self, encoding: str):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def pieces(self, text: str) -> list[str]:
        return [self.enc.decode_single_token_bytes(t).decode("utf-8", "replace") for t in self.enc.encode(text)]


class HFTok:
    def __init__(self, model_id: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.name = f"hf:{model_id}"

    def pieces(self, text: str) -> list[str]:
        ids = self.tok.encode(text, add_special_tokens=False)
        return [self.tok.decode([i]) for i in ids]


DEFAULT_REAL = [
    "tiktoken:cl100k_base",      # GPT-4 / GPT-3.5
    "tiktoken:o200k_base",       # GPT-4o family
    "hf:Qwen/Qwen2.5-0.5B",      # our retrofit target
    "hf:HuggingFaceTB/SmolLM2-135M",
    "hf:microsoft/Phi-3.5-mini-instruct",
    "hf:mistralai/Mistral-7B-v0.1",
]
RULES = ["rule:l2r3", "rule:r2l3", "rule:digit"]


def load(spec: str) -> Tok:
    kind, _, arg = spec.partition(":")
    if kind == "rule":
        return RuleChunker(arg)
    if kind == "tiktoken":
        return TiktokenTok(arg)
    if kind == "hf":
        return HFTok(arg)
    raise ValueError(f"unknown tokenizer spec {spec!r}")


def load_many(specs: list[str]) -> list[Tok]:
    """Load what is available; report (not crash on) ones that need network/auth."""
    toks = []
    for s in specs:
        try:
            toks.append(load(s))
        except Exception as e:  # noqa: BLE001 - want a summary, not a stack trace
            print(f"[skip] {s}: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
    return toks
