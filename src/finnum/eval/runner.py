"""Run a model over FinNum prompts and score its answers.

The runner takes any ``generate_fn(list[str]) -> list[str]``, so the same code
evaluates a Hugging Face model on a GPU, an API model, or a fake model in tests.
Results are appended to a JSONL file after every batch, so a Colab disconnect
loses at most one batch: re-running the cell resumes where it stopped.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from finnum.benchmark.scoring import extract_answer, is_correct

GenerateFn = Callable[[list[str]], list[str]]

_THINK = re.compile(r"<think>.*?</think>", re.S)


def clean_output(text: str) -> str:
    """Drop reasoning blocks some models emit before the answer."""
    return _THINK.sub("", text).strip()


class HFGenerator:
    """Greedy, batched generation with a Hugging Face causal LM."""

    def __init__(self, model_id: str, max_new_tokens: int = 24, batch_size: int = 16,
                 dtype: str = "auto", trust_remote_code: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left",
                                                 trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=trust_remote_code)
        self.model.eval()

    def format(self, prompt: str) -> str:
        if self.tok.chat_template:
            return self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                tokenize=False, add_generation_prompt=True)
        return prompt + "\nAnswer:"

    def count_tokens(self, prompt: str) -> int:
        """Tokens for the raw prompt (no chat wrapper) - the cost we compare across styles."""
        return len(self.tok.encode(prompt, add_special_tokens=False))

    def __call__(self, prompts: list[str]) -> list[str]:
        texts = [self.format(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(new, skip_special_tokens=True)

    def close(self) -> None:
        import gc
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {json.loads(line)["id"] for line in fh if line.strip()}


def run_eval(items: list[dict], generate_fn: GenerateFn, model_name: str, out_path: str | Path,
             batch_size: int = 16, count_tokens: Callable[[str], int] | None = None,
             progress: bool = True) -> pd.DataFrame:
    """Evaluate ``items`` and append scored predictions to ``out_path`` (resumable)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    # similar lengths per batch -> less padding -> faster
    todo.sort(key=lambda it: len(it["prompt"]))
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for b in range(0, len(todo), batch_size):
            batch = todo[b:b + batch_size]
            outputs = generate_fn([it["prompt"] for it in batch])
            for it, raw in zip(batch, outputs):
                text = clean_output(raw)
                pred = extract_answer(text, it["answer_type"], it.get("options"))
                rec = {"id": it["id"], "base_id": it["base_id"], "task": it["task"], "style": it["style"],
                       "table_format": it["table_format"], "differs_from_plain": it.get("differs_from_plain", True),
                       "model": model_name, "output": raw,
                       "pred": pred, "gold": it["gold"], "correct": is_correct(text, it),
                       "prompt_tokens": count_tokens(it["prompt"]) if count_tokens else None}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if progress:
                n = b + len(batch)
                rate = n / max(time.time() - t0, 1e-9)
                eta = (len(todo) - n) / rate if rate else 0
                print(f"\r{model_name}: {n}/{len(todo)}  ({rate:.1f}/s, ~{eta / 60:.0f} min left)", end="")
    if progress:
        print(f"\n{model_name}: done ({len(done)} already scored before this run)")
    return load_predictions(out_path)


def load_predictions(*paths: str | Path) -> pd.DataFrame:
    rows = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return pd.DataFrame(rows)
