"""Train and evaluate one tiny GPT per (tokenizer scheme, seed).

The loss is computed only on answer tokens (+ end-of-sequence), so the model
is graded on producing the answer, not on predicting random prompt numbers.
Each finished run is written to <out_dir>/<scheme>_seed<seed>.json; existing
files are skipped, so a Colab disconnect only loses the run in progress.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .model import GPT, GPTConfig
from .numtok import NumTokenizer
from .tasks import Example


@dataclass
class TrainConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    n_positions: int = 128
    steps: int = 6000
    batch_size: int = 256
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    random_pos_offset: bool = True
    eval_batch: int = 512
    log_every: int = 500
    amp: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_train(tok: NumTokenizer, examples: list[Example], device) -> dict[str, torch.Tensor]:
    """Teacher-forcing tensors: x, y (=x shifted), loss mask on answer tokens, true lengths."""
    seqs, starts = [], []
    for ex in examples:
        p = [tok.bos_id] + tok.encode(ex.prompt)
        seqs.append(p + tok.encode(ex.answer) + [tok.eos_id])
        starts.append(len(p) - 1)  # y[t] = ids[t+1]; first answer token is predicted at t = len(p)-1
    T = max(len(s) for s in seqs) - 1
    n = len(seqs)
    x = torch.full((n, T), tok.pad_id, dtype=torch.long)
    y = torch.full((n, T), tok.pad_id, dtype=torch.long)
    mask = torch.zeros((n, T), dtype=torch.bool)
    lengths = torch.zeros(n, dtype=torch.long)
    for i, (s, st) in enumerate(zip(seqs, starts)):
        L = len(s) - 1
        x[i, :L] = torch.tensor(s[:-1])
        y[i, :L] = torch.tensor(s[1:])
        mask[i, st:L] = True
        lengths[i] = L
    return {"x": x.to(device), "y": y.to(device), "mask": mask.to(device), "lengths": lengths.to(device)}


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def greedy_answers(model: GPT, tok: NumTokenizer, examples: list[Example], device, batch: int,
                   max_new: int) -> list[str]:
    """Generate answers with greedy decoding. Prompts are grouped by token length (no padding)."""
    model.eval()
    prompts = [[tok.bos_id] + tok.encode(ex.prompt) for ex in examples]
    groups: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        groups[len(p)].append(i)
    out = [""] * len(examples)
    use_amp = device.type == "cuda"
    for _, idxs in sorted(groups.items()):
        for b in range(0, len(idxs), batch):
            chunk = idxs[b:b + batch]
            seq = torch.tensor([prompts[i] for i in chunk], dtype=torch.long, device=device)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            for _ in range(max_new):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits, _ = model(seq)
                nxt = logits[:, -1, :].argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, tok.eos_id), nxt)
                seq = torch.cat([seq, nxt[:, None]], dim=1)
                done |= nxt == tok.eos_id
                if bool(done.all()):
                    break
            start = len(prompts[chunk[0]])
            for row, i in enumerate(chunk):
                out[i] = tok.decode(seq[row, start:].tolist())
    model.train()
    return out


def evaluate(model, tok, examples: list[Example], device, cfg: TrainConfig) -> dict:
    max_new = max(len(tok.encode(ex.answer)) for ex in examples) + 2
    preds = greedy_answers(model, tok, examples, device, cfg.eval_batch, max_new)
    by_task, by_len = defaultdict(list), defaultdict(list)
    rows = []
    for ex, p in zip(examples, preds):
        ok = p == ex.answer
        by_task[ex.task].append(ok)
        by_len[(ex.task, ex.n_digits)].append(ok)
        rows.append({"task": ex.task, "n_digits": ex.n_digits, "prompt": ex.prompt, "gold": ex.answer,
                     "pred": p, "correct": ok})
    return {
        "accuracy": float(np.mean([r["correct"] for r in rows])),
        "by_task": {t: float(np.mean(v)) for t, v in by_task.items()},
        "by_task_len": [{"task": t, "n_digits": d, "n": len(v), "accuracy": float(np.mean(v))}
                        for (t, d), v in sorted(by_len.items())],
        "errors": [r for r in rows if not r["correct"]][:200],
    }


def train_one(scheme: str, seed: int, data: dict[str, list[Example]], cfg: TrainConfig,
              device: torch.device | None = None, verbose: bool = True) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    tok = NumTokenizer(scheme)
    train = encode_train(tok, data["train"], device)
    longest = max(len(tok.encode(e.text)) + 2 for split in data.values() for e in split)
    if longest > cfg.n_positions:
        raise ValueError(f"{scheme}: longest sequence {longest} > n_positions {cfg.n_positions}")

    model = GPT(GPTConfig(vocab_size=tok.vocab_size, n_positions=cfg.n_positions, n_layer=cfg.n_layer,
                          n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=cfg.dropout)).to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = train["x"].shape[0]
    losses, t0 = [], time.time()

    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        idx = torch.randint(0, n, (cfg.batch_size,), generator=gen).to(device)
        L = int(train["lengths"][idx].max())
        x, y, m = (train[k][idx, :L] for k in ("x", "y", "mask"))
        offset = int(torch.randint(0, cfg.n_positions - L + 1, (1,), generator=gen)) if cfg.random_pos_offset else 0
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y, m, pos_offset=offset)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            losses.append({"step": step, "loss": float(loss)})
            if verbose:
                el = time.time() - t0
                eta = el / (step + 1) * (cfg.steps - step - 1)
                print(f"\r  [{scheme} seed {seed}] step {step + 1}/{cfg.steps} loss {float(loss):.4f} "
                      f"({el / 60:.1f} min, ~{eta / 60:.1f} min left)", end="")
    train_min = (time.time() - t0) / 60
    if verbose:
        print()

    result = {
        "scheme": scheme, "seed": seed, "config": asdict(cfg), "params": model.n_params(),
        "vocab_size": tok.vocab_size, "train_minutes": round(train_min, 2), "losses": losses,
        "n_train": len(data["train"]),
        "mean_tokens_per_example": float(np.mean([len(tok.encode(e.text)) for e in data["train"]])),
        "mean_answer_tokens": float(np.mean([len(tok.encode(e.answer)) for e in data["train"]])),
    }
    for split in ("test_id", "test_ood"):
        result[split] = evaluate(model, tok, data[split], device, cfg)
        if verbose:
            r = result[split]
            print(f"  {split}: {r['accuracy']:.1%}  " + "  ".join(f"{t}={a:.0%}" for t, a in sorted(r["by_task"].items())))
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_all(data: dict[str, list[Example]], schemes, seeds, cfg: TrainConfig, out_dir: str | Path) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:            # seed-major: one full comparison is finished before the next seed
        for scheme in schemes:
            path = out_dir / f"{scheme}_seed{seed}.json"
            if path.exists():
                print(f"already done: {path.name}")
                results.append(json.loads(path.read_text()))
                continue
            print(f"training {scheme} (seed {seed})")
            res = train_one(scheme, seed, data, cfg)
            path.write_text(json.dumps(res))
            results.append(res)
    return results
