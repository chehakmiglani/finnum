"""Retrofit a number encoding into a pretrained causal LM (Qwen2.5-0.5B by default) with LoRA.

For every method the base model gets the SAME LoRA adapters and the SAME training data;
methods differ only in how numbers in the prompt reach the model (see preprocess.METHODS).
Answers are always produced as ordinary text by the LM head, so the existing FinNum
scorer grades every method identically.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .encoders import NumberEncoder
from .preprocess import ANSWER_PREFIX, METHODS, NUM_TOKEN, encode, feature_dim, mark


@dataclass
class RetrofitConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    epochs: int = 1
    batch_size: int = 4       # x grad_accum 4 = effective batch 16; small micro-batches keep the
    grad_accum: int = 4       # 152k-vocab logits (batch x length x 151,936) within a 16 GB T4
    lr: float = 2e-4          # LoRA
    encoder_lr: float = 1e-3  # number encoder
    warmup_frac: float = 0.05
    max_len: int = 1536
    max_new_tokens: int = 16
    eval_batch: int = 16
    seed: int = 0
    log_every: int = 50


class Retrofit:
    def __init__(self, method: str, cfg: RetrofitConfig, device: torch.device | None = None):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")
        self.method, self.cfg = method, cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.tok = AutoTokenizer.from_pretrained(cfg.model_id)
        self.tok.add_special_tokens({"additional_special_tokens": [NUM_TOKEN]})
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.num_id = self.tok.convert_tokens_to_ids(NUM_TOKEN)

        base = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.float32)
        if len(self.tok) > base.get_input_embeddings().num_embeddings:
            base.resize_token_embeddings(len(self.tok))
        emb = base.get_input_embeddings().weight
        emb_std = float(emb.detach().float().std())
        hidden = emb.shape[1]
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                          task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora).to(self.device)
        kind = METHODS[method]["features"]
        self.encoder = (NumberEncoder(feature_dim(method), hidden, emb_std, multiplicative=(kind == "xval"))
                        .to(self.device) if kind else None)

    # ------------------------------------------------------------------ embeddings
    def embed(self, input_ids: torch.Tensor, feats: torch.Tensor | None) -> torch.Tensor:
        emb = self.model.get_input_embeddings()(input_ids)
        if self.encoder is None:
            return emb
        mask = input_ids == self.num_id
        n = int(mask.sum())
        if feats is None or n != feats.shape[0]:
            raise ValueError(f"{n} <num> positions but {0 if feats is None else feats.shape[0]} feature rows")
        if n:
            emb = emb.clone()
            emb[mask] = self.encoder(feats).to(emb.dtype)  # row-major order == order numbers were concatenated
        return emb

    def _feats(self, batch_feats: list) -> torch.Tensor | None:
        if self.encoder is None:
            return None
        arrs = [f for f in batch_feats if f is not None and len(f)]
        if not arrs:
            return torch.zeros((0, feature_dim(self.method)), device=self.device)
        return torch.from_numpy(np.concatenate(arrs, axis=0)).to(self.device)

    def _encode(self, item_or_prompt, with_answer: bool) -> dict:
        e = encode(item_or_prompt, self.tok, self.method, with_answer, self.num_id)
        if len(e["input_ids"]) > self.cfg.max_len:
            raise ValueError(f"example is {len(e['input_ids'])} tokens > max_len {self.cfg.max_len}")
        return e

    def count_prompt_tokens(self, prompt: str) -> int:
        marked, _ = mark(prompt, self.method)
        return len(self.tok.encode(marked + ANSWER_PREFIX, add_special_tokens=False))

    # ------------------------------------------------------------------ training
    def train(self, items: list[dict], verbose: bool = True) -> dict:
        cfg = self.cfg
        encoded = [self._encode(it, with_answer=True) for it in items]
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        groups = [{"params": lora_params, "lr": cfg.lr, "weight_decay": 0.0}]
        if self.encoder is not None:
            groups.append({"params": list(self.encoder.parameters()), "lr": cfg.encoder_lr, "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups)
        base_lrs = [g["lr"] for g in groups]
        steps_per_epoch = math.ceil(len(encoded) / (cfg.batch_size * cfg.grad_accum))
        total = steps_per_epoch * cfg.epochs
        warm = max(1, int(cfg.warmup_frac * total))
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = random.Random(cfg.seed)
        self.model.train()
        if self.encoder is not None:
            self.encoder.train()
        losses, step, t0 = [], 0, time.time()
        for _epoch in range(cfg.epochs):
            order = list(range(len(encoded)))
            rng.shuffle(order)
            micro = [order[i:i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
            for mi, idxs in enumerate(micro):
                batch = [encoded[i] for i in idxs]
                L = max(len(b["input_ids"]) for b in batch)
                pad = self.tok.pad_token_id
                ids = torch.tensor([b["input_ids"] + [pad] * (L - len(b["input_ids"])) for b in batch], device=self.device)
                att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (L - len(b["input_ids"])) for b in batch], device=self.device)
                lab = torch.tensor([b["labels"] + [-100] * (L - len(b["labels"])) for b in batch], device=self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                    emb = self.embed(ids, self._feats([b["features"] for b in batch]))
                    out = self.model(inputs_embeds=emb, attention_mask=att, labels=lab)
                    loss = out.loss / cfg.grad_accum
                scaler.scale(loss).backward()
                if (mi + 1) % cfg.grad_accum == 0 or mi == len(micro) - 1:
                    mult = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))
                    for g, lr0 in zip(opt.param_groups, base_lrs):
                        g["lr"] = lr0 * mult
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % cfg.log_every == 0 or step == total:
                        losses.append({"step": step, "loss": float(loss) * cfg.grad_accum})
                        if verbose:
                            el = time.time() - t0
                            print(f"\r  [{self.method}] step {step}/{total} loss {losses[-1]['loss']:.4f} "
                                  f"({el / 60:.1f} min, ~{el / step * (total - step) / 60:.1f} min left)", end="")
        if verbose:
            print()
        return {"train_minutes": round((time.time() - t0) / 60, 2), "steps": step, "losses": losses,
                "n_train": len(encoded),
                "trainable_params": sum(p.numel() for g in groups for p in g["params"])}

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, prompts: list[str]) -> list[str]:
        self.model.eval()
        if self.encoder is not None:
            self.encoder.eval()
        encs = [self._encode(p, with_answer=False) for p in prompts]
        L = max(len(e["input_ids"]) for e in encs)
        pad = self.tok.pad_token_id
        ids = torch.tensor([[pad] * (L - len(e["input_ids"])) + e["input_ids"] for e in encs], device=self.device)
        att = torch.tensor([[0] * (L - len(e["input_ids"])) + [1] * len(e["input_ids"]) for e in encs], device=self.device)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            emb = self.embed(ids, self._feats([e["features"] for e in encs]))
            out = self.model.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=self.cfg.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None, top_k=None,
                                      pad_token_id=pad, eos_token_id=self.tok.eos_token_id,
                                      suppress_tokens=[self.num_id])
        # with inputs_embeds (no input_ids) generate returns only the new tokens
        return self.tok.batch_decode(out, skip_special_tokens=True)

    def close(self) -> None:
        import gc
        del self.model, self.encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_method(method: str, train_items: list[dict], test_items: list[dict], cfg: RetrofitConfig,
               out_dir: str | Path) -> dict:
    """Train one method, evaluate it on FinNum test prompts, save predictions + training log (resumable)."""
    from finnum.eval.runner import run_eval

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"{method}.jsonl"
    log_path = out_dir / f"{method}_train.json"
    if preds_path.exists() and log_path.exists() and sum(1 for _ in preds_path.open()) >= len(test_items):
        print(f"already done: {method}")
        return json.loads(log_path.read_text())
    # Models are not saved between sessions, so an unfinished method restarts from scratch.
    for p in (preds_path, log_path):
        if p.exists():
            p.unlink()
    rf = Retrofit(method, cfg)
    try:
        log = rf.train(train_items)
        log.update(method=method, config=asdict(cfg))
        run_eval(test_items, rf.generate, method, preds_path, batch_size=cfg.eval_batch,
                 count_tokens=rf.count_prompt_tokens)
        log_path.write_text(json.dumps(log))
    finally:
        rf.close()
    return log
