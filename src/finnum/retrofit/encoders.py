"""Trainable number encoders: feature vector -> one embedding in the LLM's input space."""
from __future__ import annotations

import torch
import torch.nn as nn


class NumberEncoder(nn.Module):
    """Maps per-number features to an embedding for the <num> token.

    multiplicative=True is xVal: embedding = learned vector * scalar value.
    Otherwise: embedding = learned base vector + MLP(features). The last layer is
    initialised small so training starts close to "a plain <num> token".
    """

    def __init__(self, feat_dim: int, hidden_size: int, emb_std: float, multiplicative: bool = False,
                 width: int = 512):
        super().__init__()
        self.multiplicative = multiplicative
        self.base = nn.Parameter(torch.randn(hidden_size) * emb_std)
        if multiplicative:
            self.mlp = None
        else:
            self.mlp = nn.Sequential(nn.Linear(feat_dim, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                     nn.Linear(width, hidden_size))
            nn.init.normal_(self.mlp[-1].weight, std=emb_std * 0.1)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.multiplicative:
            return self.base[None, :] * feats[:, :1] * 12.0  # undo the /12 scaling so |scale| ~ log10 magnitude
        return self.base[None, :] + self.mlp(feats)
