"""Pooling strategies for collapsing a masked token sequence into one
per-event embedding."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ClsPooling(nn.Module):
    """Returns the encoded CLS token (must be at sequence position 0)."""

    def forward(self, encoded: Tensor, valid_mask: Tensor) -> Tensor:
        return encoded[:, 0]


class MeanPooling(nn.Module):
    """Mean over valid (non-padded) tokens only."""

    def forward(self, encoded: Tensor, valid_mask: Tensor) -> Tensor:
        valid = valid_mask.unsqueeze(-1).to(encoded.dtype)
        summed = (encoded * valid).sum(dim=1)
        counts = valid.sum(dim=1).clamp(min=1.0)
        return summed / counts


class AttentionPooling(nn.Module):
    """Learned attention pooling (as used in DIRECTOR), restricted to valid tokens."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, encoded: Tensor, valid_mask: Tensor) -> Tensor:
        logits = self.score(encoded).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, float("-inf"))
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (encoded * weights).sum(dim=1)


def build_pooling(kind: str, d_model: int) -> nn.Module:
    kind = kind.lower()
    if kind == "cls":
        return ClsPooling()
    if kind == "mean":
        return MeanPooling()
    if kind == "attention":
        return AttentionPooling(d_model)
    raise ValueError(f"Unknown pooling kind '{kind}'; expected 'cls', 'mean', or 'attention'.")
