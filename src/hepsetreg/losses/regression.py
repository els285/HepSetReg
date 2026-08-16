"""Base pointwise regression losses."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RegressionLoss(nn.Module):
    """Thin, picklable wrapper so this can sit inside :class:`~hepsetreg.losses.composite.CompositeLoss`
    alongside stateful terms."""

    def __init__(self, kind: str = "huber", delta: float = 1.0):
        super().__init__()
        kind = kind.lower()
        if kind not in {"huber", "mse", "mae"}:
            raise ValueError(f"Unknown regression loss kind '{kind}'.")
        self.kind = kind
        self.delta = delta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.kind == "huber":
            return F.huber_loss(pred, target, delta=self.delta)
        if self.kind == "mse":
            return F.mse_loss(pred, target)
        return F.l1_loss(pred, target)
