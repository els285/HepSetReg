"""Stateless conditional flow-matching loss.

Sampling of ``y0``, ``t``, ``x_t`` and the forward pass through the velocity
field are owned by the LightningModule (see
:class:`hepsetreg.lightning_modules.flow_matching.FlowMatchingRegressor`) so
that this term stays a simple, picklable ``nn.Module`` like the other loss
terms in :class:`hepsetreg.losses.composite.CompositeLoss`.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConditionalFlowMatchingLoss(nn.Module):
    def forward(self, v_pred: Tensor, v_target: Tensor) -> Tensor:
        return F.mse_loss(v_pred, v_target)
