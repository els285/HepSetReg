"""Pluggable physics-consistency loss.

DIRECTOR hard-codes a ttbar-specific formula: it reconstructs top/antitop/
ttbar invariant masses from 8 fixed output columns
(``top_px, top_py, top_pz, top_E, antitop_px, ...``) and penalizes deviation
from truth mass arrays baked into the HDF5 schema. That formula, the output
layout it assumes, and the truth arrays it reads are all specific to ttbar
kinematic regression.

Here the same *idea* -- "penalize predictions whose derived physical
quantities disagree with known truth/consistency constraints" -- is exposed
as a user-supplied callable, so it works for any process and any target
layout (direct mttbar regression, top four-vector regression, or anything
else).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

ConsistencyFn = Callable[[Tensor, Dict[str, Any]], Union[Tensor, Dict[str, Tensor]]]


class PhysicsConsistencyLoss(nn.Module):
    """Wraps a user callable ``fn(y_pred, batch) -> Tensor | dict[str, Tensor]``.

    ``y_pred`` is the (optionally unscaled -- see ``target_scaler`` on the
    LightningModules) model prediction. ``batch`` is the raw dataloader batch
    dict, so ``fn`` can reach any extra ground-truth arrays the user stashed
    there for exactly this purpose (analogous to DIRECTOR's ``M``/``L``/``R``
    HDF5 arrays, but under whatever key name the user chooses).

    Example -- reproducing DIRECTOR's ttbar mass-consistency term::

        def ttbar_mass_consistency(y_pred, batch):
            top = y_pred[:, 0:4]       # px, py, pz, E
            antitop = y_pred[:, 4:8]
            def mass(p4):
                return torch.sqrt(torch.clamp(p4[:, 3] ** 2 - p4[:, :3].pow(2).sum(-1), min=1e-6))
            top_m, antitop_m = mass(top), mass(antitop)
            ttbar_m = mass(top + antitop)
            truth = batch["truth_mass"]  # (B, 3): top, antitop, ttbar
            huber = torch.nn.functional.huber_loss
            return {
                "top": huber(top_m, truth[:, 0]),
                "antitop": huber(antitop_m, truth[:, 1]),
                "system": huber(ttbar_m, truth[:, 2]),
            }

        loss_terms["consistency"] = PhysicsConsistencyLoss(ttbar_mass_consistency)
    """

    def __init__(self, fn: ConsistencyFn):
        super().__init__()
        self.fn = fn

    def forward(self, y_pred: Tensor, batch: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Tensor]]:
        out = self.fn(y_pred, batch)
        if isinstance(out, dict):
            total = torch.stack([v for v in out.values()]).sum()
            return total, out
        return out, {"consistency": out}
