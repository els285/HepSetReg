"""Combines named loss terms into one training objective.

This is what makes the loss setup "modular ... in principle could include
all" (standard regression, distribution-matching, flow-matching,
physics-consistency) rather than hard-coded together in one script the way
DIRECTOR combines Huber + KL + mass loss. Any subset of terms can be present;
each has its own weight and optional linear warm-up ("ramp"), matching
DIRECTOR's ``kl_ramp_epochs`` behaviour but generalized to every term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class LossTermConfig:
    weight: float = 1.0
    ramp_epochs: int = 0  # 0 => full weight from the start; N => linear ramp over N epochs


class CompositeLoss(nn.Module):
    """``terms`` maps a term name to an ``nn.Module`` whose ``forward`` either
    returns a scalar :class:`~torch.Tensor`, or a ``(scalar, {name: tensor})``
    tuple (used by terms that report multiple named sub-components, e.g.
    :class:`~hepsetreg.losses.consistency.PhysicsConsistencyLoss`).

    Call as ``composite(inputs)`` where ``inputs`` maps term name to a kwargs
    dict for that term's ``forward``. Terms whose name is missing from
    ``inputs`` for a given call are simply skipped -- this lets a
    LightningModule decide per-step which terms it has the data to compute
    (e.g. skip an expensive ODE-sampled distribution term on steps where
    it isn't needed).
    """

    def __init__(self, terms: Dict[str, nn.Module], configs: Dict[str, LossTermConfig] = None):
        super().__init__()
        if not terms:
            raise ValueError("CompositeLoss requires at least one loss term.")
        self.terms = nn.ModuleDict(terms)
        self.configs = configs or {name: LossTermConfig() for name in terms}
        missing = set(terms) - set(self.configs)
        if missing:
            raise ValueError(f"Missing LossTermConfig for term(s): {sorted(missing)}")
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def weight_for(self, name: str) -> float:
        cfg = self.configs[name]
        if cfg.ramp_epochs <= 0:
            return cfg.weight
        return cfg.weight * min(1.0, (self._epoch + 1) / cfg.ramp_epochs)

    def forward(self, inputs: Dict[str, Dict[str, Any]]) -> Tuple[Tensor, Dict[str, Any]]:
        total = None
        logs: Dict[str, Any] = {}

        for name, term in self.terms.items():
            term_inputs = inputs.get(name)
            if term_inputs is None:
                continue

            result = term(**term_inputs)
            if isinstance(result, tuple):
                scalar, sub_terms = result
                for sub_name, sub_value in sub_terms.items():
                    logs[f"{name}/{sub_name}"] = sub_value.detach() if torch.is_tensor(sub_value) else sub_value
            else:
                scalar = result

            weight = self.weight_for(name)
            logs[f"{name}_loss"] = scalar.detach()
            logs[f"{name}_weight"] = weight
            weighted = scalar * weight
            total = weighted if total is None else total + weighted

        if total is None:
            raise ValueError(
                f"None of the configured loss terms {list(self.terms)} had matching "
                f"inputs (got inputs for {list(inputs)}); nothing to optimize this step."
            )

        logs["loss"] = total.detach()
        return total, logs
