"""Example user-supplied physics-consistency loss hooks.

These are referenced from the example configs via a dotted path
(``loss.terms.consistency.fn: "physics_hooks:toy_mass_consistency"``) and
resolved at build time by :func:`hepsetreg.factory._resolve_callable`. This
is the generalization of DIRECTOR's hard-coded ttbar mass-consistency
formula: same idea, but supplied by the user rather than baked into the
training script, so it can be swapped per-process/per-target-layout.
"""

from __future__ import annotations

import torch


def toy_mass_consistency(y_pred: torch.Tensor, batch: dict) -> torch.Tensor:
    """Penalizes the predicted system mass (output column 0, already
    unscaled to physical units) for disagreeing with the independently
    stored ``truth_mass`` extra array from the toy dataset."""
    truth_mass = batch["truth_mass"].squeeze(-1)
    pred_mass = y_pred[:, 0]
    return torch.nn.functional.huber_loss(pred_mass, truth_mass)


def ttbar_mass_consistency(y_pred: torch.Tensor, batch: dict) -> dict:
    """Reproduces DIRECTOR's ttbar kinematic mass-consistency loss as a
    pluggable hook, for reference. Assumes an 8-dim output layout
    ``[top_px, top_py, top_pz, top_E, antitop_px, antitop_py, antitop_pz, antitop_E]``
    (already unscaled to GeV) and a ``batch["truth_mass"]`` array shaped
    ``(B, 3)`` = ``[top_mass, antitop_mass, ttbar_mass]``, matching DIRECTOR's
    ``M`` HDF5 dataset.
    """

    def mass(p4: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(p4[:, 3] ** 2 - p4[:, :3].pow(2).sum(-1), min=1e-6))

    top, antitop = y_pred[:, 0:4], y_pred[:, 4:8]
    top_m, antitop_m = mass(top), mass(antitop)
    ttbar_m = mass(top + antitop)

    truth = batch["truth_mass"]
    huber = torch.nn.functional.huber_loss
    return {
        "top": huber(top_m, truth[:, 0]),
        "antitop": huber(antitop_m, truth[:, 1]),
        "system": huber(ttbar_m, truth[:, 2]),
    }
