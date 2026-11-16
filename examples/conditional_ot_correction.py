"""Frozen-baseline correction example using the HepSetReg backbone/head interface.

Run from an installed checkout: python examples/conditional_ot_correction.py
The synthetic events exercise the full correction objective on CPU. For real
data, pass an EventRegressor and its ordinary input batch to the same wrapper.
"""

from __future__ import annotations

import json

import torch
from torch import nn
from torch.nn import functional as F

from hepsetreg.losses.conditional_ot import (
    GroupDefinition, assign_groups, conditional_ot_loss, fit_groups, make_directions,
)
from hepsetreg.losses.shape_losses import emd_loss
from hepsetreg.models import EventRegressor


class FrozenBaselineCorrection(nn.Module):
    """Refine a frozen EventRegressor in its original standardised target space.

    The baseline must expose ``backbone(batch)`` and ``head(features)`` and
    predict the four observables in documented order. Feature dimension is 128
    in the recorded study, giving 132 inputs to the correction network.
    """

    def __init__(self, baseline: nn.Module, feature_dim: int = 128):
        super().__init__()
        self.baseline = baseline.requires_grad_(False).eval()
        self.correction = nn.Sequential(
            nn.Linear(feature_dim + 4, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 4),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen encoder must also keep dropout/normalisation in evaluation mode.
        self.baseline.eval()
        return self

    def forward(self, batch):
        with torch.no_grad():
            features = self.baseline.backbone(batch)
            baseline_z = self.baseline.head(features)
        if baseline_z.ndim != 2 or baseline_z.shape[1] != 4:
            raise ValueError("The baseline must predict [batch, 4] standardised observables")
        condition = torch.cat((features.detach(), baseline_z.detach()), dim=1)
        delta_z = 0.5 * torch.tanh(self.correction(condition))
        return baseline_z + delta_z, baseline_z, delta_z


def correction_objective(prediction_z, target_z, groups, reference, directions):
    """Recorded objective: 2 Huber(delta=1) + 0.1 marginal W1 + 0.1 COT."""
    with torch.autocast(device_type=prediction_z.device.type, enabled=False):
        prediction_z, target_z = prediction_z.float(), target_z.detach().float()
        point = 2.0 * F.huber_loss(prediction_z, target_z, delta=1.0)
        marginal = emd_loss(prediction_z, target_z)
        conditional = conditional_ot_loss(
            prediction_z, target_z, groups, reference.probabilities, directions,
        )
        return point + 0.1 * marginal + 0.1 * conditional, {
            "2_huber": point.detach(),
            "marginal_w1": marginal.detach(),
            "conditional_w2_squared": conditional.detach(),
        }


def main():
    torch.manual_seed(42)
    # Replace this illustrative backbone and head with the loaded EventRegressor.
    # Real EventRegressor backbones accept the normal HepSetReg batch dictionary.
    baseline = EventRegressor(
        nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Dropout(0.1)),
        nn.Linear(128, 4),
    )
    model = FrozenBaselineCorrection(baseline)
    training_inputs = torch.randn(1000, 6)
    with torch.no_grad():
        initial_z, baseline_z, _ = model(training_inputs)
        torch.testing.assert_close(initial_z, baseline_z, rtol=0, atol=0)
        training_targets_z = baseline_z + 0.15 * torch.randn_like(baseline_z)
    # Mean/std must come from the original model's training-only target scaler.
    # Mass is column zero. These numerical values serve this synthetic example.
    target_mean = torch.tensor([500.0, 0.0, 0.0, 0.0])
    target_std = torch.tensor([150.0, 0.8, 0.5, 0.5])
    training_baseline_mass = baseline_z[:, 0] * target_std[0] + target_mean[0]
    reference = fit_groups(training_baseline_mass.cpu().numpy())
    # Persist this metadata and the directions in the actual model checkpoint.
    reference = GroupDefinition.from_dict(json.loads(json.dumps(reference.to_dict())))
    fixed_training_groups = assign_groups(training_baseline_mass, reference)
    directions = make_directions()
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=3e-4)

    before = {key: value.detach().clone() for key, value in baseline.state_dict().items()}
    model.train()
    for indices in torch.randperm(len(training_inputs)).split(256):
        optimizer.zero_grad(set_to_none=True)
        prediction_z, _, _ = model(training_inputs[indices])
        loss, components = correction_objective(
            prediction_z, training_targets_z[indices], fixed_training_groups[indices],
            reference, directions,
        )
        loss.backward()
        optimizer.step()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in baseline.parameters())
    print("Frozen training group counts:", reference.counts)
    print("Final batch objective:", float(loss.detach()))
    print("Components:", {key: float(value) for key, value in components.items()})

    model.eval()
    with torch.no_grad():
        _, _, delta_z = model(training_inputs)
        # Add the correction to the exact saved physical baseline to preserve
        # its zero-correction identity through floating-point round trips.
        baseline_physical = baseline_z.double() * target_std.double() + target_mean.double()
        corrected_physical = baseline_physical + delta_z.double() * target_std.double()
    print("Physical predictions:", tuple(corrected_physical.shape))


if __name__ == "__main__":
    main()
