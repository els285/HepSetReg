"""Conditional flow-matching training, generalizing DIRECTOR's ``flowmatch_train.py``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import lightning as L
import torch

from hepsetreg.data.scaling import TargetScaler
from hepsetreg.lightning_modules.optim import build_optimizer
from hepsetreg.losses.composite import CompositeLoss
from hepsetreg.models.flow_matching import ConditionalVelocityField, sample_flow


@dataclass
class SamplingConfig:
    n_steps: int = 50
    n_samples: int = 8
    n_steps_training: int = 8
    n_samples_training: int = 4


class FlowMatchingRegressor(L.LightningModule):
    """Trains a :class:`ConditionalVelocityField` with conditional flow matching.

    Distribution-matching and physics-consistency loss terms (if configured)
    are computed from ODE-sampled predictions during training, exactly as in
    DIRECTOR -- but with a cheaper ``sampling.n_steps_training``/
    ``n_samples_training`` used for the training-time samples, and the full
    ``sampling.n_steps``/``n_samples`` reserved for inference via
    :meth:`sample`.
    """

    def __init__(
        self,
        velocity_field: ConditionalVelocityField,
        loss: CompositeLoss,
        output_dim: int,
        train_cfg: Optional[Dict[str, Any]] = None,
        sampling_cfg: Optional[SamplingConfig] = None,
        target_scaler: Optional[TargetScaler] = None,
    ):
        super().__init__()
        self.velocity_field = velocity_field
        self.loss = loss
        self.output_dim = output_dim
        self.train_cfg = train_cfg or {}
        self.sampling_cfg = sampling_cfg or SamplingConfig()
        self.target_scaler = target_scaler

    def _needs_samples(self) -> bool:
        return ("distribution_kl" in self.loss.terms) or ("consistency" in self.loss.terms)

    def _shared_step(self, batch, stage: str):
        y1 = batch["target"]
        y0 = torch.randn_like(y1)
        t = torch.rand(y1.shape[0], 1, device=y1.device, dtype=y1.dtype)
        xt = (1 - t) * y0 + t * y1

        v_pred = self.velocity_field(xt, t, batch["objects"], batch["mask"])
        v_target = y1 - y0
        inputs: Dict[str, Dict[str, Any]] = {"flow_matching": {"v_pred": v_pred, "v_target": v_target}}

        if self._needs_samples():
            samples = sample_flow(
                self.velocity_field,
                batch["objects"],
                batch["mask"],
                n_steps=self.sampling_cfg.n_steps_training,
                n_samples=self.sampling_cfg.n_samples_training,
            )
            y_pred_mean = samples.mean(dim=1)

            if "distribution_kl" in self.loss.terms:
                inputs["distribution_kl"] = {"pred": y_pred_mean, "target": y1}
            if "consistency" in self.loss.terms:
                y_unscaled = (
                    self.target_scaler.unscale(y_pred_mean) if self.target_scaler is not None else y_pred_mean
                )
                inputs["consistency"] = {"y_pred": y_unscaled, "batch": batch}

        total, logs = self.loss(inputs)

        batch_size = y1.shape[0]
        for name, value in logs.items():
            self.log(
                f"{stage}_{name}",
                value,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=(name == "loss"),
                batch_size=batch_size,
            )
        return total

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def sample(self, batch) -> torch.Tensor:
        """Full-fidelity inference sampling using ``sampling_cfg.n_steps``/``n_samples``."""
        samples = sample_flow(
            self.velocity_field,
            batch["objects"],
            batch["mask"],
            n_steps=self.sampling_cfg.n_steps,
            n_samples=self.sampling_cfg.n_samples,
        )
        y_pred = samples.mean(dim=1)
        if self.target_scaler is not None:
            y_pred = self.target_scaler.unscale(y_pred)
        return y_pred

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self.sample(batch)

    def on_train_epoch_start(self) -> None:
        self.loss.set_epoch(self.current_epoch)

    def configure_optimizers(self):
        return build_optimizer(self.parameters(), self.train_cfg)
