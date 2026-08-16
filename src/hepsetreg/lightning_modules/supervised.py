"""Standard supervised regression training (Huber/MSE, + optional
distribution-matching KL and physics-consistency terms via
:class:`~hepsetreg.losses.composite.CompositeLoss`)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import lightning as L
import torch

from hepsetreg.data.scaling import TargetScaler
from hepsetreg.lightning_modules.optim import build_optimizer
from hepsetreg.losses.composite import CompositeLoss
from hepsetreg.models.backbone import ObjectSetEncoder
from hepsetreg.models.heads import RegressionHead


class SupervisedRegressor(L.LightningModule):
    """Encoder + regression head, trained with a :class:`CompositeLoss`.

    Expects batches shaped like :class:`hepsetreg.data.dataset.PaddedObjectDataset`
    items: ``{"objects": {...}, "mask": {...}, "target": Tensor, **extras}``.
    """

    def __init__(
        self,
        encoder: ObjectSetEncoder,
        head: RegressionHead,
        loss: CompositeLoss,
        train_cfg: Optional[Dict[str, Any]] = None,
        target_scaler: Optional[TargetScaler] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.loss = loss
        self.train_cfg = train_cfg or {}
        self.target_scaler = target_scaler

    def forward(self, objects, mask):
        pooled = self.encoder(objects, mask)
        return self.head(pooled)

    def _loss_inputs(self, y_pred, batch) -> Dict[str, Dict[str, Any]]:
        inputs: Dict[str, Dict[str, Any]] = {}
        if "regression" in self.loss.terms:
            inputs["regression"] = {"pred": y_pred, "target": batch["target"]}
        if "distribution_kl" in self.loss.terms:
            inputs["distribution_kl"] = {"pred": y_pred, "target": batch["target"]}
        if "consistency" in self.loss.terms:
            y_unscaled = self.target_scaler.unscale(y_pred) if self.target_scaler is not None else y_pred
            inputs["consistency"] = {"y_pred": y_unscaled, "batch": batch}
        return inputs

    def _shared_step(self, batch, stage: str):
        y_pred = self(batch["objects"], batch["mask"])
        inputs = self._loss_inputs(y_pred, batch)
        total, logs = self.loss(inputs)

        batch_size = batch["target"].shape[0]
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

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        y_pred = self(batch["objects"], batch["mask"])
        if self.target_scaler is not None:
            y_pred = self.target_scaler.unscale(y_pred)
        return y_pred

    def on_train_epoch_start(self) -> None:
        self.loss.set_epoch(self.current_epoch)

    def configure_optimizers(self):
        return build_optimizer(self.parameters(), self.train_cfg)
