"""
LightningModule wiring a backbone+head model into a training loop: a plain
pointwise regression loss (MSE by default, via `losses.RegressionLoss`)
against EventDataset5's (already z-scored) "target".

Model-agnostic: works with `EventRegressor` (backbone.py + head.py) as-is,
or any `nn.Module` whose `forward(batch) -> (B, output_dim)` matches that
shape.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from losses import RegressionLoss


class EventRegressionModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, weight_decay=0.0, loss_kind="mse"):
        super().__init__()
        self.model = model
        self.loss_fn = RegressionLoss(kind=loss_kind)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch):
        return self.model(batch)

    def _shared_step(self, batch, stage):
        pred = self.model(batch)
        loss = self.loss_fn(pred, batch["target"])
        self.log(
            f"{stage}_loss", loss,
            on_step=(stage == "train"), on_epoch=True, prog_bar=True,
            batch_size=batch["target"].shape[0],
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, 
                                 weight_decay=self.weight_decay,
                                 fused=True)  # Requires PyTorch 2.0+ and CUDA)
