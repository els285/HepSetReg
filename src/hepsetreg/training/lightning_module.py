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

from hepsetreg.backbones.flow_matching import flow_matching_loss, sample_flow
from hepsetreg.losses.losses import RegressionLoss


class EventRegressionModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, weight_decay=0.0, loss_kind="mse", scheduler_fn=None):
        super().__init__()
        self.model = model
        self.loss_fn = RegressionLoss(kind=loss_kind)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_fn = scheduler_fn  # optimizer -> LR scheduler, or None for a constant LR
        self.save_hyperparameters(ignore=["model", "scheduler_fn"])

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

    def predict_step(self, batch, batch_idx):
        """Returns the (still z-scored) prediction and target for one batch,
        moved to CPU. The run_*.py predict() functions concatenate these
        across all batches from Trainer.predict()'s returned list, then
        unscale once at the end -- Trainer.predict() already handles model
        eval-mode, no_grad, and per-batch device transfer, so there's no
        manual loop or .to(device) needed on the caller's side."""
        pred = self.model(batch)
        return {"pred": pred.cpu(), "target": batch["target"].cpu()}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay,
                                      fused=True)  # Requires PyTorch 2.0+ and CUDA)
        if self.scheduler_fn is None:
            return optimizer
        scheduler = self.scheduler_fn(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


class FlowMatchingModule(pl.LightningModule):
    """Trains a `FlowRegressor` (flow_matching.py) via conditional flow
    matching instead of direct regression: each step samples noise and a
    random `t`, linearly interpolates towards the (already z-scored)
    target, and matches the velocity field's prediction to the true
    interpolation velocity -- see `flow_matching.flow_matching_loss`.

    A training/validation step here calls `model.backbone` and
    `model.velocity_field` directly rather than `model(batch)`, since the
    loss needs the pooled condition vector and the raw velocity field
    separately -- `model.forward` is reserved for inference-time sampling
    (see `FlowRegressor.forward`), not used during training at all.
    """

    def __init__(
        self, model, learning_rate=1e-3, weight_decay=0.0,
        val_n_steps=20, val_n_samples=4,
        predict_n_steps=50, predict_n_samples=8, scheduler_fn=None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.val_n_steps = val_n_steps
        self.val_n_samples = val_n_samples
        self.predict_n_steps = predict_n_steps
        self.predict_n_samples = predict_n_samples
        self.scheduler_fn = scheduler_fn  # optimizer -> LR scheduler, or None for a constant LR
        self.save_hyperparameters(ignore=["model", "scheduler_fn"])

    def forward(self, batch, n_steps=50, n_samples=1):
        return self.model(batch, n_steps=n_steps, n_samples=n_samples)

    def training_step(self, batch, batch_idx):
        cond = self.model.backbone(batch)
        loss = flow_matching_loss(self.model.velocity_field, batch["target"], cond)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
            batch_size=batch["target"].shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        cond = self.model.backbone(batch)
        loss = flow_matching_loss(self.model.velocity_field, batch["target"], cond)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
            batch_size=batch["target"].shape[0],
        )

        # The flow-matching loss above is a velocity MSE on a random noise
        # draw -- not directly comparable across runs the way a point-
        # estimate loss is. Also report an actual sampled point-estimate
        # MSE (sample mean over val_n_samples draws, val_n_steps ODE steps
        # each) so validation curves are comparable to EventRegressionModule's.
        samples = sample_flow(
            self.model.velocity_field, cond, self.model.velocity_field.output_dim,
            n_steps=self.val_n_steps, n_samples=self.val_n_samples,
        )
        point_pred = samples.mean(dim=1)  # (B, output_dim)
        sample_mse = (point_pred - batch["target"]).pow(2).mean()
        self.log(
            "val_sample_mse", sample_mse, on_step=False, on_epoch=True, prog_bar=True,
            batch_size=batch["target"].shape[0],
        )
        return loss

    def predict_step(self, batch, batch_idx):
        """See EventRegressionModule.predict_step -- same contract (a CPU
        {"pred", "target"} dict per batch, still z-scored), except "pred" is
        the sample-mean point estimate from `predict_n_steps`/
        `predict_n_samples` ODE integration, since this model doesn't have
        a single deterministic forward pass."""
        samples = self.model(batch, n_steps=self.predict_n_steps, n_samples=self.predict_n_samples)  # (B, n_samples, F)
        pred = samples.mean(dim=1)
        return {"pred": pred.cpu(), "target": batch["target"].cpu()}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, fused=True,
        )
        if self.scheduler_fn is None:
            return optimizer
        scheduler = self.scheduler_fn(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
