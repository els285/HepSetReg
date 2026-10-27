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
from hepsetreg.losses.shape_losses import SHAPE_LOSSES


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

    `shape_metric` (None by default, same choices as `SHAPE_LOSSES`/
    `ShapeRegressionModule`) adds a shape-matching term on top of the
    velocity loss:

        total loss = flow_matching_loss + shape_weight * shape_loss(point_pred, target)

    Unlike plain regression (`ShapeRegressionModule`), where `pred` is a
    free side effect of the single forward pass the event loss already
    needed, flow matching has no deterministic forward pass at all -- a
    point estimate only exists by actually ODE-sampling the velocity field
    (`sample_flow`), which costs `shape_n_steps * shape_n_samples` extra
    velocity-field evaluations on *every* training step, not just once per
    epoch like the (full-cost) `val_n_steps`/`val_n_samples` sampling below.
    `shape_n_steps`/`shape_n_samples` default much lower than
    `val_n_steps`/`val_n_samples` for exactly this reason -- a rough,
    cheap sample is enough to get a useful shape-matching gradient signal;
    it doesn't need predict-quality integration.
    """

    def __init__(
        self, model, learning_rate=1e-3, weight_decay=0.0,
        shape_metric=None, shape_weight=0.1, shape_n_steps=10, shape_n_samples=2,
        val_n_steps=20, val_n_samples=4,
        predict_n_steps=50, predict_n_samples=8, keep_all_samples=False, scheduler_fn=None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.shape_loss_fn = SHAPE_LOSSES[shape_metric] if shape_metric is not None else None
        self.shape_weight = shape_weight
        self.shape_n_steps = shape_n_steps
        self.shape_n_samples = shape_n_samples
        self.val_n_steps = val_n_steps
        self.val_n_samples = val_n_samples
        self.predict_n_steps = predict_n_steps
        self.predict_n_samples = predict_n_samples
        self.keep_all_samples = keep_all_samples
        self.scheduler_fn = scheduler_fn  # optimizer -> LR scheduler, or None for a constant LR
        self.save_hyperparameters(ignore=["model", "scheduler_fn"])

    def forward(self, batch, n_steps=50, n_samples=1):
        return self.model(batch, n_steps=n_steps, n_samples=n_samples)

    def training_step(self, batch, batch_idx):
        cond = self.model.backbone(batch)
        loss = flow_matching_loss(self.model.velocity_field, batch["target"], cond)
        batch_size = batch["target"].shape[0]

        if self.shape_loss_fn is not None:
            samples = sample_flow(
                self.model.velocity_field, cond, self.model.velocity_field.output_dim,
                n_steps=self.shape_n_steps, n_samples=self.shape_n_samples,
            )
            shape_loss = self.shape_loss_fn(samples.mean(dim=1), batch["target"])
            loss = loss + self.shape_weight * shape_loss
            self.log("train_shape_loss", shape_loss, on_step=True, on_epoch=True, batch_size=batch_size)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        cond = self.model.backbone(batch)
        loss = flow_matching_loss(self.model.velocity_field, batch["target"], cond)
        batch_size = batch["target"].shape[0]

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
            "val_sample_mse", sample_mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size,
        )

        if self.shape_loss_fn is not None:
            # Reuses the samples drawn above rather than sampling again --
            # same point_pred val_sample_mse was computed from.
            shape_loss = self.shape_loss_fn(point_pred, batch["target"])
            loss = loss + self.shape_weight * shape_loss
            self.log("val_shape_loss", shape_loss, on_step=False, on_epoch=True, batch_size=batch_size)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def predict_step(self, batch, batch_idx):
        """See EventRegressionModule.predict_step -- same contract (a CPU
        {"pred", "target"} dict per batch, still z-scored), except "pred" is
        from `predict_n_steps`/`predict_n_samples` ODE integration, since
        this model doesn't have a single deterministic forward pass: the
        sample-mean point estimate by default, or -- if `keep_all_samples`
        -- every individual draw, shape (B, n_samples, F) instead of
        (B, F), for looking at the predicted distribution itself rather
        than just its mean (e.g. calibration, per-event spread)."""
        samples = self.model(batch, n_steps=self.predict_n_steps, n_samples=self.predict_n_samples)  # (B, n_samples, F)
        pred = samples if self.keep_all_samples else samples.mean(dim=1)
        return {"pred": pred.cpu(), "target": batch["target"].cpu()}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, fused=True,
        )
        if self.scheduler_fn is None:
            return optimizer
        scheduler = self.scheduler_fn(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
