"""
ShapeRegressionModule: EventRegressionModule plus a shape-matching term.

    total loss = event-by-event loss + shape_weight * shape_loss(pred, target)

The shape term compares the batch distribution of each output column
(see shape_losses.py). Everything else -- optimizer, scheduler, logging of
"<stage>_loss" which early stopping and LR scheduling monitor -- is inherited
unchanged from EventRegressionModule.
"""

from __future__ import annotations

from hepsetreg.losses.shape_losses import SHAPE_LOSSES
from hepsetreg.training.lightning_module import EventRegressionModule


class ShapeRegressionModule(EventRegressionModule):
    def __init__(
        self, model, shape_metric, shape_weight=0.1,
        learning_rate=1e-3, weight_decay=0.0, loss_kind="mse", scheduler_fn=None,
    ):
        super().__init__(
            model, learning_rate=learning_rate, weight_decay=weight_decay,
            loss_kind=loss_kind, scheduler_fn=scheduler_fn,
        )
        self.shape_loss_fn = SHAPE_LOSSES[shape_metric]
        self.shape_weight = shape_weight

    def _shared_step(self, batch, stage):
        pred = self.model(batch)
        event_loss = self.loss_fn(pred, batch["target"])
        shape_loss = self.shape_loss_fn(pred, batch["target"])
        loss = event_loss + self.shape_weight * shape_loss
        batch_size = batch["target"].shape[0]
        self.log(
            f"{stage}_loss", loss,
            on_step=(stage == "train"), on_epoch=True, prog_bar=True, batch_size=batch_size,
        )
        self.log(f"{stage}_shape_loss", shape_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss
