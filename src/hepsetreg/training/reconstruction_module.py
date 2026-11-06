"""LightningModule training reconstruction heads on one shared encoder pass."""

import pytorch_lightning as pl
import torch


class ReconstructionModule(pl.LightningModule):
    def __init__(self, backbone, assignment, learning_rate=1e-3):
        super().__init__()
        self.backbone = backbone
        self.assignment = assignment
        self.learning_rate = learning_rate

    def losses(self, batch):
        _, tokens, _ = self.backbone(batch, return_tokens=True)
        return {"assignment": self.assignment.loss(tokens, batch)}

    def _step(self, batch, stage):
        losses = self.losses(batch)
        total = sum(losses.values())
        batch_size = len(batch["global"])
        self.log_dict({f"{stage}_{k}_loss": v for k, v in losses.items()}, batch_size=batch_size)
        self.log(f"{stage}_loss", total, prog_bar=True, batch_size=batch_size)
        return total

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def predict_step(self, batch, batch_idx):
        _, tokens, _ = self.backbone(batch, return_tokens=True)
        index, prob = self.assignment.predict(tokens, batch)
        return {"assignment": index.cpu(), "assignment_prob": prob.cpu()}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
