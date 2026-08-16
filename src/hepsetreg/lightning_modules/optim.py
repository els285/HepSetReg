"""Shared optimizer/scheduler construction for the LightningModules."""

from __future__ import annotations

from typing import Any, Dict

import torch


def build_optimizer(parameters, cfg: Dict[str, Any]):
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("learning_rate", 1.0e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))

    if name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=cfg.get("momentum", 0.9))
    else:
        raise ValueError(f"Unsupported optimizer '{name}'.")

    scheduler_cfg = cfg.get("lr_scheduler")
    if not scheduler_cfg or not scheduler_cfg.get("enabled", False):
        return optimizer

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler_cfg.get("mode", "min")),
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 5)),
        min_lr=float(scheduler_cfg.get("min_lr", 1.0e-7)),
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": str(scheduler_cfg.get("monitor", "val_loss")),
            "interval": "epoch",
            "frequency": 1,
        },
    }
