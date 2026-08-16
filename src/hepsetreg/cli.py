"""``hepsetreg-train`` entry point: build+train a model straight from a YAML config."""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from hepsetreg.config import load_config
from hepsetreg.factory import build_datamodule, build_lightning_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def build_trainer(cfg: DictConfig) -> L.Trainer:
    train_cfg = cfg.train
    monitor = str(train_cfg.get("monitor_metric", "val_loss"))
    mode = str(train_cfg.get("monitor_mode", "min"))

    callbacks = [
        ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            filename="{epoch:03d}-{" + monitor + ":.6f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    patience = train_cfg.get("early_stopping_patience", None)
    if patience:
        callbacks.append(EarlyStopping(monitor=monitor, mode=mode, patience=int(patience)))

    kwargs = {}
    grad_clip = train_cfg.get("gradient_clip_val", None)
    if grad_clip:
        kwargs["gradient_clip_val"] = float(grad_clip)

    return L.Trainer(
        max_epochs=int(train_cfg.get("max_epochs", 100)),
        accelerator=str(train_cfg.get("accelerator", "auto")),
        devices=train_cfg.get("devices", "auto"),
        callbacks=callbacks,
        default_root_dir=str(train_cfg.get("output_dir", "training_logs")),
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 50)),
        **kwargs,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train a hepsetreg model from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Overrides cfg.seed.")
    parser.add_argument(
        "overrides", nargs="*", help="Dotlist config overrides, e.g. train.max_epochs=5 model.d_model=64"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    if args.seed is not None:
        cfg.seed = args.seed

    L.seed_everything(int(cfg.get("seed", 42)), workers=True)

    datamodule = build_datamodule(cfg)
    model = build_lightning_module(cfg)
    trainer = build_trainer(cfg)

    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    trainer.fit(model, datamodule=datamodule)

    if cfg.data.get("test_file", None):
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
