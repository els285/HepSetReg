"""
First end-to-end training run: standard TransformerBackbone + RegressionHead
(EventRegressor), trained with plain MSE against EventDataset5's z-scored
targets.
"""

from __future__ import annotations

import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from backbone import TransformerBackbone
from EventDataset5 import (
     GLOBAL_FEATURES, JET_FEATURES,
    LEPTON_FEATURES, TARGET_FEATURES,
)
from EventDataset6 import EventDataset, EventDataModule

from head import EventRegressor, RegressionHead
from lightning_module import EventRegressionModule


def main(
    train_path, val_path, batch_size=256, max_epochs=20, fast_dev_run=False,
    accelerator="auto", log_dir="tb_logs", run_name="event_regression",
    num_workers=0, persistent_workers=False, main_process_threads=4,
    dataset_device=None, compile_model=False,
):
    torch.set_num_threads(main_process_threads)
    torch.set_float32_matmul_precision('medium')

    dm = EventDataModule(
        train_path, val_path, batch_size=batch_size,
        num_workers=num_workers, persistent_workers=persistent_workers,
        device=dataset_device,
    )

    backbone = TransformerBackbone(
        n_global_features=len(GLOBAL_FEATURES),
        n_jet_features=len(JET_FEATURES),
        n_lepton_features=len(LEPTON_FEATURES),
        d_model=128,
        nhead=8,
        num_layers=6,
    )
    head = RegressionHead(d_model=128, output_dim=len(TARGET_FEATURES))
    model = EventRegressor(backbone, head)
    
    # Compile the model for improved performance if requested
    if compile_model:
        model = torch.compile(model)
        
    # Wrap the model in a LightningModule for training
    module = EventRegressionModule(model, learning_rate=1e-3, loss_kind="mse")

    # Set up TensorBoard logger and suppress specific warnings
    logger = TensorBoardLogger(save_dir=log_dir, name=run_name)
    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
    
    # Define the trainer with bf16-mixed precision for faster training on compatible GPUs
    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator=accelerator, devices="auto",
        fast_dev_run=fast_dev_run, logger=logger, precision="bf16-mixed"
    )
    
    # Start the training process
    trainer.fit(module, datamodule=dm)

    # Persist the z-score stats + max_jets alongside this run's checkpoints:
    # inference needs to scale/pad new data EXACTLY the way this run did, and
    # can't re-fit them from scratch (that would use different values, and
    # silently produce wrong predictions).
    stats_path = f"{logger.log_dir}/stats.pt"
    torch.save({"stats": dm.stats, "max_jets": dm.train_dataset.max_jets}, stats_path)
    print(f"saved stats to {stats_path}")

    return trainer, module, dm


if __name__ == "__main__":
    main(
        "ttbar-2L_train.root_180926.parquet",
        "ttbar-2L_val.root_180926.parquet",
        batch_size=4096,
        fast_dev_run=False,
        accelerator="gpu",
        dataset_device=None,
        compile_model=True,
        num_workers=4,
    )
