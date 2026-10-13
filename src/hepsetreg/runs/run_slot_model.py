"""
Train or run inference for the per-slot (top, antitop) regression task.

The model predicts each top quark's (px, py, pz, E). Slot 0 is always the
top and slot 1 is always the antitop (see EventDatasetSlot.py) -- a fixed,
physically meaningful order, not an interchangeable one -- so we use
CrossAttentionDecoder (fixed learned queries) and a plain per-slot loss,
not SlotAttention/matching.

Usage:
    python -m hepsetreg.runs.run_slot_model --train --config configs/run_slot_model.yaml
    python -m hepsetreg.runs.run_slot_model --predict --config configs/run_slot_model.yaml
"""

import sys

from hepsetreg.config import check_args

# Checked here, before the heavy imports below, so a bad command line
# (missing --train/--predict or --config) fails in milliseconds rather than
# only being reported after paying the full torch/pytorch_lightning/... import cost.
if __name__ == "__main__":
    check_args("usage: python -m hepsetreg.runs.run_slot_model --train | --predict --config <path.yaml>")

import logging

import awkward as ak
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from hepsetreg.backbones.backbone import TransformerBackbone
from hepsetreg.backbones.decoder import CrossAttentionDecoder
from hepsetreg.backbones.head import RegressionHead, SetEventRegressor
from hepsetreg.config import load_config
from hepsetreg.datasets.EventDatasetSlot import SLOT_NAMES, SlotEventDataModule, SlotEventDataset, _identity_collate
from hepsetreg.losses.shape_losses import SHAPE_LOSSES
from hepsetreg.training.lightning_module import EventRegressionModule
from hepsetreg.training.shape_module import ShapeRegressionModule

MODEL_KEYS = ["d_model", "n_enc_heads", "n_dec_heads", "num_enc_layers", "num_dec_layers", "n_queries"]


def build_model(config):
    """Builds the model architecture from a config dict (see MODEL_KEYS,
    config["features"] and config["slot_target_features"]). train() and
    predict() both call this with the same model settings, so a saved
    checkpoint's weights always match the model they're loaded into."""
    features = config["features"]
    backbone = TransformerBackbone(
        n_global_features=len(features["global"]),
        n_jet_features=len(features["jet"]),
        n_lepton_features=len(features["lepton"]),
        d_model=config["d_model"],
        nhead=config["n_enc_heads"],
        num_layers=config["num_enc_layers"],
    )
    decoder = CrossAttentionDecoder(
        d_model=config["d_model"],
        n_queries=config["n_queries"],
        nhead=config["n_dec_heads"],
        num_layers=config["num_dec_layers"],
    )
    head = RegressionHead(d_model=config["d_model"], output_dim=len(config["slot_target_features"]))
    return SetEventRegressor(backbone, decoder, head)


def train(config):
    # Not a hyperparameter -- an environment setting. PyTorch defaults to
    # using every visible CPU core for every op, even tiny ones; on a
    # many-core shared machine that causes severe thread-oversubscription
    # overhead (measured ~20x slower elsewhere in this project for small
    # per-op work). Capping it avoids that regardless of model size.
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("medium")  # allows TF32 matmuls on Ampere+ GPUs

    dm = SlotEventDataModule(
        config["train_path"], config["val_path"], features=config["features"],
        slot_target_features=config["slot_target_features"], batch_size=config["batch_size"],
    )

    model = build_model(config)
    if config["compile_model"]:
        model = torch.compile(model)
    # halve the learning rate when val_loss stops improving
    scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=config["lr_patience"],
    )
    if config["shape_metric"] is None:
        module = EventRegressionModule(
            model, learning_rate=config["learning_rate"], loss_kind="huber", scheduler_fn=scheduler_fn,
        )
    else:
        module = ShapeRegressionModule(
            model, shape_metric=config["shape_metric"], shape_weight=config["shape_weight"],
            learning_rate=config["learning_rate"], loss_kind="huber", scheduler_fn=scheduler_fn,
        )

    # stop once val_loss hasn't improved for early_stop_patience epochs, and
    # keep the best-val_loss checkpoint rather than the last one
    early_stop = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=config["early_stop_patience"])
    best_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

    logger = TensorBoardLogger(save_dir="tb_logs", name="slot_regression")
    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"], accelerator=config["accelerator"],
        logger=logger, precision="bf16-mixed", callbacks=[early_stop, best_checkpoint],
    )
    trainer.fit(module, datamodule=dm)

    # Save the z-score stats, max_jets, and model architecture alongside the
    # checkpoint: predict() needs all three to scale new data and rebuild
    # the model exactly the way this run did -- it can't guess or refit them.
    model_config = {key: config[key] for key in MODEL_KEYS}
    model_config["features"] = config["features"]
    model_config["slot_target_features"] = config["slot_target_features"]
    torch.save(
        {"stats": dm.stats, "max_jets": dm.train_dataset.max_jets, "model_config": model_config},
        f"{logger.log_dir}/stats.pt",
    )
    print(f"done training. checkpoint + stats saved under {logger.log_dir}")


def predict(config):
    torch.set_num_threads(4)  # see the comment in train()

    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    saved = torch.load(config["stats"], map_location="cpu")
    stats, max_jets, model_config = saved["stats"], saved["max_jets"], saved["model_config"]

    model = build_model(model_config)
    module = EventRegressionModule(model)
    # if the checkpoint was saved from a torch.compile(model)-wrapped run,
    # its keys are prefixed "_orig_mod." -- strip it so they match this
    # plain, uncompiled model's keys.
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in checkpoint["state_dict"].items()}
    module.load_state_dict(state_dict)

    slot_target_features = model_config["slot_target_features"]
    ds = SlotEventDataset(
        config["data_path"], features=model_config["features"], slot_target_features=slot_target_features,
        stats=stats, max_jets=max_jets,
    )
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False, collate_fn=_identity_collate)

    # Trainer.predict() puts the module in eval mode, disables gradients,
    # and moves both module and batches to the accelerator -- no manual
    # .eval()/.to(device)/no_grad()/per-batch transfer loop needed.
    trainer = pl.Trainer(accelerator=config["accelerator"], devices=1, logger=False)
    outputs = trainer.predict(module, dataloaders=loader)  # list of per-batch {"pred", "target"} dicts
    preds_scaled = torch.cat([out["pred"] for out in outputs])  # (N, 2, len(slot_target_features))
    targets_scaled = torch.cat([out["target"] for out in outputs])

    t_mean, t_std = stats["target"]  # (2, len(slot_target_features))
    t_mean, t_std = torch.as_tensor(t_mean), torch.as_tensor(t_std)
    preds = preds_scaled * (t_std + 1e-8) + t_mean
    targets = targets_scaled * (t_std + 1e-8) + t_mean

    mae = (preds - targets).abs().mean(dim=(0, 2))  # mean over events and features, per slot
    for i, slot in enumerate(SLOT_NAMES):
        print(f"{slot:10s} MAE: {mae[i]:.4f}")

    if config["shape_metric"] is not None:
        shape = SHAPE_LOSSES[config["shape_metric"]](preds_scaled, targets_scaled)
        print(f"{config['shape_metric']} shape distance over all slot/feature columns (z-scored): {shape:.4f}")

    # save predictions + targets to parquet, nested by slot -- e.g.
    # arr.predictions.top.px, arr.targets.antitop.E
    def slot_group(values):
        return ak.zip(
            {
                slot: ak.zip(
                    {feat: ak.Array(values[:, i, j].contiguous().numpy()) for j, feat in enumerate(slot_target_features)}
                )
                for i, slot in enumerate(SLOT_NAMES)
            },
            depth_limit=1,
        )

    out = ak.zip({"targets": slot_group(targets), "predictions": slot_group(preds)}, depth_limit=1)
    ak.to_parquet(out, config["output"])
    print(f"saved predictions to {config['output']}")


if __name__ == "__main__":
    if "--train" in sys.argv:
        train(load_config("train"))
    elif "--predict" in sys.argv:
        predict(load_config("predict"))
    else:
        print("usage: python -m hepsetreg.runs.run_slot_model --train | --predict --config <path.yaml>")
        sys.exit(1)
