"""
Train or run inference for the per-slot (top, antitop) regression task.

The model predicts each top quark's (px, py, pz, E). Slot 0 is always the
top and slot 1 is always the antitop (see EventDatasetSlot.py) -- a fixed,
physically meaningful order, not an interchangeable one -- so we use
CrossAttentionDecoder (fixed learned queries) and a plain per-slot loss,
not SlotAttention/matching.

Usage:
    python run_slot_model.py --train
    python run_slot_model.py --predict
"""

import logging
import sys

import awkward as ak
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from backbone import TransformerBackbone
from decoder import CrossAttentionDecoder
from EventDatasetSlot import SLOT_NAMES, SlotEventDataModule, SlotEventDataset, _identity_collate
from head import RegressionHead, SetEventRegressor
from lightning_module import EventRegressionModule

GLOBAL_FEATURES = [
    "global_njet", "global_nelectron", "global_nmuon",
    "global_nbtagged", "global_met_met", "global_met_phi",
]
JET_FEATURES = [
    "jet_pt", "jet_eta", "jet_phi", "jet_e", "jet_m",
    "jet_px", "jet_py", "jet_pz", "jet_bTag",
]
LEPTON_FEATURES = [
    "pt", "eta", "phi", "e", "m", "px", "py", "pz", "charge", "is_electron",
]
SLOT_TARGET_FEATURES = ["px", "py", "pz", "E"]
FEATURES = {"global": GLOBAL_FEATURES, "jet": JET_FEATURES, "lepton": LEPTON_FEATURES}

MODEL_KEYS = ["d_model", "n_enc_heads", "n_dec_heads", "num_enc_layers", "num_dec_layers", "n_queries"]


def build_model(config):
    """Builds the model architecture from a config dict (see MODEL_KEYS).
    train() and predict() both call this with the same model settings, so a
    saved checkpoint's weights always match the model they're loaded into."""
    backbone = TransformerBackbone(
        n_global_features=len(GLOBAL_FEATURES),
        n_jet_features=len(JET_FEATURES),
        n_lepton_features=len(LEPTON_FEATURES),
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
    head = RegressionHead(d_model=config["d_model"], output_dim=len(SLOT_TARGET_FEATURES))
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
        config["train_path"], config["val_path"], features=FEATURES,
        slot_target_features=SLOT_TARGET_FEATURES, batch_size=config["batch_size"],
    )

    model = build_model(config)
    if config["compile_model"]:
        model = torch.compile(model)
    # halve the learning rate when val_loss stops improving
    scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=config["lr_patience"],
    )
    module = EventRegressionModule(
        model, learning_rate=config["learning_rate"], loss_kind="huber", scheduler_fn=scheduler_fn,
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

    ds = SlotEventDataset(
        config["data_path"], features=FEATURES, slot_target_features=SLOT_TARGET_FEATURES,
        stats=stats, max_jets=max_jets,
    )
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False, collate_fn=_identity_collate)

    # Trainer.predict() puts the module in eval mode, disables gradients,
    # and moves both module and batches to the accelerator -- no manual
    # .eval()/.to(device)/no_grad()/per-batch transfer loop needed.
    trainer = pl.Trainer(accelerator=config["accelerator"], devices=1, logger=False)
    outputs = trainer.predict(module, dataloaders=loader)  # list of per-batch {"pred", "target"} dicts
    preds_scaled = torch.cat([out["pred"] for out in outputs])  # (N, 2, len(SLOT_TARGET_FEATURES))
    targets_scaled = torch.cat([out["target"] for out in outputs])

    t_mean, t_std = stats["target"]  # (2, len(SLOT_TARGET_FEATURES))
    t_mean, t_std = torch.as_tensor(t_mean), torch.as_tensor(t_std)
    preds = preds_scaled * (t_std + 1e-8) + t_mean
    targets = targets_scaled * (t_std + 1e-8) + t_mean

    mae = (preds - targets).abs().mean(dim=(0, 2))  # mean over events and features, per slot
    for i, slot in enumerate(SLOT_NAMES):
        print(f"{slot:10s} MAE: {mae[i]:.4f}")

    # save predictions + targets to parquet, nested by slot -- e.g.
    # arr.predictions.top.px, arr.targets.antitop.E
    def slot_group(values):
        return ak.zip(
            {
                slot: ak.zip(
                    {feat: ak.Array(values[:, i, j].contiguous().numpy()) for j, feat in enumerate(SLOT_TARGET_FEATURES)}
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
        train({
            # data
            "train_path": "ttbar-2L_train_220926_slot.parquet",
            "val_path": "ttbar-2L_val_220926_slot.parquet",
            "batch_size": 4096,
            # model architecture
            "d_model": 128,
            "n_enc_heads": 4,
            "n_dec_heads": 4,
            "num_enc_layers": 4,
            "num_dec_layers": 4,
            "n_queries": 2,
            # training
            "max_epochs": 40,
            "learning_rate": 1e-3,
            "lr_patience": 2,
            "early_stop_patience": 5,
            "accelerator": "gpu",
            "compile_model": True,
        })
    elif "--predict" in sys.argv:
        # fill in the actual run's version_N and checkpoint filename
        # (Lightning names it "epoch=X-step=Y.ckpt" by default)
        predict({
            "checkpoint": "tb_logs/slot_regression/version_0/checkpoints/epoch=X-step=Y.ckpt",
            "stats": "tb_logs/slot_regression/version_0/stats.pt",
            "data_path": "ttbar-2L_test_220926_slot.parquet",
            "batch_size": 4096,
            "accelerator": "gpu",
            "output": "ttbar-2L-slot-inference.parquet",
        })
    else:
        print("usage: python run_slot_model.py --train | --predict")
        sys.exit(1)
