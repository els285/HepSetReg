"""
Train or run inference for the system-level (ttbar_mass/pt/DR) regression
task using conditional flow matching (flow_matching.py) instead of a direct
point-estimate regression head: same TransformerBackbone as run_regression.py,
but the "head" is a velocity field trained to match a noise-to-target
interpolation, and inference draws samples via ODE integration (sample_flow)
instead of one forward pass. `TransformerVelocityField` is used here; the
other two architectures in flow_matching.py (DNNVelocityField,
DiTVelocityField) are drop-in alternatives -- swap the import and the
constructor call in build_model().

Usage:
    python -m hepsetreg.runs.run_regression_flow --train --config configs/run_regression_flow.yaml
    python -m hepsetreg.runs.run_regression_flow --predict --config configs/run_regression_flow.yaml
"""

import sys

from hepsetreg.config import check_args

# Checked here, before the heavy imports below, so a bad command line
# (missing --train/--predict or --config) fails in milliseconds rather than
# only being reported after paying the full torch/pytorch_lightning/... import cost.
if __name__ == "__main__":
    check_args("usage: python -m hepsetreg.runs.run_regression_flow --train | --predict --config <path.yaml>")

import logging

import awkward as ak
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from hepsetreg.backbones.backbone import TransformerBackbone
from hepsetreg.backbones.flow_matching import FlowRegressor, TransformerVelocityField
from hepsetreg.config import load_config
from hepsetreg.datasets.EventDatasetBase import EventDataModule, EventDataset, _identity_collate
from hepsetreg.losses.shape_losses import SHAPE_LOSSES
from hepsetreg.training.lightning_module import FlowMatchingModule

# backbone and velocity field share one set of size knobs -- simplest config,
# and a reasonable default since they're both transformer encoders over a
# similarly-sized problem.
MODEL_KEYS = ["d_model", "nhead", "num_layers"]


def build_model(config):
    """Builds the backbone + velocity-field model from a config dict (see
    MODEL_KEYS and config["features"]). train() and predict() both call this
    with the same model settings, so a saved checkpoint's weights always
    match the model they're loaded into."""
    features = config["features"]
    backbone = TransformerBackbone(
        n_global_features=len(features["global"]),
        n_jet_features=len(features["jet"]),
        n_lepton_features=len(features["lepton"]),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
    )
    velocity_field = TransformerVelocityField(
        output_dim=len(features["target"]),
        cond_dim=config["d_model"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
    )
    return FlowRegressor(backbone, velocity_field)


def train(config):
    # Not a hyperparameter -- an environment setting. PyTorch defaults to
    # using every visible CPU core for every op, even tiny ones; on a
    # many-core shared machine that causes severe thread-oversubscription
    # overhead (measured ~20x slower elsewhere in this project for small
    # per-op work). Capping it avoids that regardless of model size.
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("medium")  # allows TF32 matmuls on Ampere+ GPUs

    dm = EventDataModule(
        config["train_path"], config["val_path"], features=config["features"], batch_size=config["batch_size"],
    )

    model = build_model(config)
    if config["compile_model"]:
        # FlowMatchingModule's training/validation steps call
        # model.backbone(...) and model.velocity_field(...) directly, never
        # model.forward() (that's reserved for inference-time sampling) --
        # so compiling the two submodules that are actually called during
        # training, rather than the wrapper's forward, is what actually
        # gets torch.compile's speedup here.
        model.backbone = torch.compile(model.backbone)
        model.velocity_field = torch.compile(model.velocity_field)
    # halve the learning rate when val_loss stops improving
    scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=config["lr_patience"],
    )
    # shape_metric: null (default, plain flow matching) or one of
    # SHAPE_LOSSES's keys (emd, mmd, kl, mmd_joint, sliced_emd) to add a
    # shape-matching term on top of the velocity loss -- see
    # FlowMatchingModule's docstring for the extra sampling cost this adds
    # to every training step, controlled by shape_n_steps/shape_n_samples.
    module = FlowMatchingModule(
        model, learning_rate=config["learning_rate"], scheduler_fn=scheduler_fn,
        shape_metric=config.get("shape_metric"), shape_weight=config.get("shape_weight", 0.1),
        shape_n_steps=config.get("shape_n_steps", 10), shape_n_samples=config.get("shape_n_samples", 2),
    )

    # stop once val_loss hasn't improved for early_stop_patience epochs, and
    # keep the best-val_loss checkpoint rather than the last one
    early_stop = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=config["early_stop_patience"])
    best_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

    logger = TensorBoardLogger(save_dir="tb_logs", name="event_regression_flow")
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

    # keep_all_samples: false (default) saves the sample-mean point estimate,
    # one row per event, same as a direct-regression predict() output. true
    # saves every individual ODE-sampled draw instead (shape (n_samples,)
    # per event, per target feature) -- e.g. for calibration or looking at
    # the predicted distribution's spread, not just its mean.
    keep_all_samples = config.get("keep_all_samples", False)
    model = build_model(model_config)
    module = FlowMatchingModule(
        model, predict_n_steps=config["n_steps"], predict_n_samples=config["n_samples"],
        keep_all_samples=keep_all_samples,
    )
    # if the checkpoint was saved from a torch.compile(...)-wrapped
    # submodule (train.py's compile_model=True compiles model.backbone and
    # model.velocity_field individually), those parameter keys are prefixed
    # "_orig_mod." -- strip it so they match this plain, uncompiled model's
    # keys, regardless of which submodule the prefix landed on.
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in checkpoint["state_dict"].items()}
    module.load_state_dict(state_dict)

    target_features = model_config["features"]["target"]
    ds = EventDataset(config["data_path"], features=model_config["features"], stats=stats, max_jets=max_jets)
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False, collate_fn=_identity_collate)

    # Trainer.predict() puts the module in eval mode, disables gradients,
    # and moves both module and batches to the accelerator -- no manual
    # .eval()/.to(device)/no_grad()/per-batch transfer loop needed.
    trainer = pl.Trainer(accelerator=config["accelerator"], devices=1, logger=False)
    outputs = trainer.predict(module, dataloaders=loader)  # list of per-batch {"pred", "target"} dicts
    preds_scaled = torch.cat([out["pred"] for out in outputs])
    targets_scaled = torch.cat([out["target"] for out in outputs])

    t_mean, t_std = stats["target"]
    t_mean, t_std = torch.as_tensor(t_mean), torch.as_tensor(t_std)
    preds = preds_scaled * (t_std + 1e-8) + t_mean
    targets = targets_scaled * (t_std + 1e-8) + t_mean

    # preds is (N, F) normally, or (N, n_samples, F) if keep_all_samples --
    # MAE is always reported against the sample mean, so this printout means
    # the same thing (and is comparable across runs) either way.
    point_pred = preds.mean(dim=1) if keep_all_samples else preds
    mae = (point_pred - targets).abs().mean(dim=0)  # mean over events, per target feature
    for name, value in zip(target_features, mae):
        print(f"{name:12s} MAE: {value:.4f}")

    if config.get("shape_metric") is not None:
        point_pred_scaled = preds_scaled.mean(dim=1) if keep_all_samples else preds_scaled
        shape_loss_fn = SHAPE_LOSSES[config["shape_metric"]]
        for i, name in enumerate(target_features):
            shape = shape_loss_fn(point_pred_scaled[:, i:i + 1], targets_scaled[:, i:i + 1])
            print(f"{name:12s} {config['shape_metric']} shape distance (z-scored): {shape:.4f}")

    # save predictions + targets to parquet, as two record groups -- e.g.
    # arr.targets.ttbar_mass, arr.predictions.ttbar_mass (or, with
    # keep_all_samples, arr.predictions.ttbar_mass is a list of n_samples
    # draws per event instead of a single value).
    # .contiguous(): a column slice of a 2D/3D tensor is a strided,
    # non-contiguous view, which pyarrow rejects with "ndarray is not contiguous"
    target_fields = {name: ak.Array(targets[:, i].contiguous().numpy()) for i, name in enumerate(target_features)}
    if keep_all_samples:
        pred_fields = {name: ak.Array(preds[:, :, i].contiguous().numpy()) for i, name in enumerate(target_features)}
    else:
        pred_fields = {name: ak.Array(preds[:, i].contiguous().numpy()) for i, name in enumerate(target_features)}
    out = ak.zip({"targets": ak.zip(target_fields), "predictions": ak.zip(pred_fields)}, depth_limit=1)
    ak.to_parquet(out, config["output"])
    print(f"saved predictions to {config['output']}")


if __name__ == "__main__":
    if "--train" in sys.argv:
        train(load_config("train"))
    elif "--predict" in sys.argv:
        predict(load_config("predict"))
    else:
        print("usage: python -m hepsetreg.runs.run_regression_flow --train | --predict --config <path.yaml>")
        sys.exit(1)
