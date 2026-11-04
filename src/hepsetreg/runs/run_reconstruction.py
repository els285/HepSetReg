"""
Train or run event reconstruction heads on the TransformerBackbone: which
measured objects belong to each configured truth role (`assignment:` in the
config, see backbones/assignment.py).

Usage:
    python -m hepsetreg.runs.run_reconstruction --train --config configs/run_reconstruction.yaml
    python -m hepsetreg.runs.run_reconstruction --predict --config configs/run_reconstruction.yaml
"""

import sys

from hepsetreg.config import check_args

if __name__ == "__main__":
    check_args("usage: python -m hepsetreg.runs.run_reconstruction --train | --predict --config <path.yaml>")

import awkward as ak
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from hepsetreg.backbones.assignment import AssignmentHead, AssignmentTargets
from hepsetreg.backbones.backbone import TransformerBackbone
from hepsetreg.config import load_config
from hepsetreg.datasets.EventDatasetBase import _identity_collate
from hepsetreg.datasets.ReconstructionDataset import ReconstructionDataset
from hepsetreg.training.reconstruction_module import ReconstructionModule

MODEL_KEYS = ["features", "d_model", "nhead", "num_layers", "assignment"]


def build_model(config):
    features = config["features"]
    backbone = TransformerBackbone(
        n_global_features=len(features["global"]),
        n_jet_features=len(features["jet"]),
        n_lepton_features=len(features["lepton"]),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
    )
    assignment = AssignmentHead(config["d_model"], AssignmentTargets(config["assignment"]))
    return ReconstructionModule(backbone, assignment, learning_rate=config.get("learning_rate", 1e-3))


def train(config):
    torch.set_num_threads(4)  # see run_regression_flow.train()
    targets = AssignmentTargets(config["assignment"])
    train_data = ReconstructionDataset(config["train_path"], features=config["features"], assignment=targets,
                                       max_jets=config.get("max_jets"))
    val_data = ReconstructionDataset(config["val_path"], features=config["features"], assignment=targets,
                                     stats=train_data.stats, max_jets=train_data.max_jets)
    loader = dict(batch_size=config["batch_size"], collate_fn=_identity_collate)

    module = build_model(config)
    logger = TensorBoardLogger(save_dir="tb_logs", name="event_reconstruction")
    best_checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(max_epochs=config["max_epochs"], accelerator=config["accelerator"],
                         logger=logger, callbacks=[best_checkpoint])
    trainer.fit(module, train_dataloaders=DataLoader(train_data, shuffle=True, **loader),
                val_dataloaders=DataLoader(val_data, **loader))

    model_config = {key: config[key] for key in MODEL_KEYS if key in config}
    torch.save({"stats": train_data.stats, "max_jets": train_data.max_jets, "model_config": model_config},
               f"{logger.log_dir}/stats.pt")
    print(f"done training. checkpoint + stats saved under {logger.log_dir}")


def predict(config):
    """Writes predictions.<target>: the selected object index of each of the
    target's roles (within its own collection; -1 if none was available),
    and predictions.<target>_prob: those roles' softmax probabilities."""
    torch.set_num_threads(4)
    saved = torch.load(config["stats"], map_location="cpu", weights_only=False)
    model_config = saved["model_config"]
    module = build_model(model_config)
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint["state_dict"])

    # Measured inputs only; no jet is dropped (max_jets=None), since every
    # real jet is a candidate for the assignment.
    data = ReconstructionDataset(config["data_path"], features=model_config["features"], stats=saved["stats"])
    loader = DataLoader(data, batch_size=config["batch_size"], shuffle=False, collate_fn=_identity_collate)
    trainer = pl.Trainer(accelerator=config["accelerator"], devices=1, logger=False)
    outputs = trainer.predict(module, dataloaders=loader)

    index = torch.cat([out["assignment"] for out in outputs]).numpy()
    prob = torch.cat([out["assignment_prob"] for out in outputs]).numpy()
    fields = {}
    for name, roles in module.assignment.targets.targets.items():
        fields[name] = ak.Array(index[:, roles])
        fields[f"{name}_prob"] = ak.Array(prob[:, roles])
    ak.to_parquet(ak.zip({"predictions": ak.zip(fields, depth_limit=1)}, depth_limit=1), config["output"])
    print(f"saved predictions to {config['output']}")


if __name__ == "__main__":
    if "--train" in sys.argv:
        train(load_config("train"))
    elif "--predict" in sys.argv:
        predict(load_config("predict"))
