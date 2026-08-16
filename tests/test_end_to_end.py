"""Full config -> datamodule -> LightningModule -> Trainer smoke tests, for
both training modes, run with Lightning's ``fast_dev_run`` (a couple of
batches, no checkpointing) so they stay fast while still exercising the
whole path the CLI (``hepsetreg-train``) takes."""

import sys
from pathlib import Path

import lightning as L
import numpy as np
import pytest
from omegaconf import OmegaConf

from hepsetreg.data.preprocessing import pad_and_mask, write_padded_hdf5, write_scaler_hdf5
from hepsetreg.factory import build_datamodule, build_lightning_module


def _write_toy_dataset(out_dir: Path, n_events=32, seed=0):
    rng = np.random.default_rng(seed)
    events = [rng.normal(size=(int(rng.integers(2, 6)), 5)).astype(np.float32) for _ in range(n_events)]
    jets_features, jets_mask = pad_and_mask(events, max_count=6, feature_dim=5)
    leptons = rng.normal(size=(n_events, 1, 4)).astype(np.float32)
    met = rng.normal(size=(n_events, 1, 2)).astype(np.float32)
    target = rng.normal(size=(n_events, 2)).astype(np.float32)
    truth_mass = rng.normal(size=(n_events, 1)).astype(np.float32)

    write_padded_hdf5(
        out_dir / "data.h5",
        {
            "jets": (jets_features, jets_mask),
            "leptons": (leptons, np.ones((n_events, 1), dtype=bool)),
            "met": (met, np.ones((n_events, 1), dtype=bool)),
        },
        target,
        extras={"truth_mass": truth_mass},
    )
    write_scaler_hdf5(out_dir / "scaler.h5", target.mean(axis=0), target.std(axis=0) + 1e-3)


def _base_cfg(out_dir: Path, mode: str):
    data_file = str(out_dir / "data.h5")
    return OmegaConf.create(
        {
            "seed": 0,
            "model": {
                "mode": mode,
                "d_model": 16,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
                "pooling": "cls",
                "use_type_embedding": True,
                "output_dim": 2,
                "groups": [
                    {"name": "jets", "in_features": 5},
                    {"name": "leptons", "in_features": 4},
                    {"name": "met", "in_features": 2},
                ],
                "head": {"n_layers": 1, "start_neurons": 16, "dropout": 0.0},
            },
            "loss": {
                "terms": {
                    ("regression" if mode == "supervised" else "flow_matching"): {"weight": 1.0},
                    "consistency": {"fn": "physics_hooks_fixture:toy_consistency", "weight": 0.1},
                }
            },
            "data": {
                "group_names": ["jets", "leptons", "met"],
                "train_file": data_file,
                "val_file": data_file,
                "target_key": "target",
                "extra_keys": ["truth_mass"],
                "scaler_file": str(out_dir / "scaler.h5"),
                "batch_size": 8,
                "num_workers": 0,
            },
            "train": {
                "optimizer": "adamw",
                "learning_rate": 1.0e-3,
                "max_epochs": 1,
            },
            "sampling": {"n_steps": 2, "n_samples": 2, "n_steps_training": 2, "n_samples_training": 2},
        }
    )


@pytest.fixture()
def toy_dir(tmp_path):
    physics_hooks_src = tmp_path / "physics_hooks_fixture.py"
    physics_hooks_src.write_text(
        "import torch\n"
        "def toy_consistency(y_pred, batch):\n"
        "    return torch.nn.functional.mse_loss(y_pred[:, 0], batch['truth_mass'].squeeze(-1))\n"
    )
    sys.path.insert(0, str(tmp_path))
    _write_toy_dataset(tmp_path)
    yield tmp_path
    sys.path.remove(str(tmp_path))


@pytest.mark.parametrize("mode", ["supervised", "flow_matching"])
def test_fast_dev_run_does_not_crash(toy_dir, mode):
    cfg = _base_cfg(toy_dir, mode)

    datamodule = build_datamodule(cfg)
    model = build_lightning_module(cfg)

    trainer = L.Trainer(fast_dev_run=2, accelerator="cpu", logger=False, enable_checkpointing=False)
    trainer.fit(model, datamodule=datamodule)
