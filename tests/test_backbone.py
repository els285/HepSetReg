"""
Example test for TransformerBackbone/EventRegressor -- use this as a
template when you add a new backbone (e.g. a tokenized one): copy this
file, swap in your own model construction, and keep both checks below.

Deliberately loads one real batch through the actual dataset class
(EventDatasetBase.EventDataset) and the real config
(configs/run_regression.yaml), rather than inventing synthetic feature
counts/shapes by hand -- that way the test exercises exactly the dict
shape/dtypes your model will actually be called with, not a guess at it.
It's skipped automatically if the data hasn't been downloaded yet
(./setup_bsc does this), rather than failing.

Why these two checks specifically:
  - test_output_shape: catches the most common bug when wiring up a new
    backbone -- a wrong d_model/output_dim/feature count mismatch would
    either crash here or silently return the wrong shape.
  - test_permutation_invariance: the model is supposed to treat the list
    of jets (and leptons) as an unordered *set*, not a sequence -- the
    physical order jets happen to appear in the input has no meaning.
    Shuffling the jets (features + mask together) and checking the
    prediction doesn't change is the standard way to test that, and is
    exactly the property a tokenized backbone needs to preserve too.
"""

import os

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from hepsetreg.backbones.head import EventRegressor, RegressionHead
from hepsetreg.backbones.backbone import TransformerBackbone
from hepsetreg.datasets.EventDatasetBase import EventDataset, _identity_collate

CONFIG_PATH = "configs/run_regression.yaml"
BATCH_SIZE = 8


@pytest.fixture(scope="module")
def real_batch():
    """Loads the val parquet and returns (batch, features) ONCE, shared by
    both tests below -- EventDataset re-reads and re-z-scores the whole
    file on every construction, which is slow on this filesystem, so
    there's no reason to pay that cost twice for two tests on the same
    8 events."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    features = config["features"]
    val_path = config["train"]["val_path"]
    if not os.path.exists(val_path):
        pytest.skip(f"{val_path} not downloaded -- run ./setup_bsc first.")

    dataset = EventDataset(val_path, features=features)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_identity_collate)
    batch = next(iter(loader))
    return batch, features


def _build_model(features):
    torch.manual_seed(0)
    backbone = TransformerBackbone(
        n_global_features=len(features["global"]), n_jet_features=len(features["jet"]),
        n_lepton_features=len(features["lepton"]), d_model=32, nhead=4, num_layers=2,
    )
    head = RegressionHead(d_model=32, output_dim=len(features["target"]), n_layers=2, start_neurons=32)
    model = EventRegressor(backbone, head)
    model.eval()  # turn off dropout -- without this, two forward passes on
    # the exact same input wouldn't even agree with each other, let alone
    # a permuted one.
    return model


def test_output_shape(real_batch):
    batch, features = real_batch
    model = _build_model(features)
    with torch.no_grad():
        pred = model(batch)
    assert pred.shape == (batch["global"].shape[0], len(features["target"]))


def test_permutation_invariance(real_batch):
    batch, features = real_batch
    model = _build_model(features)

    # Same permutation applied to the jet features AND the jet mask -- this
    # is the same reordering a different (but physically identical) jet
    # collection would arrive in.
    perm = torch.randperm(batch["jet"].shape[1])
    permuted_batch = dict(batch)
    permuted_batch["jet"] = batch["jet"][:, perm]
    permuted_batch["jet_mask"] = batch["jet_mask"][:, perm]

    with torch.no_grad():
        pred = model(batch)
        permuted_pred = model(permuted_batch)

    assert torch.allclose(pred, permuted_pred, atol=1e-5)
