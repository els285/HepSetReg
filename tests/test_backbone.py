"""
Example test for TransformerBackbone/EventRegressor -- use this as a
template when you add a new backbone (e.g. a tokenized one): copy this
file, swap in your own model construction, and keep both checks below.

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

import torch

from hepsetreg.backbones.backbone import TransformerBackbone
from hepsetreg.backbones.head import EventRegressor, RegressionHead

N_GLOBAL, N_JET, N_LEPTON, N_TARGET = 6, 9, 10, 1


def _build_model():
    torch.manual_seed(0)
    backbone = TransformerBackbone(
        n_global_features=N_GLOBAL, n_jet_features=N_JET, n_lepton_features=N_LEPTON,
        d_model=32, nhead=4, num_layers=2,
    )
    head = RegressionHead(d_model=32, output_dim=N_TARGET, n_layers=2, start_neurons=32)
    model = EventRegressor(backbone, head)
    model.eval()  # turn off dropout -- without this, two forward passes on
    # the exact same input wouldn't even agree with each other, let alone
    # a permuted one.
    return model


def _random_batch(batch_size=4, n_jets=5, n_leptons=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    # A real event has a variable number of jets -- pad up to n_jets and
    # mark the last two slots as padding (jet_mask=False) to exercise that
    # path, instead of only testing the all-real-jets case.
    jet_mask = torch.ones(batch_size, n_jets, dtype=torch.bool)
    jet_mask[:, -2:] = False
    return {
        "global": torch.randn(batch_size, N_GLOBAL, generator=g),
        "jet": torch.randn(batch_size, n_jets, N_JET, generator=g),
        "jet_mask": jet_mask,
        "lepton": torch.randn(batch_size, n_leptons, N_LEPTON, generator=g),
        "lepton_mask": torch.ones(batch_size, n_leptons, dtype=torch.bool),
    }


def test_output_shape():
    model = _build_model()
    batch = _random_batch()
    with torch.no_grad():
        pred = model(batch)
    assert pred.shape == (batch["global"].shape[0], N_TARGET)


def test_permutation_invariance():
    model = _build_model()
    batch = _random_batch()

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
