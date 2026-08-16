"""These tests directly exercise the fix over DIRECTOR's fixed-slice design:
the encoder's output must not depend on (a) how many padding slots are
present, or (b) the order same-type objects (e.g. jets) appear in."""

import pytest
import torch

from hepsetreg.models.backbone import ObjectSetEncoder
from hepsetreg.models.tokenizer import ObjectGroupSpec


def _random_batch(batch_size=3, n_jets=6, n_leptons=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    jets = torch.randn(batch_size, n_jets, 4, generator=g)
    jets_mask = torch.zeros(batch_size, n_jets, dtype=torch.bool)
    n_real = torch.randint(1, n_jets + 1, (batch_size,), generator=g)
    for i, n in enumerate(n_real.tolist()):
        jets_mask[i, :n] = True
    leptons = torch.randn(batch_size, n_leptons, 3, generator=g)
    leptons_mask = torch.ones(batch_size, n_leptons, dtype=torch.bool)
    return {"jets": jets, "leptons": leptons}, {"jets": jets_mask, "leptons": leptons_mask}


@pytest.mark.parametrize("pooling", ["cls", "mean", "attention"])
def test_extra_padding_does_not_change_output(pooling):
    groups = [ObjectGroupSpec("jets", 4), ObjectGroupSpec("leptons", 3)]
    encoder = ObjectSetEncoder(groups, d_model=16, nhead=4, num_layers=2, pooling=pooling)
    encoder.eval()

    objects, mask = _random_batch(batch_size=3, n_jets=6)
    with torch.no_grad():
        pooled_a = encoder(objects, mask)

    # Widen the jets slot count with garbage-valued (not even zero) padding;
    # since the padding mask marks these as invalid, the output must be
    # numerically identical. This is exactly the scenario DIRECTOR's fixed
    # `feature_groups` slicing could not represent (a variable jet count).
    extra_pad = 5
    objects_wide = {
        "jets": torch.cat([objects["jets"], torch.randn(3, extra_pad, 4)], dim=1),
        "leptons": objects["leptons"],
    }
    mask_wide = {
        "jets": torch.cat([mask["jets"], torch.zeros(3, extra_pad, dtype=torch.bool)], dim=1),
        "leptons": mask["leptons"],
    }
    with torch.no_grad():
        pooled_b = encoder(objects_wide, mask_wide)

    torch.testing.assert_close(pooled_a, pooled_b, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("pooling", ["cls", "mean", "attention"])
def test_permutation_invariance_within_object_group(pooling):
    groups = [ObjectGroupSpec("jets", 4)]
    encoder = ObjectSetEncoder(groups, d_model=16, nhead=4, num_layers=2, pooling=pooling)
    encoder.eval()

    n_jets = 5
    jets = torch.randn(2, n_jets, 4)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])

    with torch.no_grad():
        out_a = encoder({"jets": jets}, {"jets": mask})

    perm = torch.randperm(n_jets)
    with torch.no_grad():
        out_b = encoder({"jets": jets[:, perm]}, {"jets": mask[:, perm]})

    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-4)


def test_d_model_must_be_divisible_by_nhead():
    groups = [ObjectGroupSpec("jets", 4)]
    with pytest.raises(ValueError):
        ObjectSetEncoder(groups, d_model=17, nhead=4)


def test_all_padding_event_raises():
    groups = [ObjectGroupSpec("jets", 4)]
    encoder = ObjectSetEncoder(groups, d_model=16, nhead=4, num_layers=1, pooling="mean")
    jets = torch.randn(1, 3, 4)
    mask = torch.zeros(1, 3, dtype=torch.bool)  # no real objects at all, no CLS token either
    with pytest.raises(ValueError):
        encoder({"jets": jets}, {"jets": mask})
