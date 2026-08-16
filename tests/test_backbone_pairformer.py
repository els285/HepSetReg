"""Mirrors test_backbone.py's invariance checks for PairformerBackbone, plus
shape/finiteness checks for the triangle-attention path and the optional
explicit pairwise-features input."""

import pytest
import torch

from hepsetreg.models.backbone_pairformer import PairformerBackbone
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
@pytest.mark.parametrize("use_triangle_attention", [False, True])
def test_extra_padding_does_not_change_output(pooling, use_triangle_attention):
    groups = [ObjectGroupSpec("jets", 4), ObjectGroupSpec("leptons", 3)]
    backbone = PairformerBackbone(
        groups,
        d_model=16,
        d_pair=8,
        nhead_single=4,
        nhead_pair=2,
        num_blocks=2,
        pooling=pooling,
        use_triangle_attention=use_triangle_attention,
    )
    backbone.eval()

    objects, mask = _random_batch(batch_size=3, n_jets=6)
    with torch.no_grad():
        pooled_a = backbone(objects, mask)

    extra_pad = 4
    objects_wide = {
        "jets": torch.cat([objects["jets"], torch.randn(3, extra_pad, 4)], dim=1),
        "leptons": objects["leptons"],
    }
    mask_wide = {
        "jets": torch.cat([mask["jets"], torch.zeros(3, extra_pad, dtype=torch.bool)], dim=1),
        "leptons": mask["leptons"],
    }
    with torch.no_grad():
        pooled_b = backbone(objects_wide, mask_wide)

    torch.testing.assert_close(pooled_a, pooled_b, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("pooling", ["cls", "mean", "attention"])
def test_permutation_invariance_within_object_group(pooling):
    groups = [ObjectGroupSpec("jets", 4)]
    backbone = PairformerBackbone(
        groups, d_model=16, d_pair=8, nhead_single=4, nhead_pair=2, num_blocks=2, pooling=pooling
    )
    backbone.eval()

    n_jets = 5
    jets = torch.randn(2, n_jets, 4)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])

    with torch.no_grad():
        out_a = backbone({"jets": jets}, {"jets": mask})

    perm = torch.randperm(n_jets)
    with torch.no_grad():
        out_b = backbone({"jets": jets[:, perm]}, {"jets": mask[:, perm]})

    torch.testing.assert_close(out_a, out_b, atol=1e-4, rtol=1e-3)


def test_pairwise_features_are_used():
    groups = [ObjectGroupSpec("jets", 4)]
    backbone = PairformerBackbone(
        groups,
        d_model=16,
        d_pair=8,
        nhead_single=4,
        num_blocks=2,
        pairwise_feature_dim=2,
        pooling="mean",
    )
    backbone.eval()

    batch_size, n_jets = 2, 4
    objects = {"jets": torch.randn(batch_size, n_jets, 4)}
    mask = {"jets": torch.ones(batch_size, n_jets, dtype=torch.bool)}

    pairwise_a = torch.zeros(batch_size, n_jets, n_jets, 2)
    pairwise_b = torch.randn(batch_size, n_jets, n_jets, 2)

    with torch.no_grad():
        out_a = backbone(objects, mask, pairwise_features=pairwise_a)
        out_b = backbone(objects, mask, pairwise_features=pairwise_b)

    assert not torch.allclose(out_a, out_b)


def test_pairwise_features_required_when_configured():
    groups = [ObjectGroupSpec("jets", 4)]
    backbone = PairformerBackbone(groups, d_model=16, d_pair=8, nhead_single=4, pairwise_feature_dim=3)
    objects = {"jets": torch.randn(2, 3, 4)}
    mask = {"jets": torch.ones(2, 3, dtype=torch.bool)}
    with pytest.raises(ValueError):
        backbone(objects, mask)


def test_all_padding_event_raises():
    groups = [ObjectGroupSpec("jets", 4)]
    backbone = PairformerBackbone(groups, d_model=16, d_pair=8, nhead_single=4, num_blocks=1, pooling="mean")
    jets = torch.randn(1, 3, 4)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError):
        backbone({"jets": jets}, {"jets": mask})


def test_output_is_finite_with_triangle_attention():
    groups = [ObjectGroupSpec("jets", 4), ObjectGroupSpec("met", 2)]
    backbone = PairformerBackbone(
        groups,
        d_model=16,
        d_pair=8,
        nhead_single=4,
        nhead_pair=2,
        num_blocks=2,
        use_triangle_attention=True,
        pooling="cls",
    )
    objects, mask = _random_batch(batch_size=4, n_jets=7)
    objects["met"] = torch.randn(4, 1, 2)
    mask["met"] = torch.ones(4, 1, dtype=torch.bool)

    pooled = backbone(objects, mask)
    assert torch.isfinite(pooled).all()
    assert pooled.shape == (4, 16)
