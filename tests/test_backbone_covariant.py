"""The property that matters for CovariantParticleTransformer: applying a
longitudinal boost + azimuthal rotation to every object in an event (the
residual symmetry of hadron-collider final states) must leave the pooled
output exactly unchanged, since it's built entirely from invariant
quantities. That's checked directly, not just inferred from the code."""

import pytest
import torch

from hepsetreg.models.backbone_covariant import (
    CovariantParticleTransformer,
    KinematicGroupSpec,
    apply_beam_symmetry_transform,
)


def _random_four_vectors(batch_size, n_obj, seed=0, min_mass=0.0):
    g = torch.Generator().manual_seed(seed)
    pt = torch.rand(batch_size, n_obj, generator=g) * 100.0 + 5.0
    eta = torch.randn(batch_size, n_obj, generator=g) * 1.5
    phi = (torch.rand(batch_size, n_obj, generator=g) * 2 - 1) * torch.pi
    mass = torch.rand(batch_size, n_obj, generator=g) * 5.0 + min_mass

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    e = torch.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return torch.stack([px, py, pz, e], dim=-1)


def _random_batch(batch_size=4, n_jets=6, n_leptons=1, seed=0):
    jets = _random_four_vectors(batch_size, n_jets, seed=seed)
    jets_mask = torch.zeros(batch_size, n_jets, dtype=torch.bool)
    g = torch.Generator().manual_seed(seed + 1)
    n_real = torch.randint(1, n_jets + 1, (batch_size,), generator=g)
    for i, n in enumerate(n_real.tolist()):
        jets_mask[i, :n] = True

    leptons = _random_four_vectors(batch_size, n_leptons, seed=seed + 2, min_mass=0.0)
    leptons_mask = torch.ones(batch_size, n_leptons, dtype=torch.bool)

    return {"jets": jets, "leptons": leptons}, {"jets": jets_mask, "leptons": leptons_mask}


def _build_backbone(pooling="mean", extra_features=0):
    groups = [KinematicGroupSpec("jets", extra_features=extra_features), KinematicGroupSpec("leptons")]
    backbone = CovariantParticleTransformer(groups, d_model=16, nhead=4, num_blocks=2, pooling=pooling)
    backbone.eval()
    return backbone


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_boost_and_rotation_invariance(pooling):
    backbone = _build_backbone(pooling=pooling)
    four_vectors, mask = _random_batch(batch_size=4, n_jets=6)

    with torch.no_grad():
        pooled_a = backbone(four_vectors, mask)

    # One random longitudinal boost + azimuthal rotation *per event*, applied
    # identically to every object in that event -- exactly the transform two
    # different (equally valid) lab-frame views of the same collision differ by.
    g = torch.Generator().manual_seed(123)
    delta_rapidity = (torch.rand(4, 1, generator=g) * 4 - 2)  # per-event, broadcasts over objects
    delta_phi = (torch.rand(4, 1, generator=g) * 2 - 1) * torch.pi

    transformed = {
        name: apply_beam_symmetry_transform(fv, delta_rapidity, delta_phi) for name, fv in four_vectors.items()
    }
    with torch.no_grad():
        pooled_b = backbone(transformed, mask)

    torch.testing.assert_close(pooled_a, pooled_b, atol=1e-4, rtol=1e-3)


def test_boost_and_rotation_invariance_with_extra_features():
    backbone = _build_backbone(pooling="attention", extra_features=2)
    four_vectors, mask = _random_batch(batch_size=3, n_jets=5)
    extra = {"jets": torch.randn(3, 5, 2)}

    with torch.no_grad():
        pooled_a = backbone(four_vectors, mask, extra_features=extra)

    g = torch.Generator().manual_seed(7)
    delta_rapidity = torch.rand(3, 1, generator=g) * 3 - 1.5
    delta_phi = torch.rand(3, 1, generator=g) * 2 * torch.pi
    transformed = {name: apply_beam_symmetry_transform(fv, delta_rapidity, delta_phi) for name, fv in four_vectors.items()}

    with torch.no_grad():
        pooled_b = backbone(transformed, mask, extra_features=extra)

    torch.testing.assert_close(pooled_a, pooled_b, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_extra_padding_does_not_change_output(pooling):
    backbone = _build_backbone(pooling=pooling)
    four_vectors, mask = _random_batch(batch_size=3, n_jets=6)
    with torch.no_grad():
        pooled_a = backbone(four_vectors, mask)

    extra_pad = 4
    wide_jets = torch.cat([four_vectors["jets"], torch.randn(3, extra_pad, 4) * 10.0], dim=1)
    wide_mask = torch.cat([mask["jets"], torch.zeros(3, extra_pad, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        pooled_b = backbone({"jets": wide_jets, "leptons": four_vectors["leptons"]}, {"jets": wide_mask, "leptons": mask["leptons"]})

    torch.testing.assert_close(pooled_a, pooled_b, atol=1e-4, rtol=1e-3)


def test_permutation_invariance_within_object_group():
    groups = [KinematicGroupSpec("jets")]
    backbone = CovariantParticleTransformer(groups, d_model=16, nhead=4, num_blocks=2, pooling="mean")
    backbone.eval()

    n_jets = 5
    jets = _random_four_vectors(2, n_jets, seed=1)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])

    with torch.no_grad():
        out_a = backbone({"jets": jets}, {"jets": mask})

    perm = torch.randperm(n_jets)
    with torch.no_grad():
        out_b = backbone({"jets": jets[:, perm]}, {"jets": mask[:, perm]})

    torch.testing.assert_close(out_a, out_b, atol=1e-4, rtol=1e-3)


def test_cls_pooling_is_rejected():
    with pytest.raises(ValueError):
        CovariantParticleTransformer([KinematicGroupSpec("jets")], d_model=16, nhead=4, pooling="cls")


def test_all_padding_event_raises():
    backbone = _build_backbone()
    jets = _random_four_vectors(1, 3, seed=2)
    leptons = _random_four_vectors(1, 1, seed=3)
    mask = {"jets": torch.zeros(1, 3, dtype=torch.bool), "leptons": torch.zeros(1, 1, dtype=torch.bool)}
    with pytest.raises(ValueError):
        backbone({"jets": jets, "leptons": leptons}, mask)


def test_missing_extra_features_raises():
    backbone = _build_backbone(extra_features=2)
    four_vectors, mask = _random_batch(batch_size=2, n_jets=3)
    with pytest.raises(ValueError):
        backbone(four_vectors, mask)  # extra_features configured but not passed
