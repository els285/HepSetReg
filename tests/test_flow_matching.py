import torch

from hepsetreg.models.flow_matching import ConditionalVelocityField, sample_flow
from hepsetreg.models.tokenizer import ObjectGroupSpec


def test_velocity_field_output_shape():
    groups = [ObjectGroupSpec("jets", 4), ObjectGroupSpec("met", 2)]
    field = ConditionalVelocityField(groups, output_dim=3, d_model=16, nhead=4, num_layers=2)
    field.eval()

    batch_size, n_jets = 5, 4
    objects = {"jets": torch.randn(batch_size, n_jets, 4), "met": torch.randn(batch_size, 1, 2)}
    mask = {
        "jets": torch.ones(batch_size, n_jets, dtype=torch.bool),
        "met": torch.ones(batch_size, 1, dtype=torch.bool),
    }
    xt = torch.randn(batch_size, 3)
    t = torch.rand(batch_size, 1)

    v = field(xt, t, objects, mask)
    assert v.shape == (batch_size, 3)


def test_velocity_field_handles_variable_jet_counts():
    groups = [ObjectGroupSpec("jets", 4)]
    field = ConditionalVelocityField(groups, output_dim=2, d_model=8, nhead=2, num_layers=1)
    field.eval()

    objects = {"jets": torch.randn(2, 5, 4)}
    mask = {"jets": torch.tensor([[True, True, False, False, False], [True, True, True, True, True]])}
    xt = torch.randn(2, 2)
    t = torch.rand(2, 1)

    v = field(xt, t, objects, mask)
    assert v.shape == (2, 2)
    assert torch.isfinite(v).all()


def test_sample_flow_shape():
    groups = [ObjectGroupSpec("jets", 4)]
    field = ConditionalVelocityField(groups, output_dim=2, d_model=8, nhead=2, num_layers=1)
    field.eval()

    batch_size, n_jets = 3, 4
    objects = {"jets": torch.randn(batch_size, n_jets, 4)}
    mask = {"jets": torch.ones(batch_size, n_jets, dtype=torch.bool)}

    samples = sample_flow(field, objects, mask, n_steps=4, n_samples=6)
    assert samples.shape == (batch_size, 6, 2)
    assert torch.isfinite(samples).all()
