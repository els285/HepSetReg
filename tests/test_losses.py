import pytest
import torch

from hepsetreg.losses import (
    CompositeLoss,
    ConditionalFlowMatchingLoss,
    HistogramKLDivergenceLoss,
    LossTermConfig,
    PhysicsConsistencyLoss,
    RegressionLoss,
)


@pytest.mark.parametrize("kind", ["huber", "mse", "mae"])
def test_regression_loss_kinds_are_nonnegative(kind):
    pred = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.5, 2.5]])
    value = RegressionLoss(kind=kind)(pred, target)
    assert value.item() >= 0


def test_regression_loss_zero_when_exact():
    pred = torch.randn(8, 3)
    value = RegressionLoss(kind="mse")(pred, pred.clone())
    assert value.item() == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_near_zero_for_identical_distributions():
    torch.manual_seed(0)
    x = torch.randn(512, 2)
    loss_fn = HistogramKLDivergenceLoss(bins=64, sigma=0.3)
    value = loss_fn(x, x.clone())
    assert value.item() < 1e-4


def test_kl_divergence_positive_for_different_distributions():
    torch.manual_seed(0)
    x = torch.randn(512, 1)
    y = torch.randn(512, 1) + 3.0  # shifted distribution
    loss_fn = HistogramKLDivergenceLoss(bins=64, sigma=0.3, hist_min=-6.0, hist_max=6.0)
    value = loss_fn(x, y)
    assert value.item() > 0.1


def test_conditional_flow_matching_loss_is_mse():
    loss_fn = ConditionalFlowMatchingLoss()
    v_pred = torch.zeros(4, 2)
    v_target = torch.ones(4, 2)
    value = loss_fn(v_pred, v_target)
    assert value.item() == pytest.approx(1.0)


def test_physics_consistency_loss_with_dict_output():
    def fn(y_pred, batch):
        return {"a": (y_pred - batch["truth"]).pow(2).mean(), "b": y_pred.mean()}

    loss_fn = PhysicsConsistencyLoss(fn)
    y_pred = torch.ones(4, 1)
    batch = {"truth": torch.zeros(4, 1)}
    total, sub_terms = loss_fn(y_pred, batch)
    assert set(sub_terms) == {"a", "b"}
    assert torch.isclose(total, sub_terms["a"] + sub_terms["b"])


def test_physics_consistency_loss_with_scalar_output():
    def fn(y_pred, batch):
        return y_pred.mean()

    loss_fn = PhysicsConsistencyLoss(fn)
    total, sub_terms = loss_fn(torch.ones(4, 1), {})
    assert list(sub_terms) == ["consistency"]
    assert torch.isclose(total, torch.tensor(1.0))


def test_composite_loss_ramps_weight_linearly():
    terms = {"regression": RegressionLoss(kind="mse")}
    configs = {"regression": LossTermConfig(weight=2.0, ramp_epochs=4)}
    composite = CompositeLoss(terms, configs)

    pred, target = torch.zeros(4, 1), torch.ones(4, 1)
    inputs = {"regression": {"pred": pred, "target": target}}

    composite.set_epoch(0)  # (0+1)/4 = 0.25 -> weight 0.5
    _, logs0 = composite(inputs)
    composite.set_epoch(3)  # (3+1)/4 = 1.0 -> weight 2.0 (capped)
    _, logs3 = composite(inputs)
    composite.set_epoch(10)  # still capped at 2.0
    _, logs10 = composite(inputs)

    assert logs0["regression_weight"] == pytest.approx(0.5)
    assert logs3["regression_weight"] == pytest.approx(2.0)
    assert logs10["regression_weight"] == pytest.approx(2.0)


def test_composite_loss_skips_terms_missing_from_inputs():
    terms = {
        "regression": RegressionLoss(kind="mse"),
        "distribution_kl": HistogramKLDivergenceLoss(),
    }
    composite = CompositeLoss(terms)

    pred, target = torch.zeros(4, 1), torch.ones(4, 1)
    total, logs = composite({"regression": {"pred": pred, "target": target}})

    assert "distribution_kl_loss" not in logs
    assert "regression_loss" in logs


def test_composite_loss_requires_at_least_one_matching_term():
    composite = CompositeLoss({"regression": RegressionLoss(kind="mse")})
    with pytest.raises(ValueError):
        composite({"distribution_kl": {"pred": torch.zeros(2, 1), "target": torch.zeros(2, 1)}})
