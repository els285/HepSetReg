"""Numerical and gradient checks for the fixed-group projection objective."""

import json
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import torch

from hepsetreg.losses.conditional_ot import GroupDefinition, assign_groups, conditional_ot_loss, fit_groups, make_directions


class ConditionalOTTests(unittest.TestCase):
    def test_reference_json_roundtrip_and_fixed_edges(self):
        reference = fit_groups(np.arange(100, dtype=np.float32))
        restored = GroupDefinition.from_dict(json.loads(json.dumps(reference.to_dict())))
        self.assertEqual(reference, restored)
        self.assertEqual(reference.counts, (60, 30, 10))
        edges = torch.tensor(restored.internal_edges)
        self.assertEqual(assign_groups(edges, restored).tolist(), [1, 2])
        before = restored.to_dict()
        self.assertEqual(assign_groups([-100.0, 1000.0], restored).tolist(), [0, 2])
        self.assertEqual(restored.to_dict(), before)

    def test_tied_training_quantiles_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_groups(np.ones(100))
        with self.assertRaises(ValueError):
            fit_groups(np.arange(10), quantiles=[np.nan])

    def test_reference_cannot_be_relabelled_as_validation(self):
        state = fit_groups(np.arange(100)).to_dict()
        state['fit_split'] = 'validation'
        with self.assertRaises(ValueError):
            GroupDefinition.from_dict(state)

    def test_directions_are_fixed_unit_vectors_with_private_rng(self):
        before = torch.random.get_rng_state().clone()
        directions = make_directions()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        torch.testing.assert_close(directions.norm(dim=1), torch.ones(32))
        torch.testing.assert_close(directions, make_directions())

    def test_translation_cost_and_prediction_gradient(self):
        target = torch.arange(24, dtype=torch.float32).reshape(6, 4).requires_grad_()
        shift = torch.tensor([1.0, 2.0, 3.0, 4.0])
        prediction = (target.detach() + shift).requires_grad_()
        value = conditional_ot_loss(prediction, target, torch.zeros(6, dtype=torch.long), [1.0], torch.eye(4))
        torch.testing.assert_close(value, shift.square().mean())
        value.backward()
        torch.testing.assert_close(prediction.grad, 2 * shift.expand_as(prediction) / prediction.numel())
        self.assertIsNone(target.grad)

    def test_cross_group_exchange_is_detected(self):
        target = torch.tensor([[0.] * 4, [1.] * 4, [10.] * 4, [11.] * 4])
        prediction = target[[2, 3, 0, 1]]
        global_cost = conditional_ot_loss(prediction, target, torch.zeros(4, dtype=torch.long), [1.], torch.eye(4))
        grouped_cost = conditional_ot_loss(prediction, target, torch.tensor([0, 0, 1, 1]), [.5, .5], torch.eye(4))
        self.assertEqual(global_cost.item(), 0.)
        self.assertEqual(grouped_cost.item(), 100.)

    def test_within_group_permutation_does_not_replace_paired_loss(self):
        target = torch.tensor([[0.] * 4, [1.] * 4, [10.] * 4, [11.] * 4])
        prediction = target[[1, 0, 3, 2]].clone()
        original = prediction.clone()
        cost = conditional_ot_loss(prediction, target, torch.tensor([0, 0, 1, 1]), [.5, .5], torch.eye(4))
        self.assertEqual(cost.item(), 0.)
        self.assertGreater((prediction - target).square().mean().item(), 0.)
        torch.testing.assert_close(prediction, original)

    def test_missing_singleton_groups_keep_training_weights(self):
        target = torch.zeros(3, 4)
        prediction = torch.ones(3, 4, requires_grad=True)
        cost = conditional_ot_loss(prediction, target, torch.tensor([0, 0, 1]), [.2, .3, .5], torch.eye(4))
        self.assertAlmostEqual(cost.item(), .2, places=6)
        cost.backward()
        self.assertEqual(prediction.grad[2].abs().sum().item(), 0.)

    def test_all_singletons_produce_differentiable_zero(self):
        prediction = torch.ones(3, 4, requires_grad=True)
        cost = conditional_ot_loss(prediction, torch.zeros_like(prediction), torch.arange(3), [.6, .3, .1], make_directions())
        self.assertEqual(cost.item(), 0.)
        cost.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_identical_tied_samples_zero_and_finite_gradient(self):
        prediction = torch.zeros(6, 4, requires_grad=True)
        value = conditional_ot_loss(prediction, prediction.detach().clone(), torch.zeros(6, dtype=torch.long), [1.], make_directions())
        value.backward()
        self.assertEqual(value.item(), 0.)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_mixed_precision_loss_accumulates_in_float32(self):
        prediction = torch.ones(4, 4, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            cost = conditional_ot_loss(prediction, torch.zeros_like(prediction), torch.zeros(4, dtype=torch.long), [1.], torch.eye(4))
        self.assertEqual(cost.dtype, torch.float32)
        self.assertEqual(cost.item(), 1.)
        cost.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_invalid_weights_and_group_ids_rejected(self):
        p = torch.zeros(4, 4)
        for weights in ([.5], [float('nan')], [-1.]):
            with self.assertRaises(ValueError):
                conditional_ot_loss(p, p, torch.zeros(4, dtype=torch.long), weights, torch.eye(4))
        with self.assertRaises(ValueError):
            conditional_ot_loss(p, p, torch.ones(4, dtype=torch.long), [1.], torch.eye(4))


class FrozenCorrectionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'examples' / 'conditional_ot_correction.py'
        spec = importlib.util.spec_from_file_location('cot_correction_example', path)
        cls.example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.example)

    def test_frozen_baseline_zero_identity_and_parameter_update(self):
        baseline = self.example.EventRegressor(
            torch.nn.Sequential(torch.nn.Linear(6, 128), torch.nn.Dropout(.8)),
            torch.nn.Linear(128, 4),
        )
        model = self.example.FrozenBaselineCorrection(baseline).train()
        self.assertFalse(baseline.training)
        self.assertFalse(baseline.backbone[1].training)
        self.assertEqual(sum(p.numel() for p in model.correction.parameters()), 34052)
        x = torch.randn(10, 6)
        prediction, point, delta = model(x)
        torch.testing.assert_close(prediction, point, rtol=0, atol=0)
        self.assertEqual(delta.abs().sum().item(), 0.)
        state = {k: v.clone() for k, v in baseline.state_dict().items()}
        optimizer = torch.optim.SGD(model.correction.parameters(), lr=.1)
        (prediction - 1.).square().mean().backward()
        optimizer.step()
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in baseline.parameters()))
        _, _, delta = model(x)
        self.assertGreater(delta.abs().sum().item(), 0.)
        self.assertLessEqual(delta.abs().max().item(), .5)

    def test_full_objective_has_recorded_scaling(self):
        # A shift of one in every coordinate gives Huber=.5, W1=1, W2^2=1.
        reference = fit_groups(np.arange(100))
        prediction = torch.ones(6, 4, requires_grad=True)
        target = torch.zeros_like(prediction)
        groups = torch.tensor([0, 0, 1, 1, 2, 2])
        loss, pieces = self.example.correction_objective(
            prediction, target, groups, reference, torch.eye(4),
        )
        self.assertAlmostEqual(loss.item(), 1.2, places=6)
        self.assertAlmostEqual(pieces['2_huber'].item(), 1., places=6)
        self.assertAlmostEqual(pieces['marginal_w1'].item(), 1., places=6)
        self.assertAlmostEqual(pieces['conditional_w2_squared'].item(), 1., places=6)


if __name__ == '__main__':
    unittest.main()
