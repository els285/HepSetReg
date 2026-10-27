"""
Loss functions for the regressors in head.py.

- `RegressionLoss`: plain pointwise loss (Huber/MSE/MAE) for
  `EventRegressor`'s single (B, output_dim) prediction against a single
  (B, output_dim) target.
- `SetRegressionLoss`: permutation-invariant loss for `SetEventRegressor`/
  `SlotEventRegressor`'s (B, N, output_dim) per-query/per-slot predictions,
  matched via the Hungarian algorithm against (B, M, output_dim) per-object
  ground truth -- e.g. per-top-quark kinematics -- so query/slot identity
  doesn't need to line up with any particular target ordering.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class RegressionLoss(nn.Module):
    """Pointwise regression loss: pred (B, output_dim) vs target (B, output_dim)."""

    def __init__(self, kind="huber", delta=1.0):
        super().__init__()
        kind = kind.lower()
        if kind not in {"huber", "mse", "mae"}:
            raise ValueError(f"Unknown regression loss kind '{kind}'.")
        self.kind = kind
        self.delta = delta

    def forward(self, pred, target):
        if self.kind == "huber":
            return F.huber_loss(pred, target, delta=self.delta)
        if self.kind == "mse":
            return F.mse_loss(pred, target)
        return F.l1_loss(pred, target)


class SetRegressionLoss(nn.Module):
    """Permutation-invariant loss between N predicted objects and M true
    objects per event, via per-event Hungarian (bipartite) matching.

    For each event: build an (N, M) pairwise cost matrix (pointwise
    distance between every predicted/target pair), solve the assignment
    problem for the lowest-cost one-to-one pairing
    (`scipy.optimize.linear_sum_assignment`, run under `torch.no_grad()` --
    it's a discrete combinatorial step with no gradient of its own), then
    compute the actual (differentiable) loss only over the matched pairs.

    N and M need not be equal: for a rectangular cost matrix,
    `linear_sum_assignment` matches `min(N, M)` pairs optimally, so extra
    predictions (N > M) simply go unmatched and unpenalized here, and extra
    targets (M > N) go uncovered. If you want unmatched predictions to be
    actively suppressed, add your own "no-object" / objectness term on top
    of this loss -- that's a separate concern from the matching itself.

    `target_mask` (B, M) bool, True = real target, lets M vary per event
    (same padding convention as `jet_mask`/`lepton_mask` elsewhere in this
    repo).
    """

    def __init__(self, kind="huber", delta=1.0):
        super().__init__()
        kind = kind.lower()
        if kind not in {"huber", "mse", "mae"}:
            raise ValueError(f"Unknown regression loss kind '{kind}'.")
        self.kind = kind
        self.delta = delta

    def _pairwise_cost(self, pred, target):
        # pred: (N, D), target: (M, D) -> (N, M) matrix of per-pair losses.
        diff = pred.unsqueeze(1) - target.unsqueeze(0)  # (N, M, D)
        if self.kind == "mae":
            return diff.abs().mean(dim=-1)
        if self.kind == "huber":
            abs_diff = diff.abs()
            quadratic = torch.clamp(abs_diff, max=self.delta)
            linear = abs_diff - quadratic
            return (0.5 * quadratic.pow(2) + self.delta * linear).mean(dim=-1)
        return diff.pow(2).mean(dim=-1)  # mse

    def forward(self, pred, target, target_mask=None):
        """
        pred : (B, N, D) per-query/per-slot predictions.
        target : (B, M, D) per-object ground truth.
        target_mask : (B, M) bool or None, True = real target object.
        """
        losses = []
        for b in range(pred.shape[0]):
            p = pred[b]  # (N, D)
            t = target[b]  # (M, D)
            if target_mask is not None:
                t = t[target_mask[b]]
            if t.shape[0] == 0:
                continue  # no real targets this event -- nothing to match

            cost = self._pairwise_cost(p, t)  # (N, M_valid)
            with torch.no_grad():
                row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())

            matched_cost = cost[row_idx, col_idx]  # differentiable w.r.t. `pred`
            losses.append(matched_cost.mean())

        if not losses:
            return pred.sum() * 0.0  # keep the autograd graph well-defined, with zero grad
        return torch.stack(losses).mean()


if __name__ == "__main__":
    batch_size, n_pred, n_targets, d = 4, 3, 2, 5

    pred = torch.randn(batch_size, n_pred, d, requires_grad=True)
    target = torch.randn(batch_size, n_targets, d)

    reg_loss = RegressionLoss(kind="huber")
    pooled_pred = torch.randn(batch_size, d, requires_grad=True)
    pooled_target = torch.randn(batch_size, d)
    loss = reg_loss(pooled_pred, pooled_target)
    print(f"RegressionLoss: {loss.item():.4f}")

    set_loss = SetRegressionLoss(kind="huber")
    loss = set_loss(pred, target)
    loss.backward()
    print(f"SetRegressionLoss: {loss.item():.4f}, grad flows: {pred.grad is not None}")

    # Permutation-invariance check: shuffling the N predicted slots (a pure
    # relabeling of query identity) must not change the loss.
    perm = torch.randperm(n_pred)
    loss_perm = set_loss(pred.detach()[:, perm], target)
    print(f"loss after permuting predicted slots: {loss_perm.item():.4f} (should match {set_loss(pred.detach(), target).item():.4f})")

    # target_mask: pad targets to a common M within a batch, mask off the padding.
    padded_target = torch.randn(batch_size, 4, d)
    target_mask = torch.zeros(batch_size, 4, dtype=torch.bool)
    target_mask[:, :n_targets] = True
    padded_target[:, :n_targets] = target
    loss_masked = set_loss(pred.detach(), padded_target, target_mask=target_mask)
    print(f"loss with padded+masked targets: {loss_masked.item():.4f} (should match {set_loss(pred.detach(), target).item():.4f})")
