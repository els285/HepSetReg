"""Distribution-matching losses: compare the *distribution* of a batch of
predictions against the distribution of the corresponding batch of targets,
rather than pairing up individual (pred_i, target_i) points the way
:class:`~hepsetreg.losses.regression.RegressionLoss` does.

Four options, in increasing order of how much they trust individual samples
vs. binning/kernel structure:

* :class:`HistogramKLDivergenceLoss` -- the original, generalized from
  DIRECTOR's ``distribution_considering_loss``. Builds a soft (Gaussian-
  kernel) histogram per target *dimension* and takes KL(target || pred).
  Needs a bin range and bin count; compares marginals only (it can't see
  correlations between target dimensions).
* :class:`KNNKLDivergenceLoss` -- an *unbinned* / non-parametric KL
  divergence estimator based on k-nearest-neighbour distances (Pérez-Cruz
  2008; Wang, Kulkarni & Verdú 2009). No bin range to tune, and estimates
  the true joint KL divergence (it does see cross-dimension correlations).
* :class:`MMDLoss` -- unbinned, kernel-based (Maximum Mean Discrepancy,
  Gretton et al. 2012) distance between the joint distributions. Bounded
  and well-behaved even for small batches; a good default when
  ``HistogramKLDivergenceLoss``'s bin range is awkward to pick or
  ``KNNKLDivergenceLoss``'s "batch size >> k" requirement isn't met.
* :class:`SlicedWassersteinLoss` -- unbinned, transport-based (average
  exact 1-D Wasserstein distance over random projections; Rabin et al.
  2011). Cheap, and degrades gracefully with dimensionality.

All four share the same ``forward(pred, target) -> Tensor`` contract (a
single scalar), so any of them can be dropped into
:class:`~hepsetreg.losses.composite.CompositeLoss` under the
``distribution_kl`` term name interchangeably -- no other code needs to
change to swap one for another.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class HistogramKLDivergenceLoss(nn.Module):
    """Per-dimension soft-histogram KL(target || pred). See module docstring."""

    def __init__(
        self,
        bins: int = 100,
        sigma: float = 0.4,
        eps: float = 1e-8,
        hist_min: Optional[float] = None,
        hist_max: Optional[float] = None,
        dynamic_padding: float = 0.25,
    ):
        super().__init__()
        self.bins = bins
        self.sigma = sigma
        self.eps = eps
        self.hist_min = hist_min
        self.hist_max = hist_max
        self.dynamic_padding = dynamic_padding

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}.")
        n_dims = pred.shape[1]

        if self.hist_min is not None and self.hist_max is not None:
            hist_min = pred.new_full((n_dims,), float(self.hist_min))
            hist_max = pred.new_full((n_dims,), float(self.hist_max))
        else:
            hist_min = target.min(dim=0).values - self.dynamic_padding
            hist_max = target.max(dim=0).values + self.dynamic_padding

        # centers: (n_dims, bins)
        centers = torch.stack(
            [torch.linspace(lo.item(), hi.item(), self.bins, device=pred.device, dtype=pred.dtype)
             for lo, hi in zip(hist_min, hist_max)],
            dim=0,
        )

        # (B, n_dims, 1) vs (1, n_dims, bins) -> (B, n_dims, bins)
        pred_kernel = torch.exp(-0.5 * ((pred.unsqueeze(-1) - centers.unsqueeze(0)) / self.sigma) ** 2)
        target_kernel = torch.exp(-0.5 * ((target.unsqueeze(-1) - centers.unsqueeze(0)) / self.sigma) ** 2)

        pred_hist = pred_kernel.mean(dim=0) + self.eps  # (n_dims, bins)
        target_hist = target_kernel.mean(dim=0) + self.eps

        pred_hist = pred_hist / pred_hist.sum(dim=-1, keepdim=True)
        target_hist = target_hist / target_hist.sum(dim=-1, keepdim=True)

        kl_per_dim = torch.sum(target_hist * (torch.log(target_hist) - torch.log(pred_hist)), dim=-1)
        return kl_per_dim.mean()


class KNNKLDivergenceLoss(nn.Module):
    """Unbinned KL(pred || target) via k-nearest-neighbour distances.

    Estimator (Pérez-Cruz 2008; Wang, Kulkarni & Verdú 2009): treating
    ``pred`` as ``n`` samples from distribution P and ``target`` as ``m``
    samples from Q, both in ``d`` dimensions,

        KL(P || Q) ~= (d / n) * sum_i log(s_k(i) / r_k(i)) + log(m / (n - 1))

    where ``r_k(i)`` is the distance from ``pred[i]`` to its k-th nearest
    neighbour within ``pred`` (excluding itself), and ``s_k(i)`` is the
    distance from ``pred[i]`` to its k-th nearest neighbour within
    ``target``. No histogram/bin range is needed, and -- unlike
    :class:`HistogramKLDivergenceLoss` -- this estimates the KL divergence
    of the full joint ``d``-dimensional distribution, so it's sensitive to
    correlations between target dimensions, not just their marginals.

    Caveats inherent to k-NN density estimation: needs ``n > k`` samples per
    batch (rule of thumb: batch size >> k, and more so as ``d`` grows -- the
    curse of dimensionality affects k-NN density estimates too, just less
    abruptly than fixed-width histograms). Being a finite-sample estimator,
    it can occasionally return a small negative value even though the true
    KL divergence is >= 0; this is left unclamped (rather than
    ``clamp(min=0)``) so gradients stay informative near zero.
    """

    def __init__(self, k: int = 3, eps: float = 1e-8):
        super().__init__()
        if k < 1:
            raise ValueError("k must be >= 1.")
        self.k = k
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape[1] != target.shape[1]:
            raise ValueError(f"pred/target dimensionality mismatch: {pred.shape[1]} vs {target.shape[1]}.")
        n, d = pred.shape
        m = target.shape[0]
        if n <= self.k:
            raise ValueError(f"KNNKLDivergenceLoss needs more than k={self.k} predicted samples per batch (got {n}).")

        self_dist = torch.cdist(pred, pred)
        self_mask = torch.eye(n, device=pred.device, dtype=torch.bool)
        self_dist = self_dist.masked_fill(self_mask, float("inf"))
        r_k = self_dist.kthvalue(self.k, dim=1).values  # (n,)

        cross_dist = torch.cdist(pred, target)  # (n, m)
        s_k = cross_dist.kthvalue(self.k, dim=1).values  # (n,)

        ratio = (s_k + self.eps) / (r_k + self.eps)
        kl = (d / n) * torch.log(ratio).sum() + math.log(m / (n - 1))
        return kl


class MMDLoss(nn.Module):
    """Unbinned distribution-matching loss via squared Maximum Mean
    Discrepancy with a multi-scale Gaussian/RBF kernel (Gretton et al.
    2012), computed over the full joint ``d``-dimensional distribution.

    ``MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]``, estimated with
    the usual (nearly-)unbiased U-statistic (excluding self-pairs from the
    within-set terms). Always close to symmetric-positive in expectation and
    bounded by the kernel, so it stays well-behaved for small batches where
    :class:`KNNKLDivergenceLoss` would be noisy or inapplicable.

    Bandwidth: either pass fixed ``bandwidths`` (kernel length scales), or
    leave it as the default ``bandwidth_multipliers``, which scales the
    per-batch median pairwise distance (the standard "median heuristic") by
    each multiplier and sums the resulting multi-scale kernels -- this needs
    no manual tuning and adapts automatically to the current scale of your
    targets.
    """

    def __init__(
        self,
        bandwidths: Optional[List[float]] = None,
        bandwidth_multipliers: Optional[List[float]] = None,
    ):
        super().__init__()
        if bandwidths is not None and bandwidth_multipliers is not None:
            raise ValueError(
                "Pass either fixed `bandwidths` or `bandwidth_multipliers` (median-heuristic "
                "scale factors), not both."
            )
        self.bandwidths = bandwidths
        self.bandwidth_multipliers = None if bandwidths is not None else (
            bandwidth_multipliers or [0.25, 0.5, 1.0, 2.0, 4.0]
        )

    @staticmethod
    def _kernel_sum(sq_dists: Tensor, bandwidths: Tensor) -> Tensor:
        kernel = torch.zeros_like(sq_dists)
        for bandwidth in bandwidths:
            kernel = kernel + torch.exp(-sq_dists / (2.0 * bandwidth**2))
        return kernel

    def _resolve_bandwidths(self, xx: Tensor, yy: Tensor, xy: Tensor) -> Tensor:
        if self.bandwidths is not None:
            return torch.as_tensor(self.bandwidths, device=xx.device, dtype=xx.dtype)

        with torch.no_grad():
            pooled = torch.cat([xx.flatten(), yy.flatten(), xy.flatten()])
            nonzero = pooled[pooled > 0]
            median_sq = nonzero.median() if nonzero.numel() > 0 else pooled.median()
            base_bandwidth = median_sq.clamp(min=1e-12).sqrt()
        multipliers = torch.as_tensor(self.bandwidth_multipliers, device=xx.device, dtype=xx.dtype)
        return base_bandwidth * multipliers

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape[1] != target.shape[1]:
            raise ValueError(f"pred/target dimensionality mismatch: {pred.shape[1]} vs {target.shape[1]}.")
        n, m = pred.shape[0], target.shape[0]
        if n < 2 or m < 2:
            raise ValueError(f"MMDLoss needs >=2 samples in both pred and target (got n={n}, m={m}).")

        xx = torch.cdist(pred, pred) ** 2
        yy = torch.cdist(target, target) ** 2
        xy = torch.cdist(pred, target) ** 2

        bandwidths = self._resolve_bandwidths(xx, yy, xy)

        k_xx = self._kernel_sum(xx, bandwidths)
        k_yy = self._kernel_sum(yy, bandwidths)
        k_xy = self._kernel_sum(xy, bandwidths)

        # Exclude self-similarity (diagonal) from the within-set terms.
        mmd_xx = (k_xx.sum() - k_xx.diagonal().sum()) / (n * (n - 1))
        mmd_yy = (k_yy.sum() - k_yy.diagonal().sum()) / (m * (m - 1))
        mmd_xy = k_xy.sum() / (n * m)

        return mmd_xx + mmd_yy - 2.0 * mmd_xy


class SlicedWassersteinLoss(nn.Module):
    """Unbinned distribution-matching loss via the sliced Wasserstein
    distance (Rabin et al. 2011): project ``pred`` and ``target`` onto many
    random 1-D directions and average the *exact* 1-D Wasserstein distance
    (sorted-samples matching) along each one.

    Cheap (no pairwise kernel/distance matrix beyond the projections
    themselves), needs no bin range or bandwidth, and degrades gracefully
    with both batch size and target dimensionality. Requires ``pred`` and
    ``target`` to have the same number of samples per batch (so the sorted
    1-D matching along each projection is well defined).
    """

    def __init__(self, n_projections: int = 128, p: int = 2):
        super().__init__()
        if p not in (1, 2):
            raise ValueError("p must be 1 or 2.")
        self.n_projections = n_projections
        self.p = p

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                "SlicedWassersteinLoss compares two equal-size sets of samples along each "
                f"random projection; got pred shape {tuple(pred.shape)} vs target shape {tuple(target.shape)}."
            )
        _, d = pred.shape

        directions = torch.randn(d, self.n_projections, device=pred.device, dtype=pred.dtype)
        directions = directions / directions.norm(dim=0, keepdim=True).clamp(min=1e-12)

        pred_proj = pred @ directions  # (n, n_projections)
        target_proj = target @ directions

        pred_sorted, _ = torch.sort(pred_proj, dim=0)
        target_sorted, _ = torch.sort(target_proj, dim=0)

        diff = (pred_sorted - target_sorted).abs()
        if self.p == 2:
            return diff.pow(2).mean(dim=0).mean().sqrt()
        return diff.mean()
