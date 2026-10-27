"""
Shape-matching losses: distances between the DISTRIBUTION of predictions and
the distribution of targets over a batch, one marginal per output column.

Each function takes pred and target of shape (B, ...) and returns a scalar.
Trailing dimensions are flattened, so every (feature) or (slot, feature)
column is compared as its own 1D marginal and the results are averaged.

Inputs should be z-scored (as the model is trained on), so that scales are
comparable across features and the KL histogram range below makes sense.
These are estimates of a distribution, so use large batches (e.g. 4096).
"""

import torch


def _columns(x):
    return x.reshape(x.shape[0], -1).float()  # (B, C)


def emd_loss(pred, target):
    """1D earth mover's (Wasserstein-1) distance per column: sort both
    batches, then average the absolute gap between matched quantiles."""
    p = _columns(pred).sort(dim=0).values
    t = _columns(target).sort(dim=0).values
    return (p - t).abs().mean()


def mmd_loss(pred, target, bandwidths=(0.5, 1.0, 2.0), max_samples=1024):
    """Squared maximum mean discrepancy per column, with a Gaussian kernel
    summed over `bandwidths`. Uses a random subset of at most `max_samples`
    events, since the kernel matrix grows with the square of the batch."""
    n = min(pred.shape[0], target.shape[0], max_samples)
    p = _columns(pred[torch.randperm(pred.shape[0])[:n]]).T.unsqueeze(-1)  # (C, n, 1)
    t = _columns(target[torch.randperm(target.shape[0])[:n]]).T.unsqueeze(-1)

    def mean_kernel(a, b):
        d2 = (a - b.transpose(1, 2)).pow(2)  # (C, n, n)
        return sum(torch.exp(-d2 / (2 * h * h)) for h in bandwidths).mean(dim=(1, 2))  # (C,)

    mmd2 = mean_kernel(p, p) + mean_kernel(t, t) - 2 * mean_kernel(p, t)
    return mmd2.mean()


def kl_loss(pred, target, n_bins=50, lo=-3.0, hi=3.0):
    """KL(target || pred) per column, from histograms over the z-scored range
    [lo, hi]. Binning is Gaussian-kernel soft-assignment so the prediction
    histogram stays differentiable; the target histogram is a constant.
    Values outside [lo, hi] are simply not counted."""
    centers = torch.linspace(lo, hi, n_bins, device=pred.device)
    sigma = (hi - lo) / n_bins

    def soft_hist(x):  # (B, C) -> (C, n_bins), each row sums to 1
        w = torch.exp(-((x.T.unsqueeze(-1) - centers) ** 2) / (2 * sigma * sigma))  # (C, B, n_bins)
        h = w.sum(dim=1) + 1e-8
        return h / h.sum(dim=1, keepdim=True)

    q = soft_hist(_columns(pred))
    with torch.no_grad():
        p = soft_hist(_columns(target))
    return (p * (torch.log(p) - torch.log(q))).sum(dim=1).mean()


def joint_mmd_loss(pred, target, scales=(0.5, 1.0, 2.0), max_samples=1024):
    """Squared MMD over the full joint distribution: each event is one point
    in C dimensions, so correlations between columns are matched too. The
    Gaussian bandwidth comes from the median pairwise distance of the targets
    (median heuristic), times each of `scales`."""
    n = min(pred.shape[0], target.shape[0], max_samples)
    p = _columns(pred[torch.randperm(pred.shape[0])[:n]])  # (n, C)
    t = _columns(target[torch.randperm(target.shape[0])[:n]])

    def sq_dists(a, b):
        return (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)  # (n, n)

    with torch.no_grad():
        d2_tt = sq_dists(t, t)
        median = d2_tt[d2_tt > 0].median()

    def mean_kernel(a, b):
        d2 = sq_dists(a, b)
        return sum(torch.exp(-d2 / (s * median)) for s in scales).mean()

    return mean_kernel(p, p) + mean_kernel(t, t) - 2 * mean_kernel(p, t)


def sliced_emd_loss(pred, target, n_projections=64):
    """Sliced Wasserstein-1: project both batches onto random unit directions
    in C dimensions, take the 1D earth mover's distance along each one (the
    same sort-based computation as emd_loss), and average. Cost is
    O(n_projections * n log n), and the joint structure is only approximated
    by the projections."""
    p = _columns(pred)
    t = _columns(target)
    directions = torch.randn(p.shape[1], n_projections, device=p.device)
    directions = directions / directions.norm(dim=0, keepdim=True)
    p_proj = (p @ directions).sort(dim=0).values
    t_proj = (t @ directions).sort(dim=0).values
    return (p_proj - t_proj).abs().mean()


SHAPE_LOSSES = {
    "emd": emd_loss,
    "mmd": mmd_loss,
    "kl": kl_loss,
    "mmd_joint": joint_mmd_loss,
    "sliced_emd": sliced_emd_loss,
}
