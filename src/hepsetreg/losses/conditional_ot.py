"""Fixed reconstructed-group, projected distribution loss for four observables.

This module preserves the grouped sliced-Wasserstein-2 squared objective used
in the 2026-11-12 correction-head study. Its groups depend on immutable
baseline predictions, with boundaries and population weights fitted on training
events only. It does not fit a transport map or a conditional density.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class GroupDefinition:
    """JSON-serializable training reference for fixed predicted-mass groups."""

    internal_edges: tuple[float, ...]
    counts: tuple[int, ...]
    probabilities: tuple[float, ...]
    quantiles: tuple[float, ...]
    fit_rows: int
    fit_split: str = "train"
    group_source: str = "frozen baseline prediction only"

    def __post_init__(self) -> None:
        edges = np.asarray(self.internal_edges, dtype=float)
        q = np.asarray(self.quantiles, dtype=float)
        counts = np.asarray(self.counts)
        weights = np.asarray(self.probabilities, dtype=float)
        if self.fit_split != "train":
            raise ValueError("Group reference must be fitted on the training split")
        if edges.ndim != 1 or not len(edges) or not np.isfinite(edges).all():
            raise ValueError("Expected finite, nonempty internal group boundaries")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("Internal group boundaries must be strictly increasing")
        if q.shape != edges.shape or np.any((q <= 0) | (q >= 1)) or not np.isfinite(q).all() or np.any(np.diff(q) <= 0):
            raise ValueError("Quantiles must be strictly increasing inside (0, 1)")
        if counts.shape != (len(edges) + 1,) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
            raise ValueError("Every fitted training group must have a positive integer count")
        if self.fit_rows <= 0 or int(counts.sum()) != self.fit_rows:
            raise ValueError("Training counts must sum to fit_rows")
        if weights.shape != counts.shape or not np.isfinite(weights).all() or not np.allclose(weights, counts / self.fit_rows, rtol=1e-7, atol=1e-9):
            raise ValueError("Group probabilities must equal full-training population fractions")

    def to_dict(self) -> dict[str, Any]:
        """Return only native Python types suitable for ``json.dump``."""
        return {
            "internal_edges": list(self.internal_edges),
            "counts": list(self.counts),
            "probabilities": list(self.probabilities),
            "quantiles": list(self.quantiles),
            "fit_rows": self.fit_rows,
            "fit_split": self.fit_split,
            "group_source": self.group_source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GroupDefinition":
        """Restore and validate a saved reference without fitting new edges."""
        return cls(
            internal_edges=tuple(value["internal_edges"]),
            counts=tuple(value["counts"]),
            probabilities=tuple(value["probabilities"]),
            quantiles=tuple(value["quantiles"]),
            fit_rows=value["fit_rows"],
            fit_split=value["fit_split"],
            group_source=value["group_source"],
        )


def fit_groups(
    training_baseline_mass: Sequence[float] | np.ndarray,
    quantiles: Sequence[float] = (0.6, 0.9),
) -> GroupDefinition:
    """Fit boundaries and population weights using frozen training predictions.

    The caller must provide the complete training split, never validation or
    test labels/predictions. Float32 boundaries and NumPy linear quantiles
    reproduce the original experiment. Tied quantiles yielding equal edges or
    empty groups are rejected instead of silently changing the grouping rule.
    """
    mass = np.asarray(training_baseline_mass, dtype=np.float32)
    q = np.asarray(quantiles, dtype=float)
    if mass.ndim != 1 or not len(mass) or not np.isfinite(mass).all():
        raise ValueError("Expected finite, nonempty training baseline masses")
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or np.any((q <= 0) | (q >= 1)) or np.any(np.diff(q) <= 0):
        raise ValueError("Quantiles must be strictly increasing inside (0, 1)")
    edges = np.quantile(mass, q, method="linear").astype(np.float32)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Training quantiles do not define distinct boundaries")
    labels = np.searchsorted(edges, mass, side="right")
    counts = np.bincount(labels, minlength=len(edges) + 1)
    return GroupDefinition(
        tuple(edges.tolist()), tuple(counts.tolist()),
        tuple((counts / len(mass)).tolist()), tuple(q.tolist()), len(mass),
    )


def assign_groups(baseline_mass: Tensor | Sequence[float], reference: GroupDefinition) -> Tensor:
    """Assign fixed groups from baseline predictions; equality enters upper group."""
    mass = torch.as_tensor(baseline_mass)
    if not mass.is_floating_point():
        mass = mass.float()
    if mass.ndim != 1 or not torch.isfinite(mass).all():
        raise ValueError("Expected a finite vector of frozen baseline masses")
    edges = torch.as_tensor(reference.internal_edges, dtype=mass.dtype, device=mass.device)
    return torch.bucketize(mass.detach().contiguous(), edges, right=True)


def make_directions(count: int = 32, dimensions: int = 4, seed: int = 4201) -> Tensor:
    """Generate reproducible unit projections without changing the global RNG."""
    if count < 1 or dimensions < 1:
        raise ValueError("Projection count and dimension must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return F.normalize(torch.randn(count, dimensions, generator=generator), dim=1)


def conditional_ot_loss(
    prediction: Tensor,
    target: Tensor,
    groups: Tensor,
    group_probabilities: Tensor | Sequence[float],
    directions: Tensor,
) -> Tensor:
    """Weighted squared distance between sorted projections within each group.

    Inputs are paired ``[batch, 4]`` values in the same training-standardised
    coordinate system. One fixed group mask selects both predictions and their
    corresponding targets. Sorting is separate for the two projected lists.
    Each group with at least two events contributes its full-training weight
    times the average squared difference across its events and directions.
    Missing and singleton groups contribute zero; remaining weights are never
    renormalised. Targets are detached, while prediction gradients are retained.
    """
    if prediction.ndim != 2 or prediction.shape != target.shape or prediction.shape[1] != 4 or not len(prediction):
        raise ValueError("Expected equal nonempty [batch, 4] prediction and target")
    if prediction.device != target.device:
        raise ValueError("Prediction and target must share a device")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("Prediction and target must be floating-point tensors")
    # Match the study's float32 loss, including inside mixed-precision training.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        p, t = prediction.float(), target.detach().float()
        labels = torch.as_tensor(groups, device=p.device)
        weights = torch.as_tensor(group_probabilities, dtype=torch.float32, device=p.device)
        d = torch.as_tensor(directions, dtype=torch.float32, device=p.device).detach()
        if labels.shape != (len(p),) or labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected one integer group label per event")
        if weights.ndim != 1 or not len(weights) or not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError("Expected positive finite full-training group weights")
        if not torch.isclose(weights.sum(), weights.new_tensor(1.0), atol=1e-6, rtol=0):
            raise ValueError("Group weights must sum to one")
        if ((labels < 0) | (labels >= len(weights))).any():
            raise ValueError("A group label is outside the fitted reference")
        if d.ndim != 2 or d.shape[1] != 4 or not len(d) or not torch.isfinite(d).all():
            raise ValueError("Expected finite [directions, 4] projection vectors")
        norms = torch.linalg.vector_norm(d, dim=1)
        if not (norms > 0).all():
            raise ValueError("Projection vectors must be nonzero")
        d = d / norms[:, None]
        if not torch.isfinite(p).all() or not torch.isfinite(t).all():
            raise ValueError("Prediction and target must contain finite values")
        value = p.sum() * 0.0
        for group in range(len(weights)):
            mask = labels == group
            if int(mask.sum()) < 2:
                continue
            a = (p[mask] @ d.T).sort(dim=0).values
            b = (t[mask] @ d.T).sort(dim=0).values
            value = value + weights[group] * (a - b).square().mean()
        return value
