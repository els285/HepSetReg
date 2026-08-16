"""Target standardisation, mirroring DIRECTOR's ``Y_mean``/``Y_scale`` HDF5 convention."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from torch import Tensor


class TargetScaler:
    """``y_scaled = (y - mean) / scale``; ``unscale`` inverts it.

    Kept as a tiny standalone object (not an ``nn.Module``) so it can live on
    a LightningModule without being treated as a trainable parameter or
    showing up in checkpoints' optimizer state.
    """

    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.scale = torch.as_tensor(scale, dtype=torch.float32)

    @classmethod
    def from_hdf5(cls, path: Union[str, Path], mean_key: str = "Y_mean", scale_key: str = "Y_scale") -> "TargetScaler":
        with h5py.File(path, "r") as f:
            return cls(mean=f[mean_key][:], scale=f[scale_key][:])

    @classmethod
    def fit(cls, targets: np.ndarray) -> "TargetScaler":
        mean = targets.mean(axis=0)
        scale = targets.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def to(self, device) -> "TargetScaler":
        self.mean = self.mean.to(device)
        self.scale = self.scale.to(device)
        return self

    def scale_(self, y: Tensor) -> Tensor:
        return (y - self.mean.to(y.device, y.dtype)) / self.scale.to(y.device, y.dtype)

    def unscale(self, y: Tensor) -> Tensor:
        return y * self.scale.to(y.device, y.dtype) + self.mean.to(y.device, y.dtype)
