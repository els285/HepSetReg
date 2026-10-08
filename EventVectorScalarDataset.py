"""
Same padded/pre-scaled construction as EventDataset6, but splits each of the
jet/lepton per-object feature vectors into two groups:

  - VECTOR features: the object's 4-momentum (pt/eta/phi/e/m/px/py/pz) --
    everything a pairwise physics quantity (deltaR, kT, z, invariant mass...)
    can be computed from.
  - SCALAR features: everything else (e.g. jet_bTag, lepton charge/is_electron)
    -- object properties with no role in a pairwise kinematic bias.

This split exists for a Particle-Transformer-style pairwise attention bias:
that bias is computed from the VECTOR features only, but both groups are
still relevant to the per-token embedding, so both are returned (as
`{jet,lepton}_vector` and `{jet,lepton}_scalar`) rather than only keeping
one. `global` isn't split -- it's a single per-event summary token, not a
per-particle 4-momentum, so it has no "pairwise" role either way.

Feature names are NOT defined here: the caller passes a `features` dict
with keys "global", "jet_vector", "jet_scalar", "lepton_vector",
"lepton_scalar", "target" (see run_regression_ParT.py).
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from EventDataset6 import load_arrays, zscore


def _stack_fields(arr, fields):
    return np.stack([ak.to_numpy(arr[f]).astype(np.float32) for f in fields], axis=-1)


def _raw_features(features):
    """Converts the vector/scalar split into the global/jet/lepton/target
    layout `EventDataset6.load_arrays` expects."""
    return {
        "global": features["global"],
        "jet": features["jet_vector"] + features["jet_scalar"],
        "lepton": features["lepton_vector"] + features["lepton_scalar"],
        "target": features["target"],
    }


class EventDataset(Dataset):
    """
    Parameters
    ----------
    source : str or (global, jet, lepton, target) tuple
        Parquet path, or the arrays already split out.
    features : dict
        Keys "global", "jet_vector", "jet_scalar", "lepton_vector",
        "lepton_scalar", "target" -- field names to use.
    indices, stats, keep_unscaled_data, max_jets : see EventDataset6.EventDataset.
    """

    def __init__(self, source, *, features, indices=None, stats=None, keep_unscaled_data=False, max_jets=None):
        global_arr, jet_arr, lepton_arr, target_arr = (
            load_arrays(source, _raw_features(features)) if isinstance(source, str) else source
        )

        if indices is None:
            indices = np.arange(len(global_arr))
        indices = np.asarray(indices)
        n_events = len(indices)

        global_arr = global_arr[indices]
        target_arr = target_arr[indices]
        jet_sub = jet_arr[indices]
        lepton_sub = lepton_arr[indices]
        self.keep_unscaled_data = keep_unscaled_data

        # fit/reuse stats on the RAGGED jets (real objects only) -- must
        # happen before padding, or the padded zeros would skew the mean/std
        self.stats = stats or self._compute_stats(global_arr, jet_sub, lepton_sub, target_arr, features)
        g_mean, g_std = self.stats["global"]
        jv_mean, jv_std = self.stats["jet_vector"]
        js_mean, js_std = self.stats["jet_scalar"]
        lv_mean, lv_std = self.stats["lepton_vector"]
        ls_mean, ls_std = self.stats["lepton_scalar"]
        t_mean, t_std = self.stats["target"]

        # ---- global / target: already fixed-size, zscore the whole array at once ----
        self.global_tensor = torch.from_numpy(zscore(global_arr, g_mean, g_std))
        self.target_tensor = torch.from_numpy(zscore(target_arr, t_mean, t_std))
        if keep_unscaled_data:
            self.global_unscaled_tensor = torch.from_numpy(global_arr)
            self.target_unscaled_tensor = torch.from_numpy(target_arr)

        # ---- leptons: always exactly 2/event -> flatten, zscore, reshape ----
        lepton_flat = ak.flatten(lepton_sub)  # flatten once, reuse for both groups
        lep_vec_flat = _stack_fields(lepton_flat, features["lepton_vector"])
        lep_sca_flat = _stack_fields(lepton_flat, features["lepton_scalar"])
        self.lepton_vector_tensor = torch.from_numpy(zscore(lep_vec_flat, lv_mean, lv_std)).reshape(
            n_events, 2, len(features["lepton_vector"])
        )
        self.lepton_scalar_tensor = torch.from_numpy(zscore(lep_sca_flat, ls_mean, ls_std)).reshape(
            n_events, 2, len(features["lepton_scalar"])
        )
        # always exactly 2 real leptons/event -> mask is trivially all-True,
        # but TransformerBackbone-style forwards read it unconditionally.
        self.lepton_mask = torch.ones(n_events, 2, dtype=torch.bool)
        if keep_unscaled_data:
            self.lepton_vector_unscaled_tensor = torch.from_numpy(lep_vec_flat).reshape(
                n_events, 2, len(features["lepton_vector"])
            )
            self.lepton_scalar_unscaled_tensor = torch.from_numpy(lep_sca_flat).reshape(
                n_events, 2, len(features["lepton_scalar"])
            )

        # ---- jets: ragged -> pad/truncate to `max_jets` fixed slots, ONCE ----
        counts = ak.num(jet_sub, axis=1).to_numpy()
        self.max_jets = max_jets if max_jets is not None else int(counts.max())

        def pad_dense(fields):
            return np.stack(
                [
                    ak.to_numpy(ak.fill_none(ak.pad_none(jet_sub[f], self.max_jets, clip=True), 0.0)).astype(np.float32)
                    for f in fields
                ],
                axis=-1,
            )  # (n_events, max_jets, n_features)

        jet_vec_dense = pad_dense(features["jet_vector"])
        jet_sca_dense = pad_dense(features["jet_scalar"])

        mask = np.arange(self.max_jets)[None, :] < counts[:, None]  # True = real jet
        self.jet_mask = torch.from_numpy(mask)

        if keep_unscaled_data:
            self.jet_vector_unscaled_tensor = torch.from_numpy(jet_vec_dense.copy())
            self.jet_scalar_unscaled_tensor = torch.from_numpy(jet_sca_dense.copy())

        # In-place zscore: jets are the biggest tensors here, so avoid
        # allocating second full-size copies just to hold the scaled version.
        jet_vec_dense -= jv_mean
        jet_vec_dense /= jv_std + 1e-8
        jet_vec_dense[~mask] = 0.0  # padded slots stay exactly 0, not zscore(0)
        self.jet_vector_tensor = torch.from_numpy(jet_vec_dense)

        jet_sca_dense -= js_mean
        jet_sca_dense /= js_std + 1e-8
        jet_sca_dense[~mask] = 0.0
        self.jet_scalar_tensor = torch.from_numpy(jet_sca_dense)

    @staticmethod
    def _compute_stats(global_arr, jet_arr, lepton_arr, target_arr, features):
        jet_flat = ak.flatten(jet_arr)
        lep_flat = ak.flatten(lepton_arr)
        jet_vec_flat = _stack_fields(jet_flat, features["jet_vector"])
        jet_sca_flat = _stack_fields(jet_flat, features["jet_scalar"])
        lep_vec_flat = _stack_fields(lep_flat, features["lepton_vector"])
        lep_sca_flat = _stack_fields(lep_flat, features["lepton_scalar"])
        return {
            "global": (global_arr.mean(0), global_arr.std(0)),
            "jet_vector": (jet_vec_flat.mean(0), jet_vec_flat.std(0)),
            "jet_scalar": (jet_sca_flat.mean(0), jet_sca_flat.std(0)),
            "lepton_vector": (lep_vec_flat.mean(0), lep_vec_flat.std(0)),
            "lepton_scalar": (lep_sca_flat.mean(0), lep_sca_flat.std(0)),
            "target": (target_arr.mean(0), target_arr.std(0)),
        }

    def to(self, device):
        """Move every cached tensor onto `device` in place -- see
        EventDataset6.EventDataset.to. Returns `self` for chaining."""
        self.global_tensor = self.global_tensor.to(device)
        self.target_tensor = self.target_tensor.to(device)
        self.lepton_vector_tensor = self.lepton_vector_tensor.to(device)
        self.lepton_scalar_tensor = self.lepton_scalar_tensor.to(device)
        self.lepton_mask = self.lepton_mask.to(device)
        self.jet_vector_tensor = self.jet_vector_tensor.to(device)
        self.jet_scalar_tensor = self.jet_scalar_tensor.to(device)
        self.jet_mask = self.jet_mask.to(device)
        if self.keep_unscaled_data:
            self.global_unscaled_tensor = self.global_unscaled_tensor.to(device)
            self.target_unscaled_tensor = self.target_unscaled_tensor.to(device)
            self.lepton_vector_unscaled_tensor = self.lepton_vector_unscaled_tensor.to(device)
            self.lepton_scalar_unscaled_tensor = self.lepton_scalar_unscaled_tensor.to(device)
            self.jet_vector_unscaled_tensor = self.jet_vector_unscaled_tensor.to(device)
            self.jet_scalar_unscaled_tensor = self.jet_scalar_unscaled_tensor.to(device)
        return self

    def __len__(self):
        return self.global_tensor.shape[0]

    def __getitem__(self, idx):
        item = {
            "global": self.global_tensor[idx],
            "jet_vector": self.jet_vector_tensor[idx],  # (max_jets, n_jet_vector_features)
            "jet_scalar": self.jet_scalar_tensor[idx],  # (max_jets, n_jet_scalar_features)
            "jet_mask": self.jet_mask[idx],  # (max_jets,)
            "lepton_vector": self.lepton_vector_tensor[idx],  # (2, n_lepton_vector_features)
            "lepton_scalar": self.lepton_scalar_tensor[idx],  # (2, n_lepton_scalar_features)
            "lepton_mask": self.lepton_mask[idx],  # (2,), always True
            "target": self.target_tensor[idx],
        }
        if self.keep_unscaled_data:
            item["global_unscaled"] = self.global_unscaled_tensor[idx]
            item["jet_vector_unscaled"] = self.jet_vector_unscaled_tensor[idx]
            item["jet_scalar_unscaled"] = self.jet_scalar_unscaled_tensor[idx]
            item["lepton_vector_unscaled"] = self.lepton_vector_unscaled_tensor[idx]
            item["lepton_scalar_unscaled"] = self.lepton_scalar_unscaled_tensor[idx]
            item["target_unscaled"] = self.target_unscaled_tensor[idx]
        return item

    def __getitems__(self, indices):
        """Batched fetch -- see EventDataset6.EventDataset.__getitems__: one
        vectorized fancy-index per field instead of a Python loop over the
        batch. Returns the batch already collated as a dict of stacked
        tensors; pair with `_identity_collate`."""
        idx = torch.as_tensor(indices, device=self.global_tensor.device)
        batch = {
            "global": self.global_tensor[idx],
            "jet_vector": self.jet_vector_tensor[idx],
            "jet_scalar": self.jet_scalar_tensor[idx],
            "jet_mask": self.jet_mask[idx],
            "lepton_vector": self.lepton_vector_tensor[idx],
            "lepton_scalar": self.lepton_scalar_tensor[idx],
            "lepton_mask": self.lepton_mask[idx],
            "target": self.target_tensor[idx],
        }
        if self.keep_unscaled_data:
            batch["global_unscaled"] = self.global_unscaled_tensor[idx]
            batch["jet_vector_unscaled"] = self.jet_vector_unscaled_tensor[idx]
            batch["jet_scalar_unscaled"] = self.jet_scalar_unscaled_tensor[idx]
            batch["lepton_vector_unscaled"] = self.lepton_vector_unscaled_tensor[idx]
            batch["lepton_scalar_unscaled"] = self.lepton_scalar_unscaled_tensor[idx]
            batch["target_unscaled"] = self.target_unscaled_tensor[idx]
        return batch


def _identity_collate(batch):
    """No-op collate: `EventDataset.__getitems__` already returns a fully
    batched dict. A plain module-level function (not a lambda) so it can be
    pickled to DataLoader workers when num_workers>0."""
    return batch


class EventDataModule(pl.LightningDataModule):
    """Same design as EventDataset6.EventDataModule: fits stats (and
    max_jets) on train, reuses them for val; no custom collate_fn needed
    since every item is already a fixed shape. See EventDataset6.EventDataModule
    for the reasoning behind num_workers/persistent_workers/device defaults.
    """

    def __init__(
        self, train_path, val_path, *, features, batch_size=256, num_workers=0,
        persistent_workers=False, keep_unscaled_data=False, max_jets=None,
        device=None,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.features = features
        self.batch_size = batch_size
        self.device = torch.device(device) if device is not None else None
        if self.device is not None and self.device.type == "cuda":
            num_workers = 0
            persistent_workers = False
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.keep_unscaled_data = keep_unscaled_data
        self.max_jets = max_jets

    def setup(self, stage=None):
        self.train_dataset = EventDataset(
            self.train_path, features=self.features,
            keep_unscaled_data=self.keep_unscaled_data, max_jets=self.max_jets,
        )  # fits stats (and, if not given, max_jets) on train
        self.stats = self.train_dataset.stats
        self.val_dataset = EventDataset(
            self.val_path, features=self.features, stats=self.stats,
            keep_unscaled_data=self.keep_unscaled_data, max_jets=self.train_dataset.max_jets,
        )  # reuse them
        if self.device is not None:
            self.train_dataset.to(self.device)
            self.val_dataset.to(self.device)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, persistent_workers=self.persistent_workers,
            pin_memory=self.device is None, collate_fn=_identity_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, persistent_workers=self.persistent_workers,
            pin_memory=self.device is None, collate_fn=_identity_collate,
        )
