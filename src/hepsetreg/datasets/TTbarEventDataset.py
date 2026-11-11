"""TTbarEventDataset: a ttbar-dilepton-specific extension of
`EventDatasetBase.EventDataset` that adds the EMtops input modality --
Ellipse Method top/antitop candidate solutions (see
HanData/EllipseMethod.cpp, HanData/make_ttbar2L_parquet.py). Top and
antitop candidates share one token type, distinguished by an `is_top`
feature -- the same convention `EventDatasetBase.load_arrays` already uses
for electrons/muons in the `lepton` modality (`is_electron`).

Kept out of EventDatasetBase itself, which stays a generic, physics-agnostic
global/jet/lepton/target loader -- EMtops is specific to ttbar dilepton
events reconstructed with the Ellipse Method, not something every user of
EventDataset has or needs.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .EventDatasetBase import EventDataset, _identity_collate, _require_fields, load_arrays, zscore


def load_ttbar_arrays(parquet_path, features):
    """Like `EventDatasetBase.load_arrays`, but always returns the 5-tuple
    (global, jet, lepton, EMtops, target) -- `features["EMtops"]` is
    required here (a non-empty list of field names), not optional.
    """
    if not features.get("EMtops"):
        raise KeyError('TTbarEventDataset requires features["EMtops"] (a non-empty list of field names)')

    global_arr, jet_arr, lepton_arr, target_arr = load_arrays(parquet_path, features)

    # load_arrays() only reads global/jet/lepton/target -- re-open the file
    # to also pull out the EMtops columns.
    events = ak.from_parquet(parquet_path)
    inputs = events["inputs"]
    _require_fields(inputs, features["EMtops"], f"{parquet_path} inputs")
    emtops_arr = ak.zip({f: inputs[f] for f in features["EMtops"]})

    return global_arr, jet_arr, lepton_arr, emtops_arr, target_arr


def _pad_ragged(ragged, field_names, max_slots, stats=None):
    """Pads/truncates a ragged (per-event, variable-count) awkward record to
    `max_slots` fixed slots, z-scored -- the same recipe `EventDataset` uses
    for jets, pulled out here as a function since `EventDataset` itself
    isn't touched (EMtops is ttbar-specific, not generic).

    `stats`: reuse an already-fit (mean, std) (e.g. val reusing train's)
    instead of fitting fresh ones on `ragged`'s real (unpadded) entries.

    Returns (dense, mask, unscaled, stats) tensors/arrays; `mask` is True
    for real (non-padded) slots.
    """
    if stats is None:
        flat = np.stack([ak.to_numpy(ak.flatten(ragged[f])).astype(np.float32) for f in field_names], axis=1)
        stats = (flat.mean(0), flat.std(0))
    mean, std = stats

    dense = np.stack(
        [
            ak.to_numpy(ak.fill_none(ak.pad_none(ragged[f], max_slots, clip=True), 0.0)).astype(np.float32)
            for f in field_names
        ],
        axis=-1,
    )  # (n_events, max_slots, n_fields)
    counts = ak.num(ragged, axis=1).to_numpy()
    mask = np.arange(max_slots)[None, :] < counts[:, None]

    unscaled = torch.from_numpy(dense.copy())
    dense = zscore(dense, mean, std)
    dense[~mask] = 0.0  # padded slots stay exactly 0, not zscore(0)
    return torch.from_numpy(dense), torch.from_numpy(mask), unscaled, stats


class TTbarEventDataset(EventDataset):
    """`EventDataset` plus a ragged EMtops token modality.

    Parameters are the same as `EventDataset`, plus `max_emtops` (same role
    as `max_jets`: the fixed EMtops-slot count every event is padded/
    truncated to; None auto-detects the largest count actually present).
    `features["EMtops"]` must be set to a non-empty list of field names.
    """

    def __init__(
        self, source, *, features, indices=None, stats=None, keep_unscaled_data=False,
        max_jets=None, max_emtops=None,
    ):
        if not features.get("EMtops"):
            raise KeyError('TTbarEventDataset requires features["EMtops"] (a non-empty list of field names)')

        global_arr, jet_arr, lepton_arr, emtops_arr, target_arr = (
            load_ttbar_arrays(source, features) if isinstance(source, str) else source
        )

        if indices is None:
            indices = np.arange(len(global_arr))
        indices = np.asarray(indices)
        emtops_sub = emtops_arr[indices]

        base_features = {k: v for k, v in features.items() if k != "EMtops"}
        super().__init__(
            (global_arr, jet_arr, lepton_arr, target_arr),
            features=base_features, indices=indices, stats=stats,
            keep_unscaled_data=keep_unscaled_data, max_jets=max_jets,
        )

        self.max_emtops = max_emtops if max_emtops is not None else int(ak.num(emtops_sub, axis=1).to_numpy().max())
        self.emtops_tensor, self.emtops_mask, emtops_unscaled, em_stats = _pad_ragged(
            emtops_sub, features["EMtops"], self.max_emtops, stats=self.stats.get("EMtops"),
        )
        self.stats["EMtops"] = em_stats
        if self.keep_unscaled_data:
            self.emtops_unscaled_tensor = emtops_unscaled

    def to(self, device):
        super().to(device)
        self.emtops_tensor = self.emtops_tensor.to(device)
        self.emtops_mask = self.emtops_mask.to(device)
        if self.keep_unscaled_data:
            self.emtops_unscaled_tensor = self.emtops_unscaled_tensor.to(device)
        return self

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["EMtops"] = self.emtops_tensor[idx]
        item["EMtops_mask"] = self.emtops_mask[idx]
        if self.keep_unscaled_data:
            item["EMtops_unscaled"] = self.emtops_unscaled_tensor[idx]
        return item

    def __getitems__(self, indices):
        batch = super().__getitems__(indices)
        idx = torch.as_tensor(indices, device=self.global_tensor.device)
        batch["EMtops"] = self.emtops_tensor[idx]
        batch["EMtops_mask"] = self.emtops_mask[idx]
        if self.keep_unscaled_data:
            batch["EMtops_unscaled"] = self.emtops_unscaled_tensor[idx]
        return batch


class TTbarEventDataModule(pl.LightningDataModule):
    """`EventDatasetBase.EventDataModule`, but builds `TTbarEventDataset`
    instead of `EventDataset` -- same train/val stats-and-max_jets-fit-on-
    train-reused-on-val behaviour, extended to also fit/reuse `max_emtops`.
    See `EventDataModule`'s docstring for the rest (num_workers, device, ...).
    """

    def __init__(
        self, train_path, val_path, *, features, batch_size=256, num_workers=0,
        persistent_workers=False, keep_unscaled_data=False, max_jets=None, max_emtops=None,
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
        self.max_emtops = max_emtops

    def setup(self, stage=None):
        self.train_dataset = TTbarEventDataset(
            self.train_path, features=self.features,
            keep_unscaled_data=self.keep_unscaled_data, max_jets=self.max_jets, max_emtops=self.max_emtops,
        )  # fits stats (and, if not given, max_jets/max_emtops) on train
        self.stats = self.train_dataset.stats
        self.val_dataset = TTbarEventDataset(
            self.val_path, features=self.features, stats=self.stats,
            keep_unscaled_data=self.keep_unscaled_data,
            max_jets=self.train_dataset.max_jets, max_emtops=self.train_dataset.max_emtops,
        )  # reuse them
        if self.device is not None:
            self.train_dataset.to(self.device)
            self.val_dataset.to(self.device)
        print("Setup done")

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
