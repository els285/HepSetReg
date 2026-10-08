"""
SlotEventDataset: same padded/pre-scaled construction as EventDataset6, but
carries the per-slot (top, antitop) targets instead of the combined
ttbar_mass/pt/DR system-level target -- pairs with `SetEventRegressor` +
`CrossAttentionDecoder(n_queries=2)` (head.py/decoder.py) for fixed-slot
per-object regression.

The slot order is FIXED, not permutation-invariant: index 0 is always the
top, index 1 always the antitop (see prep_data.py, which reads
truth_parent[:, 0]/[:, 1] in that same fixed order from the truth record --
a real physical distinction, not an arbitrary combinatorial one). Because of
that, no Hungarian-matching loss is needed -- a plain per-slot MSE/Huber
(RegressionLoss, unchanged) works directly on the (B, 2, F) prediction and
target, since it's shape-agnostic.

Feature names are NOT defined here: the caller passes `features` (keys
"global", "jet", "lepton") and `slot_target_features` from its own run_*.py.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from EventDataset6 import _require_fields, load_arrays, zscore

SLOT_NAMES = ["top", "antitop"]  # fixed order, matches prep_data.py's truth_parent[:, 0]/[:, 1]


def load_slot_targets(parquet_path, slot_target_features):
    """Reads the per-slot (top, antitop) target fields -- fixed order --
    into a (n_events, 2, len(slot_target_features)) array."""
    events = ak.from_parquet(parquet_path)
    targets = events["targets"]
    required = [f"{slot}_{f}" for slot in SLOT_NAMES for f in slot_target_features]
    _require_fields(targets, required, f"{parquet_path} targets")
    return np.stack(
        [
            np.stack(
                [ak.to_numpy(targets[f"{slot}_{f}"]).astype(np.float32) for f in slot_target_features], axis=1
            )
            for slot in SLOT_NAMES
        ],
        axis=1,
    )  # (n_events, 2, len(slot_target_features))


class SlotEventDataset(Dataset):
    """
    Parameters
    ----------
    source : str or (global_arr, jet_arr, lepton_arr, slot_target_arr)
        A parquet path, or the tuple already split out (e.g. for a
        synthetic/test source, or to share one parse across train/val).
    features : dict
        Keys "global", "jet", "lepton" -- field names to use.
    slot_target_features : list[str]
        Per-slot target field names, e.g. ["px", "py", "pz", "E"].
    indices, stats, keep_unscaled_data, max_jets : see EventDataset6.EventDataset.
    """

    def __init__(
        self, source, *, features, slot_target_features,
        indices=None, stats=None, keep_unscaled_data=False, max_jets=None,
    ):
        if isinstance(source, str):
            global_arr, jet_arr, lepton_arr, _ = load_arrays(source, {**features, "target": []})
            slot_target_arr = load_slot_targets(source, slot_target_features)
        else:
            global_arr, jet_arr, lepton_arr, slot_target_arr = source

        if indices is None:
            indices = np.arange(len(global_arr))
        indices = np.asarray(indices)
        n_events = len(indices)

        global_arr = global_arr[indices]
        slot_target_arr = slot_target_arr[indices]
        jet_sub = jet_arr[indices]
        lepton_sub = lepton_arr[indices]
        self.keep_unscaled_data = keep_unscaled_data

        self.stats = stats or self._compute_stats(
            global_arr, jet_sub, lepton_sub, slot_target_arr, features,
        )
        g_mean, g_std = self.stats["global"]
        j_mean, j_std = self.stats["jet"]
        l_mean, l_std = self.stats["lepton"]
        t_mean, t_std = self.stats["target"]  # each (2, len(slot_target_features)) -- separate per slot

        # ---- global: fixed-size, zscore the whole array at once ----
        self.global_tensor = torch.from_numpy(zscore(global_arr, g_mean, g_std))
        if keep_unscaled_data:
            self.global_unscaled_tensor = torch.from_numpy(global_arr)

        # ---- target: (n_events, 2, F) -- broadcasts against (2, F) stats correctly ----
        self.target_tensor = torch.from_numpy(zscore(slot_target_arr, t_mean, t_std))
        if keep_unscaled_data:
            self.target_unscaled_tensor = torch.from_numpy(slot_target_arr)

        # ---- leptons: always exactly 2/event -> flatten, zscore, reshape ----
        lep_flat = np.stack(
            [ak.to_numpy(ak.flatten(lepton_sub[f])).astype(np.float32) for f in features["lepton"]], axis=1
        )
        self.lepton_tensor = torch.from_numpy(zscore(lep_flat, l_mean, l_std)).reshape(
            n_events, 2, len(features["lepton"])
        )
        self.lepton_mask = torch.ones(n_events, 2, dtype=torch.bool)
        if keep_unscaled_data:
            self.lepton_unscaled_tensor = torch.from_numpy(lep_flat).reshape(n_events, 2, len(features["lepton"]))

        # ---- jets: ragged -> pad/truncate to `max_jets` fixed slots, ONCE ----
        counts = ak.num(jet_sub, axis=1).to_numpy()
        self.max_jets = max_jets if max_jets is not None else int(counts.max())

        jet_dense = np.stack(
            [
                ak.to_numpy(ak.fill_none(ak.pad_none(jet_sub[f], self.max_jets, clip=True), 0.0)).astype(np.float32)
                for f in features["jet"]
            ],
            axis=-1,
        )  # (n_events, max_jets, n_jet_features)

        mask = np.arange(self.max_jets)[None, :] < counts[:, None]
        self.jet_mask = torch.from_numpy(mask)

        if keep_unscaled_data:
            self.jet_unscaled_tensor = torch.from_numpy(jet_dense.copy())

        jet_dense -= j_mean
        jet_dense /= j_std + 1e-8
        jet_dense[~mask] = 0.0
        self.jet_tensor = torch.from_numpy(jet_dense)

    @staticmethod
    def _compute_stats(global_arr, jet_arr, lepton_arr, slot_target_arr, features):
        jet_flat = np.stack(
            [ak.to_numpy(ak.flatten(jet_arr[f])).astype(np.float32) for f in features["jet"]], axis=1
        )
        lep_flat = np.stack(
            [ak.to_numpy(ak.flatten(lepton_arr[f])).astype(np.float32) for f in features["lepton"]], axis=1
        )
        return {
            "global": (global_arr.mean(0), global_arr.std(0)),
            "jet": (jet_flat.mean(0), jet_flat.std(0)),
            "lepton": (lep_flat.mean(0), lep_flat.std(0)),
            # mean/std over axis 0 (events) only -> (2, F): separate stats
            # per slot, since top/antitop are physically distinct particles.
            "target": (slot_target_arr.mean(0), slot_target_arr.std(0)),
        }

    def __len__(self):
        return self.global_tensor.shape[0]

    def __getitem__(self, idx):
        item = {
            "global": self.global_tensor[idx],
            "jet": self.jet_tensor[idx],
            "jet_mask": self.jet_mask[idx],
            "lepton": self.lepton_tensor[idx],
            "lepton_mask": self.lepton_mask[idx],
            "target": self.target_tensor[idx],  # (2, len(slot_target_features))
        }
        if self.keep_unscaled_data:
            item["global_unscaled"] = self.global_unscaled_tensor[idx]
            item["jet_unscaled"] = self.jet_unscaled_tensor[idx]
            item["lepton_unscaled"] = self.lepton_unscaled_tensor[idx]
            item["target_unscaled"] = self.target_unscaled_tensor[idx]
        return item

    def __getitems__(self, indices):
        """See EventDataset6.EventDataset.__getitems__: one vectorized
        fancy-index per field instead of a Python loop over the batch."""
        idx = torch.as_tensor(indices, device=self.global_tensor.device)
        batch = {
            "global": self.global_tensor[idx],
            "jet": self.jet_tensor[idx],
            "jet_mask": self.jet_mask[idx],
            "lepton": self.lepton_tensor[idx],
            "lepton_mask": self.lepton_mask[idx],
            "target": self.target_tensor[idx],
        }
        if self.keep_unscaled_data:
            batch["global_unscaled"] = self.global_unscaled_tensor[idx]
            batch["jet_unscaled"] = self.jet_unscaled_tensor[idx]
            batch["lepton_unscaled"] = self.lepton_unscaled_tensor[idx]
            batch["target_unscaled"] = self.target_unscaled_tensor[idx]
        return batch


def _identity_collate(batch):
    """No-op collate: `__getitems__` already returns a fully batched dict."""
    return batch


class SlotEventDataModule(pl.LightningDataModule):
    """Same design as EventDataset6.EventDataModule: fits stats (and
    max_jets) on train, reuses them for val; no custom collate_fn needed
    since every item is already a fixed shape."""

    def __init__(
        self, train_path, val_path, *, features, slot_target_features, batch_size=256, num_workers=0,
        persistent_workers=False, keep_unscaled_data=False, max_jets=None,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.features = features
        self.slot_target_features = slot_target_features
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.keep_unscaled_data = keep_unscaled_data
        self.max_jets = max_jets

    def setup(self, stage=None):
        self.train_dataset = SlotEventDataset(
            self.train_path, features=self.features, slot_target_features=self.slot_target_features,
            keep_unscaled_data=self.keep_unscaled_data, max_jets=self.max_jets,
        )
        self.stats = self.train_dataset.stats
        self.val_dataset = SlotEventDataset(
            self.val_path, features=self.features, slot_target_features=self.slot_target_features,
            stats=self.stats, keep_unscaled_data=self.keep_unscaled_data,
            max_jets=self.train_dataset.max_jets,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, persistent_workers=self.persistent_workers,
            pin_memory=True, collate_fn=_identity_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, persistent_workers=self.persistent_workers,
            pin_memory=True, collate_fn=_identity_collate,
        )
