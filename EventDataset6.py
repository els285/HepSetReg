"""
EventDataset6: takes PreCachedDataset's "scale everything once, up front"
idea all the way -- jets are also PADDED to a fixed `max_jets` slots per
event at construction, using vectorized awkward calls, not a per-batch
Python loop.

Every item this Dataset returns already has a FIXED shape (unlike
EventDataset5's ragged-per-event `jet` tensor), so a DataLoader can stack a
batch with PyTorch's own default collation -- no custom `collate_fn`, no
per-batch padding cost during training at all.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from EventDataset5 import GLOBAL_FEATURES, JET_FEATURES, LEPTON_FEATURES, load_arrays, zscore


class EventDataset(Dataset):
    """
    Parameters
    ----------
    source, indices, stats, keep_unscaled_data : see EventDataset5.EventDataset.
    max_jets : int or None
        Fixed jet-slot count every event is padded/truncated to. None (the
        default) auto-detects the largest jet count actually present in
        `source` -- no wasted padding, no risk of silently truncating a
        real jet.
    """

    def __init__(self, source, indices=None, stats=None, keep_unscaled_data=False, max_jets=None):
        global_arr, jet_arr, lepton_arr, target_arr = (
            load_arrays(source) if isinstance(source, str) else source
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
        self.stats = stats or self._compute_stats(global_arr, jet_sub, lepton_sub, target_arr)
        g_mean, g_std = self.stats["global"]
        j_mean, j_std = self.stats["jet"]
        l_mean, l_std = self.stats["lepton"]
        t_mean, t_std = self.stats["target"]

        # ---- global / target: already fixed-size, zscore the whole array at once ----
        self.global_tensor = torch.from_numpy(zscore(global_arr, g_mean, g_std))
        self.target_tensor = torch.from_numpy(zscore(target_arr, t_mean, t_std))
        if keep_unscaled_data:
            self.global_unscaled_tensor = torch.from_numpy(global_arr)
            self.target_unscaled_tensor = torch.from_numpy(target_arr)

        # ---- leptons: always exactly 2/event -> flatten, zscore, reshape ----
        lep_flat = np.stack(
            [ak.to_numpy(ak.flatten(lepton_sub[f])).astype(np.float32) for f in LEPTON_FEATURES], axis=1
        )
        self.lepton_tensor = torch.from_numpy(zscore(lep_flat, l_mean, l_std)).reshape(
            n_events, 2, len(LEPTON_FEATURES)
        )
        # always exactly 2 real leptons/event -> mask is trivially all-True,
        # but TransformerBackbone.forward reads batch["lepton_mask"]
        # unconditionally, so it still needs to be present.
        self.lepton_mask = torch.ones(n_events, 2, dtype=torch.bool)
        if keep_unscaled_data:
            self.lepton_unscaled_tensor = torch.from_numpy(lep_flat).reshape(n_events, 2, len(LEPTON_FEATURES))

        # ---- jets: ragged -> pad/truncate to `max_jets` fixed slots, ONCE ----
        counts = ak.num(jet_sub, axis=1).to_numpy()
        self.max_jets = max_jets if max_jets is not None else int(counts.max())

        # One vectorized pad+fill+convert call per feature (9 features, not
        # 9 * n_events) -- ak.pad_none pads every event's jet list up to
        # max_jets with None (truncating longer ones, since clip=True);
        # ak.fill_none turns those into 0.0.
        jet_dense = np.stack(
            [
                ak.to_numpy(ak.fill_none(ak.pad_none(jet_sub[f], self.max_jets, clip=True), 0.0)).astype(np.float32)
                for f in JET_FEATURES
            ],
            axis=-1,
        )  # (n_events, max_jets, n_jet_features)

        mask = np.arange(self.max_jets)[None, :] < counts[:, None]  # True = real jet
        self.jet_mask = torch.from_numpy(mask)

        if keep_unscaled_data:
            self.jet_unscaled_tensor = torch.from_numpy(jet_dense.copy())

        # In-place zscore: jets are the biggest tensor here, so avoid
        # allocating a second full-size (n_events, max_jets, n_features)
        # array just to hold the scaled copy.
        jet_dense -= j_mean
        jet_dense /= j_std + 1e-8
        jet_dense[~mask] = 0.0  # padded slots stay exactly 0, not zscore(0)
        self.jet_tensor = torch.from_numpy(jet_dense)

    @staticmethod
    def _compute_stats(global_arr, jet_arr, lepton_arr, target_arr):
        jet_flat = np.stack(
            [ak.to_numpy(ak.flatten(jet_arr[f])).astype(np.float32) for f in JET_FEATURES], axis=1
        )
        lep_flat = np.stack(
            [ak.to_numpy(ak.flatten(lepton_arr[f])).astype(np.float32) for f in LEPTON_FEATURES], axis=1
        )
        return {
            "global": (global_arr.mean(0), global_arr.std(0)),
            "jet": (jet_flat.mean(0), jet_flat.std(0)),
            "lepton": (lep_flat.mean(0), lep_flat.std(0)),
            "target": (target_arr.mean(0), target_arr.std(0)),
        }

    def __len__(self):
        return self.global_tensor.shape[0]

    def __getitem__(self, idx):
        item = {
            "global": self.global_tensor[idx],
            "jet": self.jet_tensor[idx],  # (max_jets, n_jet_features), fixed shape
            "jet_mask": self.jet_mask[idx],  # (max_jets,)
            "lepton": self.lepton_tensor[idx],
            "lepton_mask": self.lepton_mask[idx],  # (2,), always True
            "target": self.target_tensor[idx],
        }
        if self.keep_unscaled_data:
            item["global_unscaled"] = self.global_unscaled_tensor[idx]
            item["jet_unscaled"] = self.jet_unscaled_tensor[idx]
            item["lepton_unscaled"] = self.lepton_unscaled_tensor[idx]
            item["target_unscaled"] = self.target_unscaled_tensor[idx]
        return item

    def __getitems__(self, indices):
        """Batched fetch: when a `Dataset` defines this, `DataLoader` calls
        it ONCE per batch with the whole index list, instead of calling
        `__getitem__` once per index (see
        `torch.utils.data._utils.fetch._MapDatasetFetcher`). Since
        everything here is already a dense tensor, a batch is just one
        vectorized fancy-index per field -- no Python-level loop over the
        batch at all, unlike the default `[dataset[i] for i in indices]` +
        `default_collate` path (measured ~13x faster at batch_size=4096).

        Returns the batch already collated as a dict of stacked tensors --
        `EventDataModule` pairs this with an identity `collate_fn` so
        nothing tries to re-collate it.
        """
        idx = torch.as_tensor(indices)
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
    """No-op collate: `EventDataset.__getitems__` already returns a fully
    batched dict, so nothing further needs collating. A plain module-level
    function (not a lambda) so it can be pickled to DataLoader workers when
    num_workers>0."""
    return batch


class EventDataModule(pl.LightningDataModule):
    """Separate train/val parquet files. Fits the Z-score stats (inputs AND
    targets) on the train file only, and reuses those exact stats for val
    (no leakage from val into the scaling). `max_jets` is likewise fit on
    train and reused for val, so both splits pad to the same jet-slot count.

    No `collate_fn` is passed to either DataLoader: every item this
    Dataset's `__getitem__` returns is already a fixed shape, so PyTorch's
    own default collation handles batching.

    `num_workers` defaults to 0 -- with EventDataset6, `__getitem__` is
    pure tensor indexing (even cheaper than EventDataset5's
    PreCachedDataset), so the case for multiprocess workers is weaker than
    ever; a prior attempt at num_workers>0 + per-worker thread-capping on
    EventDataset5's DataLoader made real training slower, not faster (the
    IPC cost of shipping batches back from worker processes outweighed the
    now-tiny per-item work). Left configurable, but not defaulted on.

    `persistent_workers` only has an effect when `num_workers>0` (PyTorch
    raises otherwise if left True at num_workers=0, so it's silently
    ignored here instead). If you do use num_workers>0, this matters more
    than usual: without it, DataLoader tears down and re-forks worker
    processes every epoch, and forking after CUDA is already initialized in
    the main process (which Lightning does before the first dataloader
    iteration) can hang -- this is what caused an earlier "epochs never
    start" issue. persistent_workers=True forks workers once, up front,
    avoiding repeated forks into that danger zone.
    """

    def __init__(
        self, train_path, val_path, batch_size=256, num_workers=0,
        persistent_workers=False, keep_unscaled_data=False, max_jets=None,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers and num_workers > 0
        self.keep_unscaled_data = keep_unscaled_data
        self.max_jets = max_jets

    def setup(self, stage=None):
        self.train_dataset = EventDataset(
            self.train_path, keep_unscaled_data=self.keep_unscaled_data, max_jets=self.max_jets,
        )  # fits stats (and, if not given, max_jets) on train
        self.stats = self.train_dataset.stats
        self.val_dataset = EventDataset(
            self.val_path, stats=self.stats, keep_unscaled_data=self.keep_unscaled_data,
            max_jets=self.train_dataset.max_jets,
        )  # reuse them
        print("Setup done")

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


# if __name__ == "__main__":
#     import time

#     n = 5000
#     rng = np.random.default_rng(0)
#     target_features = ["ttbar_mass", "ttbar_pt", "ttbar_DR"]
#     global_arr = rng.normal(size=(n, len(GLOBAL_FEATURES))).astype(np.float32)
#     target_arr = rng.normal(size=(n, len(target_features))).astype(np.float32)

#     def ragged(feats, counts):
#         return ak.zip({f: ak.Array([rng.normal(size=c).tolist() for c in counts]) for f in feats})

#     counts = rng.integers(2, 14, size=n)
#     jet_arr = ragged(JET_FEATURES, counts)
#     lepton_arr = ragged(LEPTON_FEATURES, np.full(n, 2))
#     source = (global_arr, jet_arr, lepton_arr, target_arr)

#     start = time.perf_counter()
#     ds = EventDataset(source, keep_unscaled_data=True)
#     print(f"construction: {(time.perf_counter() - start) * 1000:.1f} ms for {n} events")
#     print(f"auto-detected max_jets: {ds.max_jets} (true max was {counts.max()})")

#     item0 = ds[0]
#     print("item shapes:", {k: tuple(v.shape) for k, v in item0.items()})

#     from torch.utils.data import DataLoader

#     loader = DataLoader(ds, batch_size=64, shuffle=True)  # no collate_fn needed
#     batch = next(iter(loader))
#     print("batch shapes (default collate, no custom collate_fn):", {k: tuple(v.shape) for k, v in batch.items()})

#     # EventDataModule accepts a (global, jet, lepton, target) tuple directly
#     # (same as EventDataset), so this smoke test doesn't need real files.
#     n_val = 800
#     val_counts = rng.integers(2, 14, size=n_val)
#     val_source = (
#         rng.normal(size=(n_val, len(GLOBAL_FEATURES))).astype(np.float32),
#         ragged(JET_FEATURES, val_counts),
#         ragged(LEPTON_FEATURES, np.full(n_val, 2)),
#         rng.normal(size=(n_val, len(target_features))).astype(np.float32),
#     )

#     dm = EventDataModule(source, val_source, batch_size=64)
#     dm.setup()
#     print(f"train events: {len(dm.train_dataset)}, val events: {len(dm.val_dataset)}")
#     print("val reuses train stats:", dm.val_dataset.stats is dm.train_dataset.stats)
#     print("val reuses train max_jets:", dm.val_dataset.max_jets == dm.train_dataset.max_jets)

#     train_batch = next(iter(dm.train_dataloader()))
#     val_batch = next(iter(dm.val_dataloader()))
#     print("train batch jet shape:", tuple(train_batch["jet"].shape))
#     print("val batch jet shape:  ", tuple(val_batch["jet"].shape))
