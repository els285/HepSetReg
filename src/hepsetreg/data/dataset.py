"""HDF5-backed dataset of pre-padded, masked object sets.

Expected file layout (see :mod:`hepsetreg.data.preprocessing` for a helper
that builds files in this layout from jagged/variable-length per-event
arrays)::

    /{group}/features   float32 (n_events, n_group_max, n_group_features)
    /{group}/mask       bool    (n_events, n_group_max)     True = real object
    /target             float32 (n_events, n_targets)
    /extras/{name}      any     (n_events, ...)             optional, passed
                                                             through to the
                                                             batch dict for
                                                             physics-consistency
                                                             loss hooks.

This generalizes DIRECTOR's flat ``X``/``Y``/``M`` HDF5 convention: ``X`` (one
flat feature vector) becomes one padded array + mask per object group, and
``M`` (the ttbar-specific mass array) becomes an arbitrary named ``extras``
entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import torch
from torch.utils.data import Dataset


class PaddedObjectDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        group_names: List[str],
        target_key: str = "target",
        extra_keys: Optional[List[str]] = None,
    ):
        self.path = str(path)
        self.group_names = list(group_names)
        self.target_key = target_key
        self.extra_keys = list(extra_keys or [])
        self._file: Optional[h5py.File] = None

        with h5py.File(self.path, "r") as f:
            for name in self.group_names:
                if f"{name}/features" not in f or f"{name}/mask" not in f:
                    raise KeyError(
                        f"HDF5 file '{self.path}' is missing '{name}/features' or '{name}/mask'."
                    )
            if self.target_key not in f:
                raise KeyError(f"HDF5 file '{self.path}' is missing target dataset '{self.target_key}'.")
            self._length = f[self.target_key].shape[0]
            for name in self.extra_keys:
                if f"extras/{name}" not in f:
                    raise KeyError(f"HDF5 file '{self.path}' is missing 'extras/{name}'.")

    def __len__(self) -> int:
        return self._length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None  # h5py.File handles are not picklable across DataLoader workers
        return state

    @property
    def _h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getitem__(self, idx: int) -> Dict:
        f = self._h5
        objects = {name: torch.from_numpy(f[f"{name}/features"][idx]).float() for name in self.group_names}
        mask = {name: torch.from_numpy(f[f"{name}/mask"][idx]).bool() for name in self.group_names}
        target = torch.from_numpy(f[self.target_key][idx]).float()
        item = {"objects": objects, "mask": mask, "target": target}
        for name in self.extra_keys:
            item[name] = torch.from_numpy(f[f"extras/{name}"][idx]).float()
        return item
