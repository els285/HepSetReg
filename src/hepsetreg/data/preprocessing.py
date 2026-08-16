"""Helpers for turning jagged, variable-length per-event object arrays into
the padded ``features``/``mask`` HDF5 layout :class:`~hepsetreg.data.dataset.PaddedObjectDataset`
expects.

This is the direct generalization of DIRECTOR's ``preprocess.py``: instead of
writing a fixed-width flat ``X`` row per event (padding missing jets with
hard-coded zero *columns*, capping the jet count implicitly at whatever the
column layout allows), each object group gets its own padded array plus a
boolean mask recording which entries are real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np


def pad_and_mask(
    events: List[np.ndarray],
    max_count: Optional[int] = None,
    feature_dim: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pads a list of per-event ``(n_objects_i, feature_dim)`` arrays.

    Parameters
    ----------
    events:
        One array per event; ``n_objects_i`` may differ freely between events.
    max_count:
        Width to pad to. Defaults to the largest ``n_objects_i`` in ``events``.
        Events with more objects than ``max_count`` are truncated (a warning
        is not raised here -- pick ``max_count`` deliberately).
    feature_dim:
        Per-object feature width. Inferred from the first non-empty event if
        not given.

    Returns
    -------
    features: ``(n_events, max_count, feature_dim)`` float32, zero-padded.
    mask: ``(n_events, max_count)`` bool, ``True`` where an entry is real.
    """
    n_events = len(events)
    if feature_dim is None:
        non_empty = next((e for e in events if e.shape[0] > 0), None)
        if non_empty is None:
            raise ValueError("Cannot infer feature_dim: every event has zero objects.")
        feature_dim = non_empty.shape[1]
    if max_count is None:
        max_count = max((e.shape[0] for e in events), default=0)

    features = np.zeros((n_events, max_count, feature_dim), dtype=np.float32)
    mask = np.zeros((n_events, max_count), dtype=bool)
    for i, event in enumerate(events):
        n = min(event.shape[0], max_count)
        if n > 0:
            features[i, :n] = event[:n]
            mask[i, :n] = True
    return features, mask


def write_padded_hdf5(
    path: Union[str, Path],
    groups: Dict[str, Tuple[np.ndarray, np.ndarray]],
    target: np.ndarray,
    extras: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Writes one dataset file in the layout :class:`PaddedObjectDataset` expects.

    ``groups`` maps object-group name -> ``(features, mask)`` as returned by
    :func:`pad_and_mask`.
    """
    with h5py.File(path, "w") as f:
        for name, (features, mask) in groups.items():
            f.create_dataset(f"{name}/features", data=np.asarray(features, dtype=np.float32))
            f.create_dataset(f"{name}/mask", data=np.asarray(mask, dtype=bool))
        f.create_dataset("target", data=np.asarray(target, dtype=np.float32))
        if extras:
            for name, arr in extras.items():
                f.create_dataset(f"extras/{name}", data=np.asarray(arr, dtype=np.float32))


def write_scaler_hdf5(path: Union[str, Path], mean: np.ndarray, scale: np.ndarray) -> None:
    """Writes a ``Y_mean``/``Y_scale`` file, matching DIRECTOR's scaler convention."""
    with h5py.File(path, "w") as f:
        f.create_dataset("Y_mean", data=np.asarray(mean, dtype=np.float32))
        f.create_dataset("Y_scale", data=np.asarray(scale, dtype=np.float32))
