"""EventDataset plus optional truth for the reconstruction heads."""

import awkward as ak
import numpy as np
import torch

from hepsetreg.datasets.EventDatasetBase import EventDataset, load_arrays, _require_fields

# Integer truth labels aligned with each measured collection, concatenated
# like the lepton features (electrons, then muons).
LABEL_FIELDS = {"jet": ["jet_labels"], "lepton": ["el_labels", "mu_labels"]}


class ReconstructionDataset(EventDataset):
    """Same inputs, scaling and padding as EventDataset.

    With `assignment` (an AssignmentTargets), batches also contain
    "assignment" [B, n_roles]: see AssignmentTargets.truth. Labels are read
    from targets.jet_labels, targets.el_labels and targets.mu_labels, with
    0 for other objects and a negative value where truth is unknown.
    Without it, only measured inputs are read, e.g. for prediction.
    """

    def __init__(self, path, *, features, assignment=None, stats=None, max_jets=None, keep_unscaled_data=False):
        features = {**features, "target": []}
        arrays = load_arrays(path, features)
        if not ak.all(ak.num(arrays[2], axis=1) == 2):
            raise ValueError(f"{path}: EventDataset requires exactly two leptons per event")
        super().__init__(arrays, features=features, stats=stats, max_jets=max_jets,
                         keep_unscaled_data=keep_unscaled_data)
        self.truth = {}
        if assignment is None:
            return
        required_sources = {source for source, _, _ in assignment.roles}
        required_fields = [field for source, names in LABEL_FIELDS.items()
                           if source in required_sources for field in names]
        targets = ak.from_parquet(path, columns=[f"targets.{f}" for f in required_fields])
        _require_fields(targets["targets"], required_fields, f"{path} targets")
        labels = {}
        for source, objects, mask in (("jet", arrays[1], self.jet_mask), ("lepton", arrays[2], self.lepton_mask)):
            if source not in required_sources:
                continue
            values = ak.concatenate([targets["targets"][f] for f in LABEL_FIELDS[source]], axis=1)
            if not ak.all(ak.num(values, axis=1) == ak.num(objects, axis=1)):
                raise ValueError(f"{path}: {LABEL_FIELDS[source]} are not aligned with the measured {source}s")
            if ak.to_numpy(ak.flatten(values)).dtype.kind not in "iu":
                raise ValueError(f"{path}: {LABEL_FIELDS[source]} must be integers")
            dense = ak.fill_none(ak.pad_none(values, mask.shape[1], clip=True), 0)
            labels[source] = (ak.to_numpy(dense), mask.numpy())
        self.truth["assignment"] = torch.from_numpy(assignment.truth(labels))

    def to(self, device):
        super().to(device)
        self.truth = {k: v.to(device) for k, v in self.truth.items()}
        return self

    def __getitem__(self, idx):
        return {**super().__getitem__(idx), **{k: v[idx] for k, v in self.truth.items()}}

    def __getitems__(self, indices):
        idx = torch.as_tensor(indices, device=self.global_tensor.device)
        return {**super().__getitems__(indices), **{k: v[idx] for k, v in self.truth.items()}}
