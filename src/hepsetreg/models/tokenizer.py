"""Turns heterogeneous, variable-count sets of physics objects into a single
masked token sequence.

This is the piece that fixes DIRECTOR's main limitation: instead of slicing a
flat feature vector into fixed ``feature_groups`` column ranges (one linear
projection per *slot*, so the number of jets etc. had to be fixed and padded
with hard-coded zero columns), each *object type* ("jets", "leptons", "MET",
...) gets ONE shared linear projection applied to every instance of that
type. The number of instances per event is arbitrary; padding is handled
with a boolean mask that is threaded through to the transformer encoder's
``src_key_padding_mask`` (and to masked pooling), so padded slots never
influence the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ObjectGroupSpec:
    """Describes one object type ("jets", "leptons", "met", "global", ...).

    Parameters
    ----------
    name:
        Key used to look this group up in the ``objects``/``mask`` dicts
        passed to the model.
    in_features:
        Width of the raw per-object feature vector for this group (e.g. 5 for
        (pt, eta, phi, E, b-tag)). Every instance of the group shares one
        ``nn.Linear(in_features, d_model)`` projection, so this must be
        constant across instances -- the *count* of instances is free to
        vary per event.
    """

    name: str
    in_features: int


class ObjectTokenizer(nn.Module):
    """Projects ``{group_name: (B, N_group, F_group)}`` into one token stream.

    Returns ``tokens`` of shape ``(B, sum(N_group), d_model)`` and a boolean
    ``valid_mask`` of shape ``(B, sum(N_group))`` where ``True`` means "real
    object" (as opposed to padding). Groups are concatenated in the order
    they were given at construction time, so callers must supply ``objects``
    and ``mask`` with consistent per-event object counts across the batch
    (i.e. already padded to a common per-group width -- see
    :mod:`hepsetreg.data.preprocessing`).

    ``n_extra_types`` reserves additional type-embedding ids for tokens the
    caller will add itself (e.g. a CLS token, or the time/xt tokens used by
    :class:`hepsetreg.models.flow_matching.ConditionalVelocityField`).
    """

    def __init__(
        self,
        groups: List[ObjectGroupSpec],
        d_model: int,
        use_type_embedding: bool = True,
        n_extra_types: int = 0,
    ):
        super().__init__()
        if len(groups) == 0:
            raise ValueError("ObjectTokenizer requires at least one object group.")
        names = [g.name for g in groups]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate object group names: {names}")

        self.group_order: List[str] = names
        self.projections = nn.ModuleDict(
            {g.name: nn.Linear(g.in_features, d_model) for g in groups}
        )
        n_types = len(groups) + n_extra_types
        self.type_embedding = nn.Embedding(n_types, d_model) if use_type_embedding else None
        self.d_model = d_model

    def forward(
        self, objects: Dict[str, Tensor], mask: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        token_chunks = []
        mask_chunks = []
        for type_id, name in enumerate(self.group_order):
            if name not in objects or name not in mask:
                raise KeyError(
                    f"ObjectTokenizer expected object group '{name}' in `objects`/`mask`; "
                    f"got objects={list(objects)} mask={list(mask)}."
                )
            feats = objects[name]
            valid = mask[name]
            if feats.dim() != 3:
                raise ValueError(
                    f"objects['{name}'] must have shape (batch, n_objects, n_features), got {tuple(feats.shape)}."
                )
            if valid.shape != feats.shape[:2]:
                raise ValueError(
                    f"mask['{name}'] shape {tuple(valid.shape)} does not match "
                    f"objects['{name}'] shape {tuple(feats.shape)}[:2]."
                )

            proj = self.projections[name](feats)
            if self.type_embedding is not None:
                type_ids = torch.full(
                    valid.shape, type_id, dtype=torch.long, device=feats.device
                )
                proj = proj + self.type_embedding(type_ids)

            # Zero out padded slots defensively: they are excluded from attention
            # via the key-padding mask below, but masked mean-pooling and any
            # downstream numerical hygiene benefits from them being exactly zero.
            proj = proj * valid.unsqueeze(-1).to(proj.dtype)

            token_chunks.append(proj)
            mask_chunks.append(valid.bool())

        tokens = torch.cat(token_chunks, dim=1)
        valid_mask = torch.cat(mask_chunks, dim=1)
        return tokens, valid_mask
