"""The shared transformer-encoder backbone.

Generalizes both reference repos: DIRECTOR's per-feature-group linear
projections + MHA/MLA transformer encoder + attention pooling, and
ReconstructionAndSB's padded-token + type-embedding + key-padding-mask
transformer encoder -- but with a variable number of objects per group
instead of a fixed flat feature vector.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from hepsetreg.models.pooling import build_pooling
from hepsetreg.models.tokenizer import ObjectGroupSpec, ObjectTokenizer


class ObjectSetEncoder(nn.Module):
    """Encodes a variable-length set of physics objects into one event embedding.

    Parameters
    ----------
    groups:
        List of :class:`ObjectGroupSpec`, one per object type.
    d_model, nhead, num_layers, dim_feedforward, dropout:
        Standard ``nn.TransformerEncoder`` hyperparameters.
    pooling:
        ``"cls"``, ``"mean"``, or ``"attention"``.
    use_type_embedding:
        Add a learned embedding identifying which object group (and, for CLS
        pooling, the CLS token itself) each token came from.
    """

    def __init__(
        self,
        groups: List[ObjectGroupSpec],
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        pooling: str = "cls",
        use_type_embedding: bool = True,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        pooling = pooling.lower()
        if pooling not in {"cls", "mean", "attention"}:
            raise ValueError("pooling must be one of 'cls', 'mean', 'attention'.")

        n_extra_types = 1 if (pooling == "cls" and use_type_embedding) else 0
        self.tokenizer = ObjectTokenizer(
            groups, d_model=d_model, use_type_embedding=use_type_embedding, n_extra_types=n_extra_types
        )
        self._cls_type_id = len(groups) if n_extra_types else None

        self.pooling_kind = pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        dim_feedforward = dim_feedforward or 4 * d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.pool = build_pooling(pooling, d_model)
        self.d_model = d_model

    def forward(
        self,
        objects: Dict[str, Tensor],
        mask: Dict[str, Tensor],
        return_tokens: bool = False,
    ):
        tokens, valid_mask = self.tokenizer(objects, mask)

        if self.cls_token is not None:
            batch_size = tokens.shape[0]
            cls = self.cls_token.expand(batch_size, -1, -1)
            if self.tokenizer.type_embedding is not None:
                cls_type_ids = torch.full(
                    (batch_size, 1), self._cls_type_id, dtype=torch.long, device=tokens.device
                )
                cls = cls + self.tokenizer.type_embedding(cls_type_ids)
            tokens = torch.cat([cls, tokens], dim=1)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
            valid_mask = torch.cat([cls_mask, valid_mask], dim=1)

        if not valid_mask.any(dim=1).all():
            # Every real transformer needs at least one attendable key per row;
            # guard against fully-empty events (e.g. an event with zero jets and
            # no CLS token) producing NaNs from an all -inf softmax.
            raise ValueError(
                "Found an event with no valid tokens at all (mask is all-False). "
                "Add a CLS/global token or ensure every event has >=1 real object."
            )

        key_padding_mask = ~valid_mask
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        encoded = self.output_norm(encoded)
        pooled = self.pool(encoded, valid_mask)

        if return_tokens:
            return pooled, encoded, valid_mask
        return pooled
