
from __future__ import annotations

import torch.nn as nn


class EventRegressor(nn.Module):
    """Wires a TransformerBackbone + RegressionHead together.

    `forward(batch)` takes the exact dict produced by
    `EventDataset5.collate_fn` and returns (B, output_dim) predictions.
    """

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, batch):
        pooled = self.backbone(batch)
        return self.head(pooled)


class SetEventRegressor(nn.Module):
    """Wires a TransformerBackbone(_CLS) + CrossAttentionDecoder + RegressionHead
    together for per-query set prediction.

    `RegressionHead` is applied independently -- with weights shared across
    queries -- to each of the decoder's N query embeddings: `nn.Linear` (and
    the `GELU`/`Dropout` in between) only act on the last dimension, so
    feeding it a `(B, N, d_model)` tensor instead of `(B, d_model)` naturally
    yields `(B, N, output_dim)`, no per-query loop required. Intended for use
    with a permutation-invariant loss (e.g. Hungarian matching) over the N
    predictions.

    `forward(batch)` takes the exact dict produced by
    `EventDataset5.collate_fn` and returns (B, n_queries, output_dim)
    predictions.
    """

    def __init__(self, backbone, decoder, head):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.head = head

    def forward(self, batch):
        _, encoded, valid_mask = self.backbone(batch, return_tokens=True)
        queries = self.decoder(encoded, valid_mask)  # (B, N, d_model)
        return self.head(queries)  # (B, N, output_dim)


class SlotEventRegressor(nn.Module):
    """Wires a TransformerBackbone(_CLS) + SlotAttention + RegressionHead
    together for per-slot set prediction.

    Same idea as `SetEventRegressor`, but the query embeddings come from
    `SlotAttention` (slots *compete* for input tokens via softmax-over-slots
    attention, refined over a few GRU iterations) instead of independent
    cross-attention queries. `RegressionHead` is again applied with weights
    shared across slots, so `(B, n_slots, d_model)` in gives
    `(B, n_slots, output_dim)` out -- intended for use with a
    permutation-invariant loss over the n_slots predictions.

    `forward(batch)` takes the exact dict produced by
    `EventDataset5.collate_fn` and returns (B, n_slots, output_dim)
    predictions.
    """

    def __init__(self, backbone, slot_attention, head):
        super().__init__()
        self.backbone = backbone
        self.slot_attention = slot_attention
        self.head = head

    def forward(self, batch):
        _, encoded, valid_mask = self.backbone(batch, return_tokens=True)
        slots, _ = self.slot_attention(encoded, valid_mask)  # (B, n_slots, d_model)
        return self.head(slots)  # (B, n_slots, output_dim)
