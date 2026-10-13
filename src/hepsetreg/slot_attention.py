"""
Slot attention (Locatello et al., 2020): iteratively assigns input tokens to
a fixed set of N "slot" queries.

This differs from `decoder.py`'s `CrossAttentionDecoder` in how the
attention is normalized: there, each query independently runs a standard
softmax over the input tokens (competition is only ever among inputs, for a
given query). Here, the softmax is taken over the *slots* for each input
token -- so slots compete with each other for input tokens, which tends to
produce a cleaner one-token-to-one-slot routing (useful together with a
permutation-invariant loss over the slots, since slot identity is
exchangeable). Slots are refined over a few iterations of
attend -> weighted-mean-aggregate -> GRU update, rather than a single
attention pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SlotAttention(nn.Module):
    """
    Parameters
    ----------
    n_slots : int
        Number of slot queries.
    d_model : int
        Width of both input tokens and slots.
    n_iters : int
        Number of attend/aggregate/GRU-update refinement iterations.
    d_hidden : int or None
        Hidden width of the per-slot residual MLP (default 4 * d_model).
    eps : float
        Numerical-stability epsilon in the per-slot attention normalization.
    """

    def __init__(self, n_slots, d_model, n_iters=3, d_hidden=None, eps=1e-8):
        super().__init__()
        self.n_slots = n_slots
        self.d_model = d_model
        self.n_iters = n_iters
        self.eps = eps
        self.scale = d_model**-0.5

        self.slot_mu = nn.Parameter(torch.zeros(1, 1, d_model))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        self.norm_input = nn.LayerNorm(d_model)
        self.norm_slots = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)

        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        self.gru = nn.GRUCell(d_model, d_model)

        d_hidden = d_hidden or 4 * d_model
        self.mlp = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model))

    def forward(self, inputs, valid_mask=None):
        """
        inputs : (B, T, D) input tokens -- e.g. a backbone's per-token
            encoded sequence, from `forward(batch, return_tokens=True)`.
        valid_mask : (B, T) bool or None, True = real token, False = padding.

        Returns
        -------
        slots : (B, n_slots, D)
        attn : (B, n_slots, T), the final input-normalized attention weights
            -- how much of each input token was routed to each slot.
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        slots = self.slot_mu + self.slot_log_sigma.exp() * torch.randn(
            batch_size, self.n_slots, self.d_model, device=device, dtype=inputs.dtype
        )

        attn = None
        for _ in range(self.n_iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))  # (B, n_slots, D)

            logits = torch.einsum("bsd,btd->bst", q, k) * self.scale  # (B, n_slots, T)
            attn = torch.softmax(logits, dim=1)  # competition over SLOTS, per input token

            if valid_mask is not None:
                attn = attn * valid_mask.unsqueeze(1).to(attn.dtype)  # padded tokens route nowhere

            # Per-slot weighted mean over the (real) input tokens it won.
            attn_weighted = attn / (attn.sum(dim=2, keepdim=True) + self.eps)
            updates = torch.einsum("bst,btd->bsd", attn_weighted, v)  # (B, n_slots, D)

            slots = self.gru(
                updates.reshape(-1, self.d_model), slots_prev.reshape(-1, self.d_model)
            ).reshape(batch_size, self.n_slots, self.d_model)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn


if __name__ == "__main__":
    from backbone import TransformerBackbone
    from EventDataset5 import GLOBAL_FEATURES, JET_FEATURES, LEPTON_FEATURES

    d_model = 32
    backbone = TransformerBackbone(
        n_global_features=len(GLOBAL_FEATURES),
        n_jet_features=len(JET_FEATURES),
        n_lepton_features=len(LEPTON_FEATURES),
        d_model=d_model,
        nhead=4,
        num_layers=2,
    )
    slot_attn = SlotAttention(n_slots=5, d_model=d_model, n_iters=3)

    batch_size, n_jets, n_leptons = 8, 5, 2
    dummy_batch = {
        "global": torch.randn(batch_size, len(GLOBAL_FEATURES)),
        "jet": torch.randn(batch_size, n_jets, len(JET_FEATURES)),
        "jet_mask": torch.ones(batch_size, n_jets, dtype=torch.bool),
        "lepton": torch.randn(batch_size, n_leptons, len(LEPTON_FEATURES)),
        "lepton_mask": torch.ones(batch_size, n_leptons, dtype=torch.bool),
    }

    _, encoded, valid_mask = backbone(dummy_batch, return_tokens=True)
    slots, attn = slot_attn(encoded, valid_mask)
    print(f"slots shape: {tuple(slots.shape)}")
    print(f"attn shape: {tuple(attn.shape)}, per-token sum over slots ~= 1: {attn.sum(dim=1)[0, :3]}")
