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

`FixedSlotAttention` below is the de-randomized variant: same competitive
attend/aggregate/GRU refinement, but slots start from a fixed, learned
per-slot embedding (like `decoder.py`'s `query_embed`) instead of a fresh
Gaussian sample every forward call, so slot 0 has a stable identity across
events -- suited to fixed-order supervision (e.g. slot 0 = top, slot 1 =
antitop), not permutation-invariant matching.
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

    def _init_slots(self, batch_size, device, dtype):
        """The one piece `FixedSlotAttention` overrides: a fresh Gaussian
        sample per forward call, which is what makes slot identity
        exchangeable between events (no slot is tied to any particular
        role)."""
        return self.slot_mu + self.slot_log_sigma.exp() * torch.randn(
            batch_size, self.n_slots, self.d_model, device=device, dtype=dtype
        )

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

        slots = self._init_slots(batch_size, device, inputs.dtype)

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


class FixedSlotAttention(SlotAttention):
    """De-randomized `SlotAttention`: identical competitive
    attend -> weighted-mean-aggregate -> GRU refinement, but slots start
    from a fixed, learned per-slot embedding (one per `n_slots`, shared
    across every event) instead of a fresh Gaussian sample each forward
    call -- so slot 0 has a stable, trainable identity across events, the
    same way `decoder.py`'s `CrossAttentionDecoder.query_embed` does.

    Use this (not `SlotAttention`) when the targets have a fixed, physically
    meaningful order (e.g. slot 0 = top, slot 1 = antitop) rather than an
    exchangeable/permutation-invariant one -- a plain per-slot loss then
    works directly, no Hungarian matching needed. It keeps `SlotAttention`'s
    competitive routing (tokens are "won" by one slot via softmax-over-
    slots), which is the real architectural difference from
    `CrossAttentionDecoder` -- fixed identity alone doesn't make the two
    equivalent.

    Same constructor signature as `SlotAttention`.
    """

    def __init__(self, n_slots, d_model, n_iters=3, d_hidden=None, eps=1e-8):
        super().__init__(n_slots, d_model, n_iters=n_iters, d_hidden=d_hidden, eps=eps)
        del self.slot_log_sigma  # no randomness left to parameterize

        # The base class's slot_mu is a single (1, 1, D) vector shared by
        # every slot -- fine there, since the random noise term is what
        # actually differentiates slots at init. With that noise gone, all
        # slots would start (and, with identical inputs, stay) completely
        # identical -- nothing left to break the symmetry. Replace it with
        # one DISTINCT learned embedding per slot instead, matching
        # decoder.py's CrossAttentionDecoder.query_embed exactly.
        del self.slot_mu
        self.slot_mu = nn.Parameter(torch.zeros(n_slots, d_model))
        nn.init.normal_(self.slot_mu, mean=0.0, std=0.02)

    def _init_slots(self, batch_size, device, dtype):
        return self.slot_mu.unsqueeze(0).expand(batch_size, self.n_slots, self.d_model).to(dtype)


if __name__ == "__main__":
    from hepsetreg.backbones.backbone import TransformerBackbone

    # feature counts only -- see any run_*.py for the actual field names
    n_global_features, n_jet_features, n_lepton_features = 6, 9, 10

    d_model = 32
    backbone = TransformerBackbone(
        n_global_features=n_global_features,
        n_jet_features=n_jet_features,
        n_lepton_features=n_lepton_features,
        d_model=d_model,
        nhead=4,
        num_layers=2,
    )
    slot_attn = SlotAttention(n_slots=5, d_model=d_model, n_iters=3)

    batch_size, n_jets, n_leptons = 8, 5, 2
    dummy_batch = {
        "global": torch.randn(batch_size, n_global_features),
        "jet": torch.randn(batch_size, n_jets, n_jet_features),
        "jet_mask": torch.ones(batch_size, n_jets, dtype=torch.bool),
        "lepton": torch.randn(batch_size, n_leptons, n_lepton_features),
        "lepton_mask": torch.ones(batch_size, n_leptons, dtype=torch.bool),
    }

    _, encoded, valid_mask = backbone(dummy_batch, return_tokens=True)
    slots, attn = slot_attn(encoded, valid_mask)
    print(f"slots shape: {tuple(slots.shape)}")
    print(f"attn shape: {tuple(attn.shape)}, per-token sum over slots ~= 1: {attn.sum(dim=1)[0, :3]}")

    fixed_slot_attn = FixedSlotAttention(n_slots=2, d_model=d_model, n_iters=3)
    fixed_slots, fixed_attn = fixed_slot_attn(encoded, valid_mask)
    print(f"\nFixedSlotAttention slots shape: {tuple(fixed_slots.shape)}")
    fixed_slots2, _ = fixed_slot_attn(encoded, valid_mask)
    print(f"deterministic across calls: {torch.equal(fixed_slots, fixed_slots2)}")
    print(f"slot 0 != slot 1: {not torch.allclose(fixed_slots[:, 0], fixed_slots[:, 1])}")
