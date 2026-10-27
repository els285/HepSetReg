"""
Particle-Transformer-style backbone: same global/jet/lepton token setup as
`backbone.py`, but self-attention is biased with a learned pairwise
interaction term computed from each particle's 4-momentum (Qu, Li & Qian,
"Particle Transformer for Jet Tagging", https://arxiv.org/abs/2202.03772).

Standard self-attention treats jets/leptons as an unordered set with no
notion of how close two particles are in momentum space. The ParT bias adds
that back in directly: for every pair of particles (i, j), a handful of
physically meaningful pairwise features (angular separation, relative
transverse momentum, pairwise invariant mass) are embedded through a small
shared pointwise network into one bias value per attention head, and added
to the attention logits before softmax -- exactly like a learned relative-
position bias, but built from physics instead of sequence position. The same
bias is reused at every encoder layer (not recomputed per layer), matching
the paper.

Requires `EventVectorScalarDataset.EventDataset(..., keep_unscaled_data=True)`:
the pairwise features need real physical values (not z-scored ones), and
this backbone reads them from the `jet_vector_unscaled` / `lepton_vector_unscaled`
batch entries specifically. The (possibly scaled) `jet_vector`/`jet_scalar`/
`lepton_vector`/`lepton_scalar` entries are still used for the per-token
embedding as usual -- both vector and scalar features are relevant there,
only the pairwise bias is vector-only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Column order `ParticlePairwiseBias` assumes for any (B, N, 8) vector
# tensor it's given -- must match EventVectorScalarDataset.JET_VECTOR_FEATURES
# / LEPTON_VECTOR_FEATURES (both share this order; see the assert there).
VECTOR_COLUMNS = ["pt", "eta", "phi", "e", "m", "px", "py", "pz"]


def _mlp_embedder(in_features, d_model, hidden_dim=None, dropout=0.0):
    hidden_dim = hidden_dim or d_model
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, d_model),
    )


class AttentionPooling(nn.Module):
    """Learned-query attention pooling over a masked token sequence."""

    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, tokens, valid_mask):
        # tokens: (B, T, D), valid_mask: (B, T) bool, True = real token
        logits = self.score(tokens).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, float("-inf"))
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (tokens * weights).sum(dim=1)


class ParticlePairwiseBias(nn.Module):
    """
    Computes the ParT pairwise interaction bias for one set of particles.

    For every pair (i, j) in the (already padded) particle sequence, builds
    4 physically-motivated features from their 4-momenta:
      - log(deltaR_ij)      angular separation
      - log(kT_ij)          = log(min(pT_i, pT_j) * deltaR_ij)
      - log(z_ij)           = log(min(pT_i, pT_j) / (pT_i + pT_j))
      - log(m2_ij)          pairwise invariant mass squared, from the
                             summed 4-vector (E_i+E_j, p_i+p_j)

    and passes them through a small pointwise network (shared across all
    pairs, applied identically at every position) that outputs one bias
    value per attention head. This is implemented as a stack of 1x1 Conv1d
    layers over the flattened N*N pairs -- mathematically identical to a
    shared per-pair MLP, but expressed as batched matmuls over the whole
    pair grid at once (no Python loop over pairs, no loop over layers of
    the encoder that will reuse this bias).

    Cost: O(N^2) pairs per event. For this project's particle counts
    (max_jets + 2 leptons, typically ~10-20 tokens) that's a few hundred
    pairs/event -- negligible next to the O(N^2 * d_model) attention
    matmuls the encoder already does; the dominant cost of adding this bias
    at all is that PyTorch's fused/nested-tensor attention fast path can't
    be used with a custom float attn_mask. That fast path is already off
    in this project's encoders (`norm_first=True` disables it), so there's
    no extra regression from that here.
    """

    def __init__(self, nhead, hidden_dims=(64, 64, 64), eps=1e-8):
        super().__init__()
        self.nhead = nhead
        self.eps = eps

        layers = []
        in_channels = 4  # log(deltaR), log(kT), log(z), log(m2)
        for hidden in hidden_dims:
            layers += [
                nn.Conv1d(in_channels, hidden, kernel_size=1),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
            ]
            in_channels = hidden
        layers += [nn.Conv1d(in_channels, nhead, kernel_size=1)]
        self.embed = nn.Sequential(*layers)

    def forward(self, vector, valid_mask):
        """
        vector : (B, N, 8) float tensor, columns == VECTOR_COLUMNS, RAW
            (unscaled) physical units -- z-scored inputs would make
            deltaR/kT/z/m2 physically meaningless.
        valid_mask : (B, N) bool, True = real (non-padded) particle.

        Returns (B, nhead, N, N) float bias, exactly 0 for any pair
        touching a padded particle.
        """
        B, N, _ = vector.shape
        # the pairwise physics math is numerically sensitive (log of small
        # values, differences of squares for m2) -- always do it in fp32,
        # regardless of the ambient autocast precision the rest of the
        # model runs under.
        vector = vector.float()
        pt, eta, phi, e, m, px, py, pz = vector.unbind(dim=-1)

        deta = eta.unsqueeze(2) - eta.unsqueeze(1)  # (B, N, N)
        dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
        dphi = (dphi + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
        delta_r = torch.sqrt(deta**2 + dphi**2 + self.eps)

        pt_i = pt.unsqueeze(2)  # (B, N, 1)
        pt_j = pt.unsqueeze(1)  # (B, 1, N)
        pt_min = torch.minimum(pt_i, pt_j)  # broadcasts to (B, N, N)
        kt = pt_min * delta_r
        z = pt_min / (pt_i + pt_j + self.eps)

        e_sum = e.unsqueeze(2) + e.unsqueeze(1)
        px_sum = px.unsqueeze(2) + px.unsqueeze(1)
        py_sum = py.unsqueeze(2) + py.unsqueeze(1)
        pz_sum = pz.unsqueeze(2) + pz.unsqueeze(1)
        m2 = e_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
        m2 = torch.clamp(m2, min=self.eps)  # guards fp-error negatives near 0

        feats = torch.stack(
            [
                torch.log(delta_r + self.eps),
                torch.log(kt + self.eps),
                torch.log(z + self.eps),
                torch.log(m2),
            ],
            dim=1,
        )  # (B, 4, N, N)

        pair_valid = (valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)).unsqueeze(1)  # (B, 1, N, N)
        feats = feats * pair_valid.to(feats.dtype)  # padded pairs -> exactly 0 in, not log(eps)

        bias = self.embed(feats.reshape(B, 4, N * N)).reshape(B, self.nhead, N, N)
        # Conv/BatchNorm bias terms mean a 0 input doesn't guarantee a 0
        # output -- re-zero padded pairs after the network too.
        bias = bias * pair_valid.to(bias.dtype)
        return bias


class TransformerBackboneParT(nn.Module):
    """
    Parameters
    ----------
    n_global_features : int
        Width of the global feature vector (e.g. `len(GLOBAL_FEATURES)`).
    n_jet_vector_features, n_jet_scalar_features,
    n_lepton_vector_features, n_lepton_scalar_features : int
        Widths of the split jet/lepton feature groups, e.g.
        `len(JET_VECTOR_FEATURES)`, `len(JET_SCALAR_FEATURES)`, etc. from
        `EventVectorScalarDataset.py`. The per-token embedding sees
        vector+scalar concatenated; the pairwise bias sees vector only.
    d_model, nhead, num_layers, dim_feedforward, dropout :
        Standard `nn.TransformerEncoder` hyperparameters. `nhead` also
        sets how many per-head bias channels `ParticlePairwiseBias` learns.
    use_type_embedding : bool
        Add a learned embedding identifying which group (global/jet/lepton)
        a token came from, so self-attention can tell object types apart.
    pairwise_hidden_dims : tuple[int, ...]
        Hidden layer widths of the pairwise bias network (see
        `ParticlePairwiseBias`).
    """

    def __init__(
        self,
        n_global_features,
        n_jet_vector_features,
        n_jet_scalar_features,
        n_lepton_vector_features,
        n_lepton_scalar_features,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=None,
        dropout=0.1,
        use_type_embedding=True,
        pairwise_hidden_dims=(64, 64, 64),
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.global_embed = _mlp_embedder(n_global_features, d_model, dropout=dropout)
        self.jet_embed = _mlp_embedder(n_jet_vector_features + n_jet_scalar_features, d_model, dropout=dropout)
        self.lepton_embed = _mlp_embedder(
            n_lepton_vector_features + n_lepton_scalar_features, d_model, dropout=dropout
        )

        self.type_embedding = nn.Embedding(3, d_model) if use_type_embedding else None

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
        self.pairwise_bias = ParticlePairwiseBias(nhead, hidden_dims=pairwise_hidden_dims)
        self.output_norm = nn.LayerNorm(d_model)
        self.pool = AttentionPooling(d_model)
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, batch, return_tokens: bool = False):
        """
        batch : dict with keys "global" (B, F_g), "jet_vector" (B, N_j, F_jv),
            "jet_scalar" (B, N_j, F_js), "jet_vector_unscaled" (B, N_j, F_jv),
            "jet_mask" (B, N_j), "lepton_vector" (B, N_l, F_lv),
            "lepton_scalar" (B, N_l, F_ls), "lepton_vector_unscaled" (B, N_l, F_lv),
            "lepton_mask" (B, N_l) -- exactly what
            `EventVectorScalarDataset.EventDataset(..., keep_unscaled_data=True)`
            produces.
        return_tokens : bool
            If True, also return the full per-token encoded sequence and its
            validity mask (e.g. for a cross-attention decoder).

        Returns
        -------
        Tensor of shape (B, d_model): one pooled embedding per event. If
        `return_tokens` is True, instead returns
        `(pooled, encoded, valid_mask)`.
        """
        global_feat = batch["global"]
        jet_mask, lepton_mask = batch["jet_mask"], batch["lepton_mask"]

        batch_size = global_feat.shape[0]
        device = global_feat.device

        global_tok = self.global_embed(global_feat).unsqueeze(1)  # (B, 1, D)
        global_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        jet_in = torch.cat([batch["jet_vector"], batch["jet_scalar"]], dim=-1)
        lepton_in = torch.cat([batch["lepton_vector"], batch["lepton_scalar"]], dim=-1)
        jet_tok = self.jet_embed(jet_in)  # (B, N_j, D)
        lepton_tok = self.lepton_embed(lepton_in)  # (B, N_l, D)

        if self.type_embedding is not None:
            global_tok = global_tok + self.type_embedding.weight[0]
            jet_tok = jet_tok + self.type_embedding.weight[1]
            lepton_tok = lepton_tok + self.type_embedding.weight[2]

        # Zero out padded slots defensively -- see backbone.py.
        jet_tok = jet_tok * jet_mask.unsqueeze(-1).to(jet_tok.dtype)
        lepton_tok = lepton_tok * lepton_mask.unsqueeze(-1).to(lepton_tok.dtype)

        tokens = torch.cat([global_tok, jet_tok, lepton_tok], dim=1)
        valid_mask = torch.cat([global_mask, jet_mask, lepton_mask], dim=1)
        n_tokens = tokens.shape[1]

        # ---- pairwise interaction bias, jets+leptons only ----
        particle_vector = torch.cat([batch["jet_vector_unscaled"], batch["lepton_vector_unscaled"]], dim=1)
        particle_valid = torch.cat([jet_mask, lepton_mask], dim=1)
        particle_bias = self.pairwise_bias(particle_vector, particle_valid)  # (B, nhead, N_j+N_l, N_j+N_l)

        # global token has no 4-momentum -> zero interaction bias to/from it.
        # F.pad adds one zero row/col at the front of the last two dims,
        # i.e. exactly the "global" position at index 0 of `tokens`.
        full_bias = torch.nn.functional.pad(particle_bias, (1, 0, 1, 0), value=0.0)  # (B, nhead, T, T)

        # fold key-padding directly into the same float mask (rather than
        # also passing a separate bool src_key_padding_mask): mixing a bool
        # key_padding_mask with a float attn_mask is a deprecated path in
        # recent PyTorch, and this keeps the masking logic in one place.
        key_bias = torch.zeros(batch_size, n_tokens, dtype=full_bias.dtype, device=device)
        key_bias = key_bias.masked_fill(~valid_mask, float("-inf"))
        full_bias = full_bias + key_bias.view(batch_size, 1, 1, n_tokens)

        attn_mask = full_bias.reshape(batch_size * self.nhead, n_tokens, n_tokens).to(tokens.dtype)

        encoded = self.encoder(tokens, mask=attn_mask)
        encoded = self.output_norm(encoded)
        pooled = self.pool(encoded, valid_mask)

        if return_tokens:
            return pooled, encoded, valid_mask
        return pooled


# if __name__ == "__main__":
#     # feature counts only -- see run_regression_ParT.py for the actual field names
#     n_global, n_jet_vec, n_jet_sca, n_lep_vec, n_lep_sca = 6, 8, 1, 8, 2
#
#     backbone = TransformerBackboneParT(
#         n_global_features=n_global,
#         n_jet_vector_features=n_jet_vec,
#         n_jet_scalar_features=n_jet_sca,
#         n_lepton_vector_features=n_lep_vec,
#         n_lepton_scalar_features=n_lep_sca,
#         d_model=32, nhead=4, num_layers=2,
#     )
#
#     batch_size, n_jets, n_leptons = 8, 5, 2
#     dummy_batch = {
#         "global": torch.randn(batch_size, n_global),
#         "jet_vector": torch.randn(batch_size, n_jets, n_jet_vec),
#         "jet_scalar": torch.randn(batch_size, n_jets, n_jet_sca),
#         "jet_vector_unscaled": torch.randn(batch_size, n_jets, n_jet_vec).abs(),
#         "jet_mask": torch.ones(batch_size, n_jets, dtype=torch.bool),
#         "lepton_vector": torch.randn(batch_size, n_leptons, n_lep_vec),
#         "lepton_scalar": torch.randn(batch_size, n_leptons, n_lep_sca),
#         "lepton_vector_unscaled": torch.randn(batch_size, n_leptons, n_lep_vec).abs(),
#         "lepton_mask": torch.ones(batch_size, n_leptons, dtype=torch.bool),
#     }
#     pooled = backbone(dummy_batch)
#     print(f"pooled embedding shape: {tuple(pooled.shape)}")
