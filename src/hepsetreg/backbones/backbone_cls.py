"""
Simple transformer backbone for the global/jet/lepton event format produced
by `EventDataset5.collate_fn` -- CLS-token variant of `backbone.py`.

Each object type gets its own MLP embedder into a shared `d_model` space:
  - `global` is a single fixed-size vector per event -> one always-valid token
  - `jet` / `lepton` are padded, variable-length collections -> one token per
    (real) object, with padded slots excluded via the attention mask

A learned CLS token is prepended to the sequence. All tokens are run through
a stack of `nn.TransformerEncoder` self-attention layers (padding-aware),
and the encoded CLS token is used directly as the per-event embedding,
instead of pooling over the other tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp_embedder(in_features, d_model, hidden_dim=None, dropout=0.0):
    hidden_dim = hidden_dim or d_model
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, d_model),
    )


class TransformerBackboneCLS(nn.Module):
    """
    Parameters
    ----------
    n_global_features, n_jet_features, n_lepton_features : int
        Widths of the raw per-object feature vectors, e.g.
        `len(GLOBAL_FEATURES)`, `len(JET_FEATURES)`, `len(LEPTON_FEATURES)`
        from EventDataset5.py.
    d_model, nhead, num_layers, dim_feedforward, dropout :
        Standard `nn.TransformerEncoder` hyperparameters.
    use_type_embedding : bool
        Add a learned embedding identifying which group (CLS/global/jet/lepton)
        a token came from, so self-attention can tell object types apart.
    """

    def __init__(
        self,
        n_global_features,
        n_jet_features,
        n_lepton_features,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=None,
        dropout=0.1,
        use_type_embedding=True,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.global_embed = _mlp_embedder(n_global_features, d_model, dropout=dropout)
        self.jet_embed = _mlp_embedder(n_jet_features, d_model, dropout=dropout)
        self.lepton_embed = _mlp_embedder(n_lepton_features, d_model, dropout=dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # 4 types: cls, global, jet, lepton
        self.type_embedding = nn.Embedding(4, d_model) if use_type_embedding else None

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
        self.d_model = d_model

    def forward(self, batch, return_tokens: bool = False):
        """
        batch : dict with keys "global" (B, F_g), "jet" (B, N_j, F_j),
            "jet_mask" (B, N_j), "lepton" (B, N_l, F_l), "lepton_mask" (B, N_l)
            -- exactly what `EventDataset5.collate_fn` produces.
        return_tokens : bool
            If True, also return the full per-token encoded sequence (CLS
            token included, at position 0) and its validity mask -- e.g. for
            `decoder.py`'s cross-attention decoder.

        Returns
        -------
        Tensor of shape (B, d_model): the encoded CLS token, one per event.
        If `return_tokens` is True, instead returns
        `(pooled, encoded, valid_mask)` where `encoded` is (B, T, d_model)
        and `valid_mask` is (B, T) bool (True = real token).
        """
        global_feat = batch["global"]
        jet_feat, jet_mask = batch["jet"], batch["jet_mask"]
        lepton_feat, lepton_mask = batch["lepton"], batch["lepton_mask"]

        batch_size = global_feat.shape[0]
        device = global_feat.device

        cls_tok = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        global_tok = self.global_embed(global_feat).unsqueeze(1)  # (B, 1, D)
        global_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        jet_tok = self.jet_embed(jet_feat)  # (B, N_j, D)
        lepton_tok = self.lepton_embed(lepton_feat)  # (B, N_l, D)

        if self.type_embedding is not None:
            cls_tok = cls_tok + self.type_embedding.weight[0]
            global_tok = global_tok + self.type_embedding.weight[1]
            jet_tok = jet_tok + self.type_embedding.weight[2]
            lepton_tok = lepton_tok + self.type_embedding.weight[3]

        # Zero out padded slots defensively: they're already excluded from
        # attention via the key-padding mask, but this keeps them exactly
        # zero for numerical hygiene.
        jet_tok = jet_tok * jet_mask.unsqueeze(-1).to(jet_tok.dtype)
        lepton_tok = lepton_tok * lepton_mask.unsqueeze(-1).to(lepton_tok.dtype)

        tokens = torch.cat([cls_tok, global_tok, jet_tok, lepton_tok], dim=1)
        valid_mask = torch.cat([cls_mask, global_mask, jet_mask, lepton_mask], dim=1)

        key_padding_mask = ~valid_mask
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        encoded = self.output_norm(encoded)
        pooled = encoded[:, 0]  # encoded CLS token

        if return_tokens:
            return pooled, encoded, valid_mask
        return pooled


# if __name__ == "__main__":
#     from EventDataset5 import GLOBAL_FEATURES, JET_FEATURES, LEPTON_FEATURES

#     backbone = TransformerBackboneCLS(
#         n_global_features=len(GLOBAL_FEATURES),
#         n_jet_features=len(JET_FEATURES),
#         n_lepton_features=len(LEPTON_FEATURES),
#         d_model=32,
#         nhead=4,
#         num_layers=2,
#     )

#     batch_size, n_jets, n_leptons = 8, 5, 2
#     dummy_batch = {
#         "global": torch.randn(batch_size, len(GLOBAL_FEATURES)),
#         "jet": torch.randn(batch_size, n_jets, len(JET_FEATURES)),
#         "jet_mask": torch.ones(batch_size, n_jets, dtype=torch.bool),
#         "lepton": torch.randn(batch_size, n_leptons, len(LEPTON_FEATURES)),
#         "lepton_mask": torch.ones(batch_size, n_leptons, dtype=torch.bool),
#     }
#     pooled = backbone(dummy_batch)
#     print(f"pooled embedding shape: {tuple(pooled.shape)}")
