"""
Simple cross-attention decoder: N learned query embeddings attend into an
encoded token sequence (e.g. the per-token output of `TransformerBackbone` /
`TransformerBackboneCLS`, obtained via `return_tokens=True`, before pooling)
and produce N output embeddings -- one per query.

Useful when a single pooled event embedding isn't the right shape for the
downstream task, e.g. predicting several targets that each want their own
view of the event, or per-object outputs (one query per object slot).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionDecoder(nn.Module):
    """
    Parameters
    ----------
    d_model, nhead, num_layers, dim_feedforward, dropout :
        Standard `nn.TransformerDecoderLayer` hyperparameters.
    n_queries : int
        Number of learned query tokens. Each one cross-attends into the
        memory sequence and self-attends against the other queries (standard
        `nn.TransformerDecoderLayer` behaviour), producing one output
        embedding per query.
    """

    def __init__(
        self,
        d_model,
        n_queries,
        nhead=8,
        num_layers=4,
        dim_feedforward=None,
        dropout=0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.query_embed = nn.Parameter(torch.zeros(n_queries, d_model))
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)

        dim_feedforward = dim_feedforward or 4 * d_model
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.n_queries = n_queries
        self.d_model = d_model

    def forward(self, memory, valid_mask=None):
        """
        memory : (B, T, D) encoded token sequence to cross-attend into --
            e.g. the `encoded` tensor returned by a backbone's
            `forward(batch, return_tokens=True)`.
        valid_mask : (B, T) bool or None
            True = real token, False = padding. Same convention used
            throughout this repo (the inverse is passed to PyTorch's
            `memory_key_padding_mask`, which expects True = ignore).

        Returns
        -------
        Tensor of shape (B, n_queries, d_model): one output embedding per
        query.
        """
        batch_size = memory.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, D)

        memory_key_padding_mask = ~valid_mask if valid_mask is not None else None
        decoded = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_norm(decoded)


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
    decoder = CrossAttentionDecoder(d_model=d_model, n_queries=5, nhead=4, num_layers=2)

    batch_size, n_jets, n_leptons = 8, 5, 2
    dummy_batch = {
        "global": torch.randn(batch_size, n_global_features),
        "jet": torch.randn(batch_size, n_jets, n_jet_features),
        "jet_mask": torch.ones(batch_size, n_jets, dtype=torch.bool),
        "lepton": torch.randn(batch_size, n_leptons, n_lepton_features),
        "lepton_mask": torch.ones(batch_size, n_leptons, dtype=torch.bool),
    }

    _, encoded, valid_mask = backbone(dummy_batch, return_tokens=True)
    query_out = decoder(encoded, valid_mask)
    print(f"query output shape: {tuple(query_out.shape)}")
