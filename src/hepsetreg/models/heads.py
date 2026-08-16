"""Regression head(s) applied to the pooled event embedding."""

from __future__ import annotations

import torch.nn as nn


class RegressionHead(nn.Module):
    """Simple MLP mapping a pooled embedding to the regression targets.

    Hidden width halves each layer (floored at ``min_neurons``), mirroring
    DIRECTOR's ``build_mlp`` classifier head.
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        n_layers: int = 3,
        start_neurons: int = 128,
        dropout: float = 0.05,
        min_neurons: int = 8,
    ):
        super().__init__()
        layers = []
        prev_dim = d_model
        for i in range(n_layers):
            hidden_dim = max(start_neurons // (2**i), min_neurons)
            layers += [nn.Linear(prev_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
