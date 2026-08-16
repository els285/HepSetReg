"""Conditional flow-matching velocity network + ODE sampler.

Generalizes DIRECTOR's ``flowmatch_train.py`` (a fixed-slice-conditioned
velocity transformer with a hard-coded ttbar output layout) to a variable
number of conditioning objects, reusing the same masked
:class:`~hepsetreg.models.tokenizer.ObjectTokenizer` as the supervised
backbone. The condition (jets/leptons/MET/...) is attended to token-by-token
rather than being pooled first, exactly like DIRECTOR -- the network can look
at individual objects when predicting the velocity field.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from hepsetreg.models.tokenizer import ObjectGroupSpec, ObjectTokenizer


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding of the flow-matching time ``t in [0, 1]``."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        # t: (B, 1)
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class ConditionalVelocityField(nn.Module):
    """Predicts the flow-matching velocity ``v(x_t, t | context objects)``.

    The context objects, a time token, and an ``x_t`` token are all fed as one
    masked token sequence into a transformer encoder; the encoded ``x_t``
    token position is read out and mapped to the velocity prediction.
    """

    def __init__(
        self,
        groups: List[ObjectGroupSpec],
        output_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        use_type_embedding: bool = True,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.output_dim = output_dim
        self.tokenizer = ObjectTokenizer(
            groups, d_model=d_model, use_type_embedding=use_type_embedding, n_extra_types=2
        )
        self._time_type_id = len(groups)
        self._xt_type_id = len(groups) + 1

        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.xt_proj = nn.Linear(output_dim, d_model)

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
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, output_dim)
        )

    def forward(self, xt: Tensor, t: Tensor, objects: Dict[str, Tensor], mask: Dict[str, Tensor]) -> Tensor:
        batch_size = xt.shape[0]
        t = t.reshape(batch_size, 1).to(xt.dtype)

        ctx_tokens, ctx_mask = self.tokenizer(objects, mask)

        t_token = self.time_proj(self.time_embed(t)).unsqueeze(1)
        xt_token = self.xt_proj(xt).unsqueeze(1)
        if self.tokenizer.type_embedding is not None:
            t_type = torch.full((batch_size, 1), self._time_type_id, dtype=torch.long, device=xt.device)
            xt_type = torch.full((batch_size, 1), self._xt_type_id, dtype=torch.long, device=xt.device)
            t_token = t_token + self.tokenizer.type_embedding(t_type)
            xt_token = xt_token + self.tokenizer.type_embedding(xt_type)

        tokens = torch.cat([xt_token, t_token, ctx_tokens], dim=1)
        always_valid = torch.ones(batch_size, 2, dtype=torch.bool, device=xt.device)
        valid_mask = torch.cat([always_valid, ctx_mask], dim=1)

        encoded = self.encoder(tokens, src_key_padding_mask=~valid_mask)
        encoded = self.output_norm(encoded)
        xt_out = encoded[:, 0]
        return self.velocity_head(xt_out)


def _expand_context(objects: Dict[str, Tensor], mask: Dict[str, Tensor], n_samples: int):
    """Repeats each event's context ``n_samples`` times: (B, ...) -> (B*S, ...)."""

    def expand(t: Tensor) -> Tensor:
        b = t.shape[0]
        rest = t.shape[1:]
        return t.unsqueeze(1).expand(b, n_samples, *rest).reshape(b * n_samples, *rest)

    return {k: expand(v) for k, v in objects.items()}, {k: expand(v) for k, v in mask.items()}


@torch.no_grad()
def sample_flow(
    velocity_field: ConditionalVelocityField,
    objects: Dict[str, Tensor],
    mask: Dict[str, Tensor],
    n_steps: int = 50,
    n_samples: int = 1,
) -> Tensor:
    """Midpoint-method ODE integration from noise (t=0) to data (t=1).

    Returns a tensor of shape ``(batch, n_samples, output_dim)``.
    """

    any_tensor = next(iter(objects.values()))
    device = any_tensor.device
    batch_size = any_tensor.shape[0]
    output_dim = velocity_field.output_dim

    exp_objects, exp_mask = _expand_context(objects, mask, n_samples)
    x = torch.randn(batch_size * n_samples, output_dim, device=device)
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = step / n_steps
        t = torch.full((batch_size * n_samples, 1), t_val, device=device)
        v1 = velocity_field(x, t, exp_objects, exp_mask)

        x_half = x + 0.5 * dt * v1
        t_half = torch.full((batch_size * n_samples, 1), t_val + 0.5 * dt, device=device)
        v2 = velocity_field(x_half, t_half, exp_objects, exp_mask)

        x = x + dt * v2

    return x.reshape(batch_size, n_samples, output_dim)
