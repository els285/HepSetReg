"""
Simple conditional flow matching: learn a velocity field
`v(x_t, t | cond)` from a linear interpolation between noise and data,
`x_t = (1 - t) * y0 + t * y1` with `y0 ~ N(0, I)`, `v_target = y1 - y0`.

`cond` is just a plain (B, cond_dim) vector -- e.g. the pooled embedding
from `TransformerBackbone`/`TransformerBackboneCLS` in backbone.py -- so
this file has no dependency on EventDataset5's global/jet/lepton layout.

Three interchangeable velocity-network architectures are provided, all
sharing the same `forward(xt, t, cond) -> v` signature:
  - `DNNVelocityField`: concatenate [xt, time embedding, cond] and pass
    through a plain MLP.
  - `TransformerVelocityField`: embed xt/time/cond each into one token and
    run a small `nn.TransformerEncoder` over the resulting 3-token sequence.
  - `DiTVelocityField`: each scalar of `xt` becomes its own token
    ("patchify"), and a stack of adaLN-Zero-conditioned transformer blocks
    -- modulated by a single (time + cond) vector -- refines them,
    DiT-style, before a zero-initialized final layer maps each token back
    to a scalar.

`flow_matching_loss` and `sample_flow` are architecture-agnostic: any of the
three velocity fields above can be dropped in.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding of the flow-matching time `t in [0, 1]`."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B, 1)
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class DNNVelocityField(nn.Module):
    """Plain-MLP velocity field: concatenate [xt, time embedding, cond]."""

    def __init__(self, output_dim, cond_dim, d_model=128, n_layers=4, dropout=0.0):
        super().__init__()
        self.output_dim = output_dim
        self.time_embed = SinusoidalTimeEmbedding(d_model)

        layers = []
        prev_dim = output_dim + d_model + cond_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev_dim, d_model), nn.GELU(), nn.Dropout(dropout)]
            prev_dim = d_model
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xt, t, cond):
        t = t.reshape(xt.shape[0], 1).to(xt.dtype)
        t_emb = self.time_embed(t)
        x = torch.cat([xt, t_emb, cond], dim=-1)
        return self.net(x)


class TransformerVelocityField(nn.Module):
    """xt/time/cond each become one token; a small transformer encoder mixes
    them and the encoded xt-token position is read out as the velocity."""

    def __init__(
        self,
        output_dim,
        cond_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=None,
        dropout=0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        self.output_dim = output_dim

        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.xt_proj = nn.Linear(output_dim, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.type_embedding = nn.Embedding(3, d_model)  # 0=xt, 1=time, 2=cond

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

    def forward(self, xt, t, cond):
        t = t.reshape(xt.shape[0], 1).to(xt.dtype)

        xt_tok = self.xt_proj(xt).unsqueeze(1) + self.type_embedding.weight[0]
        t_tok = self.time_proj(self.time_embed(t)).unsqueeze(1) + self.type_embedding.weight[1]
        cond_tok = self.cond_proj(cond).unsqueeze(1) + self.type_embedding.weight[2]

        # Fixed 3-token sequence, all always valid -- no padding mask needed.
        tokens = torch.cat([xt_tok, t_tok, cond_tok], dim=1)
        encoded = self.output_norm(self.encoder(tokens))
        return self.velocity_head(encoded[:, 0])


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """adaLN-Zero-conditioned transformer block, as in the DiT paper."""

    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.0):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Linear(dim_feedforward, d_model)
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        # Zero-init so every block starts as the identity function.
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        # x: (B, T, D) tokens, c: (B, D) conditioning vector.
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)

        h = _modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out

        h = _modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


class DiTVelocityField(nn.Module):
    """Diffusion-Transformer-style velocity field: each scalar of `xt` is
    "patchified" into its own token, and a stack of adaLN-Zero-conditioned
    `DiTBlock`s -- modulated by a single (time + cond) vector -- refines
    them before a zero-initialized final layer maps each token back to a
    scalar. DiT's patch tokens, here applied to a handful of physics targets
    instead of image patches.
    """

    def __init__(
        self,
        output_dim,
        cond_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=None,
        dropout=0.0,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        self.output_dim = output_dim

        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)

        self.patch_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, output_dim, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.final_linear = nn.Linear(d_model, 1)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def forward(self, xt, t, cond):
        t = t.reshape(xt.shape[0], 1).to(xt.dtype)
        c = self.time_proj(self.time_embed(t)) + self.cond_proj(cond)  # (B, D)

        tokens = self.patch_embed(xt.unsqueeze(-1)) + self.pos_embed  # (B, output_dim, D)
        for block in self.blocks:
            tokens = block(tokens, c)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        tokens = _modulate(self.final_norm(tokens), shift, scale)
        return self.final_linear(tokens).squeeze(-1)  # (B, output_dim)


def flow_matching_loss(velocity_field, y1, cond):
    """One conditional flow-matching training step's loss (MSE against the
    linear-interpolation velocity target).

    y1 : (B, output_dim) data/target samples.
    cond : (B, cond_dim) condition vector.
    """
    y0 = torch.randn_like(y1)
    t = torch.rand(y1.shape[0], 1, device=y1.device, dtype=y1.dtype)
    xt = (1 - t) * y0 + t * y1

    v_pred = velocity_field(xt, t, cond)
    v_target = y1 - y0
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample_flow(velocity_field, cond, output_dim, n_steps=50, n_samples=1):
    """Midpoint-method ODE integration from noise (t=0) to data (t=1).

    Returns a tensor of shape (batch, n_samples, output_dim).
    """
    batch_size = cond.shape[0]
    device = cond.device
    cond_exp = cond.unsqueeze(1).expand(batch_size, n_samples, -1).reshape(batch_size * n_samples, -1)

    x = torch.randn(batch_size * n_samples, output_dim, device=device)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_val = step / n_steps
        t = torch.full((batch_size * n_samples, 1), t_val, device=device)
        v1 = velocity_field(x, t, cond_exp)

        x_half = x + 0.5 * dt * v1
        t_half = torch.full((batch_size * n_samples, 1), t_val + 0.5 * dt, device=device)
        v2 = velocity_field(x_half, t_half, cond_exp)

        x = x + dt * v2

    return x.reshape(batch_size, n_samples, output_dim)


class FlowRegressor(nn.Module):
    """Wires a TransformerBackbone + a velocity field from this file
    together for conditional-flow-matching regression.

    Unlike EventRegressor (head.py), a single `forward(batch)` call isn't
    how this is trained: `FlowMatchingModule` (lightning_module.py) calls
    `.backbone` and `.velocity_field` directly, since a training step needs
    the pooled condition vector and the raw velocity field separately to
    compute `flow_matching_loss`. `forward(batch, ...)` is for inference
    instead: it pools `batch` through the backbone into a condition vector,
    then runs `sample_flow` to draw `(B, n_samples, output_dim)` samples
    from the learned target distribution.
    """

    def __init__(self, backbone, velocity_field):
        super().__init__()
        self.backbone = backbone
        self.velocity_field = velocity_field

    def forward(self, batch, n_steps=50, n_samples=1):
        cond = self.backbone(batch)
        return sample_flow(self.velocity_field, cond, self.velocity_field.output_dim, n_steps=n_steps, n_samples=n_samples)


if __name__ == "__main__":
    output_dim, cond_dim, batch_size = 3, 32, 8

    y1 = torch.randn(batch_size, output_dim)
    cond = torch.randn(batch_size, cond_dim)

    fields = {
        "dnn": DNNVelocityField(output_dim, cond_dim, d_model=32, n_layers=2),
        "transformer": TransformerVelocityField(output_dim, cond_dim, d_model=32, nhead=4, num_layers=2),
        "dit": DiTVelocityField(output_dim, cond_dim, d_model=32, nhead=4, num_layers=2),
    }

    for name, field in fields.items():
        loss = flow_matching_loss(field, y1, cond)
        samples = sample_flow(field, cond, output_dim, n_steps=10, n_samples=4)
        print(f"{name:12s} loss={loss.item():.4f}  samples shape={tuple(samples.shape)}")
