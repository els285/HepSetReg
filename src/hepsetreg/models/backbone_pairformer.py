"""A Pairformer-style backbone (AlphaFold3, Abramson et al. 2024) adapted to
variable-length sets of physics objects.

AlphaFold3's Pairformer maintains two representations side by side: a
**single representation** ``s`` (one embedding per token) and a **pair
representation** ``z`` (one embedding per token *pair*). Each of its 48
blocks (1) refines ``z`` using only ``z`` itself, via triangle multiplicative
updates (and, optionally, triangle self-attention), then (2) uses that
refined ``z`` to *bias* ordinary multi-head self-attention over ``s`` -- a
linear projection of ``z_ij`` is added to the attention logits between
tokens ``i`` and ``j`` before softmax. Because ``z`` is updated every block,
the bias fed into the single-representation attention is itself updated
every block, not a fixed precomputed matrix.

This module reuses :class:`~hepsetreg.models.tokenizer.ObjectTokenizer` for
the same masked, variable-count, permutation-equivariant object handling as
:class:`~hepsetreg.models.backbone.ObjectSetEncoder`, then builds a pair
representation over those object tokens and runs it through a stack of
:class:`PairformerBlock`. ``z`` can optionally be seeded with explicit
per-pair physical features (e.g. ΔR, k_T, pair mass) -- the same "pairwise
interaction feature" idea used by the Particle Transformer (ParT) for jet
tagging, which is essentially a static, single-shot version of the same
attention-bias trick.

Simplifications relative to the AF3 paper (deliberate, for a small package
whose object counts -- jets, leptons, ... -- are typically single digits to
a few dozen, nowhere near protein sequence lengths):

* Triangle self-attention (steps 3-4 in AF3's block) is implemented but
  **off by default** (``use_triangle_attention=False``). At small N the
  triangle multiplicative updates alone already give every pair entry an
  O(N) receptive field per block; triangle attention adds real cost
  (O(N^3 * nhead) per block) for comparatively little benefit here. It's
  included for completeness / larger-N use cases.
* No recycling, no MSA/template representations, no confidence heads --
  just the trunk mechanism (pair-updates + pair-biased single attention)
  relevant to regressing an event-level observable from a token set.

Caveat: this module could not be executed in the sandbox it was written in
(no network access to install torch), so it has been carefully re-derived
and reviewed by hand but not run. See ``tests/test_backbone_pairformer.py``
-- please run that before relying on this in anger, especially with
``use_triangle_attention=True``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hepsetreg.models.pooling import build_pooling
from hepsetreg.models.tokenizer import ObjectGroupSpec, ObjectTokenizer


class SwiGLUTransition(nn.Module):
    """The "transition" MLP used throughout AF3 (SwiGLU, pre-LayerNorm)."""

    def __init__(self, d_model: int, expansion: float = 4.0):
        super().__init__()
        hidden = int(d_model * expansion)
        self.norm = nn.LayerNorm(d_model)
        self.proj_gate = nn.Linear(d_model, hidden)
        self.proj_value = nn.Linear(d_model, hidden)
        self.proj_out = nn.Linear(hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        xn = self.norm(x)
        return self.proj_out(F.silu(self.proj_gate(xn)) * self.proj_value(xn))


class TriangleMultiplicativeUpdate(nn.Module):
    """AF2/AF3 triangle multiplicative update ("outgoing" or "incoming").

    Refines ``z`` using only ``z``: gated projections ``a``, ``b`` are
    contracted over a shared third index (rows for "outgoing", columns for
    "incoming"), giving every pair entry information from every other pair
    entry sharing a row/column -- an O(N) receptive field per call.
    """

    def __init__(self, d_pair: int, d_hidden: Optional[int] = None, mode: str = "outgoing"):
        super().__init__()
        if mode not in {"outgoing", "incoming"}:
            raise ValueError("mode must be 'outgoing' or 'incoming'.")
        self.mode = mode
        d_hidden = d_hidden or d_pair

        self.norm_in = nn.LayerNorm(d_pair)
        self.a_proj = nn.Linear(d_pair, d_hidden)
        self.a_gate = nn.Linear(d_pair, d_hidden)
        self.b_proj = nn.Linear(d_pair, d_hidden)
        self.b_gate = nn.Linear(d_pair, d_hidden)
        self.out_gate = nn.Linear(d_pair, d_pair)
        self.norm_out = nn.LayerNorm(d_hidden)
        self.out_proj = nn.Linear(d_hidden, d_pair)

    def forward(self, z: Tensor, pair_mask: Tensor) -> Tensor:
        # z: (B, N, N, d_pair); pair_mask: (B, N, N) bool, True = valid pair.
        zn = self.norm_in(z)
        m = pair_mask.unsqueeze(-1).to(zn.dtype)

        a = torch.sigmoid(self.a_gate(zn)) * self.a_proj(zn) * m
        b = torch.sigmoid(self.b_gate(zn)) * self.b_proj(zn) * m

        if self.mode == "outgoing":
            mixed = torch.einsum("bikc,bjkc->bijc", a, b)
        else:
            mixed = torch.einsum("bkic,bkjc->bijc", a, b)

        mixed = self.out_proj(self.norm_out(mixed))
        gate = torch.sigmoid(self.out_gate(zn))
        return gate * mixed


class TriangleAttention(nn.Module):
    """AF2/AF3 triangle self-attention ("starting-node" or "ending-node").

    For a fixed row ``i``, runs self-attention among the entries of that
    row (columns), with an attention bias derived from ``z`` itself
    (independent of ``i``) -- this is the step that lets the pair
    representation reason about triangle-inequality-like consistency.
    Off by default in :class:`PairformerBackbone`; see the module docstring.
    """

    def __init__(self, d_pair: int, nhead: int = 4, mode: str = "starting_node"):
        super().__init__()
        if mode not in {"starting_node", "ending_node"}:
            raise ValueError("mode must be 'starting_node' or 'ending_node'.")
        if d_pair % nhead != 0:
            raise ValueError(f"d_pair ({d_pair}) must be divisible by nhead ({nhead}).")

        self.mode = mode
        self.nhead = nhead
        self.d_head = d_pair // nhead

        self.norm = nn.LayerNorm(d_pair)
        self.q_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.k_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.v_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.bias_proj = nn.Linear(d_pair, nhead, bias=False)
        self.gate_proj = nn.Linear(d_pair, d_pair)
        self.out_proj = nn.Linear(d_pair, d_pair)

    def forward(self, z: Tensor, pair_mask: Tensor) -> Tensor:
        if self.mode == "ending_node":
            z_in = z.transpose(1, 2)
            mask_in = pair_mask.transpose(1, 2)
        else:
            z_in, mask_in = z, pair_mask

        batch_size, n_obj, _, _ = z_in.shape
        zn = self.norm(z_in)
        q = self.q_proj(zn).view(batch_size, n_obj, n_obj, self.nhead, self.d_head)
        k = self.k_proj(zn).view(batch_size, n_obj, n_obj, self.nhead, self.d_head)
        v = self.v_proj(zn).view(batch_size, n_obj, n_obj, self.nhead, self.d_head)
        # bias[b, p, q, h] = Linear(z[b, p, q]); shared across every row i (added below).
        bias = self.bias_proj(zn)

        # logits[b, h, i, p, q] = <q[b,i,p,h,:], k[b,i,q,h,:]> / sqrt(d) + bias[b,p,q,h]
        logits = torch.einsum("biphc,biqhc->bhipq", q, k) / (self.d_head**0.5)
        logits = logits + bias.permute(0, 3, 1, 2).unsqueeze(2)

        key_valid = mask_in.unsqueeze(1).unsqueeze(3)  # (B, 1, N_i, 1, N_q)
        # Guard fully-invalid rows (token i itself is padding) against an all -inf softmax;
        # their output gets zeroed below regardless.
        row_has_valid_key = key_valid.any(dim=-1, keepdim=True)
        safe_key_valid = torch.where(row_has_valid_key, key_valid, torch.ones_like(key_valid))
        logits = logits.masked_fill(~safe_key_valid.expand_as(logits), float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        out = torch.einsum("bhipq,biqhc->biphc", attn, v)
        out = out.reshape(batch_size, n_obj, n_obj, self.nhead * self.d_head)

        gate = torch.sigmoid(self.gate_proj(zn))
        out = gate * self.out_proj(out)
        out = out * mask_in.unsqueeze(-1).to(out.dtype)

        if self.mode == "ending_node":
            out = out.transpose(1, 2)
        return out


class AttentionPairBias(nn.Module):
    """Multi-head self-attention over the single representation ``s``,
    with an additive per-head bias supplied from the (current) pair
    representation -- this is AF3's "attention with pair bias"."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate_proj = nn.Linear(d_model, d_model)

    def forward(self, s: Tensor, bias: Tensor, key_padding_mask: Tensor) -> Tensor:
        # s: (B, N, d_model); bias: (B, nhead, N, N) additive logits bias;
        # key_padding_mask: (B, N) bool, True = padding.
        sn = self.norm(s)
        batch_size, n_tok, _ = sn.shape
        attn_mask = bias.reshape(batch_size * self.nhead, n_tok, n_tok)
        out, _ = self.mha(
            sn, sn, sn, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        gate = torch.sigmoid(self.gate_proj(sn))
        return gate * out


class PairformerBlock(nn.Module):
    """One Pairformer block: refine ``z`` (pair-only updates), then use the
    refined ``z`` to bias attention over ``s``."""

    def __init__(
        self,
        d_model: int,
        d_pair: int,
        nhead_single: int,
        nhead_pair: int = 4,
        pair_transition_expansion: float = 4.0,
        single_transition_expansion: float = 4.0,
        use_triangle_attention: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tri_mult_out = TriangleMultiplicativeUpdate(d_pair, mode="outgoing")
        self.tri_mult_in = TriangleMultiplicativeUpdate(d_pair, mode="incoming")

        self.use_triangle_attention = use_triangle_attention
        if use_triangle_attention:
            self.tri_attn_start = TriangleAttention(d_pair, nhead=nhead_pair, mode="starting_node")
            self.tri_attn_end = TriangleAttention(d_pair, nhead=nhead_pair, mode="ending_node")

        self.pair_transition = SwiGLUTransition(d_pair, expansion=pair_transition_expansion)
        self.bias_proj = nn.Linear(d_pair, nhead_single, bias=False)
        self.attn_pair_bias = AttentionPairBias(d_model, nhead_single, dropout=dropout)
        self.single_transition = SwiGLUTransition(d_model, expansion=single_transition_expansion)

    def forward(
        self, s: Tensor, z: Tensor, pair_mask: Tensor, key_padding_mask: Tensor, cls_offset: int = 0
    ):
        z = z + self.tri_mult_out(z, pair_mask)
        z = z + self.tri_mult_in(z, pair_mask)
        if self.use_triangle_attention:
            z = z + self.tri_attn_start(z, pair_mask)
            z = z + self.tri_attn_end(z, pair_mask)
        z = z + self.pair_transition(z)

        bias = self.bias_proj(z).permute(0, 3, 1, 2)  # (B, nhead_single, N_obj, N_obj)
        if cls_offset:
            # Prepend a zero row/column so the CLS token (position 0 in `s`) gets no
            # pair bias -- it isn't part of the physical pair grid.
            bias = F.pad(bias, (cls_offset, 0, cls_offset, 0), value=0.0)

        s = s + self.attn_pair_bias(s, bias, key_padding_mask)
        s = s + self.single_transition(s)
        return s, z


class PairformerBackbone(nn.Module):
    """Drop-in alternative to :class:`~hepsetreg.models.backbone.ObjectSetEncoder`
    with the same ``forward(objects, mask) -> pooled`` contract (plus an
    optional ``pairwise_features`` argument), so it can be used with the
    existing :class:`~hepsetreg.models.heads.RegressionHead` /
    :class:`~hepsetreg.lightning_modules.supervised.SupervisedRegressor`
    unchanged.

    Parameters
    ----------
    groups, use_type_embedding:
        Same object-group contract as :class:`ObjectSetEncoder`.
    d_model:
        Single-representation width.
    d_pair:
        Pair-representation width (independent of ``d_model``; AF3 uses a
        narrower pair rep than single rep, e.g. 128 vs 384).
    nhead_single, nhead_pair:
        Head counts for the single (pair-biased) attention and, if enabled,
        triangle attention.
    num_blocks:
        Number of stacked :class:`PairformerBlock`. AF3 uses 48; for small
        object sets a handful is typically enough.
    pairwise_feature_dim:
        If > 0, :meth:`forward` requires a ``pairwise_features`` tensor of
        shape ``(B, N_obj, N_obj, pairwise_feature_dim)`` (e.g. per-jet-pair
        ΔR / k_T / pair mass), linearly embedded and added into the initial
        pair representation alongside the learned ``Linear(s_i) + Linear(s_j)``
        term AF3 uses.
    use_triangle_attention:
        Off by default -- see the module docstring.
    pooling:
        ``"cls"``, ``"mean"``, or ``"attention"`` (same semantics as
        :class:`ObjectSetEncoder`; CLS token, if used, sits outside the pair
        grid with a zero attention bias to/from every real object).
    """

    def __init__(
        self,
        groups: List[ObjectGroupSpec],
        d_model: int = 128,
        d_pair: int = 64,
        nhead_single: int = 8,
        nhead_pair: int = 4,
        num_blocks: int = 4,
        pairwise_feature_dim: int = 0,
        use_triangle_attention: bool = False,
        pooling: str = "cls",
        use_type_embedding: bool = True,
        dropout: float = 0.0,
        pair_transition_expansion: float = 4.0,
        single_transition_expansion: float = 4.0,
    ):
        super().__init__()
        if d_model % nhead_single != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead_single ({nhead_single}).")
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

        # Pair representation init, AF3-style: z_ij = Linear(s_i) + Linear(s_j),
        # optionally plus an embedding of explicit physical pairwise features
        # (the ParT-style "interaction features").
        self.pair_row_proj = nn.Linear(d_model, d_pair)
        self.pair_col_proj = nn.Linear(d_model, d_pair)
        self.pairwise_feature_dim = pairwise_feature_dim
        self.pairwise_feature_proj = (
            nn.Linear(pairwise_feature_dim, d_pair) if pairwise_feature_dim > 0 else None
        )
        self.pair_init_norm = nn.LayerNorm(d_pair)

        self.blocks = nn.ModuleList(
            [
                PairformerBlock(
                    d_model,
                    d_pair,
                    nhead_single=nhead_single,
                    nhead_pair=nhead_pair,
                    pair_transition_expansion=pair_transition_expansion,
                    single_transition_expansion=single_transition_expansion,
                    use_triangle_attention=use_triangle_attention,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.pool = build_pooling(pooling, d_model)
        self.d_model = d_model
        self.d_pair = d_pair

    def forward(
        self,
        objects: Dict[str, Tensor],
        mask: Dict[str, Tensor],
        pairwise_features: Optional[Tensor] = None,
        return_tokens: bool = False,
    ):
        tokens, valid_mask = self.tokenizer(objects, mask)  # (B, N_obj, d_model), (B, N_obj)
        batch_size = tokens.shape[0]
        pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)  # (B, N_obj, N_obj)

        z = self.pair_row_proj(tokens).unsqueeze(2) + self.pair_col_proj(tokens).unsqueeze(1)
        if self.pairwise_feature_proj is not None:
            if pairwise_features is None:
                raise ValueError(
                    "PairformerBackbone was built with pairwise_feature_dim > 0; "
                    "forward() requires a `pairwise_features` tensor of shape "
                    "(batch, n_objects, n_objects, pairwise_feature_dim)."
                )
            z = z + self.pairwise_feature_proj(pairwise_features)
        z = self.pair_init_norm(z)
        z = z * pair_mask.unsqueeze(-1).to(z.dtype)

        s = tokens
        key_padding_mask = ~valid_mask
        cls_offset = 0
        if self.cls_token is not None:
            cls = self.cls_token.expand(batch_size, -1, -1)
            if self.tokenizer.type_embedding is not None:
                cls_type_ids = torch.full(
                    (batch_size, 1), self._cls_type_id, dtype=torch.long, device=tokens.device
                )
                cls = cls + self.tokenizer.type_embedding(cls_type_ids)
            s = torch.cat([cls, s], dim=1)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device)
            key_padding_mask = ~torch.cat([cls_mask, valid_mask], dim=1)
            cls_offset = 1

        if not (~key_padding_mask).any(dim=1).all():
            raise ValueError(
                "Found an event with no valid tokens at all (mask is all-False). "
                "Add a CLS/global token or ensure every event has >=1 real object."
            )

        for block in self.blocks:
            s, z = block(s, z, pair_mask, key_padding_mask, cls_offset=cls_offset)

        s = self.output_norm(s)
        pooled = self.pool(s, ~key_padding_mask)

        if return_tokens:
            return pooled, s, ~key_padding_mask, z
        return pooled
