"""EfficientDualAxis — a sub-quadratic dual-axis forecaster for multivariate time-series.

Core idea (extends QuantFormer-E's dual-axis design, made efficient):
  patchify each variable's series  ->  for each layer, mix information
    (1) along TIME (across patches), and
    (2) across VARIABLES (at each patch position),
  both with LINEAR ATTENTION — O(n) instead of attention's O(n²).
Then a linear head maps the patch representations to the forecast horizon.

Why linear attention: it replaces softmax(QKᵀ)V with φ(Q)·(φ(K)ᵀV), computed
associatively, so cost is linear in sequence length (and in #variables on the other axis).
That's the sub-quadratic claim. The temporal mixer can later be swapped for a state-space
block as an ablation — that's a planned experiment, not a dependency.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """Non-causal linear attention over the sequence axis. x: [B, N, D] -> [B, N, D]."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shp = (B, N, self.h, self.dh)
        q = q.view(shp).transpose(1, 2)            # [B, h, N, dh]
        k = k.view(shp).transpose(1, 2)
        v = v.view(shp).transpose(1, 2)
        q, k = F.elu(q) + 1.0, F.elu(k) + 1.0      # feature map φ (keeps things positive)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)            # [B, h, dh, dh]  — O(N·dh²)
        ksum = k.sum(dim=2)                                   # [B, h, dh]
        num = torch.einsum("bhnd,bhde->bhne", q, kv)          # [B, h, N, dh]
        den = torch.einsum("bhnd,bhd->bhn", q, ksum).clamp_min(1e-6).unsqueeze(-1)
        out = (num / den).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class DualAxisBlock(nn.Module):
    """One layer: sub-quadratic mixing along time, then across variables, then an MLP."""

    def __init__(self, dim: int, heads: int, cross_variable: bool = True):
        super().__init__()
        self.cross_variable = cross_variable
        self.temporal = LinearAttention(dim, heads)
        self.variable = LinearAttention(dim, heads) if cross_variable else None
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: [B, V, N, D]
        B, V, N, D = x.shape
        # (1) temporal mixing — each variable independently, across the N patches
        xt = x.reshape(B * V, N, D)
        xt = xt + self.temporal(self.n1(xt))
        x = xt.reshape(B, V, N, D)
        # (2) cross-variable mixing — at each patch position, across the V variables
        if self.cross_variable:
            xv = x.permute(0, 2, 1, 3).reshape(B * N, V, D)
            xv = xv + self.variable(self.n2(xv))
            x = xv.reshape(B, N, V, D).permute(0, 2, 1, 3)
        # (3) channel MLP
        return x + self.mlp(self.n3(x))


class EfficientDualAxis(nn.Module):
    def __init__(self, n_vars: int, seq_len: int, pred_len: int,
                 patch: int = 16, stride: int = 8, dim: int = 64, depth: int = 2, heads: int = 4,
                 cross_variable: bool = True):
        super().__init__()
        self.patch, self.stride, self.seq_len = patch, stride, seq_len
        self.n_patches = (seq_len - patch) // stride + 1
        self.embed = nn.Linear(patch, dim)
        self.pos = nn.Parameter(torch.randn(1, 1, self.n_patches, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [DualAxisBlock(dim, heads, cross_variable) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(self.n_patches * dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: [B, seq_len, n_vars]
        B, L, V = x.shape
        # instance norm (standard for LTSF): subtract per-series mean/std, add back at the end
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True) + 1e-5
        x = (x - mean) / std
        # patchify per variable -> [B, V, N, patch]
        xp = x.permute(0, 2, 1).unfold(dimension=2, size=self.patch, step=self.stride)
        h = self.embed(xp) + self.pos                       # [B, V, N, D]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h).reshape(B, V, -1)                  # [B, V, N*D]
        out = self.head(h).permute(0, 2, 1)                 # [B, pred_len, V]
        return out * std + mean
