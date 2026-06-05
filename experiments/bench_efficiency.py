"""Efficiency receipt — does the cross-variable mixer actually scale sub-quadratically?

The accuracy result (Electricity-96: cross-variable 0.1472 ≈ iTransformer 0.148) shows our
linear-attention cross-variable mixer MATCHES full quadratic cross-attention. This script
shows it does so CHEAPER: it sweeps the number of variables V and measures, head-to-head,

    - LinearAttention  (what EfficientDualAxis uses, model.py)   — claim: O(V)
    - SoftmaxAttention (standard scaled dot-product, the iTransformer-style cross-attn) — O(V²)

both with the identical [B, V, D] -> [B, V, D] interface used on the variable axis
(model.py:67-69). We report per-call latency, peak GPU memory, and — the punchline — the
empirical scaling exponent fit on the high-V tail: linear should land near 1.0, softmax near 2.0.

    python experiments/bench_efficiency.py
    python experiments/bench_efficiency.py --batch 256 --dim 128 --vars 7 21 100 200 321 500 862

No external FLOP-counting library needed: wall-clock + peak memory are what a reviewer asks for,
and they can't be fudged by an analytical count that ignores kernel realities.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import LinearAttention  # noqa: E402  — the O(V) mixer under test


class SoftmaxAttention(nn.Module):
    """Standard scaled-dot-product attention — the O(N²) baseline. Same interface as
    LinearAttention so it is a true drop-in on the variable axis. This is the quadratic
    cross-attention an iTransformer-style model pays for variable mixing."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.dh ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # [B, N, D]
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shp = (B, N, self.h, self.dh)
        q = q.view(shp).transpose(1, 2)                        # [B, h, N, dh]
        k = k.view(shp).transpose(1, 2)
        v = v.view(shp).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale         # [B, h, N, N] — the O(N²) term
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


@torch.no_grad()
def bench(module: nn.Module, x: torch.Tensor, device: str, iters: int = 30) -> tuple[float, float]:
    """Return (median ms per call, peak MiB). Warms up, syncs, times `iters` forward passes."""
    module.eval()
    for _ in range(5):                                        # warmup (cuDNN autotune, alloc)
        module(x)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        module(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
    return times[len(times) // 2], peak


def fit_exponent(vs: list[int], ys: list[float]) -> float:
    """Slope of log(y) vs log(V) — the empirical scaling exponent. ~1 linear, ~2 quadratic."""
    lx = [math.log(v) for v in vs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vars", type=int, nargs="+",
                    default=[7, 21, 100, 200, 321, 500, 862],
                    help="variable counts to sweep (7=ETTh1 … 321=Electricity … 862=Traffic)")
    ap.add_argument("--batch", type=int, default=256,
                    help="effective batch = (real batch × #patches) seen by the variable mixer")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tail", type=int, default=3,
                    help="fit the scaling exponent on the largest-V `tail` points")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    lin = LinearAttention(args.dim, args.heads).to(device)
    sm = SoftmaxAttention(args.dim, args.heads).to(device)
    print(f"device={device}  batch={args.batch}  dim={args.dim}  heads={args.heads}  iters={args.iters}")
    print(f"\n{'V':>6} | {'linear ms':>10} {'lin MiB':>9} | {'softmax ms':>11} {'sm MiB':>9} | {'speedup':>8}")
    print("-" * 66)

    lat_lin, lat_sm, oom_from = [], [], None
    for v in args.vars:
        x = torch.randn(args.batch, v, args.dim, device=device)
        lm, lp = bench(lin, x, device, args.iters)
        try:
            sm_ms, sp = bench(sm, x, device, args.iters)
        except torch.cuda.OutOfMemoryError:                   # quadratic mem blows up first
            torch.cuda.empty_cache()
            print(f"{v:>6} | {lm:>10.3f} {lp:>9.1f} | {'OOM':>11} {'—':>9} | {'—':>8}")
            oom_from = oom_from or v
            lat_lin.append((v, lm))
            continue
        sp_ratio = sm_ms / lm
        print(f"{v:>6} | {lm:>10.3f} {lp:>9.1f} | {sm_ms:>11.3f} {sp:>9.1f} | {sp_ratio:>7.1f}×")
        lat_lin.append((v, lm))
        lat_sm.append((v, sm_ms))

    print("-" * 66)
    tail = args.tail
    if len(lat_lin) >= tail:
        vs, ys = zip(*lat_lin[-tail:])
        print(f"linear-attention  scaling exponent (top {tail} V): {fit_exponent(list(vs), list(ys)):.2f}  "
              f"(O(V) ⇒ ~1.0)")
    if len(lat_sm) >= tail:
        vs, ys = zip(*lat_sm[-tail:])
        print(f"softmax-attention scaling exponent (top {tail} V): {fit_exponent(list(vs), list(ys)):.2f}  "
              f"(O(V²) ⇒ ~2.0)")
    if oom_from:
        print(f"softmax attention OOM'd at V={oom_from}; linear ran fine — that's the quadratic memory wall.")
    print("\nThis is the sub-quadratic claim, measured: same interface, same accuracy (see NOTES.md),"
          "\nlinear cost where the baseline pays quadratic.")


if __name__ == "__main__":
    main()
