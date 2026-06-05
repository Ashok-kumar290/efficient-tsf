"""Smoke experiment: train EfficientDualAxis on synthetic multivariate data that has BOTH
temporal autocorrelation AND cross-variable coupling — so a model that captures both axes
should clearly beat a naive baseline. This validates the architecture end-to-end before we
port it into the standard LTSF benchmark harness on real datasets (ETT, Weather, ...).

    python experiments/train_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import EfficientDualAxis  # noqa: E402


def simulate(n_steps=8000, n_vars=8, seed=0):
    """A VAR(1)-style system: next step depends on its own past (temporal) AND on the other
    variables' past (cross-variable), plus noise."""
    rng = np.random.default_rng(seed)
    A = 0.5 * np.eye(n_vars) + 0.15 * rng.standard_normal((n_vars, n_vars)) / n_vars
    x = np.zeros((n_steps, n_vars), dtype=np.float32)
    x[0] = rng.standard_normal(n_vars)
    for t in range(1, n_steps):
        x[t] = x[t - 1] @ A.T + 0.1 * rng.standard_normal(n_vars)
    return x


def make_windows(series, seq_len, pred_len):
    xs, ys = [], []
    for i in range(len(series) - seq_len - pred_len):
        xs.append(series[i:i + seq_len])
        ys.append(series[i + seq_len:i + seq_len + pred_len])
    return torch.tensor(np.array(xs)), torch.tensor(np.array(ys))


def main():
    torch.manual_seed(0)
    seq_len, pred_len, n_vars = 96, 24, 8
    series = simulate(n_vars=n_vars)
    X, Y = make_windows(series, seq_len, pred_len)
    n_train = int(0.8 * len(X))
    tr = DataLoader(TensorDataset(X[:n_train], Y[:n_train]), batch_size=64, shuffle=True)
    va_x, va_y = X[n_train:], Y[n_train:]

    # naive baseline: predict the last observed value, repeated over the horizon
    naive = X[n_train:, -1:, :].repeat(1, pred_len, 1)
    naive_mse = nn.functional.mse_loss(naive, va_y).item()

    model = EfficientDualAxis(n_vars=n_vars, seq_len=seq_len, pred_len=pred_len,
                              patch=16, stride=8, dim=64, depth=2, heads=4)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"model params: {n_params:,} · naive(last-value) val MSE: {naive_mse:.4f}")

    for epoch in range(15):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_mse = nn.functional.mse_loss(model(va_x), va_y).item()
        print(f"epoch {epoch:2d}  val MSE {val_mse:.4f}  "
              f"({'beats' if val_mse < naive_mse else 'worse than'} naive)")

    print(f"\nfinal val MSE {val_mse:.4f}  vs naive {naive_mse:.4f}  "
          f"-> {naive_mse / val_mse:.2f}× better"
          if val_mse < naive_mse else "\n(model did not beat naive — needs tuning)")


if __name__ == "__main__":
    main()
