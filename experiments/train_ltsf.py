"""Train EfficientDualAxis on the ETTh1 benchmark — the real experiment.

Proper protocol: train on train split, early-stop on validation (prevents overfitting),
report MSE/MAE on the held-out test split. Runs on GPU if available.

    python experiments/train_ltsf.py --pred-len 96 --epochs 20

Compare the test MSE/MAE against published baselines (DLinear, PatchTST, iTransformer) at
the same horizon. A linear baseline is included for an internal sanity check.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import CustomCSV, ETTh1  # noqa: E402
from model import EfficientDualAxis  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    se = ae = n = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x)
        se += ((p - y) ** 2).mean().item() * len(x)
        ae += (p - y).abs().mean().item() * len(x)
        n += len(x)
    return se / n, ae / n


class LinearBaseline(nn.Module):
    """Per-variable linear map from history to forecast (a DLinear-style sanity baseline)."""
    def __init__(self, seq_len, pred_len, n_vars):
        super().__init__()
        self.fc = nn.Linear(seq_len, pred_len)

    def forward(self, x):                      # [B, L, V]
        return self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)


def train(model, tr, va, device, epochs, lr, patience):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()
        val_mse, _ = evaluate(model, va, device)
        if val_mse < best - 1e-5:
            best, best_state, bad = val_mse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        print(f"  epoch {ep:2d}  val MSE {val_mse:.4f}{'  *' if bad == 0 else ''}")
        if bad >= patience:
            print(f"  early stop (no val improvement in {patience} epochs)")
            break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--pred-len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dataset", default="etth1",
                    help="'etth1' (auto-download) or a path to a custom CSV (Electricity/Traffic)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sl, pl = args.seq_len, args.pred_len
    if args.dataset.lower() == "etth1":
        mk, n_vars, name = (lambda sp: ETTh1(sp, sl, pl)), ETTh1.N_VARS, "ETTh1"
    else:
        mk = lambda sp: CustomCSV(args.dataset, sp, sl, pl)  # noqa: E731
        n_vars, name = mk("train").n_vars, Path(args.dataset).stem
    tr = DataLoader(mk("train"), batch_size=args.batch, shuffle=True)
    va = DataLoader(mk("val"), batch_size=args.batch)
    te = DataLoader(mk("test"), batch_size=args.batch)
    print(f"device={device}  {name} ({n_vars} vars)  seq_len={sl} pred_len={pl}  "
          f"train/val/test batches: {len(tr)}/{len(va)}/{len(te)}")

    results = {}
    for mname, model in [
        ("LinearBaseline", LinearBaseline(sl, pl, n_vars)),
        ("DualAxis (channel-indep)",
         EfficientDualAxis(n_vars, sl, pl, dim=args.dim, depth=args.depth, cross_variable=False)),
        ("DualAxis (cross-variable)",
         EfficientDualAxis(n_vars, sl, pl, dim=args.dim, depth=args.depth, cross_variable=True)),
    ]:
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n=== {mname}  ({n_params:,} params) ===")
        model = train(model, tr, va, device, args.epochs, args.lr, args.patience)
        mse, mae = evaluate(model, te, device)
        results[mname] = (mse, mae)
        print(f"  TEST  MSE {mse:.4f}  MAE {mae:.4f}")

    print(f"\n{'='*48}\n{name}  pred_len={pl}   (lower = better)")
    for name, (mse, mae) in results.items():
        print(f"  {name:<20} MSE {mse:.4f}  MAE {mae:.4f}")
    print("compare vs published DLinear/PatchTST/iTransformer at this horizon.")


if __name__ == "__main__":
    main()
