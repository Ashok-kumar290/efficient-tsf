"""Aggregate multi-seed results into the paper table: mean ± std per (dataset, horizon, model).

Reads the JSONL written by train_ltsf.py --out, groups across seeds, and prints MSE/MAE as
mean ± std. This is what turns a single lucky number into a defensible result.

    python experiments/aggregate.py results.jsonl
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)        # sample std (Bessel) — honest error bars
    return m, math.sqrt(var)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.jsonl"
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        print(f"no results in {path}")
        return

    # group: (dataset, pred_len, model) -> list of (mse, mae) across seeds
    g: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: {"mse": [], "mae": []})
    for r in rows:
        key = (r["dataset"], r["pred_len"], r["model"])
        g[key]["mse"].append(r["mse"])
        g[key]["mae"].append(r["mae"])

    last_ds_pl = None
    for (ds, pl, model) in sorted(g, key=lambda k: (k[0], k[1], k[2])):
        if (ds, pl) != last_ds_pl:
            seeds = len(g[(ds, pl, model)]["mse"])
            print(f"\n{'='*64}\n{ds}  pred_len={pl}   ({seeds} seed(s), lower = better)")
            print(f"  {'model':<26} {'MSE (mean±std)':<22} {'MAE (mean±std)':<22}")
            last_ds_pl = (ds, pl)
        mse_m, mse_s = mean_std(g[(ds, pl, model)]["mse"])
        mae_m, mae_s = mean_std(g[(ds, pl, model)]["mae"])
        print(f"  {model:<26} {mse_m:.4f} ± {mse_s:.4f}      {mae_m:.4f} ± {mae_s:.4f}")
    print()


if __name__ == "__main__":
    main()
