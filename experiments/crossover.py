"""Crossover study: at how many variables V does cross-variable mixing start to help?

Reads the JSONL from train_ltsf.py --out, run at several --max-vars on one dataset/horizon.
Prints, per V, the mean±std TEST MSE of each model and the cross-variable vs channel-independent
gap. Negative gap = cross-variable HELPS at that V; positive = it HURTS. The V where the sign
flips is the crossover — the headline of this study.

    python experiments/crossover.py crossover.jsonl
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

CH, CV, LB = "DualAxis (channel-indep)", "DualAxis (cross-variable)", "LinearBaseline"


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "crossover.jsonl"
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        print(f"no results in {path}")
        return

    # dedupe (n_vars, model, seed) -> latest, then group MSE across seeds
    dedup = {}
    for r in rows:
        dedup[(r["n_vars"], r["model"], r["seed"])] = r
    g: dict[tuple, list[float]] = defaultdict(list)
    for r in dedup.values():
        g[(r["n_vars"], r["model"])].append(r["mse"])

    Vs = sorted({k[0] for k in g})
    print(f"{'V':>6} | {'linear':>15} {'chan-indep':>15} {'cross-var':>15} | {'cross−chan gap':>18}")
    print("-" * 80)
    flip = None
    for V in Vs:
        def fmt(m):
            if (V, m) in g:
                mu, sd = mean_std(g[(V, m)])
                return f"{mu:.4f}±{sd:.4f}"
            return "n/a"
        gap = ""
        if (V, CH) in g and (V, CV) in g:
            chm, _ = mean_std(g[(V, CH)])
            cvm, _ = mean_std(g[(V, CV)])
            d = cvm - chm
            gap = f"{d:+.4f} ({100 * d / chm:+.1f}%)"
            if flip is None and d < 0:
                flip = V
        print(f"{V:>6} | {fmt(LB):>15} {fmt(CH):>15} {fmt(CV):>15} | {gap:>18}")
    print("-" * 80)
    print("gap < 0  => cross-variable mixing HELPS at that V;  > 0 => it HURTS.")
    if flip is not None:
        print(f"crossover: cross-variable mixing starts helping at V ≈ {flip}.")
    else:
        print("no crossover observed in this range (cross-variable never beat channel-indep).")


if __name__ == "__main__":
    main()
