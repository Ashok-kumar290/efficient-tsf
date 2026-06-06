# Status & Resume Notes

_Last updated: 2026-06-06_

## What this is
`EfficientDualAxis` — a sub-quadratic dual-axis forecaster — + a validated LTSF benchmark
harness. Core question: **does cross-variable mixing help, and when — and can it be done cheaply?**

## RESULT — both halves done (3 seeds, A100, batch 128)

### 1. Accuracy — Electricity (321 vars), mean ± std over seeds {2021, 2022, 2023}
TEST MSE (lower = better). Cross-variable wins at **every** horizon, by **many std** — unambiguous.

| pred_len | LinearBaseline | DualAxis (channel-indep) | **DualAxis (cross-variable)** | cross vs chan-indep | cross vs linear |
|---|---|---|---|---|---|
| 96  | 0.1950 ± 0.0002 | 0.1646 ± 0.0004 | **0.1491 ± 0.0013** | −9.4% | −23.5% |
| 192 | 0.1943 ± 0.0001 | 0.1742 ± 0.0014 | **0.1639 ± 0.0002** | −5.9% | −15.6% |
| 336 | 0.2072 ± 0.0003 | 0.1912 ± 0.0010 | **0.1763 ± 0.0005** | −7.8% | −14.9% |
| 720 | 0.2428 ± 0.0003 | 0.2342 ± 0.0030 | **0.2051 ± 0.0017** | −12.4% | −15.5% |

The std (≤0.003) is far smaller than the gaps between models → the effect is statistically solid,
not seed luck. Cross-variable MSE is in the same range as published **iTransformer** (a full
*quadratic* cross-attention model) — i.e. matched/competitive, achieved sub-quadratically.

### 2. Efficiency — cross-variable mixer, linear vs softmax attention, scaling in #variables V
(A100, batch=256, dim=128, heads=4, 30 iters; same [B,V,D]->[B,V,D] interface.)

| V | linear ms | linear MiB | softmax ms | softmax MiB | speedup |
|---|---|---|---|---|---|
| 7   | 0.61 | 21.7  | 0.34 | 15.1   | 0.6× |
| 21  | 0.61 | 37.5  | 0.34 | 27.1   | 0.6× |
| 100 | 0.96 | 127.1 | 0.94 | 140.1  | 1.0× |
| 200 | 1.80 | 243.5 | 2.46 | 423.1  | 1.4× |
| 321 | 2.79 | 376.8 | 5.13 | 975.1  | 1.8× |
| 500 | 4.17 | 578.2 | 9.91 | 2212.8 | 2.4× |
| 862 | 7.07 | 986.9 | 26.89| 6245.7 | **3.8×** |

Fitted scaling exponent (top-3 V): **linear 0.94 (≈ O(V))** vs **softmax 1.68 (super-linear → O(V²))**.
Memory is the cleanest tell: at V=862 softmax uses **6.3× more memory** (the V×V attention matrix).
Honest crossover: at tiny V (≤~50) linear attention is *slower* (feature-map/kv overhead dominates);
it wins from V≈100 up, exactly where it matters (Electricity 321, Traffic 862).

## The thesis (now fully supported)
**Cross-variable mixing hurts on low-dim data (ETTh1, 7 vars), helps decisively on high-dim
(Electricity, 321 vars) at all horizons — and is done in ~O(V) instead of O(V²), matching a
quadratic cross-attention model's accuracy at a fraction of the compute/memory.**

Model size: ~0.67M–1.55M params (grows with pred_len). Cross-variable mechanism adds only ~132K
params over channel-indep. Tiny vs iTransformer/PatchTST (several M–tens of M).

## Reproduce
```bash
git clone https://github.com/Ashok-kumar290/efficient-tsf.git && cd efficient-tsf
pip install -q torch numpy pandas gdown
gdown 1FHH0S3d6IK_UOpg6taBRavx4MragRLo1 -O electricity.zip
unzip -oq electricity.zip -d ltsf_data
CSV=$(find ltsf_data -name electricity.csv | head -1)
bash experiments/run_paper.sh "$CSV" /content/drive/MyDrive/efficient-tsf/results.jsonl  # persist to Drive
python experiments/bench_efficiency.py
```
`run_paper.sh` is resumable (skips finished seed×horizon) and aggregates to mean±std at the end.

## Next (writing the paper)
1. **Drop published baselines into the accuracy table** (DLinear/PatchTST/iTransformer/Crossformer
   at each horizon) for direct comparison — sanity: our Linear ≈ published DLinear.
2. **Confirm on Traffic (862 vars)**, batch 64 — the highest-dim dataset, where efficiency matters most.
3. **Ablation:** swap temporal linear-attention for an SSM (Mamba-style) block.
4. Draft: intro / method (dual-axis, linear attention) / results (the two tables above) /
   related work / honest limitations (linear attn loses at tiny V; single dataset family so far).
