# Status & Resume Notes

_Last updated: 2026-06-05_

## What this is
`EfficientDualAxis` — a sub-quadratic dual-axis forecaster — + a validated LTSF benchmark
harness. Core question: **does cross-variable mixing help, and when?**

## Findings so far
- **Harness validated** — LinearBaseline MSE ≈ published DLinear (ETTh1 ~0.39; Electricity 0.1948 ≈ published 0.197). Numbers are trustworthy and comparable to the literature.
- **ETTh1 (7 vars)** — cross-variable mixing **HURTS** (0.41 vs 0.39); channel-independent ≈ linear (tie). Simple wins on low-dimensional data.
- **Electricity (321 vars), pred_len=96 — RESOLVED ✅ (A100, batch 128, 2026-06-05):**
  | Model | TEST MSE | TEST MAE |
  |---|---|---|
  | LinearBaseline | 0.1948 | 0.2773 |
  | DualAxis (channel-indep) | 0.1635 | 0.2512 |
  | **DualAxis (cross-variable)** | **0.1472** | **0.2423** |
  Clean monotonic ordering. **Cross-variable beats channel-indep by ~10%, beats linear by ~24%.**
  Kicker: 0.1472 ≈ published **iTransformer** (0.148, full quadratic cross-attention) — *matched
  sub-quadratically.* This is the paper's core finding.

## The thesis (now supported by data)
**Cross-variable mixing hurts on low-dim data (7 vars), helps decisively on high-dim (321 vars)
— and can be done in O(n) without losing accuracy vs full attention.**

## Resume here (Colab A100)
```bash
cd /content
git clone https://github.com/Ashok-kumar290/efficient-tsf.git && cd efficient-tsf
pip install -q numpy pandas gdown
gdown 1FHH0S3d6IK_UOpg6taBRavx4MragRLo1 -O electricity.zip   # just the electricity file
unzip -oq electricity.zip -d ltsf_data
CSV=$(find ltsf_data -name electricity.csv | head -1)
python experiments/train_ltsf.py --dataset "$CSV" --epochs 30 --pred-len 96 --batch 128
```
Read all three TEST numbers. **If cross-variable > channel-independent → that's the paper's core finding** ("cross-variable modeling hurts on low-dim data, helps on high-dim — done sub-quadratically").

## Next (to make it paper-ready — in priority order)
1. **Error bars** — rerun Electricity-96 with 3 seeds. The 0.1635→0.1472 gap should survive; show it.
2. **Full horizon table** — pred_len 192/336/720 on Electricity (one horizon isn't a result).
3. **Efficiency metrics (FLOPs / throughput vs a quadratic cross-attention baseline)** — NOT optional.
   The whole claim is "sub-quadratic." Accuracy is shown; cheaper-than-quadratic is NOT yet measured. This measurement *is* the contribution.
4. Confirm the trend on **Traffic (862 vars)**, batch 64.
5. Ablation: swap the temporal linear-attention for an **SSM (Mamba-style)** block.
- `--batch 128` for Electricity (321 vars) on A100; `--batch 64` for Traffic (862). 512 OOMs.
