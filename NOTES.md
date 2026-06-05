# Status & Resume Notes

_Last updated: 2026-06-05_

## What this is
`EfficientDualAxis` — a sub-quadratic dual-axis forecaster — + a validated LTSF benchmark
harness. Core question: **does cross-variable mixing help, and when?**

## Findings so far
- **Harness validated** — LinearBaseline MSE ≈ published DLinear (~0.39 on ETTh1). Numbers are trustworthy.
- **ETTh1 (7 vars)** — cross-variable mixing **HURTS** (0.41 vs 0.39); channel-independent ≈ linear (tie). Simple wins on low-dimensional data.
- **Electricity (321 vars)** — channel-independent **decisively beats linear** (val ~0.144 vs linear test 0.195). ✅ The architecture earns its keep on high-dimensional data.
- **OPEN — the make-or-break number:** does **cross-variable beat channel-independent** on Electricity? The cross-variable model OOM'd at batch 512 before finishing. *This single number is still unknown.*

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

## After that
- Tune training (models overfit in ~2 epochs — add dropout / weight decay).
- Confirm the trend on **Traffic (862 vars)**.
- Add **efficiency metrics** (FLOPs / throughput) — that's the sub-quadratic *claim*, not optional.
- Ablation: swap the temporal linear-attention for an **SSM (Mamba-style)** block.
- `--batch 128` for high-var datasets on A100 (512 OOMs at 321 vars).
