# efficient-tsf — Sub-Quadratic Dual-Axis Forecasting *(research, working title)*

**Research question:** Can cross-variable dependencies in multivariate time-series
forecasting be modeled **sub-quadratically** without losing accuracy versus attention?

**Hypothesis:** A dual-axis model using sub-quadratic blocks (state-space / linear-attention)
on **both** the temporal and the cross-variable axis can match attention-based methods on
standard benchmarks while scaling better (compute, # variables, context length).

**Why it's a real gap:** PatchTST treats channels independently; iTransformer / Crossformer
model cross-variable dependence but with *quadratic* attention across variables. *Efficient*
cross-variable modeling is open. Builds on the QuantFormer-E dual-axis idea.

## Experimental design
- **Baselines:** DLinear, PatchTST, iTransformer, Crossformer (+ vanilla Transformer).
- **Datasets (standard LTSF):** ETTh1/h2, ETTm1/m2, Weather, Electricity, Traffic.
- **Horizons:** 96 / 192 / 336 / 720.
- **Metrics:** MSE/MAE (accuracy) **+** FLOPs / params / memory / throughput (efficiency — the claim).

## Roadmap
1. **Reproduce baselines** (validate the harness) ← start here
2. Build the sub-quadratic dual-axis model
3. Ablate (which axis / which block actually helps?)
4. Write up → arXiv → workshop

## Scope (honest)
SSM-for-TSF is an active area — aim for a **sharp, careful, workshop-tier first paper**, not
a NeurIPS oral. That's the right, *finishable* size.

## First experiment (this week)
Build the LTSF eval pipeline and **reproduce one baseline's published numbers** (e.g. DLinear
on ETTh1). A credible result needs a trustworthy harness — reproducing a known number *is*
the foundation, and it teaches the field.

## Structure
- `src/` — models, data loaders, training loop
- `experiments/` — configs + run scripts + results
- `data/` — datasets (gitignored)
