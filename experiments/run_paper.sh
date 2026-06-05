#!/usr/bin/env bash
# Reproduce the paper's main Electricity table: 3 seeds × 4 horizons -> results.jsonl,
# then aggregate to mean±std. Run on an A100 (batch 128 fits 321 vars; 512 OOMs).
#
#   bash experiments/run_paper.sh /path/to/electricity.csv
#
set -euo pipefail
CSV="${1:?pass the path to electricity.csv (find ltsf_data -name electricity.csv)}"
OUT="results.jsonl"
: > "$OUT"   # start fresh

for SEED in 2021 2022 2023; do
  for PL in 96 192 336 720; do
    echo "===== seed=$SEED pred_len=$PL ====="
    python experiments/train_ltsf.py --dataset "$CSV" --pred-len "$PL" \
      --epochs 30 --batch 128 --seed "$SEED" --out "$OUT"
  done
done

echo; echo "##### AGGREGATED PAPER TABLE #####"
python experiments/aggregate.py "$OUT"
