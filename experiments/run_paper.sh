#!/usr/bin/env bash
# Reproduce the paper's main Electricity table: 3 seeds × 4 horizons -> results JSONL,
# then aggregate to mean±std. Run on an A100 (batch 128 fits 321 vars; 512 OOMs).
#
#   bash experiments/run_paper.sh /path/to/electricity.csv [OUT.jsonl]
#
# IMPORTANT on Colab: point OUT at Google Drive so results survive a VM disconnect, e.g.
#   bash experiments/run_paper.sh "$CSV" /content/drive/MyDrive/efficient-tsf/results.jsonl
# The sweep is RESUMABLE: re-running skips any seed×horizon already complete in OUT, so a
# mid-sweep disconnect costs at most the run that was in flight.
set -euo pipefail
CSV="${1:?pass the path to electricity.csv (find ltsf_data -name electricity.csv)}"
OUT="${2:-results.jsonl}"
mkdir -p "$(dirname "$OUT")"
touch "$OUT"   # do NOT truncate — we resume from whatever is already here

for SEED in 2021 2022 2023; do
  for PL in 96 192 336 720; do
    # a combo is "done" once its LAST model (cross-variable) is logged for this seed+horizon
    if grep -qE "\"pred_len\": $PL,.*\"seed\": $SEED,.*cross-variable" "$OUT"; then
      echo "skip seed=$SEED pred_len=$PL (already in $OUT)"
      continue
    fi
    echo "===== seed=$SEED pred_len=$PL ====="
    python experiments/train_ltsf.py --dataset "$CSV" --pred-len "$PL" \
      --epochs 30 --batch 128 --seed "$SEED" --out "$OUT"
  done
done

echo; echo "##### AGGREGATED PAPER TABLE #####"
python experiments/aggregate.py "$OUT"
