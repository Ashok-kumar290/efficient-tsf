#!/usr/bin/env bash
# Crossover study: where (in #variables V) does cross-variable mixing flip from hurting to helping?
# Subsamples Electricity to V in {7,20,50,100,200,321} at pred_len=96, 3 seeds, then maps the curve.
#
#   bash experiments/crossover.sh /path/to/electricity.csv [OUT.jsonl]
#
# Point OUT at Google Drive on Colab so results survive a disconnect. Resumable: re-running
# skips any (seed, V) already complete.
set -euo pipefail
CSV="${1:?pass the path to electricity.csv (find ltsf_data -name electricity.csv)}"
OUT="${2:-crossover.jsonl}"
mkdir -p "$(dirname "$OUT")"
touch "$OUT"

for SEED in 2021 2022 2023; do
  for V in 7 20 50 100 200 321; do
    # done = cross-variable (the last model) logged for this seed at this V (n_vars trails the line)
    if grep -E "\"seed\": $SEED," "$OUT" | grep cross-variable | grep -qE "\"n_vars\": $V}"; then
      echo "skip seed=$SEED V=$V (already in $OUT)"
      continue
    fi
    echo "===== seed=$SEED max_vars=$V ====="
    python experiments/train_ltsf.py --dataset "$CSV" --pred-len 96 --epochs 30 \
      --batch 128 --seed "$SEED" --max-vars "$V" --out "$OUT"
  done
done

echo; echo "##### CROSSOVER TABLE #####"
python experiments/crossover.py "$OUT"
