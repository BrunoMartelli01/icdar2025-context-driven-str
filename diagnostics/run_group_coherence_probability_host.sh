#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venvs/context-str-main-py311/bin/python
CFG=configs/approximate/group_coherence_probability_train_v1.json
ORACLE=artifacts/diagnostics/gt-mask-localization-oracle-train-v1/results.json
CACHE=artifacts/diagnostics/group-coherence-probabilities-cache-v1
OUT=artifacts/diagnostics/group-coherence-probability-train-v1-complete
if [[ -e "$OUT" ]]; then
    echo "Output already exists; preserved without overwriting: $OUT" >&2
    exit 1
fi
"$PY" -u -m diagnostics.export_grouping_probabilities \
    --config "$CFG" --oracle "$ORACLE" --output-dir "$CACHE" --device cuda:0
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.group_coherence_probability_train \
    --config "$CFG" --oracle "$ORACLE" --probability-cache "$CACHE" --output-dir "$OUT"
