#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${1:?Usage: bash diagnostics/run_curve_rectification_train_host.sh /extra1/bmartelli}
PY=.venvs/context-str-main-py311/bin/python
OUT=artifacts/diagnostics/fixed-v1-curve-rectification-train-v1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
"$PY" -u -m diagnostics.curve_rectification_train prepare \
  --config configs/approximate/curve_rectification_train_v1.json \
  --source artifacts/diagnostics/ocr-target-bound-battery-train-v2 \
  --oracle artifacts/diagnostics/gt-mask-localization-oracle-train-v1/results.json \
  --data-root "$ROOT/datasets/context-str/icdar13" --output-dir "$OUT"
"$PY" -u -m diagnostics.curve_rectification_train infer \
  --artifact-root "$ROOT/context-str-artifacts" --device cuda:0 --output-dir "$OUT"
