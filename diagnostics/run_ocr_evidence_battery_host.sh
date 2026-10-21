#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${1:?Usage: bash diagnostics/run_ocr_evidence_battery_host.sh USER_DATA_ROOT}"
PY=.venvs/context-str-main-py311/bin/python
MODULE=diagnostics.ocr_evidence_generation_battery_train
OUT=artifacts/diagnostics/ocr-evidence-generation-battery-train-v1
COMPACT="$ROOT/context-str-runs/practical-context-train-v1"
COMPACT="$COMPACT/train-practical-remap-v1-parseq-compact-v1"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' "$PY" -u -m "$MODULE" prepare \
  --config configs/approximate/ocr_evidence_generation_battery_train_v1.json \
  --source-root artifacts/diagnostics/geometry-context-reconstruction-train-v1 \
  --compact-root "$COMPACT" \
  --coverage artifacts/diagnostics/parseq-beam-candidate-coverage-train-v1/results.json \
  --audit artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json \
  --hf-cache "$ROOT/context-str-artifacts/huggingface" --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m "$MODULE" evaluate --baseline-only --output-dir "$OUT"
"$PY" -u -m "$MODULE" infer --device cuda:0 \
  --hf-cache "$ROOT/context-str-artifacts/huggingface" --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m "$MODULE" evaluate --output-dir "$OUT"
