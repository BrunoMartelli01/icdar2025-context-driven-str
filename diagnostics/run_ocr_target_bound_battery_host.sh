#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${1:?Usage: bash diagnostics/run_ocr_target_bound_battery_host.sh USER_DATA_ROOT}"
PY=.venvs/context-str-main-py311/bin/python
MODULE=diagnostics.ocr_target_bound_battery_train
OUT=artifacts/diagnostics/ocr-target-bound-battery-train-v2
CACHE="$ROOT/context-str-artifacts/huggingface"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' "$PY" -u -m "$MODULE" prepare \
  --config configs/approximate/ocr_target_bound_battery_train_v2.json \
  --source-root artifacts/diagnostics/ocr-evidence-generation-battery-train-v1 \
  --hf-cache "$CACHE" --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m "$MODULE" evaluate \
  --baseline-only --hf-cache "$CACHE" --output-dir "$OUT"
"$PY" -u -m "$MODULE" infer --phase propose --device cuda:0 \
  --hf-cache "$CACHE" --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m "$MODULE" prepare-verifiers \
  --hf-cache "$CACHE" --output-dir "$OUT"
"$PY" -u -m "$MODULE" infer --phase verify --device cuda:0 \
  --hf-cache "$CACHE" --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m "$MODULE" evaluate \
  --hf-cache "$CACHE" --output-dir "$OUT"
