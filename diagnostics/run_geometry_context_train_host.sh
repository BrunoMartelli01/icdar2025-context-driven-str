#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${1:?Usage: bash diagnostics/run_geometry_context_train_host.sh USER_DATA_ROOT}"
PY=.venvs/context-str-main-py311/bin/python
OUT=artifacts/diagnostics/geometry-context-reconstruction-train-v1
MODULE=diagnostics.geometry_context_reconstruction_train
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' "$PY" -u -m "$MODULE" prepare \
  --config configs/approximate/geometry_context_reconstruction_train_v1.json \
  --routing-root "$ROOT/context-str-runs/caption-routing-five-models-train-v1" \
  --model-source artifacts/diagnostics/contextual-ocr-plaintext-train-v2 \
  --audit artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json \
  --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m "$MODULE" preflight \
  --hf-cache "$ROOT/context-str-artifacts/huggingface" --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m "$MODULE" evaluate --baseline-only --output-dir "$OUT"
"$PY" -u -m "$MODULE" infer --device cuda:0 \
  --hf-cache "$ROOT/context-str-artifacts/huggingface" --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m "$MODULE" evaluate --output-dir "$OUT"
