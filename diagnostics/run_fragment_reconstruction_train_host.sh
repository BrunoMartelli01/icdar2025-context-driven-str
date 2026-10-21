#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${1:?Usage: bash diagnostics/run_fragment_reconstruction_train_host.sh USER_DATA_ROOT}"
PY=.venvs/context-str-main-py311/bin/python
OUT=artifacts/diagnostics/fragment-reconstruction-train-v1
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.fragment_reconstruction_train prepare \
  --config configs/approximate/fragment_reconstruction_train_v1.json \
  --source-root artifacts/diagnostics/contextual-ocr-plaintext-train-v2 \
  --audit artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json \
  --output-dir "$OUT"
"$PY" -u -m diagnostics.fragment_reconstruction_train infer \
  --hf-cache "$ROOT/context-str-artifacts/huggingface" --device cuda:0 --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.fragment_reconstruction_train evaluate --output-dir "$OUT"
