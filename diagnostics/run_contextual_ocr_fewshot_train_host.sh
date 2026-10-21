#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${1:?Usage: bash diagnostics/run_contextual_ocr_fewshot_train_host.sh USER_DATA_ROOT}"
PY=.venvs/context-str-main-py311/bin/python
OUT=artifacts/diagnostics/contextual-ocr-fewshot-train-v1
SOURCE=artifacts/diagnostics/contextual-ocr-plaintext-train-v2
AUDIT=artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.contextual_ocr_fewshot_train prepare \
  --config configs/approximate/contextual_ocr_fewshot_train_v1.json \
  --source-root "$SOURCE" --audit "$AUDIT" --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.contextual_ocr_fewshot_train evaluate \
  --audit "$AUDIT" --output-dir "$OUT" --baseline-only
"$PY" -u -m diagnostics.contextual_ocr_fewshot_train infer \
  --model qwen2.5-1.5b --hf-cache "$ROOT/context-str-artifacts/huggingface" \
  --device cuda:0 --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.contextual_ocr_fewshot_train evaluate \
  --audit "$AUDIT" --output-dir "$OUT"
