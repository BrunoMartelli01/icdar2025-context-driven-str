#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${1:?Usage: bash diagnostics/run_contextual_ocr_plaintext_train_host.sh USER_DATA_ROOT}"
PY=.venvs/context-str-main-py311/bin/python
OUT=artifacts/diagnostics/contextual-ocr-plaintext-train-v2
SOURCE=artifacts/diagnostics/contextual-ocr-correction-train-v1
AUDIT=artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.contextual_ocr_plaintext_train prepare \
  --config configs/approximate/contextual_ocr_plaintext_train_v2.json \
  --source-root "$SOURCE" --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.contextual_ocr_plaintext_train evaluate \
  --audit "$AUDIT" --output-dir "$OUT" --baseline-only
for MODEL in flan-t5-base qwen2.5-0.5b qwen2.5-1.5b; do
  "$PY" -u -m diagnostics.contextual_ocr_plaintext_train infer \
    --model "$MODEL" --hf-cache "$ROOT/context-str-artifacts/huggingface" \
    --device cuda:0 --output-dir "$OUT"
done
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.contextual_ocr_plaintext_train evaluate \
  --audit "$AUDIT" --output-dir "$OUT"
