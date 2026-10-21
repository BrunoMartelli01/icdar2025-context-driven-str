#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${1:?Usage: bash diagnostics/run_ocr_prompt_comparison_train_host.sh USER_DATA_ROOT}"
PY=.venvs/context-str-main-py311/bin/python
OUT=artifacts/diagnostics/ocr-prompt-comparison-train-v1
SOURCE=artifacts/diagnostics/contextual-ocr-conservative-crop-train-v1
AUDIT=artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.ocr_prompt_comparison_train prepare \
  --config configs/approximate/ocr_prompt_comparison_train_v1.json \
  --source-root "$SOURCE" --audit "$AUDIT" --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.ocr_prompt_comparison_train evaluate \
  --audit "$AUDIT" --output-dir "$OUT" --baseline-only
"$PY" -u -m diagnostics.ocr_prompt_comparison_train infer \
  --model qwen2.5-1.5b --hf-cache "$ROOT/context-str-artifacts/huggingface" \
  --device cuda:0 --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.ocr_prompt_comparison_train evaluate \
  --audit "$AUDIT" --output-dir "$OUT"
