#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${1:?Usage: bash diagnostics/run_contextual_ocr_correction_host.sh USER_DATA_ROOT}"
PY=.venvs/context-str-main-py311/bin/python
CFG=configs/approximate/contextual_ocr_correction_train_v1.json
OUT=artifacts/diagnostics/contextual-ocr-correction-train-v1
AUDIT=artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.contextual_ocr_correction prepare \
  --config "$CFG" --source-root "$ROOT/context-str-runs/caption-routing-five-models-train-v1" \
  --output-dir "$OUT"
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.contextual_ocr_correction evaluate \
  --audit "$AUDIT" --output-dir "$OUT" --baseline-only
for MODEL in flan-t5-base qwen2.5-0.5b qwen2.5-1.5b; do
  "$PY" -u -m diagnostics.contextual_ocr_correction infer \
    --model "$MODEL" --hf-cache "$ROOT/context-str-artifacts/huggingface" \
    --device cuda:0 --output-dir "$OUT" --allow-download
done
CUDA_VISIBLE_DEVICES='' "$PY" -u -m diagnostics.contextual_ocr_correction evaluate \
  --audit "$AUDIT" --output-dir "$OUT"
