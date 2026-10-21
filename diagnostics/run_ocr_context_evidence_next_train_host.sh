#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${1:?Usage: bash diagnostics/run_ocr_context_evidence_next_train_host.sh /extra1/bmartelli}
PY=.venvs/context-str-main-py311/bin/python
OUT=artifacts/diagnostics/ocr-context-evidence-next-train-v1
HF="$ROOT/context-str-artifacts/huggingface"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
"$PY" -u -m diagnostics.ocr_context_evidence_next_train prepare \
  --config configs/approximate/ocr_context_evidence_next_train_v1.json \
  --source artifacts/diagnostics/ocr-target-bound-battery-train-v2 \
  --coverage artifacts/diagnostics/parseq-beam-candidate-coverage-train-v1/results.json \
  --score-root artifacts/diagnostics/parseq-context-reranker-train-v1 \
  --hf-cache "$HF" --output-dir "$OUT"
"$PY" -u -m diagnostics.ocr_context_evidence_next_train infer \
  --hf-cache "$HF" --device cuda:0 --output-dir "$OUT"
