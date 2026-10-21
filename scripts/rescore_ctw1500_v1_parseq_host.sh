#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${1:-/extra1/bmartelli}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/.venvs/context-str-main-py311/bin/python"
DATA="$STORAGE_ROOT/datasets/context-str/scene-segmentation-v1"
RUNS="$STORAGE_ROOT/context-str-runs/scene-text-v1-parseq-test-v1"
OUTPUT="$STORAGE_ROOT/context-str-runs/scene-text-v1-parseq-test-v2-ctw-recognition.json"

PYTHONUNBUFFERED=1 "$PY" -u -m scripts.rescore_ctw1500_v1_parseq \
  --recognition-manifest "$DATA/manifests/ctw1500_recognition_test.json" \
  --data-root "$DATA" \
  --frozen-run "$RUNS/ctw1500-specialized" \
  --frozen-run "$RUNS/ctw1500-combined" \
  --output "$OUTPUT"
