#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /extra1/bmartelli (or /extra1)" >&2
  exit 2
fi

ROOT=${1%/}
if [[ ! -d "$ROOT/projects/icdar2025-context-driven-str" && -d "$ROOT/bmartelli/projects/icdar2025-context-driven-str" ]]; then
  ROOT="$ROOT/bmartelli"
fi
REPO="$ROOT/projects/icdar2025-context-driven-str"
PY="$REPO/.venvs/context-str-main-py311/bin/python"
ARTIFACTS="$ROOT/context-str-artifacts/parseq"
REV=1902db043c029a7e03a3818c616c06600af574be
CROPS="$ROOT/context-str-runs/scene-text-paper-algorithm1-crops-v1"
V1="$ROOT/context-str-runs/scene-text-v1-parseq-test-v1"
OUTPUT="$ROOT/context-str-runs/scene-text-paper-algorithm1-parseq-v1"

cd "$REPO"
test -x "$PY"
for dataset in ctw1500 icdar2015 total_text; do
  echo "[Algorithm1 PARSeq] dataset=$dataset" >&2
  PYTHONUNBUFFERED=1 "$PY" -u -m scripts.evaluate_scene_text_algorithm1_parseq \
    --dataset "$dataset" \
    --algorithm1-run "$CROPS/$dataset-specialized" \
    --v1-run "$V1/$dataset-specialized" \
    --parseq-source "$ARTIFACTS/parseq-$REV" \
    --parseq-source-archive "$ARTIFACTS/parseq-$REV.tar.gz" \
    --parseq-checkpoint "$ARTIFACTS/parseq-bb5792a6.pt" \
    --device cuda:0 \
    --batch-size 32 \
    --output-dir "$OUTPUT/$dataset-specialized"
done
