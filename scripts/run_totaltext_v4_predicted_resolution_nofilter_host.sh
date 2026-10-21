#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE=$(dirname "$(dirname "$ROOT")")
DATA="$BASE/datasets/context-str/scene-segmentation-v1"
PRED="$BASE/context-str-runs/totaltext-v4-findcontours-parseq-test-v1"
PARSEQ="$BASE/context-str-artifacts/parseq"
REV=1902db043c029a7e03a3818c616c06600af574be
FROZEN="$ROOT/artifacts/diagnostics/totaltext-v4-curved-component-crops-test-v1-qualified"
OUT="$ROOT/artifacts/diagnostics/totaltext-v4-predicted-mask-resolution-nofilter-test-v1"
PY="$ROOT/.venvs/context-str-main-py311/bin/python"

test -f "$OUT/prepared.json" || { echo "Missing CPU-prepared predicted resolution audit" >&2; exit 1; }
cd "$ROOT"
PYTHONUNBUFFERED=1 "$PY" -u -m scripts.sweep_totaltext_v4_predicted_mask_resolution recognize \
  --test-manifest "$DATA/manifests/total_text_test.yaml" \
  --data-root "$DATA" \
  --predicted-cache "$PRED" \
  --frozen-v4-root "$FROZEN" \
  --output-dir "$OUT" \
  --parseq-source "$PARSEQ/parseq-$REV" \
  --parseq-source-archive "$PARSEQ/parseq-$REV.tar.gz" \
  --parseq-checkpoint "$PARSEQ/parseq-bb5792a6.pt" \
  --device cuda:0
