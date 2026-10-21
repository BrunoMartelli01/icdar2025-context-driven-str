#!/usr/bin/env bash
set -euo pipefail

BASE=${1:?Pass the PAPER base directory, e.g. /extra1/bmartelli}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA="$BASE/datasets/context-str/scene-segmentation-v1"
ARTIFACTS="$BASE/context-str-artifacts/parseq"
REV=1902db043c029a7e03a3818c616c06600af574be
OUTPUT="$ROOT/artifacts/diagnostics/totaltext-v4-gt-mask-resolution-nofilter-test-v1"
FROZEN="$ROOT/artifacts/diagnostics/totaltext-v4-curved-component-crops-test-v1-qualified"
PY="$ROOT/.venvs/context-str-main-py311/bin/python"

test -f "$OUTPUT/prepared.json" || { echo "Missing CPU-prepared no-filter resolution audit" >&2; exit 1; }
PYTHONUNBUFFERED=1 "$PY" -u -m scripts.sweep_totaltext_v4_gt_mask_resolution recognize \
  --test-manifest "$DATA/manifests/total_text_test.yaml" \
  --data-root "$DATA" \
  --frozen-v4-root "$FROZEN" \
  --output-dir "$OUTPUT" \
  --filter-mode none \
  --parseq-source "$ARTIFACTS/parseq-$REV" \
  --parseq-source-archive "$ARTIFACTS/parseq-$REV.tar.gz" \
  --parseq-checkpoint "$ARTIFACTS/parseq-bb5792a6.pt" \
  --device cuda:0
