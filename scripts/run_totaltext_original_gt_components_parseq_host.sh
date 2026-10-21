#!/usr/bin/env bash
set -euo pipefail

BASE=${1:?Pass the PAPER base directory, e.g. /extra1/bmartelli}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="$ROOT/.venvs/context-str-main-py311/bin/python"
DATA="$BASE/datasets/context-str/scene-segmentation-v1"
ARTIFACTS="$BASE/context-str-artifacts/parseq"
REV=1902db043c029a7e03a3818c616c06600af574be
OUTPUT="$ROOT/artifacts/diagnostics/totaltext-original-gt-component-parseq-test-v2"

test -f "$OUTPUT/prepared.json" || { echo "Missing prepared CPU audit: $OUTPUT/prepared.json" >&2; exit 1; }
PYTHONUNBUFFERED=1 "$PY" -u -m scripts.diagnose_totaltext_original_gt_components_parseq recognize \
  --manifest "$DATA/manifests/total_text_test.yaml" \
  --data-root "$DATA" \
  --output-dir "$OUTPUT" \
  --parseq-source "$ARTIFACTS/parseq-$REV" \
  --parseq-source-archive "$ARTIFACTS/parseq-$REV.tar.gz" \
  --parseq-checkpoint "$ARTIFACTS/parseq-bb5792a6.pt" \
  --device cuda:0
