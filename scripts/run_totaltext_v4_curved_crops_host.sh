#!/usr/bin/env bash
# Foreground GPU PARSeq diagnostic on already prepared TEST-300 v4 masks.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname "$(dirname "$REPO")")"
PY="$REPO/.venvs/context-str-main-py311/bin/python"
DATA="$ROOT/datasets/context-str/scene-segmentation-v1"
PRED="$ROOT/context-str-runs/totaltext-v4-findcontours-parseq-test-v1"
PARSEQ="$ROOT/context-str-artifacts/parseq"
REV=1902db043c029a7e03a3818c616c06600af574be
OUT="$REPO/artifacts/diagnostics/totaltext-v4-curved-component-crops-test-v1-qualified"

test -x "$PY" || { echo "Missing main Python: $PY" >&2; exit 1; }
test -f "$OUT/prepared.json" || { echo "Run CPU preparation first: $OUT/prepared.json" >&2; exit 1; }
test -f "$PRED/results.json" || { echo "Missing frozen predicted masks: $PRED" >&2; exit 1; }
test -d "$PARSEQ/parseq-$REV" || { echo "Missing PARSeq source" >&2; exit 1; }
test -f "$PARSEQ/parseq-$REV.tar.gz" || { echo "Missing PARSeq source archive" >&2; exit 1; }
test -f "$PARSEQ/parseq-bb5792a6.pt" || { echo "Missing PARSeq checkpoint" >&2; exit 1; }

cd "$REPO"
export PYTHONUNBUFFERED=1
"$PY" -u -m scripts.evaluate_totaltext_v4_curved_crops recognize \
  --test-manifest "$DATA/manifests/total_text_test.yaml" \
  --data-root "$DATA" \
  --segmentation-config "$REPO/configs/approximate/segmentation_total_text_curved_word_boxes_v4.yaml" \
  --localization-config "$REPO/configs/approximate/localization_curved_components_native128_no_padding_v1.json" \
  --predicted-cache "$PRED" \
  --output-dir "$OUT" \
  --parseq-source "$PARSEQ/parseq-$REV" \
  --parseq-source-archive "$PARSEQ/parseq-$REV.tar.gz" \
  --parseq-checkpoint "$PARSEQ/parseq-bb5792a6.pt" \
  --device cuda:0
