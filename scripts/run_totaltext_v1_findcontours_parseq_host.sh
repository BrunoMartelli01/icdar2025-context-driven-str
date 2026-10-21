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
DATA="$ROOT/datasets/context-str/scene-segmentation-v1"
TRAIN_RUN="$ROOT/context-str-runs/totaltext-curved-word-regions-v1/total-text-curved-word-regions-v1"
RUN_MANIFEST="$TRAIN_RUN/run_manifest.json"
OUTPUT="$ROOT/context-str-runs/totaltext-v1-findcontours-parseq-test-v1"
ARTIFACTS="$ROOT/context-str-artifacts/parseq"
REV=1902db043c029a7e03a3818c616c06600af574be

cd "$REPO"
test -x "$PY"
test -f "$RUN_MANIFEST"
test -f "$DATA/manifests/total_text_test.yaml"
CHECKPOINT="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_output_location"])' "$RUN_MANIFEST")"
test -f "$CHECKPOINT"

PYTHONUNBUFFERED=1 "$PY" -u -m scripts.evaluate_totaltext_v1_findcontours_parseq \
  --test-manifest "$DATA/manifests/total_text_test.yaml" \
  --data-root "$DATA" \
  --segmentation-config "$REPO/configs/approximate/segmentation_total_text_curved_word_regions_v1.yaml" \
  --localization-config "$REPO/configs/approximate/localization_repository_contours_native128_v1.json" \
  --training-run-manifest "$RUN_MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --parseq-source "$ARTIFACTS/parseq-$REV" \
  --parseq-source-archive "$ARTIFACTS/parseq-$REV.tar.gz" \
  --parseq-checkpoint "$ARTIFACTS/parseq-bb5792a6.pt" \
  --device cuda:0 \
  --parseq-batch-size 32 \
  --output-dir "$OUTPUT"
