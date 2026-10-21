#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /extra1/bmartelli" >&2
  exit 2
fi

ROOT=${1%/}
if [[ ! -d "$ROOT/projects/icdar2025-context-driven-str" && -d "$ROOT/bmartelli/projects/icdar2025-context-driven-str" ]]; then
  ROOT="$ROOT/bmartelli"
fi
REPO="$ROOT/projects/icdar2025-context-driven-str"
PY="$REPO/.venvs/context-str-main-py311/bin/python"
DATA="$REPO/artifacts/totaltext-curved-word-regions-v1"
TRAIN_RUN="$ROOT/context-str-runs/totaltext-curved-word-regions-v1/total-text-curved-word-regions-v1"
RUN_MANIFEST="$TRAIN_RUN/run_manifest.json"
OUTPUT="$ROOT/context-str-runs/totaltext-train-native-padding-sweep-v1"
ARTIFACTS="$ROOT/context-str-artifacts/parseq"
REV=1902db043c029a7e03a3818c616c06600af574be

cd "$REPO"
test -x "$PY"
test -f "$RUN_MANIFEST"
test -f "$DATA/manifests/total_text_train.yaml"
CHECKPOINT="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_output_location"])' "$RUN_MANIFEST")"
test -f "$CHECKPOINT"

ARGS=(
  --train-manifest "$DATA/manifests/total_text_train.yaml"
  --data-root "$DATA"
  --segmentation-config "$REPO/configs/approximate/segmentation_total_text_curved_word_regions_v1.yaml"
  --sweep-config "$REPO/configs/approximate/totaltext_train_native_padding_sweep_v1.json"
  --training-run-manifest "$RUN_MANIFEST"
  --checkpoint "$CHECKPOINT"
  --parseq-source "$ARTIFACTS/parseq-$REV"
  --parseq-source-archive "$ARTIFACTS/parseq-$REV.tar.gz"
  --parseq-checkpoint "$ARTIFACTS/parseq-bb5792a6.pt"
  --device cuda:0
  --batch-size 32
  --output-root "$OUTPUT"
)

PYTHONUNBUFFERED=1 "$PY" -u -m scripts.evaluate_totaltext_train_padding_sweep masks "${ARGS[@]}"
PYTHONUNBUFFERED=1 "$PY" -u -m scripts.evaluate_totaltext_train_padding_sweep evaluate "${ARGS[@]}"
