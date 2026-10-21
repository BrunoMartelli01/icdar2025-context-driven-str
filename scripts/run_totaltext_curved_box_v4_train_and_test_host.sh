#!/usr/bin/env bash
# Train the isolated v4 target, then evaluate native-128 contours + PARSeq on TEST-300.
set -euo pipefail

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
  shift
fi
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--check] [host-root, default derived from repository]" >&2
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${1:-$(dirname "$(dirname "$REPO")")}"
ROOT="${ROOT%/}"
if [[ ! -d "$ROOT/projects/icdar2025-context-driven-str" && -d "$ROOT/bmartelli/projects/icdar2025-context-driven-str" ]]; then
  ROOT="$ROOT/bmartelli"
fi
PY="$REPO/.venvs/context-str-main-py311/bin/python"
TARGET="$REPO/artifacts/totaltext-curved-word-boxes-v4"
CONFIG="$REPO/configs/approximate/segmentation_total_text_curved_word_boxes_v4.yaml"
DATA="$ROOT/datasets/context-str/scene-segmentation-v1"
RUN_ROOT="$ROOT/context-str-runs/totaltext-curved-word-boxes-v4-500ep"
RUN="$RUN_ROOT/total-text-filled-curved-word-boxes-v4-500ep"
RESULTS="$ROOT/context-str-runs/totaltext-v4-findcontours-parseq-test-v1"
PARSEQ="$ROOT/context-str-artifacts/parseq"
REV=1902db043c029a7e03a3818c616c06600af574be

test -x "$PY" || { echo "Missing Python: $PY" >&2; exit 1; }
test -f "$CONFIG" || { echo "Missing v4 config: $CONFIG" >&2; exit 1; }
test -f "$DATA/manifests/total_text_test.yaml" || { echo "Missing qualified TEST manifest" >&2; exit 1; }
test -d "$PARSEQ/parseq-$REV" || { echo "Missing PARSeq source" >&2; exit 1; }
test -f "$PARSEQ/parseq-$REV.tar.gz" || { echo "Missing PARSeq source archive" >&2; exit 1; }
test -f "$PARSEQ/parseq-bb5792a6.pt" || { echo "Missing PARSeq checkpoint" >&2; exit 1; }
printf '%s  %s\n' 7ad8525d66294e07b2d63be0dd68d08a2e83dbecc04a4d55fb65cd6110c51bea "$TARGET/manifests/total_text_train.yaml" | sha256sum --check -
printf '%s  %s\n' ea4edfc47954b1158d7299a1a37742d60bf92a8e1132b64149a32fe170806856 "$TARGET/qualification.json" | sha256sum --check -
printf '%s  %s\n' d4d8c94bc84d127d487f62d811e53bfec31e5126b3091d41c103406f4fa06a13 "$TARGET/qualification_native128.json" | sha256sum --check -
"$PY" -c 'from segmentation.config import load_segmentation_config; import sys; c=load_segmentation_config(sys.argv[1]); assert c.training.epochs == 500 and c.training.protocol == "practical_totaltext_curved_word_boxes_500_v1"' "$CONFIG"
if (( CHECK_ONLY )); then
  echo "[v4] CPU preflight passed; no training or TEST inference started"
  exit 0
fi

cd "$REPO"
export PYTHONUNBUFFERED=1
if [[ ! -f "$RUN/run_manifest.json" ]]; then
  if [[ -d "$RUN" && -n "$(find "$RUN" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Incomplete v4 training run exists: $RUN. Refusing to overwrite; resume it explicitly." >&2
    exit 1
  else
    "$PY" -u -m segmentation.train --config "$CONFIG" --data-root "$TARGET" --manifest-root "$TARGET" --output-root "$RUN_ROOT" --device cuda
  fi
fi

RUN_MANIFEST="$RUN/run_manifest.json"
CHECKPOINT="$($PY -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="SUCCESS" and len(r["epochs"])==500; print(r["checkpoint_output_location"])' "$RUN_MANIFEST")"
test -f "$CHECKPOINT" || { echo "Selected checkpoint missing: $CHECKPOINT" >&2; exit 1; }
echo "[v4] 500-epoch training complete; starting official TEST-300 contour/PARSeq evaluation" >&2

"$PY" -u -m scripts.evaluate_totaltext_v1_findcontours_parseq \
  --test-manifest "$DATA/manifests/total_text_test.yaml" \
  --data-root "$DATA" \
  --segmentation-config "$CONFIG" \
  --localization-config "$REPO/configs/approximate/localization_repository_contours_native128_v1.json" \
  --training-run-manifest "$RUN_MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --parseq-source "$PARSEQ/parseq-$REV" \
  --parseq-source-archive "$PARSEQ/parseq-$REV.tar.gz" \
  --parseq-checkpoint "$PARSEQ/parseq-bb5792a6.pt" \
  --device cuda:0 \
  --parseq-batch-size 32 \
  --output-dir "$RESULTS"
