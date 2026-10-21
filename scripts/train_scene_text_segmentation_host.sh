#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {ctw1500|icdar2015|total_text|combined} [--qualify|resume-checkpoint]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/.venvs/context-str-main-py311/bin/python"
DATA_ROOT="${SCENE_TEXT_DATA_ROOT:-/extra1/bmartelli/datasets/context-str/scene-segmentation-v1}"
OUTPUT_ROOT="${SCENE_TEXT_RUN_ROOT:-/extra1/bmartelli/context-str-runs/scene-text-segmentation-v1}"
VARIANT="$1"
RESUME="${2:-}"
QUALIFY=false
if [[ "$RESUME" == "--qualify" ]]; then
  QUALIFY=true
  RESUME=""
  OUTPUT_ROOT="${SCENE_TEXT_QUALIFICATION_ROOT:-/extra1/bmartelli/context-str-runs/scene-text-segmentation-v1-qualification}"
fi

case "$VARIANT" in
  ctw1500) CONFIG="configs/approximate/segmentation_ctw1500_author_hyperparameters.yaml" ;;
  icdar2015) CONFIG="configs/approximate/segmentation_icdar2015_author_hyperparameters.yaml" ;;
  total_text) CONFIG="configs/approximate/segmentation_total_text_author_hyperparameters.yaml" ;;
  combined) CONFIG="configs/approximate/segmentation_scene_text3_balanced.yaml" ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

test -x "$PY" || { echo "Missing Python environment: $PY" >&2; exit 1; }
test -f "$DATA_ROOT/qualification.json" || {
  echo "Prepared dataset is missing: $DATA_ROOT/qualification.json" >&2
  exit 1
}

COMMAND=(
  "$PY" -u -m segmentation.train
  --config "$REPO_ROOT/$CONFIG"
  --data-root "$DATA_ROOT"
  --manifest-root "$DATA_ROOT"
  --output-root "$OUTPUT_ROOT"
  --device cuda
)
if [[ -n "$RESUME" ]]; then
  COMMAND+=(--resume "$RESUME")
fi
if [[ "$QUALIFY" == true ]]; then
  COMMAND+=(--qualification-max-steps 1)
fi

cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "${COMMAND[@]}"
