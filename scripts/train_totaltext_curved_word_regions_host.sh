#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 {standard|fusion|refined|refined-fusion} [prepared-data-root] [run-root]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/.venvs/context-str-main-py311/bin/python"
case "$1" in
  standard) CONFIG="$REPO_ROOT/configs/approximate/segmentation_total_text_curved_word_regions_v1.yaml"; SET="v1" ;;
  fusion) CONFIG="$REPO_ROOT/configs/approximate/segmentation_total_text_curved_word_regions_fusion_v1.yaml"; SET="v1" ;;
  refined) CONFIG="$REPO_ROOT/configs/approximate/segmentation_total_text_curved_word_regions_refined_v2.yaml"; SET="refined-v2" ;;
  refined-fusion) CONFIG="$REPO_ROOT/configs/approximate/segmentation_total_text_curved_word_regions_refined_fusion_v2.yaml"; SET="refined-v2" ;;
  *) echo "Unknown variant: $1" >&2; exit 2 ;;
esac
if [[ "$SET" == "v1" ]]; then
  DATA_DEFAULT="$REPO_ROOT/artifacts/totaltext-curved-word-regions-v1"
  RUN_NAME="totaltext-curved-word-regions-v1"
else
  DATA_DEFAULT="$REPO_ROOT/artifacts/totaltext-curved-word-regions-refined-v2"
  RUN_NAME="totaltext-curved-word-regions-refined-v2"
fi
STORAGE_ROOT="${CONTEXT_STR_RUNS_ROOT:-$(dirname "$(dirname "$REPO_ROOT")")/context-str-runs}"
RUN_DEFAULT="$STORAGE_ROOT/$RUN_NAME"
DATA_ROOT="${2:-$DATA_DEFAULT}"
RUN_ROOT="${3:-$RUN_DEFAULT}"
test -x "$PY" || { echo "Missing qualified Python: $PY" >&2; exit 1; }
test -f "$DATA_ROOT/qualification.json" || { echo "Missing prepared target qualification: $DATA_ROOT" >&2; exit 1; }
test -f "$CONFIG" || { echo "Missing frozen config: $CONFIG" >&2; exit 1; }
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
exec "$PY" -u -m segmentation.train --config "$CONFIG" --data-root "$DATA_ROOT" --manifest-root "$DATA_ROOT" --output-root "$RUN_ROOT" --device cuda
