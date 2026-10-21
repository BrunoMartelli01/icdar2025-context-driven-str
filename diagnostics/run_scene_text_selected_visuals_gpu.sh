#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /extra1/bmartelli" >&2
  exit 2
fi

ROOT=${1%/}
REPO="$ROOT/projects/icdar2025-context-driven-str"
PY="$REPO/.venvs/context-str-main-py311/bin/python"
DATA="$ROOT/datasets/context-str/scene-segmentation-v1"
RUNS="$ROOT/context-str-runs/scene-text-v1-parseq-test-v1"
SEG_RUNS="$ROOT/context-str-runs/scene-text-segmentation-v1"
OUT="$REPO/artifacts/diagnostics/scene-text-selected-visual-audit-with-unet-v2"
ZIP="$REPO/artifacts/diagnostics/scene-text-selected-visual-audit-with-unet-v2.zip"

cd "$REPO"

if [[ -f "$ZIP" && -f "$OUT/manifest.json" ]]; then
  unzip -t "$ZIP" >/dev/null
  sha256sum "$ZIP"
  echo "[reuse] $ZIP"
  exit 0
fi
if [[ -e "$OUT" || -e "$ZIP" ]]; then
  echo "incomplete output exists; inspect before removing: $OUT or $ZIP" >&2
  exit 1
fi

PYTHONUNBUFFERED=1 "$PY" -u -m diagnostics.export_scene_text_selected_visuals \
  --data-root "$DATA" \
  --run-root "$RUNS" \
  --segmentation-runs-root "$SEG_RUNS" \
  --configs-root "$REPO/configs/approximate" \
  --include-predicted-masks \
  --device cuda:0 \
  --output-dir "$OUT" \
  --zip-output "$ZIP"
