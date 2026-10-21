#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/.venvs/context-str-main-py311/bin/python"
DOWNLOAD_ROOT="${1:-/extra1/bmartelli/datasets/context-str}"
OUTPUT_ROOT="${2:-/extra1/bmartelli/datasets/context-str/scene-segmentation-v1}"

test -x "$PY" || { echo "Missing Python environment: $PY" >&2; exit 1; }
exec env PYTHONUNBUFFERED=1 "$PY" -u -m scripts.prepare_scene_text_segmentation \
  --download-root "$DOWNLOAD_ROOT" \
  --output-root "$OUTPUT_ROOT"
