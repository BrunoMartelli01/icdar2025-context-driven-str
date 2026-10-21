#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [storage-root]" >&2
  exit 2
fi

STORAGE_ROOT="${1:-/extra1/bmartelli}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/.venvs/context-str-main-py311/bin/python"
DOWNLOADS="$STORAGE_ROOT/datasets/context-str/ctw1500/downloads"
PREPARED="$STORAGE_ROOT/datasets/context-str/scene-segmentation-v1"

test -x "$PY"
test -f "$DOWNLOADS/ctw1500_mmocr.zip"
test -f "$PREPARED/manifests/ctw1500_train.yaml"
test -f "$PREPARED/manifests/ctw1500_test.yaml"

bash "$REPO_ROOT/scripts/download_ctw1500_recognition_host.sh" "$DOWNLOADS"
PYTHONUNBUFFERED=1 "$PY" -u -m scripts.prepare_ctw1500_recognition \
  --annotation-archive "$DOWNLOADS/ctw1500_e2e_annos.zip" \
  --detection-archive "$DOWNLOADS/ctw1500_mmocr.zip" \
  --prepared-root "$PREPARED"
