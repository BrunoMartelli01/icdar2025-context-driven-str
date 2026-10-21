#!/usr/bin/env bash
# Opt-in GPU training for the isolated TRAIN-only filled-box v3 target.
set -euo pipefail
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [run-root]" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/.venvs/context-str-main-py311/bin/python"
DATA="$REPO_ROOT/artifacts/totaltext-adaptive-word-boxes-v3"
CONFIG="$REPO_ROOT/configs/approximate/segmentation_total_text_adaptive_word_boxes_v3.yaml"
RUN_ROOT="${1:-$(dirname "$(dirname "$REPO_ROOT")")/context-str-runs/totaltext-adaptive-word-boxes-v3-500ep}"
test -x "$PY" || { echo "Missing qualified Python: $PY" >&2; exit 1; }
test -f "$DATA/qualification_native128.json" || { echo "Missing native-128 qualification" >&2; exit 1; }
printf '%s  %s\n' bd3697b63318d6412cab8aca668ca82f67c443d1313d4b03b2e3adbd6767c9bf "$DATA/manifests/total_text_train.yaml" | sha256sum --check -
printf '%s  %s\n' 19197978be1b6f82dcb9b5e8f0d252f34262e2678f37edef10d77a43274155c5 "$DATA/qualification_native128.json" | sha256sum --check -
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
exec "$PY" -u -m segmentation.train --config "$CONFIG" --data-root "$DATA" --manifest-root "$DATA" --output-root "$RUN_ROOT" --device cuda
