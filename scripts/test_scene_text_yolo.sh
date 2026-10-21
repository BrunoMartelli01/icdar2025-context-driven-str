#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/yolo-word-slots/ultralytics-config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/yolo-word-slots/matplotlib-config}"
python_bin="$repo_dir/.venvs/context-str-yolo-py311/bin/python"
# A real model smoke test precedes the complete requested TEST splits.
"$python_bin" -m scripts.evaluate_scene_text_yolo_parseq --limit 1 "$@"
"$python_bin" -m scripts.evaluate_scene_text_yolo_parseq "$@"
