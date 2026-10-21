#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/yolo-word-slots/ultralytics-config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/yolo-word-slots/matplotlib-config}"
"$repo_dir/.venvs/context-str-yolo-py311/bin/python" -m scripts.compare_yolo_parseq_four_datasets "$@"
