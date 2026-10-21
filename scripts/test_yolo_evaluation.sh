#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
"$repo_dir/.venvs/context-str-yolo-py311/bin/python" -m scripts.evaluate_yolo_dont_care "$@"
