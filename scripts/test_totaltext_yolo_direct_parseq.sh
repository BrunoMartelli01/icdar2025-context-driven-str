#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_path="$repo_dir/.venvs/context-str-main-py311/bin/python"
"$python_path" -m scripts.evaluate_totaltext_yolo_direct_parseq "$@"
