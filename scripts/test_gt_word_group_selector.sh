#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TOTALTEXT_PYTHON:-$repo/.venvs/context-str-main-py311/bin/python}"
cd "$repo"
"$python_bin" -m pytest -q tests/recognition/test_gt_word_group_selector.py
PYTHONUNBUFFERED=1 "$python_bin" -u -m diagnostics.gt_word_group_selector "$@"
