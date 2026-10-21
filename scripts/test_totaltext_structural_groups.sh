#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TOTALTEXT_PYTHON:-$repo/.venvs/context-str-main-py311/bin/python}"
cd "$repo"
"$python_bin" -m pytest -q tests/localize/test_structural_word_groups.py tests/localize/test_crop_recovery.py tests/localize/test_crop_word_groups.py
PYTHONUNBUFFERED=1 "$python_bin" -u -m scripts.analyze_totaltext_structural_groups "$@"
