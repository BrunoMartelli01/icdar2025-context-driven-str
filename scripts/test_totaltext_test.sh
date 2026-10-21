#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TOTALTEXT_PYTHON:-$repo/.venvs/context-str-main-py311/bin/python}"
cd "$repo"
"$python_bin" -m pytest -q tests/localize/test_crop_recovery.py tests/localize/test_crop_word_groups.py tests/recognition/test_group_word_selection.py tests/recognition/test_crop_group_test_suite.py
PYTHONUNBUFFERED=1 "$python_bin" -u -m scripts.run_totaltext_crop_group_test_suite "$@"
