#!/usr/bin/env bash
# Full TRAIN suite by default; pass --limit 1 for the same stages on one image.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TOTALTEXT_PYTHON:-$repo/.venvs/context-str-main-py311/bin/python}"
cd "$repo"
"$python_bin" -m pytest -q tests/localize/test_crop_recovery.py tests/recognition/test_crop_recovery_suite.py
PYTHONUNBUFFERED=1 "$python_bin" -u -m scripts.run_totaltext_crop_recovery_suite "$@"
