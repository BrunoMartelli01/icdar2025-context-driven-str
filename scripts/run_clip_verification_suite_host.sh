#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo/.venvs/context-str-main-py311/bin/python"
cd "$repo"

"$python_bin" -m pytest -q \
  tests/recognition/test_clip_word_verifier.py \
  tests/recognition/test_clip_innovative_verifier.py \
  tests/recognition/test_totaltext_crop_clip_battery.py \
  tests/recognition/test_group_word_selection.py

log_path="${CLIP_SUITE_LOG:-/tmp/clip-verification-host-suite.log}"
PYTHONUNBUFFERED=1 "$python_bin" -u -m scripts.run_clip_verification_suite "$@" 2>&1 | tee -a "$log_path"
