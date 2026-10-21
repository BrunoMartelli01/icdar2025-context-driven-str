#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
python_bin="$repo/.venvs/context-str-main-py311/bin/python"
"$python_bin" -m pytest -q \
  tests/recognition/test_clip_crop_admission.py \
  tests/recognition/test_clip_word_verifier.py \
  tests/recognition/test_clip_innovative_verifier.py
log_path="${CLIP_ADMISSION_LOG:-/tmp/clip-high-recall-admission-host.log}"
PYTHONUNBUFFERED=1 "$python_bin" -u -m scripts.run_clip_high_recall_admission "$@" 2>&1 | tee -a "$log_path"
