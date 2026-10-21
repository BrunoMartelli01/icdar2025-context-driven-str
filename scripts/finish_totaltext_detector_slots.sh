#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${TOTALTEXT_SLOTS_PYTHON:-${repo_root}/.venvs/context-str-main-py311/bin/python}"
detector_run="${TOTALTEXT_SLOTS_DETECTOR_RUN:-/tmp/totaltext-independent-detector-slots-v2/full}"
output_root="${TOTALTEXT_SLOTS_OUTPUT_ROOT:-/tmp/totaltext-independent-detector-slots-v3}"
exec "${python_bin}" -m scripts.evaluate_totaltext_detector_slots \
  --analyze-only --detector-run "${detector_run}" --output-root "${output_root}" "$@"
