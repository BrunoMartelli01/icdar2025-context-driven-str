#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${TOTALTEXT_SLOTS_PYTHON:-${repo_root}/.venvs/context-str-main-py311/bin/python}"
paths="${TOTALTEXT_SLOTS_PATHS:-${repo_root}/configs/approximate/totaltext_independent_detector_slots_local_paths_v1.json}"
exec "${python_bin}" -m scripts.evaluate_totaltext_detector_slots --paths "${paths}" "$@"
