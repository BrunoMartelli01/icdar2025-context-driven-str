#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: run_grounded_word_recovery.sh ABSOLUTE_PYTHON APPROXIMATE_CONFIG FRESH_OUTPUT' >&2
  exit 2
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
exec "$1" -u -m scripts.run_grounded_word_recovery --mode approximate --config "$2" --output "$3"
