#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo 'Usage: run_word_graph_recovery.sh MAIN_PYTHON SNAPSHOT CONFIG OUTPUT [--smoke]' >&2
  exit 2
fi
word_graph_python=$1
word_graph_snapshot=$2
word_graph_config=$3
word_graph_output=$4
shift 4
cd "$word_graph_snapshot"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
exec "$word_graph_python" -u -m scripts.run_word_graph_recovery --mode approximate \
  --config "$word_graph_config" --output "$word_graph_output" --stage all "$@"
