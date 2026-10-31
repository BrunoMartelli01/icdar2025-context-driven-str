#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf '%s\n' 'Usage: run_graph_occlusion_battery.sh PYTHON CONFIG OUTPUT [--smoke]' >&2
  exit 2
fi
task_python=$1
task_config=$2
task_output=$3
task_smoke=()
if [[ ${4:-} == --smoke ]]; then task_smoke=(--smoke); fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
for task_stage in graph_export graph segmentation report; do
  "$task_python" -m scripts.run_graph_occlusion_research --mode approximate --config "$task_config" --output "$task_output" --stage "$task_stage" "${task_smoke[@]}"
done
