#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 || $# -gt 5 ]]; then
  printf '%s\n' 'Usage: run_teacher_student_battery.sh MAIN_PYTHON CONFIG OUTPUT YOLO_PYTHON [--smoke]' >&2
  exit 2
fi
task_main_python=$1
task_config=$2
task_output=$3
task_yolo_python=$4
task_smoke=()
if [[ ${5:-} == --smoke ]]; then task_smoke=(--smoke); fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
for task_stage in selector pseudo segmentation recognition student_prepare; do
  "$task_main_python" -m scripts.run_teacher_student_tests --mode approximate --config "$task_config" --output "$task_output" --stage "$task_stage" "${task_smoke[@]}"
done
"$task_yolo_python" -m scripts.run_teacher_student_yolo --root "$task_output/student"
for task_stage in student_evaluate report; do
  "$task_main_python" -m scripts.run_teacher_student_tests --mode approximate --config "$task_config" --output "$task_output" --stage "$task_stage" "${task_smoke[@]}"
done
