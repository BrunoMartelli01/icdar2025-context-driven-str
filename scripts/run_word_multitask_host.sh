#!/usr/bin/env bash
# Preflight, GPU smoke, matched training, exports and historical comparison.
set -euo pipefail
if [[ $# -lt 4 ]]; then
    printf 'Usage: bash scripts/run_word_multitask_host.sh TARGET_ROOT NEW_OUTPUT_ROOT ENCODER_CHECKPOINT ENCODER_SHA256 [HISTORICAL_RUN_MANIFEST ...]\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
target_root="$1"
output_root="$2"
encoder_checkpoint="$3"
encoder_sha256="$4"
shift 4
history=()
for report in "$@"; do history+=(--historical-report "$report"); done
cd "$repo"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
common=(--mode approximate --target-root "$target_root" --output-root "$output_root"
        --encoder-checkpoint "$encoder_checkpoint" --encoder-sha256 "$encoder_sha256" --variant all)
"$py" -m scripts.run_word_multitask "${common[@]}" --action preflight --device cpu
for action in smoke train evaluate; do
    "$py" -m scripts.run_word_multitask "${common[@]}" --action "$action" --device cuda:0
done
"$py" -m scripts.compare_word_multitask --run-root "$output_root" \
    --output "$output_root/comparison.json" "${history[@]}"
