#!/usr/bin/env bash
# Two separate, simultaneous single-GPU TRAIN processes; wrapper stays foreground.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: bash scripts/run_totaltext_letterbox_500_separator_pair_host.sh /extra1/bmartelli\n' >&2
    exit 2
fi
host_root="${1%/}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo/.venvs/context-str-main-py311/bin/python"
target_root="$repo/artifacts/totaltext-curved-word-boxes-v4"
source_run="$host_root/context-str-runs/totaltext-v4-image-frequency-ablation-v1/letterbox"
output_root="$host_root/context-str-runs/totaltext-letterbox-500-separator-pair-v1"

for required in "$python_bin" "$target_root/manifests/total_text_train.yaml" \
    "$source_run/run_manifest.json" "$source_run/best.pt"; do
    if [[ ! -f "$required" ]]; then
        printf 'Missing frozen input: %s\n' "$required" >&2
        exit 1
    fi
done

cd "$repo"
export PYTHONUNBUFFERED=1
for branch in letterbox_500_control letterbox_500_sep5; do
    "$python_bin" -u -m scripts.train_totaltext_letterbox_500_separator_pair \
        --branch "$branch" --target-root "$target_root" \
        --source-run "$source_run" --device cpu --check
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi is required for parallel GPU free-memory preflight\n' >&2
    exit 1
fi
free_mb="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits)"
if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < 16000 )); then
    printf 'Parallel start refused: GPU 0 has %s MiB free; need at least 16000 MiB\n' "$free_mb" >&2
    exit 1
fi
printf '[parallel] GPU 0 free=%s MiB; launching control=3.0 and separator=5.0\n' "$free_mb"

"$python_bin" -u -m scripts.train_totaltext_letterbox_500_separator_pair \
    --branch letterbox_500_control --target-root "$target_root" \
    --source-run "$source_run" --output-root "$output_root" --device cuda:0 &
control_pid=$!
"$python_bin" -u -m scripts.train_totaltext_letterbox_500_separator_pair \
    --branch letterbox_500_sep5 --target-root "$target_root" \
    --source-run "$source_run" --output-root "$output_root" --device cuda:0 &
sep_pid=$!

trap 'kill "$control_pid" "$sep_pid" 2>/dev/null || true; wait "$control_pid" "$sep_pid" 2>/dev/null || true' INT TERM
control_status=0
sep_status=0
wait "$control_pid" || control_status=$?
wait "$sep_pid" || sep_status=$?
trap - INT TERM
printf '[parallel] control exit=%s separator exit=%s\n' "$control_status" "$sep_status"
if (( control_status != 0 || sep_status != 0 )); then
    exit 1
fi
