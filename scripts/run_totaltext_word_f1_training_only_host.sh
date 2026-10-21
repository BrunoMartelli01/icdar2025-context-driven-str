#!/usr/bin/env bash
# Resume/start only the two TRAIN-only branches; no TEST or PARSeq job.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: bash scripts/run_totaltext_word_f1_training_only_host.sh /extra1/bmartelli\n' >&2
    exit 2
fi
host_root="${1%/}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
target="$repo/artifacts/totaltext-curved-word-boxes-v4"
output="$host_root/context-str-runs/totaltext-letterbox-word-f1-separator-v1"

cd "$repo"
export PYTHONUNBUFFERED=1
for branch in word_f1_control word_f1_separator_head; do
    "$py" -u -m scripts.train_totaltext_letterbox_word_f1_separator_head \
        --branch "$branch" --target-root "$target" --output-root "$output" \
        --device cuda:0 --preflight-only
done

free_mb="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < 9000 )); then
    printf 'GPU 0 needs at least 9000 MiB free; observed %s MiB\n' "$free_mb" >&2
    exit 1
fi

train_branch() {
    "$py" -u -m scripts.train_totaltext_letterbox_word_f1_separator_head \
        --branch "$1" --target-root "$target" --output-root "$output" \
        --device cuda:0
}
if (( free_mb < 16000 )); then
    printf '[serial] GPU 0 free=%s MiB\n' "$free_mb"
    train_branch word_f1_control
    train_branch word_f1_separator_head
    exit 0
fi

printf '[parallel] GPU 0 free=%s MiB; starting two TRAIN branches only\n' "$free_mb"
train_branch word_f1_control & control_pid=$!
train_branch word_f1_separator_head & separator_pid=$!
trap 'kill "$control_pid" "$separator_pid" 2>/dev/null || true; wait "$control_pid" "$separator_pid" 2>/dev/null || true' INT TERM
control_status=0
separator_status=0
wait "$control_pid" || control_status=$?
wait "$separator_pid" || separator_status=$?
trap - INT TERM
printf '[parallel] control exit=%s separator exit=%s\n' "$control_status" "$separator_status"
if (( control_status != 0 || separator_status != 0 )); then
    exit 1
fi
