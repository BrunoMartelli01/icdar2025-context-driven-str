#!/usr/bin/env bash
# Foreground supervisor: TRAIN-only ablations and an independently frozen TEST threshold.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: bash scripts/run_totaltext_word_f1_separator_suite_host.sh /extra1/bmartelli\n' >&2
    exit 2
fi
host_root="${1%/}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
target="$repo/artifacts/totaltext-curved-word-boxes-v4"
data="$host_root/datasets/context-str/scene-segmentation-v1"
training="$host_root/context-str-runs/totaltext-v4-image-frequency-ablation-v1"
threshold_report="$host_root/context-str-runs/totaltext-letterbox-threshold-sweep-train-v1/results.json"
output="$host_root/context-str-runs/totaltext-letterbox-word-f1-separator-v1"
threshold_output="$host_root/context-str-runs/totaltext-letterbox-threshold065-parseq-test-v1"
parseq="$host_root/context-str-artifacts/parseq"
revision=1902db043c029a7e03a3818c616c06600af574be

for required in "$py" "$target/manifests/total_text_train.yaml" \
    "$data/manifests/total_text_test.yaml" "$training/letterbox/run_manifest.json" \
    "$training/letterbox/best.pt" "$training/validation-pad0-parseq-v1/results.json" \
    "$threshold_report" "$parseq/parseq-$revision.tar.gz" \
    "$parseq/parseq-bb5792a6.pt"; do
    if [[ ! -f "$required" ]]; then
        printf 'Missing frozen input: %s\n' "$required" >&2
        exit 1
    fi
done
if [[ ! -d "$parseq/parseq-$revision" ]]; then
    printf 'Missing qualified PARSeq source\n' >&2
    exit 1
fi
printf '%s  %s\n' \
    613957db6019d0762a4d890495e3c1c0af5f91a92ced9c232c099a103328a334 \
    "$threshold_report" | sha256sum --check -
cd "$repo"
export PYTHONUNBUFFERED=1

train_branch() {
    "$py" -u -m scripts.train_totaltext_letterbox_word_f1_separator_head \
        --branch "$1" --target-root "$target" --output-root "$output" \
        --device cuda:0
}
test_threshold() {
    "$py" -u -m scripts.evaluate_totaltext_letterbox_threshold065_test \
        --test-manifest "$data/manifests/total_text_test.yaml" \
        --data-root "$data" --target-root "$target" --training-runs "$training" \
        --parseq-source "$parseq/parseq-$revision" \
        --parseq-source-archive "$parseq/parseq-$revision.tar.gz" \
        --parseq-checkpoint "$parseq/parseq-bb5792a6.pt" \
        --threshold-selection-report "$threshold_report" \
        --output-dir "$threshold_output" --device cuda:0 --parseq-batch-size 32
}

for branch in word_f1_control word_f1_separator_head; do
    "$py" -u -m scripts.train_totaltext_letterbox_word_f1_separator_head \
        --branch "$branch" --target-root "$target" --output-root "$output" \
        --device cuda:0 --preflight-only
done
"$py" -u -m scripts.evaluate_totaltext_letterbox_threshold065_test \
    --test-manifest "$data/manifests/total_text_test.yaml" \
    --data-root "$data" --target-root "$target" --training-runs "$training" \
    --parseq-source "$parseq/parseq-$revision" \
    --parseq-source-archive "$parseq/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$parseq/parseq-bb5792a6.pt" \
    --threshold-selection-report "$threshold_report" \
    --output-dir "$threshold_output" --device cuda:0 --preflight-only

free_mb="$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < 9000 )); then
    printf 'GPU 0 does not have the minimum 9 GB free; no GPU job started (%s MB)\n' "$free_mb" >&2
    exit 1
fi
if (( free_mb >= 16000 )); then
    printf '[parallel] GPU 0 free=%s MB; two TRAIN branches and frozen threshold TEST\n' "$free_mb"
    train_branch word_f1_control & control_pid=$!
    train_branch word_f1_separator_head & separator_pid=$!
    test_threshold & threshold_pid=$!
    trap 'kill "$control_pid" "$separator_pid" "$threshold_pid" 2>/dev/null || true; wait "$control_pid" "$separator_pid" "$threshold_pid" 2>/dev/null || true' INT TERM
    status=0
    wait "$control_pid" || status=1
    wait "$separator_pid" || status=1
    wait "$threshold_pid" || status=1
    trap - INT TERM
    exit "$status"
fi
printf '[serial] GPU 0 free=%s MB; conservative execution to avoid OOM\n' "$free_mb"
train_branch word_f1_control
train_branch word_f1_separator_head
test_threshold
