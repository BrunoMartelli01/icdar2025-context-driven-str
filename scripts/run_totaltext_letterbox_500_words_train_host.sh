#!/usr/bin/env bash
# Frozen TRAIN-125 word/PARSeq audit; foreground, progress visible, resumable.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: bash scripts/run_totaltext_letterbox_500_words_train_host.sh /extra1/bmartelli\n' >&2
    exit 2
fi
host_root="${1%/}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo/.venvs/context-str-main-py311/bin/python"
target_root="$repo/artifacts/totaltext-curved-word-boxes-v4"
source_run="$host_root/context-str-runs/totaltext-v4-image-frequency-ablation-v1/letterbox"
pair_run_root="$host_root/context-str-runs/totaltext-letterbox-500-separator-pair-v1"
parseq_root="$host_root/context-str-artifacts/parseq"
revision=1902db043c029a7e03a3818c616c06600af574be
output="$host_root/context-str-runs/totaltext-letterbox-500-word-check-train-v1"

for required in "$python_bin" "$target_root/manifests/total_text_train.yaml" \
    "$source_run/run_manifest.json" "$source_run/best.pt" \
    "$pair_run_root/letterbox_500_control/run_manifest.json" \
    "$pair_run_root/letterbox_500_control/best.pt" \
    "$pair_run_root/letterbox_500_sep5/run_manifest.json" \
    "$pair_run_root/letterbox_500_sep5/best.pt" \
    "$parseq_root/parseq-$revision.tar.gz" "$parseq_root/parseq-bb5792a6.pt"; do
    if [[ ! -f "$required" ]]; then
        printf 'Missing frozen input: %s\n' "$required" >&2
        exit 1
    fi
done
if [[ ! -d "$parseq_root/parseq-$revision" ]]; then
    printf 'Missing qualified PARSeq runtime source\n' >&2
    exit 1
fi

cd "$repo"
export PYTHONUNBUFFERED=1
exec "$python_bin" -u -m scripts.evaluate_totaltext_letterbox_500_words_train \
    --target-root "$target_root" --source-run "$source_run" \
    --pair-run-root "$pair_run_root" \
    --parseq-source "$parseq_root/parseq-$revision" \
    --parseq-source-archive "$parseq_root/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$parseq_root/parseq-bb5792a6.pt" \
    --output-dir "$output" --device cuda:0 --parseq-batch-size 32
