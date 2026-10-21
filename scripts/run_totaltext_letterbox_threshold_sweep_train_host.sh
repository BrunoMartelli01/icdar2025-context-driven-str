#!/usr/bin/env bash
# TRAIN validation only. Foreground GPU run, resumable map/crop caches.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: bash scripts/run_totaltext_letterbox_threshold_sweep_train_host.sh /extra1/bmartelli\n' >&2
    exit 2
fi
host_root="${1%/}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo/.venvs/context-str-main-py311/bin/python"
training_runs="$host_root/context-str-runs/totaltext-v4-image-frequency-ablation-v1"
parseq_root="$host_root/context-str-artifacts/parseq"
revision=1902db043c029a7e03a3818c616c06600af574be
output="$host_root/context-str-runs/totaltext-letterbox-threshold-sweep-train-v1"

for required in "$python_bin" \
    "$repo/artifacts/totaltext-curved-word-boxes-v4/manifests/total_text_train.yaml" \
    "$training_runs/letterbox/run_manifest.json" "$training_runs/letterbox/best.pt" \
    "$training_runs/validation-pad0-parseq-v1/results.json" \
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
exec "$python_bin" -u -m scripts.evaluate_totaltext_letterbox_threshold_sweep_train \
    --target-root "$repo/artifacts/totaltext-curved-word-boxes-v4" \
    --training-runs "$training_runs" \
    --parseq-source "$parseq_root/parseq-$revision" \
    --parseq-source-archive "$parseq_root/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$parseq_root/parseq-bb5792a6.pt" \
    --output-dir "$output" \
    --device cuda:0 \
    --parseq-batch-size 32
