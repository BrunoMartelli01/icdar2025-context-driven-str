#!/usr/bin/env bash
# Restart only the frozen threshold-0.65 TEST job, never the two TRAIN jobs.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: bash scripts/run_totaltext_letterbox_threshold065_test_host.sh /extra1/bmartelli\n' >&2
    exit 2
fi
host_root="${1%/}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
data="$host_root/datasets/context-str/scene-segmentation-v1"
training="$host_root/context-str-runs/totaltext-v4-image-frequency-ablation-v1"
threshold_report="$host_root/context-str-runs/totaltext-letterbox-threshold-sweep-train-v1/results.json"
parseq="$host_root/context-str-artifacts/parseq"
revision=1902db043c029a7e03a3818c616c06600af574be
output="$host_root/context-str-runs/totaltext-letterbox-threshold065-parseq-test-v1"

cd "$repo"
export PYTHONUNBUFFERED=1
exec "$py" -u -m scripts.evaluate_totaltext_letterbox_threshold065_test \
    --test-manifest "$data/manifests/total_text_test.yaml" \
    --data-root "$data" \
    --target-root "$repo/artifacts/totaltext-curved-word-boxes-v4" \
    --training-runs "$training" \
    --parseq-source "$parseq/parseq-$revision" \
    --parseq-source-archive "$parseq/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$parseq/parseq-bb5792a6.pt" \
    --threshold-selection-report "$threshold_report" \
    --output-dir "$output" --device cuda:0 --parseq-batch-size 32
