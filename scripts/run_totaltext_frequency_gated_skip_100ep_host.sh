#!/usr/bin/env bash
# One foreground TRAIN-only run and held-out PARSeq check; no TEST inference.
set -euo pipefail

if [[ $# -ne 0 ]]; then
    printf 'Usage: bash scripts/run_totaltext_frequency_gated_skip_100ep_host.sh\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host_root="$(cd "$repo/../.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
target="$repo/artifacts/totaltext-curved-word-boxes-v4"
output="$host_root/context-str-runs/totaltext-frequency-gated-skip-100ep-v1"
run="$output/frequency_gated_skip_100ep"
ocr_output="$output/validation-pad0-parseq-threshold065-v1"
old_ocr="$host_root/context-str-runs/totaltext-letterbox-threshold-sweep-train-v1/ocr"
baseline="$host_root/context-str-runs/totaltext-letterbox-threshold-sweep-train-v1/results.json"
parseq="$host_root/context-str-artifacts/parseq"
revision=1902db043c029a7e03a3818c616c06600af574be

cd "$repo"
export PYTHONUNBUFFERED=1
"$py" -u -m scripts.train_totaltext_frequency_gated_skip_100ep \
    --target-root "$target" --output-root "$output" --device cuda:0 --check
"$py" -u -m scripts.train_totaltext_frequency_gated_skip_100ep \
    --target-root "$target" --output-root "$output" --device cuda:0
"$py" -u -m scripts.evaluate_totaltext_frequency_gated_skip_parseq_val \
    --target-root "$target" --training-run "$run" --baseline-report "$baseline" \
    --parseq-source "$parseq/parseq-$revision" \
    --parseq-source-archive "$parseq/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$parseq/parseq-bb5792a6.pt" \
    --reuse-ocr "$old_ocr" --output-dir "$ocr_output" \
    --device cuda:0 --parseq-batch-size 32 --preflight-only
"$py" -u -m scripts.evaluate_totaltext_frequency_gated_skip_parseq_val \
    --target-root "$target" --training-run "$run" --baseline-report "$baseline" \
    --parseq-source "$parseq/parseq-$revision" \
    --parseq-source-archive "$parseq/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$parseq/parseq-bb5792a6.pt" \
    --reuse-ocr "$old_ocr" --output-dir "$ocr_output" \
    --device cuda:0 --parseq-batch-size 32
