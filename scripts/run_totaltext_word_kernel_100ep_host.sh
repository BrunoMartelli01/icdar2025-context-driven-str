#!/usr/bin/env bash
# Foreground: target qualification, one 100-epoch TRAIN run, held-out PARSeq.
set -euo pipefail
if [[ $# -ne 0 ]]; then
    printf 'Usage: bash scripts/run_totaltext_word_kernel_100ep_host.sh\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host_root="$(cd "$repo/../.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
target="$repo/artifacts/totaltext-curved-word-boxes-v4"
output="$host_root/context-str-runs/totaltext-word-kernel-100ep-v1"
baseline="$host_root/context-str-runs/totaltext-letterbox-threshold-sweep-train-v1/results.json"
parseq="$host_root/context-str-artifacts/parseq"
revision=1902db043c029a7e03a3818c616c06600af574be
cd "$repo"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$output"
exec 9>"$output/runner.lock"
flock -n 9 || { printf 'Word-kernel experiment is already running.\n' >&2; exit 1; }
for required in "$py" "$target/manifests/total_text_train.yaml" "$baseline" \
    "$parseq/parseq-$revision.tar.gz" "$parseq/parseq-bb5792a6.pt"; do
    [[ -f "$required" ]] || { printf 'Missing input: %s\n' "$required" >&2; exit 1; }
done
"$py" -u -m scripts.train_totaltext_word_kernel_100ep \
    --target-root "$target" --output-root "$output" --device cpu --preflight-only
"$py" -u -m scripts.train_totaltext_word_kernel_100ep \
    --target-root "$target" --output-root "$output" --device cuda:0
"$py" -u -m scripts.evaluate_totaltext_word_kernel_parseq_val \
    --target-root "$target" --training-run "$output/word_kernel_100ep" \
    --baseline-report "$baseline" \
    --parseq-source "$parseq/parseq-$revision" \
    --parseq-source-archive "$parseq/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$parseq/parseq-bb5792a6.pt" \
    --reuse-ocr "$host_root/context-str-runs/totaltext-letterbox-threshold-sweep-train-v1/ocr" \
    --output-dir "$output/validation-pad0-parseq-v1" --device cuda:0 --parseq-batch-size 32
