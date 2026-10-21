#!/usr/bin/env bash
# Foreground, restartable PARSeq validation for the two completed word-F1 runs.
set -euo pipefail

if [[ $# -gt 1 ]]; then
    printf 'Usage: bash scripts/run_totaltext_word_f1_parseq_val_host.sh\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host_root="$(cd "$repo/../.." && pwd)"
if [[ $# -eq 1 && "$(cd "${1%/}" && pwd)" != "$host_root" ]]; then
    printf 'Wrong host root: expected %s; omit the argument to avoid path errors\n' "$host_root" >&2
    exit 2
fi
py="$repo/.venvs/context-str-main-py311/bin/python"
revision=1902db043c029a7e03a3818c616c06600af574be
artifacts="$host_root/context-str-artifacts/parseq"
runs="$host_root/context-str-runs"
target="$repo/artifacts/totaltext-curved-word-boxes-v4"
output="$runs/totaltext-letterbox-word-f1-separator-parseq-val-v1"

cd "$repo"
export PYTHONUNBUFFERED=1
"$py" -u -m scripts.evaluate_totaltext_word_f1_separator_parseq_val \
    --target-root "$target" \
    --training-runs "$runs/totaltext-letterbox-word-f1-separator-v1" \
    --parseq-source "$artifacts/parseq-$revision" \
    --parseq-source-archive "$artifacts/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$artifacts/parseq-bb5792a6.pt" \
    --reuse-ocr "$runs/totaltext-v4-image-frequency-ablation-v1/validation-pad0-parseq-v1/ocr" \
    --output-dir "$output" --device cuda:0 --parseq-batch-size 8 --preflight-only
"$py" -u -m scripts.evaluate_totaltext_word_f1_separator_parseq_val \
    --target-root "$target" \
    --training-runs "$runs/totaltext-letterbox-word-f1-separator-v1" \
    --parseq-source "$artifacts/parseq-$revision" \
    --parseq-source-archive "$artifacts/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$artifacts/parseq-bb5792a6.pt" \
    --reuse-ocr "$runs/totaltext-v4-image-frequency-ablation-v1/validation-pad0-parseq-v1/ocr" \
    --output-dir "$output" --device cuda:0 --parseq-batch-size 8
