#!/usr/bin/env bash
# Foreground, sequential matched TRAIN continuations plus held-out PARSeq.
set -euo pipefail
if [[ $# -ne 0 ]]; then
    printf 'Usage: bash scripts/run_totaltext_partial_encoder_50ep_host.sh\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host_root="$(cd "$repo/../.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
target="$repo/artifacts/totaltext-curved-word-boxes-v4"
source_run="$host_root/context-str-runs/totaltext-v4-image-frequency-ablation-v1/letterbox"
output="$host_root/context-str-runs/totaltext-partial-encoder-50ep-v1"
baseline="$host_root/context-str-runs/totaltext-letterbox-threshold-sweep-train-v1/results.json"
old_ocr="$host_root/context-str-runs/totaltext-letterbox-threshold-sweep-train-v1/ocr"
parseq="$host_root/context-str-artifacts/parseq"
revision=1902db043c029a7e03a3818c616c06600af574be
for file in "$py" "$baseline" "$parseq/parseq-bb5792a6.pt" "$parseq/parseq-$revision.tar.gz"; do
    [[ -f "$file" ]] || { printf 'Missing required file: %s\n' "$file" >&2; exit 2; }
done
cd "$repo"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$output"
exec 9>"$output/suite.lock"
flock -n 9 || { printf 'This suite is already running.\n' >&2; exit 2; }
"$py" -u -c 'import json,sys; from pathlib import Path; from recognition.parseq import PARSEQ_CHECKPOINT_SHA256,PARSEQ_RUNTIME_ARCHIVE_SHA256; from scripts.prune_segmentation_checkpoints import sha256_file; r=json.loads(Path(sys.argv[1]).read_text()); assert r["images"]==125 and r["gt_words"]==983 and r["thresholds"]["0.65"]["aggregate"]["exact_casefold_iou_ge_50"]==437; assert sha256_file(Path(sys.argv[2]))==PARSEQ_CHECKPOINT_SHA256; assert sha256_file(Path(sys.argv[3]))==PARSEQ_RUNTIME_ARCHIVE_SHA256; assert Path(sys.argv[4]).is_dir(); print("[preflight] baseline and PARSeq hashes PASS", flush=True)' "$baseline" "$parseq/parseq-bb5792a6.pt" "$parseq/parseq-$revision.tar.gz" "$parseq/parseq-$revision"
for branch in frozen_control partial_encoder; do
    "$py" -u -m scripts.train_totaltext_partial_encoder_50ep \
        --branch "$branch" --target-root "$target" --source-run "$source_run" \
        --output-root "$output" --device cpu --check
done
for branch in frozen_control partial_encoder; do
    printf '[TRAIN] %s: 50 additional epochs; foreground GPU execution\n' "$branch"
    "$py" -u -m scripts.train_totaltext_partial_encoder_50ep \
        --branch "$branch" --target-root "$target" --source-run "$source_run" \
        --output-root "$output" --device cuda:0
    "$py" -u -m scripts.evaluate_totaltext_partial_encoder_parseq_val \
        --target-root "$target" --training-run "$output/$branch" --baseline-report "$baseline" \
        --parseq-source "$parseq/parseq-$revision" \
        --parseq-source-archive "$parseq/parseq-$revision.tar.gz" \
        --parseq-checkpoint "$parseq/parseq-bb5792a6.pt" --reuse-ocr "$old_ocr" \
        --reuse-ocr "$output/frozen_control-validation-parseq/ocr" \
        --output-dir "$output/$branch-validation-parseq" --device cuda:0 --parseq-batch-size 32
done
"$py" -u -m scripts.summarize_totaltext_partial_encoder --output-root "$output"
