#!/usr/bin/env bash
# Four isolated TRAIN ablations + common control; foreground and resumable.
set -euo pipefail
if [[ $# -ne 0 ]]; then
    printf 'Usage: bash scripts/run_totaltext_word_ablation_50ep_host.sh\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host_root="$(cd "$repo/../.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
target="$repo/artifacts/totaltext-curved-word-boxes-v4"
source_run="$host_root/context-str-runs/totaltext-partial-encoder-50ep-v1/frozen_control"
output="$host_root/context-str-runs/totaltext-word-ablation-50ep-v1"
baseline="$host_root/context-str-runs/totaltext-letterbox-threshold-sweep-train-v1/results.json"
parseq="$host_root/context-str-artifacts/parseq"
revision=1902db043c029a7e03a3818c616c06600af574be
cd "$repo"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for file in "$py" "$baseline" "$source_run/best.pt" "$source_run/run_manifest.json" \
            "$parseq/parseq-bb5792a6.pt" "$parseq/parseq-$revision.tar.gz"; do
    [[ -f "$file" ]] || { printf 'Missing required file: %s\n' "$file" >&2; exit 2; }
done
mkdir -p "$output"
exec 9>"$output/suite.lock"
flock -n 9 || { printf 'This suite is already running.\n' >&2; exit 2; }
"$py" -u -c 'import shutil,sys; from pathlib import Path; from recognition.parseq import PARSEQ_CHECKPOINT_SHA256,PARSEQ_RUNTIME_ARCHIVE_SHA256; from scripts.prune_segmentation_checkpoints import sha256_file; p=Path(sys.argv[1]); assert sha256_file(p/"parseq-bb5792a6.pt")==PARSEQ_CHECKPOINT_SHA256; assert sha256_file(p/("parseq-"+sys.argv[2]+".tar.gz"))==PARSEQ_RUNTIME_ARCHIVE_SHA256; assert (p/("parseq-"+sys.argv[2])).is_dir(); free=shutil.disk_usage(sys.argv[3]).free; assert free>=4_000_000_000, "Need4GB free for5 best checkpoints, atomic saves and OCR"; print("[preflight] PARSeq hashes PASS; free disk GB=",round(free/1e9,2),flush=True)' "$parseq" "$revision" "$output"
for branch in control pixel_link word_balanced word_zoom deep_supervision; do
    CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.train_totaltext_word_ablation_50ep \
        --branch "$branch" --target-root "$target" --source-run "$source_run" --device cpu --check
done
for branch in control pixel_link word_balanced word_zoom deep_supervision; do
    printf '[TRAIN] %s:50 additional epochs; GPU foreground\n' "$branch"
    "$py" -u -m scripts.train_totaltext_word_ablation_50ep \
        --branch "$branch" --target-root "$target" --source-run "$source_run" \
        --output-root "$output" --device cuda:0
    "$py" -u -m scripts.evaluate_totaltext_word_ablation_parseq_val \
        --target-root "$target" --training-run "$output/$branch" --baseline-report "$baseline" \
        --parseq-source "$parseq/parseq-$revision" --parseq-source-archive "$parseq/parseq-$revision.tar.gz" \
        --parseq-checkpoint "$parseq/parseq-bb5792a6.pt" \
        --reuse-ocr "$host_root/context-str-runs/totaltext-letterbox-threshold-sweep-train-v1/ocr" \
        --reuse-ocr "$host_root/context-str-runs/totaltext-partial-encoder-50ep-v1/frozen_control-validation-parseq/ocr" \
        --reuse-ocr "$output/control-validation-parseq/ocr" \
        --output-dir "$output/$branch-validation-parseq" --device cuda:0 --parseq-batch-size 32
done
"$py" -u -m scripts.summarize_totaltext_word_ablation --output-root "$output"
