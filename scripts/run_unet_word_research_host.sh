#!/usr/bin/env bash
# Paired-target preflight, five sequential GPU branches, frozen-crop PARSeq.
set -euo pipefail
if [[ $# -ne 5 ]]; then
    printf 'Usage: bash scripts/run_unet_word_research_host.sh WORD_ROOT GLYPH_ROOT KERNEL_RUN NEW_OUTPUT_ROOT PARSEQ_ARTIFACT_ROOT\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
word="$(realpath "$1")"
glyph="$(realpath "$2")"
kernel="$(realpath "$3")"
output="$(realpath -m "$4")"
parseq="$(realpath "$5")"
cd "$repo"
[[ -f "$py" ]] || { printf 'Missing Python environment: %s\n' "$py" >&2; exit 2; }
revision="$("$py" -c 'from recognition.parseq import PARSEQ_RUNTIME_REVISION; print(PARSEQ_RUNTIME_REVISION)')"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$output"
exec 9>"$output/suite.lock"
flock -n 9 || { printf 'This suite is already running.\n' >&2; exit 2; }
for required in "$py" "$kernel/best.pt" "$kernel/run_manifest.json" \
                "$parseq/parseq-$revision.tar.gz" "$parseq/parseq-bb5792a6.pt"; do
    [[ -f "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 2; }
done
"$py" -u -c 'import shutil,sys; from pathlib import Path; from recognition.parseq import PARSEQ_CHECKPOINT_SHA256,PARSEQ_RUNTIME_ARCHIVE_SHA256; from scripts.prune_segmentation_checkpoints import sha256_file; p=Path(sys.argv[1]); assert sha256_file(p/"parseq-bb5792a6.pt")==PARSEQ_CHECKPOINT_SHA256; assert sha256_file(p/("parseq-"+sys.argv[2]+".tar.gz"))==PARSEQ_RUNTIME_ARCHIVE_SHA256; assert (p/("parseq-"+sys.argv[2])).is_dir(); assert shutil.disk_usage(sys.argv[3]).free>=10_000_000_000, "Need10GB free for five resumable runs and exported maps"; print("[preflight] local PARSeq and disk PASS",flush=True)' "$parseq" "$revision" "$output"
common=(--mode approximate --word-root "$word" --glyph-root "$glyph" --kernel-run "$kernel" --output-root "$output")
"$py" -u -m scripts.run_unet_word_research "${common[@]}" --action preflight --device cpu --num-workers 0
"$py" -u -m scripts.diagnose_unet_mask_stability --mode approximate --kernel-run "$kernel" --output-root "$output/stability"
for branch in control glyph_word embedding adaptive_threshold quality; do
    "$py" -u -m scripts.run_unet_word_research "${common[@]}" --branch "$branch" --action smoke --device cuda:0
    "$py" -u -m scripts.run_unet_word_research "${common[@]}" --branch "$branch" --action train --device cuda:0
    "$py" -u -m scripts.run_unet_word_research "${common[@]}" --branch "$branch" --action evaluate --device cuda:0
done
"$py" -u -m scripts.evaluate_unet_word_research_ocr --mode approximate --word-root "$word" --run-root "$output" \
    --parseq-source "$parseq/parseq-$revision" --parseq-source-archive "$parseq/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$parseq/parseq-bb5792a6.pt" --device cuda:0
"$py" -u -m scripts.summarize_unet_word_research --run-root "$output"
