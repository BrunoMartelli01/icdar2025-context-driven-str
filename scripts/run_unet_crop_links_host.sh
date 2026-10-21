#!/usr/bin/env bash
# Frozen predictions -> learned crop controls -> real PARSeq, with V2 untouched.
set -euo pipefail
if [[ $# -ne 5 ]]; then
    printf 'Usage: bash scripts/run_unet_crop_links_host.sh WORD_ROOT V1_ROOT V2_ROOT NEW_OUTPUT_ROOT PARSEQ_ROOT\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
word="$(realpath "$1")"
v1="$(realpath "$2")"
v2="$(realpath "$3")"
output="$(realpath -m "$4")"
parseq="$(realpath "$5")"
cd "$repo"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$output"
exec 9>"$output/suite.lock"
flock -n 9 || { printf 'The learned-crop suite is already running.\n' >&2; exit 2; }
if [[ -f "$v2/suite.lock" ]]; then
    printf '[queue] Waiting for the V2 suite to release the GPU.\n'
    exec 8<"$v2/suite.lock"
    flock -s 8
    exec 8<&-
fi
"$py" -c 'import sys,shutil; from pathlib import Path; from scripts.run_unet_crop_links import read_config,DEFAULT_CONFIG,qualify_global; c,_=read_config(DEFAULT_CONFIG,"approximate"); qualify_global(Path(sys.argv[1])/"links_isolated",c); assert shutil.disk_usage(sys.argv[2]).free>=8_000_000_000, "Need8GB disk for new checkpoints and crop exports"' "$v2" "$output"
revision="$("$py" -c 'from recognition.parseq import PARSEQ_RUNTIME_REVISION; print(PARSEQ_RUNTIME_REVISION)')"
"$py" -c 'import sys; from pathlib import Path; from recognition.parseq import PARSEQ_CHECKPOINT_SHA256,PARSEQ_RUNTIME_ARCHIVE_SHA256; from scripts.prune_segmentation_checkpoints import sha256_file; p=Path(sys.argv[1]); assert sha256_file(p/"parseq-bb5792a6.pt")==PARSEQ_CHECKPOINT_SHA256; assert sha256_file(p/("parseq-"+sys.argv[2]+".tar.gz"))==PARSEQ_RUNTIME_ARCHIVE_SHA256; assert (p/("parseq-"+sys.argv[2])).is_dir()' "$parseq" "$revision"
common=(--mode approximate --word-root "$word" --v1-root "$v1" --v2-root "$v2" --output-root "$output" --device cuda:0)
"$py" -u -m scripts.run_unet_crop_links "${common[@]}" --action prepare
for branch in control global_links; do
    "$py" -u -m scripts.run_unet_crop_links "${common[@]}" --action evaluate --branch "$branch"
done
for branch in crop_word crop_links; do
    "$py" -u -m scripts.run_unet_crop_links "${common[@]}" --action smoke --branch "$branch"
    "$py" -u -m scripts.run_unet_crop_links "${common[@]}" --action train --branch "$branch"
    "$py" -u -m scripts.run_unet_crop_links "${common[@]}" --action evaluate --branch "$branch"
done
"$py" -u -m scripts.evaluate_unet_crop_links_ocr --mode approximate --subset all \
    --word-root "$word" --run-root "$output" --parseq-source "$parseq/parseq-$revision" \
    --parseq-source-archive "$parseq/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$parseq/parseq-bb5792a6.pt" --device cuda:0
"$py" -u -m scripts.summarize_unet_crop_links --run-root "$output"
