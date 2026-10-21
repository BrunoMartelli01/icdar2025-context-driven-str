#!/usr/bin/env bash
# New methods only; completed V2/crop-link experiments are never retrained here.
set -euo pipefail
if [[ $# -lt 5 ]]; then
    printf 'Usage: bash scripts/run_unet_detection_methods_host.sh WORD_ROOT V1_ROOT NEW_OUTPUT_ROOT PARSEQ_ROOT --without-synthtext | --synthtext-root DIRECTORY [--config FILE]\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
word="$(realpath "$1")"
v1="$(realpath "$2")"
output="$(realpath -m "$3")"
parseq="$(realpath "$4")"
shift 4
synthetic=""
without=0
config="$repo/configs/approximate/unet_detection_methods_v1.json"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --without-synthtext) without=1; shift ;;
        --synthtext-root) synthetic="$(realpath "${2:?--synthtext-root needs a directory}")"; shift 2 ;;
        --config) config="$(realpath "${2:?--config needs a file}")"; shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ -n "$synthetic" && "$without" -eq 1 ]] || [[ -z "$synthetic" && "$without" -eq 0 ]]; then
    printf 'Choose exactly --without-synthtext or --synthtext-root DIRECTORY. Pretraining is never silently skipped.\n' >&2
    exit 2
fi
if [[ -n "$synthetic" && ! -f "$synthetic/gt.mat" ]]; then
    printf 'SynthText requires gt.mat in %s. No download or replacement occurs.\n' "$synthetic" >&2
    exit 2
fi
cd "$repo"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$output"
exec 9>"$output/suite.lock"
flock -n 9 || { printf 'This experiment suite is already running.\n' >&2; exit 2; }
# Validate the immutable suite plan before any training; changing the optional
# synthetic phase requires a new root, even after only preparation.
"$py" -c 'import sys,shutil; from pathlib import Path; from scripts.run_unet_detection_methods import bind,read_config; c,_=read_config(Path(sys.argv[2]),"approximate"); bind(Path(sys.argv[1])/"host_plan.json",{"config":c,"synthtext_root":sys.argv[3] or None}); assert shutil.disk_usage(sys.argv[1]).free>=12_000_000_000,"Need12GB disk for checkpoints and exports"' "$output" "$config" "$synthetic"
revision="$("$py" -c 'from recognition.parseq import PARSEQ_RUNTIME_REVISION; print(PARSEQ_RUNTIME_REVISION)')"
"$py" -c 'import sys; from pathlib import Path; from recognition.parseq import PARSEQ_CHECKPOINT_SHA256,PARSEQ_RUNTIME_ARCHIVE_SHA256; from scripts.prune_segmentation_checkpoints import sha256_file; p=Path(sys.argv[1]); assert sha256_file(p/"parseq-bb5792a6.pt")==PARSEQ_CHECKPOINT_SHA256; assert sha256_file(p/("parseq-"+sys.argv[2]+".tar.gz"))==PARSEQ_RUNTIME_ARCHIVE_SHA256; assert (p/("parseq-"+sys.argv[2])).is_dir()' "$parseq" "$revision"
common=(--mode approximate --config "$config" --word-root "$word" --v1-root "$v1" --output-root "$output" --device cuda:0)
"$py" -u -m scripts.run_unet_detection_methods "${common[@]}" --action preflight
branches=(reference control scale_fusion progressive_kernels fusion_progressive)
if [[ -n "$synthetic" ]]; then
    "$py" -u -m scripts.run_unet_detection_methods "${common[@]}" --action prepare-synthtext --synthtext-root "$synthetic"
    "$py" -u -m scripts.run_unet_detection_methods "${common[@]}" --action pretrain
    branches+=(pretrained_control pretrained_fusion_progressive)
fi
"$py" -u -m scripts.run_unet_detection_methods "${common[@]}" --action evaluate --branch reference
for branch in "${branches[@]:1}"; do
    # Smokes deliberately have a separate root/contract, so they cannot poison
    # a full training resume. Existing passed smokes are not repeated.
    if [[ ! -f "$output/gpu_smokes/$branch/$branch/smoke.json" ]]; then
        "$py" -u -m scripts.run_unet_detection_methods --mode approximate --config "$config" \
            --word-root "$word" --v1-root "$v1" --output-root "$output/gpu_smokes/$branch" \
            --pretrain-root "$output" --device cuda:0 --action smoke --branch "$branch" --limit-images 2
    fi
    "$py" -u -m scripts.run_unet_detection_methods "${common[@]}" --action train --branch "$branch"
    "$py" -u -m scripts.run_unet_detection_methods "${common[@]}" --action evaluate --branch "$branch"
done
"$py" -u -m scripts.evaluate_unet_detection_methods --mode approximate --config "$config" \
    --word-root "$word" --run-root "$output" --device cuda:0 --branches "${branches[@]}" \
    --parseq-source "$parseq/parseq-$revision" --parseq-source-archive "$parseq/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$parseq/parseq-bb5792a6.pt"
printf 'Complete: %s/comparison.md and comparison.json; crop views in ocr_inputs/; gallery in review/index.html\n' "$output"
