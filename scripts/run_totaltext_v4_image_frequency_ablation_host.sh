#!/usr/bin/env bash
set -euo pipefail

host_root="${1:?usage: bash scripts/run_totaltext_v4_image_frequency_ablation_host.sh /extra1/bmartelli}"
mode="${2:-all}"
if [[ "$mode" != "all" && "$mode" != "--validation-only" ]]; then
    printf 'Second argument must be --validation-only when provided\n' >&2
    exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venvs/context-str-main-py311/bin/python"
target_root="$repo_root/artifacts/totaltext-curved-word-boxes-v4"
run_root="$host_root/context-str-runs/totaltext-v4-image-frequency-ablation-v1"
baseline_run="$host_root/context-str-runs/totaltext-v4-highres-ablation-v1/highres_separator_residual"
parseq_revision="1902db043c029a7e03a3818c616c06600af574be"
parseq_root="$host_root/context-str-artifacts/parseq"
parseq_source="$parseq_root/parseq-$parseq_revision"
parseq_archive="$parseq_source.tar.gz"
parseq_checkpoint="$parseq_root/parseq-bb5792a6.pt"

for required_file in "$python_bin" "$target_root/manifests/total_text_train.yaml" \
    "$baseline_run/run_manifest.json" "$baseline_run/best.pt" \
    "$parseq_archive" "$parseq_checkpoint"; do
    if [[ ! -f "$required_file" ]]; then
        printf 'Missing required input: %s\n' "$required_file" >&2
        exit 1
    fi
done
if [[ ! -d "$parseq_source" ]]; then
    printf 'Missing qualified PARSeq source: %s\n' "$parseq_source" >&2
    exit 1
fi

cd "$repo_root"
export PYTHONUNBUFFERED=1
if [[ "$mode" == "all" ]]; then
    printf '[baseline validation] frozen residual checkpoint; matching control\n'
    "$python_bin" -u -m scripts.evaluate_totaltext_v4_image_frequency_baseline \
        --target-root "$target_root" --baseline-run "$baseline_run" \
        --device cuda:0 --output "$run_root/frozen_baseline_validation_geometry.json"
    for variant in photometric letterbox fourier_amplitude wavelet_haar; do
        printf '[preflight] %s\n' "$variant"
        "$python_bin" -u -m scripts.train_totaltext_v4_image_frequency_ablation \
            --variant "$variant" --target-root "$target_root" --device cuda:0 \
            --check --shape-check
        printf '[training] %s\n' "$variant"
        "$python_bin" -u -m scripts.train_totaltext_v4_image_frequency_ablation \
            --variant "$variant" --target-root "$target_root" --device cuda:0 \
            --output-root "$run_root" --stop-after-epoch 100
    done
elif [[ ! -f "$run_root/frozen_baseline_validation_geometry.json" ]]; then
    printf 'Missing frozen baseline validation report; run full host suite first\n' >&2
    exit 1
fi

printf '[validation] native geometry + pad0 PARSeq; frozen baseline and four ablations\n'
"$python_bin" -u -m scripts.evaluate_totaltext_v4_image_frequency_val_parseq \
    --target-root "$target_root" --baseline-run "$baseline_run" \
    --training-runs "$run_root" --parseq-source "$parseq_source" \
    --parseq-source-archive "$parseq_archive" --parseq-checkpoint "$parseq_checkpoint" \
    --parseq-batch-size 32 --device cuda:0 \
    --output-dir "$run_root/validation-pad0-parseq-v1"
