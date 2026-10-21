#!/usr/bin/env bash
# Local one-command Total-Text suite. Paths can be overridden with environment
# variables; the Python runner verifies hashes and the exact TRAIN split.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
host_root="${TOTALTEXT_HOST_ROOT:-$(cd "$repo/../.." && pwd)}"
python_bin="$repo/.venvs/context-str-main-py311/bin/python"
data_root="${TOTALTEXT_DATA_ROOT:-$repo/artifacts/totaltext-curved-word-boxes-v4}"
probability_cache="${TOTALTEXT_PROBABILITY_CACHE:-$host_root/context-str-runs/totaltext-letterbox-threshold-sweep-train-v1}"
baseline_ocr_cache="${TOTALTEXT_BASELINE_OCR_CACHE:-$host_root/context-str-runs/totaltext-v4-image-frequency-ablation-v1/validation-pad0-parseq-v1/ocr}"
parseq_root="${TOTALTEXT_PARSEQ_ROOT:-$host_root/context-str-artifacts/parseq}"
parseq_revision=1902db043c029a7e03a3818c616c06600af574be
parseq_source="${TOTALTEXT_PARSEQ_SOURCE:-$parseq_root/parseq-$parseq_revision}"
parseq_archive="${TOTALTEXT_PARSEQ_ARCHIVE:-$parseq_root/parseq-$parseq_revision.tar.gz}"
parseq_checkpoint="${TOTALTEXT_PARSEQ_CHECKPOINT:-$parseq_root/parseq-bb5792a6.pt}"
cd "$repo"
suite_tag="$(sha256sum \
  recognition/crop_clip_battery.py \
  recognition/parseq.py \
  scripts/crop_clip_runtime.py \
  scripts/evaluate_totaltext_highres_contour_padding.py \
  scripts/run_totaltext_crop_clip_battery.py \
  configs/approximate/totaltext_crop_clip_battery_train_v1.json \
  | sha256sum | cut -c1-12)"
suite_output="${TOTALTEXT_SUITE_OUTPUT:-/tmp/totaltext-crop-clip-local-suite-$suite_tag}"

bash scripts/run_totaltext_crop_clip_battery_host.sh \
  --data-root "$data_root" \
  --probability-cache "$probability_cache" \
  --baseline-ocr-cache "$baseline_ocr_cache" \
  --output-dir "$suite_output/cached-threshold-audit" \
  --device cpu --cached-threshold-audit

PYTHONUNBUFFERED=1 "$python_bin" -u -m scripts.run_totaltext_crop_clip_battery \
  --config configs/approximate/totaltext_crop_clip_battery_train_v1.json \
  --data-root "$data_root" \
  --probability-cache "$probability_cache" \
  --baseline-ocr-cache "$baseline_ocr_cache" \
  --parseq-source "$parseq_source" \
  --parseq-source-archive "$parseq_archive" \
  --parseq-checkpoint "$parseq_checkpoint" \
  --output-dir "$suite_output/parseq-smoke" \
  --device cpu --batch-size 16 --limit 1 --parseq-only

if [[ -n "${CLIP_MODEL_DIR:-}" ]]; then
  PYTHONUNBUFFERED=1 "$python_bin" -u -m scripts.run_totaltext_crop_clip_battery \
    --config configs/approximate/totaltext_crop_clip_battery_train_v1.json \
    --data-root "$data_root" \
    --probability-cache "$probability_cache" \
    --baseline-ocr-cache "$baseline_ocr_cache" \
    --parseq-source "$parseq_source" \
    --parseq-source-archive "$parseq_archive" \
    --parseq-checkpoint "$parseq_checkpoint" \
    --clip-model-dir "$CLIP_MODEL_DIR" \
    --clip-revision "$(basename "$CLIP_MODEL_DIR")" \
    --output-dir "$suite_output/full-clip-train" \
    --device "${CLIP_DEVICE:-cuda:0}" --batch-size "${CLIP_BATCH_SIZE:-16}"
else
  printf 'CLIP stage pending: set CLIP_MODEL_DIR to a local 40-character commit snapshot to include it.\n'
fi

printf 'Total-Text reports: %s\n' "$suite_output"
