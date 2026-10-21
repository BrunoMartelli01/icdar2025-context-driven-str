#!/usr/bin/env bash
# Foreground, persistent, resumable. No installation, no segmentation/PARSeq rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
base="${1:?Usage: bash diagnostics/run_budgeted_paired_e2e_host.sh STORAGE_ROOT [--check]}"
action="${2:-run}"
[[ "$action" == run || "$action" == --check ]] || { echo "Unknown action: $action" >&2; exit 2; }
assets="$base/context-str-artifacts"
runs="$base/context-str-runs"
out="${BUDGETED_ROUTING_OUTPUT:-$runs/instructblip-budgeted-paired-e2e-v1}"
case "$out" in /tmp|/tmp/*) echo 'Use persistent output storage' >&2; exit 2;; esac
py=.venvs/context-str-main-py311/bin/python
deep=.venvs/context-str-deepsolo-host/bin/python
[[ -x "$py" && -x "$deep" ]] || { echo 'Qualified environments missing' >&2; exit 1; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE="$assets/huggingface"
export HF_HOME="$assets/huggingface-home"
unset TRANSFORMERS_CACHE
"$py" -u -m diagnostics.budgeted_paired_e2e prepare \
  --output-dir "$out" \
  --calibration artifacts/diagnostics/instructblip-budgeted-paired-train-v1/results.json \
  --config configs/approximate/instructblip_budgeted_paired_train_v1.json \
  --train-root "$runs/caption-routing-five-models-train-v1" \
  --test-root "$runs/icdar13-routing-suite-v1" \
  --train-gt-report "$runs/repository-contours-train-v1/results.json" \
  --data-root "$base/datasets/context-str/icdar13" --assets "$assets" \
  --recognition-manifest manifests/icdar13_recognition_test.yaml
if [[ "$action" == --check ]]; then exit 0; fi
"$py" -u -m diagnostics.budgeted_paired_e2e export-deepsolo \
  --output-dir "$out" --device cuda:0 --deepsolo-python "$deep"
"$py" -u -m diagnostics.budgeted_paired_e2e evaluate --output-dir "$out" --split TRAIN
"$py" -u -m diagnostics.budgeted_paired_e2e export-captions --output-dir "$out" --device cuda:0
"$py" -u -m diagnostics.budgeted_paired_e2e evaluate --output-dir "$out" --split TEST
