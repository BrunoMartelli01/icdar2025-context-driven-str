#!/usr/bin/env bash
set -euo pipefail

TASK_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_STORAGE="$(cd -- "$TASK_REPO/../.." && pwd)"
TASK_PYTHON="${UNET_TEST_PYTHON:-$TASK_REPO/.venvs/context-str-main-py311/bin/python}"
TASK_RUNS="${UNET_TEST_RUNS:-$TASK_STORAGE/context-str-runs}"
TASK_DATA="${UNET_TEST_DATA:-$TASK_STORAGE/datasets/context-str/scene-segmentation-v1}"
TASK_OUTPUT="${UNET_TEST_OUTPUT:-$TASK_REPO/artifacts/diagnostics/unet-word-detection-test-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

cd -- "$TASK_REPO"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$TASK_PYTHON" -u -m scripts.evaluate_unet_word_detection_test \
  --mode approximate \
  --reference-run "$TASK_RUNS/unet-word-research-v1/control" \
  --text-run "$TASK_RUNS/totaltext-text-recall-resolution-v1/branches/tversky768" \
  --kernel-run "$TASK_RUNS/totaltext-word-multitask-v1/text_kernel" \
  --data-root "$TASK_DATA" \
  --test-manifest "$TASK_DATA/manifests/total_text_test.yaml" \
  --output-root "$TASK_OUTPUT" \
  --device "${UNET_TEST_DEVICE:-cuda:0}" "$@"
