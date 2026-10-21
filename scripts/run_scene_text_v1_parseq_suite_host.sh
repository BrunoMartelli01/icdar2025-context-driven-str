#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [storage-root]" >&2
  exit 2
fi

STORAGE_ROOT="${1:-/extra1/bmartelli}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/.venvs/context-str-main-py311/bin/python"
DATA="$STORAGE_ROOT/datasets/context-str/scene-segmentation-v1"
TRAIN_RUNS="$STORAGE_ROOT/context-str-runs/scene-text-segmentation-v1"
OUTPUT="$STORAGE_ROOT/context-str-runs/scene-text-v1-parseq-test-v1"
ARTIFACTS="$STORAGE_ROOT/context-str-artifacts"
CALIBRATION="$STORAGE_ROOT/context-str-runs/table5-icdar13-repository-approximate-traces/practical-remap-train-calibration-v1.json"
PARSEQ_REV="1902db043c029a7e03a3818c616c06600af574be"

test -x "$PY"
test -f "$DATA/qualification.json"
test -f "$CALIBRATION"

run_one() {
  local dataset="$1"
  local model="$2"
  local run_name config
  if [[ "$model" == "combined" ]]; then
    run_name="scene-text3-balanced-approximate"
    config="segmentation_scene_text3_balanced.yaml"
  else
    case "$dataset" in
      ctw1500)
        run_name="ctw1500-author-hyperparameters-approximate"
        config="segmentation_ctw1500_author_hyperparameters.yaml"
        ;;
      icdar2015)
        run_name="icdar2015-author-hyperparameters-approximate"
        config="segmentation_icdar2015_author_hyperparameters.yaml"
        ;;
      total_text)
        run_name="total-text-author-hyperparameters-approximate"
        config="segmentation_total_text_author_hyperparameters.yaml"
        ;;
      *) return 2 ;;
    esac
  fi
  local run_dir="$TRAIN_RUNS/$run_name"
  local run_manifest="$run_dir/run_manifest.json"
  test -f "$run_manifest"
  local checkpoint_rel
  checkpoint_rel="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["path"])' "$run_manifest")"
  local extra=()
  if [[ "$dataset" == "ctw1500" ]]; then
    extra+=(--ctw-source-archive "$STORAGE_ROOT/datasets/context-str/ctw1500/downloads/ctw1500_mmocr.zip")
  fi
  echo "[suite] dataset=$dataset model=$model" >&2
  "$PY" -u -m scripts.evaluate_scene_text_v1_parseq \
    --dataset "$dataset" \
    --test-manifest "$DATA/manifests/${dataset}_test.yaml" \
    --data-root "$DATA" \
    --segmentation-config "$REPO_ROOT/configs/approximate/$config" \
    --segmentation-run-manifest "$run_manifest" \
    --segmentation-checkpoint "$run_dir/$checkpoint_rel" \
    --v1-calibration "$CALIBRATION" \
    --parseq-source "$ARTIFACTS/parseq/parseq-$PARSEQ_REV" \
    --parseq-source-archive "$ARTIFACTS/parseq/parseq-$PARSEQ_REV.tar.gz" \
    --parseq-checkpoint "$ARTIFACTS/parseq/parseq-bb5792a6.pt" \
    --device cuda:0 \
    --parseq-batch-size 32 \
    --output-dir "$OUTPUT/$dataset-$model" \
    "${extra[@]}"
}

cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for dataset in ctw1500 icdar2015 total_text; do
  run_one "$dataset" specialized
  run_one "$dataset" combined
done

"$PY" -u -m scripts.summarize_scene_text_v1_parseq_suite --root "$OUTPUT"
