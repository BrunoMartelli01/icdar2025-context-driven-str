#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /extra1/bmartelli" >&2
  exit 2
fi

ROOT=${1%/}
if [[ ! -d "$ROOT/projects/icdar2025-context-driven-str" && -d "$ROOT/bmartelli/projects/icdar2025-context-driven-str" ]]; then
  ROOT="$ROOT/bmartelli"
fi
REPO="$ROOT/projects/icdar2025-context-driven-str"
PY="$REPO/.venvs/context-str-main-py311/bin/python"
DATA="$ROOT/datasets/context-str/scene-segmentation-v1"
TRAIN_RUNS="$ROOT/context-str-runs/scene-text-segmentation-v1"
FROZEN="$ROOT/context-str-runs/scene-text-v1-parseq-test-v1"
OUTPUT="$ROOT/context-str-runs/scene-text-paper-algorithm1-crops-v1"

run_one() {
  local dataset=$1 run_name=$2 config=$3
  local run_dir="$TRAIN_RUNS/$run_name"
  local checkpoint
  checkpoint="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_output_location"])' "$run_dir/run_manifest.json")"
  local extra=()
  if [[ "$dataset" == "ctw1500" ]]; then
    extra+=(--ctw-recognition-manifest "$DATA/manifests/ctw1500_recognition_test.json")
  fi
  echo "[paper Algorithm1] dataset=$dataset model=$run_name" >&2
  PYTHONUNBUFFERED=1 "$PY" -u -m scripts.export_scene_text_algorithm1_crops \
    --dataset "$dataset" \
    --test-manifest "$DATA/manifests/${dataset}_test.yaml" \
    --data-root "$DATA" \
    --segmentation-config "$REPO/configs/approximate/$config" \
    --localization-config "$REPO/configs/approximate/scene_text_paper_algorithm1_native128_v1.yaml" \
    --segmentation-run-manifest "$run_dir/run_manifest.json" \
    --segmentation-checkpoint "$checkpoint" \
    --frozen-v1-run "$FROZEN/${dataset}-specialized" \
    --device cuda:0 \
    --output-dir "$OUTPUT/${dataset}-specialized" \
    "${extra[@]}"
}

cd "$REPO"
test -x "$PY"
run_one ctw1500 ctw1500-author-hyperparameters-approximate segmentation_ctw1500_author_hyperparameters.yaml
run_one icdar2015 icdar2015-author-hyperparameters-approximate segmentation_icdar2015_author_hyperparameters.yaml
run_one total_text total-text-author-hyperparameters-approximate segmentation_total_text_author_hyperparameters.yaml
