#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo/.venvs/context-str-main-py311/bin/python"

cd "$repo"
"$python_bin" -m pytest -q \
  tests/recognition/test_totaltext_crop_clip_battery.py \
  tests/segmentation/test_totaltext_letterbox_threshold_sweep_train.py \
  tests/segmentation/test_totaltext_image_frequency_ablation.py
PYTHONUNBUFFERED=1 "$python_bin" -u -m scripts.run_totaltext_crop_clip_battery \
  --config configs/approximate/totaltext_crop_clip_battery_train_v1.json \
  "$@"
