#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo"
router_python=${CLIP_ROUTER_PYTHON:-"$repo/.venvs/context-str-main-py311/bin/python"}
router_device=${CLIP_ROUTER_DEVICE:-cuda:0}
"$router_python" -m pytest tests/recognition/test_clip_image_router.py tests/recognition/test_clip_image_router_execution.py tests/recognition/test_clip_crop_admission.py -q
for router_model in vit_b32 vit_b16 vit_l14; do
    "$router_python" -m scripts.run_clip_image_router_word_advantage --mode approximate --model "$router_model" --device "$router_device" "$@"
done
