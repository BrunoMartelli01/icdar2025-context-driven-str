#!/usr/bin/env bash
set -euo pipefail
router_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$router_repo"
router_python=${CLIP_ROUTER_PYTHON:-"$router_repo/.venvs/context-str-main-py311/bin/python"}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
# CPU preserves the prepared PARSeq/CLIP run identity. DeepSolo independently
# uses the qualified isolated CUDA subprocess and the original full images.
"$router_python" -m scripts.run_clip_router_e2e --mode approximate \
    --device cpu --stage complete --split both "$@"
