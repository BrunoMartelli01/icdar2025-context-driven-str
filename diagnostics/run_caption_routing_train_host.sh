#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
base="${1:?Usage: bash diagnostics/run_caption_routing_train_host.sh USER_STORAGE_ROOT [--check]}"
action="${2:-run}"
[[ "$action" == run || "$action" == --check ]] || { echo "unknown action: $action" >&2; exit 2; }
assets="$base/context-str-artifacts"
data="$base/datasets/context-str/icdar13"
source_root="$base/context-str-runs/practical-context-train-v1"
out="${CAPTION_ROUTING_TRAIN_OUTPUT:-$base/context-str-runs/caption-routing-five-models-train-v1}"
case "$out" in /tmp|/tmp/*) echo "Use persistent output storage" >&2; exit 2;; esac
py=.venvs/context-str-main-py311/bin/python
legacy=.venvs/context-str-lavis/bin/python
visual="$PWD/artifacts/diagnostics/lavis-captioner-candidates-20260914"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset TRANSFORMERS_CACHE
export HF_HUB_CACHE="$assets/huggingface"
export HF_HOME="$visual/cache/huggingface-home"
export TORCH_HOME="$visual/cache/torch"
export PYTHONPATH="$visual/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
[[ -x "$py" && -x "$legacy" ]] || { echo "Qualified main/LAVIS environments missing" >&2; exit 1; }
# No installation or download. Models/crops/segmentation are reused from their original caches.
CUDA_VISIBLE_DEVICES="" "$py" -u -m diagnostics.caption_routing_train --prepare \
  --source-root "$source_root" --data-root "$data" --asset-root "$assets" --output-dir "$out"
if [[ "$action" == --check ]]; then exit 0; fi
for stage in t1 flan-antirepetition flan-repository blip-base-coco blip2-vitl instructblip florence2-base-ft florence2-large-ft; do
  if [[ "$stage" == blip-base-coco || "$stage" == blip2-vitl ]]; then
    HF_HUB_CACHE="$HF_HOME/hub" "$legacy" -u -m diagnostics.caption_routing_train \
      --stage "$stage" --device cuda:0 --output-dir "$out"
  else
    "$py" -u -m diagnostics.caption_routing_train --stage "$stage" --device cuda:0 --output-dir "$out"
  fi
done
CUDA_VISIBLE_DEVICES="" "$py" -u -m diagnostics.caption_routing_train --evaluate --output-dir "$out"
