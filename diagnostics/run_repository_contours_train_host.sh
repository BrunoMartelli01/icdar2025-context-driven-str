#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
base="${1:?Usage: bash diagnostics/run_repository_contours_train_host.sh USER_STORAGE_ROOT [--check]}"
action="${2:-run}"
[[ "$action" == run || "$action" == --check ]] || exit 2
assets="$base/context-str-artifacts"
runs="$base/context-str-runs"
ckpt="$runs/ap-full-20260901-174246/icdar13-author-protocol-approximate/checkpoints/epoch-0098.pt"
revision=1902db043c029a7e03a3818c616c06600af574be
py=.venvs/context-str-main-py311/bin/python
out="${REPOSITORY_CONTOURS_OUTPUT:-$runs/repository-contours-train-v1}"
case "$out" in /tmp|/tmp/*) echo "Use persistent output storage" >&2; exit 2;; esac
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
args=(
  --train-manifest manifests/icdar13_segmentation_author_protocol.yaml
  --data-root "$base/datasets/context-str/icdar13"
  --checkpoint "$ckpt"
  --evaluation-config configs/approximate/segmentation_icdar13_author_protocol_test.yaml
  --repository-config configs/approximate/localization_repository_contours_v1.json
  --v1-config configs/approximate/pipeline_icdar13_practical_remap_v1.yaml
  --compact-v1 "$runs/practical-context-train-v1/train-practical-remap-v1-parseq-compact-v1"
  --parseq-source "$assets/parseq/parseq-$revision"
  --parseq-source-archive "$assets/parseq/parseq-$revision.tar.gz"
  --parseq-checkpoint "$assets/parseq/parseq-bb5792a6.pt"
  --device cuda:0 --output-dir "$out"
)
CUDA_VISIBLE_DEVICES="" "$py" -u -m diagnostics.repository_contours_train "${args[@]}" --check
if [[ "$action" == --check ]]; then exit 0; fi
"$py" -u -m diagnostics.repository_contours_train "${args[@]}"
