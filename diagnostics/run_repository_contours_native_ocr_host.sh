#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
base="${1:?Usage: bash diagnostics/run_repository_contours_native_ocr_host.sh USER_STORAGE_ROOT}"
assets="$base/context-str-artifacts"
source_report="$base/context-str-runs/repository-contours-train-v1/results.json"
out="$PWD/artifacts/diagnostics/repository-contours-native128-train-v1"
py=.venvs/context-str-main-py311/bin/python
revision=1902db043c029a7e03a3818c616c06600af574be
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
args=(
  --source-report "$source_report"
  --source-sha256 38afc5fbb93bf6c6cc9970ad90454509f1089d6249a56447e6f60aecfbf2cc1b
  --data-root "$base/datasets/context-str/icdar13"
  --train-manifest manifests/icdar13_segmentation_author_protocol.yaml
  --config configs/approximate/localization_repository_contours_native128_v1.json
  --output-dir "$out"
)
if [[ ! -f "$out/geometry.json" ]]; then
  CUDA_VISIBLE_DEVICES="" "$py" -u -m diagnostics.repository_contours_native_train --stage geometry "${args[@]}"
fi
"$py" -u -m diagnostics.repository_contours_native_train --stage ocr "${args[@]}" \
  --device cuda:0 \
  --parseq-source "$assets/parseq/parseq-$revision" \
  --parseq-archive "$assets/parseq/parseq-$revision.tar.gz" \
  --parseq-checkpoint "$assets/parseq/parseq-bb5792a6.pt"
