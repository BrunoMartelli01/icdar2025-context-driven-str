#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 ARTIFACT_ROOT ICDAR13_DATA_ROOT CACHE_ROOT" >&2
  exit 2
fi

artifact_root=$1
data_root=$2
cache_root=$3
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$repo_root/.venvs/context-str-main-py311/bin/python"
parseq_revision=1902db043c029a7e03a3818c616c06600af574be

cd "$repo_root"
PYTHONUNBUFFERED=1 "$python_bin" -u -m scripts.evaluate_practical_candidate_selection_v2_train \
  --probability-manifest "$cache_root/icdar13-train-native-probabilities-epoch0098/manifest.json" \
  --mask-manifest "$cache_root/icdar13-train-native-masks-epoch0098/manifest.json" \
  --train-manifest manifests/icdar13_segmentation_author_protocol.yaml \
  --data-root "$data_root" \
  --v1-calibration "$cache_root/practical-remap-train-calibration-v1.json" \
  --v1-reference-report "$cache_root/icdar13-train-gtbox-vs-v1-parseq-full-v1.json" \
  --context-report "$cache_root/blip-context-candidate-ranking-train-v1.json" \
  --config configs/approximate/localization_practical_candidate_selection_v2.yaml \
  --candidate-artifact-dir "$cache_root/parseq-candidate-ranking-train-v1-artifacts" \
  --parseq-source "$artifact_root/parseq/parseq-$parseq_revision" \
  --parseq-source-archive "$artifact_root/parseq/parseq-$parseq_revision.tar.gz" \
  --parseq-checkpoint "$artifact_root/parseq/parseq-bb5792a6.pt" \
  --device cuda:0 \
  --batch-size 64 \
  --artifact-dir "$cache_root/practical-candidate-selection-v2-train-artifacts" \
  --output "$cache_root/practical-candidate-selection-v2-train.json"
