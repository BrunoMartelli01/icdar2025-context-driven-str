#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_probability_threshold_crops_parseq_train_host.sh ARTIFACT_ROOT ICDAR13_DATA_ROOT" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="$(cd "$1" && pwd)"
data_root="$(cd "$2" && pwd)"
python_executable="${repository_root}/.venvs/context-str-main-py311/bin/python"
revision=1902db043c029a7e03a3818c616c06600af574be

cd "${repository_root}"
export PYTHONUNBUFFERED=1
"${python_executable}" -u -m scripts.evaluate_probability_threshold_crops_parseq_train \
  --geometry-report /tmp/probability-threshold-word-connectivity-train-v1.json \
  --crop-manifest /tmp/probability-threshold-word-connectivity-train-v1-crops/manifest.json \
  --train-manifest manifests/icdar13_segmentation_author_protocol.yaml \
  --data-root "${data_root}" \
  --parseq-source "${artifact_root}/parseq/parseq-${revision}" \
  --parseq-source-archive "${artifact_root}/parseq/parseq-${revision}.tar.gz" \
  --parseq-checkpoint "${artifact_root}/parseq/parseq-bb5792a6.pt" \
  --device cuda:0 \
  --batch-size 64 \
  --ocr-artifact-dir /tmp/probability-threshold-word-connectivity-train-v1-parseq-artifacts \
  --output /tmp/probability-threshold-word-connectivity-train-v1-parseq.json
