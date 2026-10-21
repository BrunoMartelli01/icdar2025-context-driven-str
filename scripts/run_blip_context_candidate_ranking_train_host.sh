#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_blip_context_candidate_ranking_train_host.sh ARTIFACT_ROOT ICDAR13_DATA_ROOT" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="$(cd "$1" && pwd)"
data_root="$(cd "$2" && pwd)"
python_executable="${repository_root}/.venvs/context-str-main-py311/bin/python"
source_report=/tmp/parseq-candidate-ranking-train-v1.json
candidate_root=/tmp/parseq-candidate-ranking-train-v1-artifacts
caption_root=/tmp/icdar13-train-blip2-captions-v1
result=/tmp/blip-context-candidate-ranking-train-v1.json
hf_cache="${artifact_root}/huggingface"

for required in \
  "${python_executable}" \
  "${source_report}" \
  "${candidate_root}/generation-manifest.json" \
  "${candidate_root}/candidates.jsonl" \
  "${candidate_root}/ocr-batch-0064/manifest.json"
do
  if [[ ! -e "${required}" ]]; then
    echo "Required frozen input is missing: ${required}" >&2
    exit 1
  fi
done
if [[ ! -d "${hf_cache}" ]]; then
  echo "Pinned Hugging Face cache is missing: ${hf_cache}" >&2
  exit 1
fi

export HF_HOME="${artifact_root}/huggingface-home"
export HF_HUB_CACHE="${hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}"

cd "${repository_root}"
"${python_executable}" -u -m scripts.export_train_blip2_captions \
  --train-manifest manifests/icdar13_segmentation_author_protocol.yaml \
  --data-root "${data_root}" \
  --source-report "${source_report}" \
  --candidate-generation-manifest "${candidate_root}/generation-manifest.json" \
  --hf-cache "${hf_cache}" \
  --device cuda:0 \
  --output-dir "${caption_root}"

CUDA_VISIBLE_DEVICES="" "${python_executable}" -u \
  -m scripts.diagnose_blip_context_candidate_ranking_train \
  --source-report "${source_report}" \
  --candidate-artifact-dir "${candidate_root}" \
  --caption-manifest "${caption_root}/manifest.json" \
  --hf-cache "${hf_cache}" \
  --device cpu \
  --output "${result}"
