#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/qualify_mpnet_scoring_host.sh ARTIFACT_ROOT" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="$(cd "$1" && pwd)"
python_executable="${CONTEXT_STR_MAIN_PYTHON:-${repository_root}/.venvs/context-str-main-py311/bin/python}"
hf_cache="${CONTEXT_STR_HF_CACHE:-${artifact_root}/huggingface}"
qualification_root="${CONTEXT_STR_MAIN_QUALIFICATION_ROOT:-${artifact_root}/main-host/qualification}"
config="${repository_root}/configs/approximate/pipeline_icdar13_table5_smoke.yaml"
output="${qualification_root}/mpnet-scoring-qualification.json"

if [[ ! -x "${python_executable}" ]]; then
  echo "Main Python executable is missing: ${python_executable}" >&2
  exit 1
fi
if [[ ! -d "${hf_cache}" ]]; then
  echo "Hugging Face cache is missing: ${hf_cache}" >&2
  exit 1
fi

mkdir -p "${qualification_root}" "${artifact_root}/huggingface-home"
export HF_HOME="${artifact_root}/huggingface-home"
export HF_HUB_CACHE="${hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

cd "${repository_root}"
"${python_executable}" -m scripts.qualify_mpnet_scoring \
  --config "${config}" \
  --hf-cache "${hf_cache}" \
  --output "${output}"
