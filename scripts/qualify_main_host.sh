#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/qualify_main_host.sh ARTIFACT_ROOT [EXPECTED_GPU_FRAGMENT]" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="$(cd "$1" && pwd)"
expected_gpu_fragment="${2:-RTX 3070}"
python_executable="${CONTEXT_STR_MAIN_PYTHON:-${repository_root}/.venvs/context-str-main-py311/bin/python}"
qualification_root="${CONTEXT_STR_MAIN_QUALIFICATION_ROOT:-${artifact_root}/main-host/qualification}"
hf_cache="${CONTEXT_STR_HF_CACHE:-${artifact_root}/huggingface}"

parseq_revision="1902db043c029a7e03a3818c616c06600af574be"
parseq_root="${artifact_root}/parseq"
parseq_source="${parseq_root}/parseq-${parseq_revision}"
parseq_archive="${parseq_root}/parseq-${parseq_revision}.tar.gz"
parseq_checkpoint="${parseq_root}/parseq-bb5792a6.pt"
output="${qualification_root}/context-str-main-py311-host.json"

if [[ ! -x "${python_executable}" ]]; then
  echo "Main Python executable is missing: ${python_executable}" >&2
  exit 1
fi
if [[ ! -d "${parseq_source}" ]]; then
  echo "PARSeq source directory is missing: ${parseq_source}" >&2
  exit 1
fi
if [[ ! -f "${parseq_archive}" ]]; then
  echo "PARSeq source archive is missing: ${parseq_archive}" >&2
  exit 1
fi
if [[ ! -f "${parseq_checkpoint}" ]]; then
  echo "PARSeq checkpoint is missing: ${parseq_checkpoint}" >&2
  exit 1
fi
if [[ ! -f "${parseq_source}/demo_images/cute-184.jpg" ]]; then
  echo "PARSeq qualification image is missing" >&2
  exit 1
fi
if [[ ! -d "${hf_cache}" ]]; then
  echo "Hugging Face cache is missing: ${hf_cache}" >&2
  exit 1
fi

mkdir -p "${qualification_root}"
export HF_HOME="${artifact_root}/huggingface-home"
export HF_HUB_CACHE="${hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

cd "${repository_root}"
"${python_executable}" -m scripts.check_main_environment \
  --expected-gpu-fragment "${expected_gpu_fragment}" \
  --parseq-source "${parseq_source}" \
  --parseq-source-type github_commit_archive \
  --parseq-source-archive "${parseq_archive}" \
  --parseq-checkpoint "${parseq_checkpoint}" \
  --parseq-image "${parseq_source}/demo_images/cute-184.jpg" \
  --output "${output}"
