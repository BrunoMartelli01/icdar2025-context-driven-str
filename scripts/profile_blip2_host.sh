#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/profile_blip2_host.sh ARTIFACT_ROOT DATA_ROOT" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="$(cd "$1" && pwd)"
data_root="$(cd "$2" && pwd)"
python_executable="${CONTEXT_STR_MAIN_PYTHON:-${repository_root}/.venvs/context-str-main-py311/bin/python}"
hf_cache="${CONTEXT_STR_HF_CACHE:-${artifact_root}/huggingface}"
qualification_root="${CONTEXT_STR_MAIN_QUALIFICATION_ROOT:-${artifact_root}/main-host/qualification}"
image="${data_root}/icdar13/raw/test_images/img_1.jpg"
expected_image_sha256="67118d0c38c37ed8f6f2d06862e23e0ab752c76da93728fa779cfd14b4414143"
output="${qualification_root}/blip2-fp16-single-device-icdar13-img1.json"

if [[ ! -x "${python_executable}" ]]; then
  echo "Main Python executable is missing: ${python_executable}" >&2
  exit 1
fi
if [[ ! -d "${hf_cache}" ]]; then
  echo "Hugging Face cache is missing: ${hf_cache}" >&2
  exit 1
fi
if [[ ! -f "${image}" ]]; then
  echo "ICDAR13 qualification image is missing: ${image}" >&2
  exit 1
fi
actual_image_sha256="$(sha256sum "${image}" | awk '{print $1}')"
if [[ "${actual_image_sha256}" != "${expected_image_sha256}" ]]; then
  echo "ICDAR13 qualification image SHA256 mismatch" >&2
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
"${python_executable}" -m scripts.profile_blip2 \
  --image "${image}" \
  --output "${output}" \
  --device cuda:0 \
  --dtype float16 \
  --placement single_device \
  --quantization none \
  --model-lifecycle standalone_blip2_only
