#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: bash scripts/run_icdar13_trace_batch_host.sh ARTIFACT_ROOT DATA_ROOT RUN_ROOT START_ID END_ID" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="$(cd "$1" && pwd)"
data_root="$(cd "$2" && pwd)"
run_root="$(cd "$3" && pwd)"
start_id="$4"
end_id="$5"
if [[ ! "${start_id}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${end_id}" =~ ^[1-9][0-9]*$ ]]; then
  echo "START_ID and END_ID must be decimal integers" >&2
  exit 2
fi
if (( start_id > end_id || end_id > 233 )); then
  echo "Expected 1 <= START_ID <= END_ID <= 233" >&2
  exit 2
fi

main_python="${CONTEXT_STR_MAIN_PYTHON:-${repository_root}/.venvs/context-str-main-py311/bin/python}"
hf_cache="${CONTEXT_STR_HF_CACHE:-${artifact_root}/huggingface}"
config="${CONTEXT_STR_TABLE5_CONFIG:-${repository_root}/configs/approximate/pipeline_icdar13_table5_smoke.yaml}"
recognition_manifest="${repository_root}/manifests/icdar13_recognition_test.yaml"
segmentation_config="${repository_root}/configs/approximate/segmentation_icdar13_author_protocol_test.yaml"
segmentation_checkpoint="${run_root}/ap-full-20260901-174246/icdar13-author-protocol-approximate/checkpoints/epoch-0098.pt"
parseq_revision="1902db043c029a7e03a3818c616c06600af574be"
parseq_source="${artifact_root}/parseq/parseq-${parseq_revision}"
parseq_archive="${artifact_root}/parseq/parseq-${parseq_revision}.tar.gz"
parseq_checkpoint="${artifact_root}/parseq/parseq-bb5792a6.pt"
deepsolo_python="${repository_root}/.venvs/context-str-deepsolo-host/bin/python"
deepsolo_source="${artifact_root}/deepsolo-host/DeepSolo-upstream/DeepSolo"
deepsolo_manifest="${artifact_root}/deepsolo-host/qualification/runtime.json"
qualification_root="${artifact_root}/main-host/qualification"
output_root="${CONTEXT_STR_ICDAR13_TRACE_ROOT:-${run_root}/table5-icdar13-approximate-traces}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

require_directory() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory is missing: $1" >&2
    exit 1
  fi
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  require_file "${path}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA256 mismatch for ${path}" >&2
    echo "Expected: ${expected}" >&2
    echo "Actual:   ${actual}" >&2
    exit 1
  fi
}

if [[ ! -x "${main_python}" || ! -x "${deepsolo_python}" ]]; then
  echo "One or both isolated Python executables are missing" >&2
  exit 1
fi
require_directory "${hf_cache}"
require_directory "${parseq_source}"
require_directory "${deepsolo_source}"
require_file "${config}"
require_sha256 "${recognition_manifest}" "d8ab8863fa2dd00cdbf0667fb36c20a28432ba070164b85d9dbff0827dfb20bd"
require_sha256 "${segmentation_checkpoint}" "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2"
require_sha256 "${parseq_archive}" "f57289cb1fc7fe41f488a52ea48f8a7f114252ef405b0b64efc1d528c29b48ef"
require_sha256 "${parseq_checkpoint}" "bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00"
require_sha256 "${deepsolo_manifest}" "c52a65b7d844cd35bcf6e38e3cb58c5194851c8327cbdaba9b313869ec2c7ffa"
require_sha256 "${qualification_root}/context-str-main-py311-host.json" "23f4e739966ef686b8ef3646e77164f4410d41bec94423bbba9bf9e70e9ed685"
require_sha256 "${qualification_root}/blip2-fp16-single-device-icdar13-img1.json" "0c65c3ffd3bd1d0105a28181a4a2071a54a2f10b4228d18237db0ffe3d235ccc"
require_sha256 "${qualification_root}/mpnet-scoring-qualification.json" "32ce49e73b12a0e57ba1ec91cdf6b4dc29e33716679fe51bfb0d9a13da09a1db"

mkdir -p "${artifact_root}/huggingface-home"
export HF_HOME="${artifact_root}/huggingface-home"
export HF_HUB_CACHE="${hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

cd "${repository_root}"
"${main_python}" -m scripts.check_icdar13_recognition_manifest \
  --manifest "${recognition_manifest}" \
  --data-root "${data_root}/icdar13"
"${main_python}" -m scripts.check_table5_assets \
  --component segmentation \
  --component parseq \
  --component blip2 \
  --component mpnet \
  --component python \
  --segmentation-config "${segmentation_config}" \
  --segmentation-checkpoint "${segmentation_checkpoint}" \
  --parseq-source "${parseq_source}" \
  --parseq-source-type github_commit_archive \
  --parseq-source-archive "${parseq_archive}" \
  --parseq-checkpoint "${parseq_checkpoint}" \
  --hf-cache "${hf_cache}"

"${main_python}" -m scripts.run_icdar13_trace_batch \
  --manifest "${recognition_manifest}" \
  --data-root "${data_root}/icdar13" \
  --config "${config}" \
  --output-root "${output_root}" \
  --start-id "${start_id}" \
  --end-id "${end_id}" \
  --segmentation-checkpoint "${segmentation_checkpoint}" \
  --parseq-source "${parseq_source}" \
  --parseq-source-archive "${parseq_archive}" \
  --parseq-checkpoint "${parseq_checkpoint}" \
  --deepsolo-python "${deepsolo_python}" \
  --deepsolo-runtime-manifest "${deepsolo_manifest}" \
  --deepsolo-source-root "${deepsolo_source}"
