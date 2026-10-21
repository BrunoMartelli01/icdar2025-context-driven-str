#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/run_icdar13_e2e_faithful_vs_balanced_host.sh ARTIFACT_ROOT ICDAR13_DATA_ROOT RUN_ROOT" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="$(cd "$1" && pwd)"
data_root="$(cd "$2" && pwd)"
run_root="$(cd "$3" && pwd)"
python_executable="${repository_root}/.venvs/context-str-main-py311/bin/python"
deepsolo_python="${repository_root}/.venvs/context-str-deepsolo-host/bin/python"
checkpoint="${run_root}/ap-full-20260901-174246/icdar13-author-protocol-approximate/checkpoints/epoch-0098.pt"
parseq_revision="1902db043c029a7e03a3818c616c06600af574be"
parseq_source="${artifact_root}/parseq/parseq-${parseq_revision}"
parseq_archive="${artifact_root}/parseq/parseq-${parseq_revision}.tar.gz"
parseq_checkpoint="${artifact_root}/parseq/parseq-bb5792a6.pt"
deepsolo_source="${artifact_root}/deepsolo-host/DeepSolo-upstream/DeepSolo"
deepsolo_manifest="${artifact_root}/deepsolo-host/qualification/runtime.json"
hf_cache="${artifact_root}/huggingface"
v1_calibration="/tmp/practical-remap-train-calibration-v1.json"
balanced_train="/tmp/practical-candidate-selection-v2-train.json"
artifact_dir="${CONTEXT_STR_E2E_COMPARISON_ARTIFACT_DIR:-/tmp/icdar13-e2e-faithful-vs-balanced-artifacts}"
output="${CONTEXT_STR_E2E_COMPARISON_OUTPUT:-/tmp/icdar13-e2e-faithful-vs-balanced-v1.json}"

require_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Required directory is missing: $1" >&2; exit 1; }; }
require_sha() {
  require_file "$1"
  local actual
  actual="$(sha256sum "$1" | awk '{print $1}')"
  [[ "${actual}" == "$2" ]] || {
    echo "SHA256 mismatch: $1" >&2
    echo "expected $2" >&2
    echo "actual   ${actual}" >&2
    exit 1
  }
}

[[ -x "${python_executable}" ]] || { echo "Missing main Python: ${python_executable}" >&2; exit 1; }
[[ -x "${deepsolo_python}" ]] || { echo "Missing DeepSolo Python: ${deepsolo_python}" >&2; exit 1; }
require_dir "${parseq_source}"
require_dir "${deepsolo_source}"
require_dir "${hf_cache}"
require_sha "${checkpoint}" b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2
require_sha "${parseq_archive}" f57289cb1fc7fe41f488a52ea48f8a7f114252ef405b0b64efc1d528c29b48ef
require_sha "${parseq_checkpoint}" bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00
require_sha "${v1_calibration}" 6c0e4cb36fe49f2c1d2306fc7226a5a7d7938208b7b223b4490a6dae377a659f
require_sha "${balanced_train}" 8db4dac11c9d08e956d1f11c982b97c5e538e009f3dff9d0b5bd0e393b7a94e5
require_sha "${deepsolo_manifest}" c52a65b7d844cd35bcf6e38e3cb58c5194851c8327cbdaba9b313869ec2c7ffa
require_sha "${artifact_root}/main-host/qualification/context-str-main-py311-host.json" 23f4e739966ef686b8ef3646e77164f4410d41bec94423bbba9bf9e70e9ed685
require_sha "${artifact_root}/main-host/qualification/blip2-fp16-single-device-icdar13-img1.json" 0c65c3ffd3bd1d0105a28181a4a2071a54a2f10b4228d18237db0ffe3d235ccc
require_sha "${artifact_root}/main-host/qualification/mpnet-scoring-qualification.json" 32ce49e73b12a0e57ba1ec91cdf6b4dc29e33716679fe51bfb0d9a13da09a1db

mkdir -p "${artifact_root}/huggingface-home"
export HF_HOME="${artifact_root}/huggingface-home"
export HF_HUB_CACHE="${hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

cd "${repository_root}"
"${python_executable}" -u -m scripts.check_table5_assets \
  --component segmentation --component parseq --component blip2 --component mpnet --component python \
  --segmentation-config configs/approximate/segmentation_icdar13_author_protocol_test.yaml \
  --segmentation-checkpoint "${checkpoint}" \
  --parseq-source "${parseq_source}" --parseq-source-type github_commit_archive \
  --parseq-source-archive "${parseq_archive}" --parseq-checkpoint "${parseq_checkpoint}" \
  --hf-cache "${hf_cache}"

"${python_executable}" -u -m scripts.evaluate_icdar13_e2e_faithful_vs_balanced \
  --segmentation-manifest manifests/icdar13_segmentation_test.yaml \
  --recognition-manifest manifests/icdar13_recognition_test.yaml \
  --data-root "${data_root}" \
  --evaluation-config configs/approximate/segmentation_icdar13_author_protocol_test.yaml \
  --checkpoint "${checkpoint}" \
  --v1-calibration "${v1_calibration}" \
  --balanced-train-report "${balanced_train}" \
  --comparison-config configs/approximate/pipeline_icdar13_e2e_faithful_vs_balanced_diagnostic.yaml \
  --parseq-source "${parseq_source}" \
  --parseq-source-archive "${parseq_archive}" \
  --parseq-checkpoint "${parseq_checkpoint}" \
  --deepsolo-python "${deepsolo_python}" \
  --deepsolo-runtime-manifest "${deepsolo_manifest}" \
  --deepsolo-source-root "${deepsolo_source}" \
  --device cuda:0 --batch-size 64 \
  --artifact-dir "${artifact_dir}" --output "${output}"
