#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_file=${DEEPSOLO_ENV_FILE:-"${repo_root}/docker/deepsolo/.env"}
if [[ ! -f "${env_file}" ]]; then
  echo "Missing ${env_file}; copy docker/deepsolo/.env.example and set host paths." >&2
  exit 2
fi
set -a
source "${env_file}"
set +a

: "${DEEPSOLO_BASE_IMAGE:?DEEPSOLO_BASE_IMAGE is required}"
: "${DEEPSOLO_RECONSTRUCTED_IMAGE:?DEEPSOLO_RECONSTRUCTED_IMAGE is required}"
: "${DEEPSOLO_SOURCE_ROOT:?DEEPSOLO_SOURCE_ROOT is required}"
: "${DEEPSOLO_CONTAINER_PYTHON:=/opt/context-str-deepsolo/bin/python}"

source_revision=$(git -C "${DEEPSOLO_SOURCE_ROOT}" rev-parse HEAD)
expected_revision=dbadae995035246bad3376c7a44c015c69e9b313
if [[ "${source_revision}" != "${expected_revision}" ]]; then
  echo "DeepSolo source revision mismatch: ${source_revision}" >&2
  exit 2
fi

docker build \
  --build-arg "BASE_IMAGE=${DEEPSOLO_BASE_IMAGE}" \
  --tag "${DEEPSOLO_RECONSTRUCTED_IMAGE}" \
  "${repo_root}/docker/deepsolo"

docker run --rm --gpus 'device=0' \
  --mount "type=bind,src=${DEEPSOLO_SOURCE_ROOT},dst=/workspace/DeepSolo" \
  --workdir /workspace/DeepSolo \
  --env CUDA_HOME=/usr/local/cuda \
  --env CC=/usr/bin/gcc \
  --env CXX=/usr/bin/g++ \
  --env TORCH_CUDA_ARCH_LIST=8.6 \
  --env MAX_JOBS=2 \
  --env FORCE_CUDA=1 \
  "${DEEPSOLO_RECONSTRUCTED_IMAGE}" \
  "${DEEPSOLO_CONTAINER_PYTHON}" -m pip install --no-build-isolation --editable /workspace/DeepSolo

docker run --rm --gpus 'device=0' \
  --mount "type=bind,src=${DEEPSOLO_SOURCE_ROOT},dst=/workspace/DeepSolo" \
  --workdir /workspace/DeepSolo \
  "${DEEPSOLO_RECONSTRUCTED_IMAGE}" \
  "${DEEPSOLO_CONTAINER_PYTHON}" -c 'import adet._C; print("adet._C reconstruction import: PASS")'

echo "Reconstruction completed. This image is NOT qualified until separately validated."
