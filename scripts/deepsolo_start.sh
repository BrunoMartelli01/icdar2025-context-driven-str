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

: "${DEEPSOLO_IMAGE:?DEEPSOLO_IMAGE is required}"
: "${DEEPSOLO_EXPECTED_IMAGE_ID:?DEEPSOLO_EXPECTED_IMAGE_ID is required}"
: "${DEEPSOLO_CONTAINER:?DEEPSOLO_CONTAINER is required}"
: "${DEEPSOLO_SOURCE_ROOT:?DEEPSOLO_SOURCE_ROOT is required}"
: "${DEEPSOLO_CONTAINER_SOURCE:=/workspace/DeepSolo}"
: "${DEEPSOLO_CONTAINER_CHECKPOINT_DIR:=/opt/checkpoints}"

image_id=$(docker image inspect --format '{{.Id}}' "${DEEPSOLO_IMAGE}")
if [[ "${image_id}" != "${DEEPSOLO_EXPECTED_IMAGE_ID}" ]]; then
  echo "Qualified image ID mismatch: ${image_id}" >&2
  exit 2
fi

if docker container inspect "${DEEPSOLO_CONTAINER}" >/dev/null 2>&1; then
  container_image=$(docker container inspect --format '{{.Image}}' "${DEEPSOLO_CONTAINER}")
  restart_policy=$(docker container inspect --format '{{.HostConfig.RestartPolicy.Name}}' "${DEEPSOLO_CONTAINER}")
  if [[ "${container_image}" != "${DEEPSOLO_EXPECTED_IMAGE_ID}" ]]; then
    echo "Existing container uses unexpected image ${container_image}" >&2
    exit 2
  fi
  if [[ "${restart_policy}" != "unless-stopped" ]]; then
    echo "Existing container restart policy is ${restart_policy}, expected unless-stopped" >&2
    exit 2
  fi
  docker start "${DEEPSOLO_CONTAINER}" >/dev/null
else
  # Recovery/reconstruction path only. The qualified existing snapshot already
  # contains its verified checkpoint at /opt/checkpoints and is not modified.
  : "${DEEPSOLO_CHECKPOINT_DIR:?DEEPSOLO_CHECKPOINT_DIR is required to create a container}"
  docker run --detach \
    --name "${DEEPSOLO_CONTAINER}" \
    --restart unless-stopped \
    --gpus 'device=0' \
    --mount "type=bind,src=${DEEPSOLO_SOURCE_ROOT},dst=${DEEPSOLO_CONTAINER_SOURCE}" \
    --mount "type=bind,src=${DEEPSOLO_CHECKPOINT_DIR},dst=${DEEPSOLO_CONTAINER_CHECKPOINT_DIR},readonly" \
    --workdir "${DEEPSOLO_CONTAINER_SOURCE}" \
    --env "CONTEXT_STR_DEEPSOLO_IMAGE_TAG=${DEEPSOLO_IMAGE}" \
    --env "CONTEXT_STR_DEEPSOLO_IMAGE_ID=${DEEPSOLO_EXPECTED_IMAGE_ID}" \
    "${DEEPSOLO_IMAGE}" sleep infinity >/dev/null
fi

echo "${DEEPSOLO_CONTAINER} is running with ${DEEPSOLO_EXPECTED_IMAGE_ID}"
