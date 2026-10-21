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

"${repo_root}/scripts/deepsolo_start.sh"

: "${DEEPSOLO_MAIN_PYTHON:=${repo_root}/.venvs/context-str-main/bin/python}"
: "${DEEPSOLO_CONTAINER_PYTHON:=/opt/context-str-deepsolo/bin/python}"
: "${DEEPSOLO_CONTAINER_SOURCE:=/workspace/DeepSolo}"
: "${DEEPSOLO_RUNTIME_MANIFEST:=${repo_root}/docker/deepsolo/runtime_manifest.json}"
: "${DEEPSOLO_SMOKE_IMAGE:=/tmp/context-str-deepsolo-m6-smoke.png}"
: "${DEEPSOLO_SMOKE_OUTPUT:=/tmp/context-str-deepsolo-m6-smoke.json}"
: "${DEEPSOLO_EXPECTED_DETECTIONS:=3}"

if [[ "${DEEPSOLO_MAIN_PYTHON}" != /* ]]; then
  DEEPSOLO_MAIN_PYTHON="${repo_root}/${DEEPSOLO_MAIN_PYTHON}"
fi
if [[ "${DEEPSOLO_RUNTIME_MANIFEST}" != /* ]]; then
  DEEPSOLO_RUNTIME_MANIFEST="${repo_root}/${DEEPSOLO_RUNTIME_MANIFEST}"
fi

for feature in avx avx2 fma; do
  if ! grep -m1 '^flags' /proc/cpuinfo | grep -qw "${feature}"; then
    echo "Host CPU does not expose ${feature}; Proxmox CPU type must be host." >&2
    exit 2
  fi
  if ! docker exec "${DEEPSOLO_CONTAINER}" sh -c "grep -m1 '^flags' /proc/cpuinfo | grep -qw '${feature}'"; then
    echo "Container CPU does not expose ${feature}." >&2
    exit 2
  fi
done

"${DEEPSOLO_MAIN_PYTHON}" - "${DEEPSOLO_SMOKE_IMAGE}" <<'PY'
import sys
import cv2
import numpy as np

path = sys.argv[1]
image = np.full((720, 1280, 3), 255, dtype=np.uint8)
for text, origin in (("DEEPSOLO", (90, 210)), ("TEST", (90, 420)), ("2026", (90, 630))):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 8, cv2.LINE_AA)
if not cv2.imwrite(path, image):
    raise RuntimeError("Could not write DeepSolo smoke fixture")
PY

"${DEEPSOLO_MAIN_PYTHON}" -m scripts.profile_deepsolo \
  --image "${DEEPSOLO_SMOKE_IMAGE}" \
  --output "${DEEPSOLO_SMOKE_OUTPUT}" \
  --docker-container "${DEEPSOLO_CONTAINER}" \
  --container-python "${DEEPSOLO_CONTAINER_PYTHON}" \
  --container-workdir "${DEEPSOLO_CONTAINER_SOURCE}" \
  --container-image-tag "${DEEPSOLO_IMAGE}" \
  --container-image-id "${DEEPSOLO_EXPECTED_IMAGE_ID}" \
  --runtime-manifest "${DEEPSOLO_RUNTIME_MANIFEST}" \
  --deepsolo-source-root "${DEEPSOLO_SOURCE_ROOT}" \
  --candidate ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE \
  --reduction-policy reading_order_space_join \
  --device cuda:0

"${DEEPSOLO_MAIN_PYTHON}" - "${DEEPSOLO_SMOKE_OUTPUT}" "${DEEPSOLO_EXPECTED_DETECTIONS}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected_detections = int(sys.argv[2])
assert payload["success"] is True
assert len(payload["detections"]) == expected_detections
artifact = payload["artifact"]
assert artifact["config_sha256"] == "0da913239e4b3f11ae1bad19e0b104996f7c3f3607aa2a7f4b1f52e3838b702d"
assert artifact["checkpoint_sha256"] == "ac2a01b3d8df0ee42e4632ff57a0d3a20a5706fb0ac1c5ae33b2455d0d90da68"
assert artifact["load_validation"]["total_parameters"] == 42588150
assert artifact["load_validation"]["trainable_parameters"] == 42365750
runtime = artifact["runtime_qualification"]
assert runtime["cudnn_enabled"] is True
assert runtime["cudnn_benchmark"] is False
assert runtime["cudnn_deterministic"] is False
print("DEEPSOLO M6 DOCKER SMOKE: PASS")
PY
