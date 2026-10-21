#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

: "${DEEPSOLO_HOST_PYTHON:=${repo_root}/.venvs/context-str-deepsolo-host/bin/python}"
: "${DEEPSOLO_MAIN_PYTHON:=${repo_root}/.venvs/context-str-main/bin/python}"
: "${DEEPSOLO_SOURCE_ROOT:?DEEPSOLO_SOURCE_ROOT is required}"
: "${DEEPSOLO_CHECKPOINT:?DEEPSOLO_CHECKPOINT is required}"
: "${DEEPSOLO_HOST_REPORT:=/tmp/context-str-deepsolo-host-environment.json}"
: "${DEEPSOLO_RUNTIME_MANIFEST:=/tmp/context-str-deepsolo-host-runtime.json}"
: "${DEEPSOLO_SMOKE_IMAGE:=/tmp/context-str-deepsolo-host-smoke.png}"
: "${DEEPSOLO_SMOKE_OUTPUT:=/tmp/context-str-deepsolo-host-smoke.json}"
: "${DEEPSOLO_CUDA_HOME:=/usr/local/cuda-11.1}"
: "${DEEPSOLO_SOURCE_TYPE:=git_checkout}"

source_identity_args=(--source-type "${DEEPSOLO_SOURCE_TYPE}")
if [[ "${DEEPSOLO_SOURCE_TYPE}" == "github_commit_archive" ]]; then
  : "${DEEPSOLO_SOURCE_REVISION:?DEEPSOLO_SOURCE_REVISION is required for an archive}"
  : "${DEEPSOLO_SOURCE_COMMIT_MARKER:?DEEPSOLO_SOURCE_COMMIT_MARKER is required for an archive}"
  : "${DEEPSOLO_SOURCE_ARCHIVE:?DEEPSOLO_SOURCE_ARCHIVE is required for an archive}"
  : "${DEEPSOLO_SOURCE_ARCHIVE_SHA256:?DEEPSOLO_SOURCE_ARCHIVE_SHA256 is required for an archive}"
  source_identity_args+=(
    --source-revision "${DEEPSOLO_SOURCE_REVISION}"
    --source-commit-marker "${DEEPSOLO_SOURCE_COMMIT_MARKER}"
    --source-archive "${DEEPSOLO_SOURCE_ARCHIVE}"
    --source-archive-sha256 "${DEEPSOLO_SOURCE_ARCHIVE_SHA256}"
  )
elif [[ "${DEEPSOLO_SOURCE_TYPE}" != "git_checkout" ]]; then
  echo "Unsupported DEEPSOLO_SOURCE_TYPE: ${DEEPSOLO_SOURCE_TYPE}" >&2
  exit 2
fi

export CUDA_HOME="${DEEPSOLO_CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

for feature in avx avx2 fma; do
  if ! grep -m1 '^flags' /proc/cpuinfo | grep -qw "${feature}"; then
    echo "Host CPU does not expose ${feature}." >&2
    exit 2
  fi
done

"${DEEPSOLO_HOST_PYTHON}" "${repo_root}/scripts/check_deepsolo_environment.py" \
  --source-root "${DEEPSOLO_SOURCE_ROOT}" \
  --checkpoint "${DEEPSOLO_CHECKPOINT}" \
  "${source_identity_args[@]}" \
  --output "${DEEPSOLO_HOST_REPORT}"

"${DEEPSOLO_MAIN_PYTHON}" -m scripts.write_deepsolo_runtime_manifest \
  --source-root "${DEEPSOLO_SOURCE_ROOT}" \
  "${source_identity_args[@]}" \
  --output "${DEEPSOLO_RUNTIME_MANIFEST}" \
  --candidate-checkpoint \
  "ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE=${DEEPSOLO_CHECKPOINT}" \
  --runtime-kind host_subprocess \
  --runtime-python "${DEEPSOLO_HOST_PYTHON}" \
  --runtime-environment-report "${DEEPSOLO_HOST_REPORT}"

"${DEEPSOLO_MAIN_PYTHON}" - "${DEEPSOLO_SMOKE_IMAGE}" <<'PY'
import sys
import cv2
import numpy as np

path = sys.argv[1]
image = np.full((720, 1280, 3), 255, dtype=np.uint8)
for text, origin in (("DEEPSOLO", (90, 210)), ("TEST", (90, 420)), ("2026", (90, 630))):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 8, cv2.LINE_AA)
if not cv2.imwrite(path, image):
    raise RuntimeError("Could not write DeepSolo host smoke fixture")
PY

"${DEEPSOLO_MAIN_PYTHON}" -m scripts.profile_deepsolo \
  --image "${DEEPSOLO_SMOKE_IMAGE}" \
  --output "${DEEPSOLO_SMOKE_OUTPUT}" \
  --deepsolo-python "${DEEPSOLO_HOST_PYTHON}" \
  --runtime-manifest "${DEEPSOLO_RUNTIME_MANIFEST}" \
  --deepsolo-source-root "${DEEPSOLO_SOURCE_ROOT}" \
  --candidate ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE \
  --reduction-policy reading_order_space_join \
  --device cuda:0

"${DEEPSOLO_MAIN_PYTHON}" - "${DEEPSOLO_SMOKE_OUTPUT}" <<'PY'
import json
import sys

from scripts.check_deepsolo_environment import EXPECTED

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["success"] is True
assert len(payload["detections"]) == 3
artifact = payload["artifact"]
assert artifact["config_sha256"] == "0da913239e4b3f11ae1bad19e0b104996f7c3f3607aa2a7f4b1f52e3838b702d"
assert artifact["checkpoint_sha256"] == "ac2a01b3d8df0ee42e4632ff57a0d3a20a5706fb0ac1c5ae33b2455d0d90da68"
assert artifact["load_validation"]["total_parameters"] == 42588150
assert artifact["load_validation"]["trainable_parameters"] == 42365750
versions = artifact["versions"]
assert versions["python"].startswith("3.8.")
assert versions["torch"] == EXPECTED["torch"]
assert versions["torchvision"] == EXPECTED["torchvision"]
assert versions["detectron2"] == EXPECTED["detectron2"]
assert versions["cuda_runtime"] == "11.1"
runtime = artifact["runtime_qualification"]
assert runtime["cudnn_enabled"] is True
assert runtime["cudnn_benchmark"] is False
assert runtime["cudnn_deterministic"] is False
assert runtime.get("container_image") is None
assert runtime.get("container_image_id") is None
assert isinstance(runtime["gpu_name"], str) and runtime["gpu_name"]
memory = payload["memory"]
assert memory["gpu_peak_allocated_bytes"] > 0
assert memory["gpu_peak_reserved_bytes"] > 0
print("DEEPSOLO NON-DOCKER HOST SMOKE: PASS")
PY
