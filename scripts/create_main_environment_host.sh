#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_environment="${repository_root}/.venvs/uv-0.12.5"
uv_cache="${repository_root}/.venvs/uv-cache"
python_install_directory="${repository_root}/.venvs/python"
main_environment="${repository_root}/.venvs/context-str-main-py311"
snapshot="${repository_root}/environments/main/installed.txt"

if [[ ! -x /usr/bin/python3 ]]; then
  echo "Required bootstrap interpreter is missing: /usr/bin/python3" >&2
  exit 1
fi

if [[ ! -d "${uv_environment}" ]]; then
  /usr/bin/python3 -m venv "${uv_environment}"
fi
"${uv_environment}/bin/python" -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  "uv==0.12.5"
uv_executable="${uv_environment}/bin/uv"
uv_version="$("${uv_executable}" --version)"
if [[ "${uv_version%% (*}" != "uv 0.12.5" ]]; then
  echo "The uv bootstrap does not have the frozen version 0.12.5" >&2
  exit 1
fi

export UV_PYTHON_INSTALL_DIR="${python_install_directory}"
export UV_CACHE_DIR="${uv_cache}"
export UV_HTTP_TIMEOUT=600
export UV_HTTP_RETRIES=5
export UV_CONCURRENT_DOWNLOADS=1
"${uv_executable}" python install 3.11.11 --no-bin

if [[ -e "${main_environment}" ]]; then
  if [[ ! -x "${main_environment}/bin/python" ]]; then
    echo "Refusing to overwrite an incomplete path: ${main_environment}" >&2
    exit 1
  fi
  actual_python="$("${main_environment}/bin/python" -c 'import platform; print(platform.python_version())')"
  if [[ "${actual_python}" != "3.11.11" ]]; then
    echo "Refusing to overwrite ${main_environment}: Python is ${actual_python}" >&2
    exit 1
  fi
else
  "${uv_executable}" venv --python 3.11.11 "${main_environment}"
fi

"${uv_executable}" pip sync \
  --python "${main_environment}/bin/python" \
  --default-index "https://pypi.org/simple" \
  --torch-backend cu124 \
  "${snapshot}"
"${uv_executable}" pip check --python "${main_environment}/bin/python"

expected_freeze="$(mktemp /tmp/context-str-main-expected.XXXXXX)"
actual_freeze="$(mktemp /tmp/context-str-main-actual.XXXXXX)"
trap 'rm -f "${expected_freeze}" "${actual_freeze}"' EXIT
sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' "${snapshot}" | LC_ALL=C sort > "${expected_freeze}"
"${uv_executable}" pip freeze --python "${main_environment}/bin/python" | LC_ALL=C sort > "${actual_freeze}"
if ! diff -u "${expected_freeze}" "${actual_freeze}"; then
  echo "Installed state does not match environments/main/installed.txt" >&2
  exit 1
fi

"${main_environment}/bin/python" - <<'PY'
import json
import platform
import sys

print(json.dumps({
    "environment": "context-str-main-py311",
    "executable": sys.executable,
    "installed_state_matches_snapshot": True,
    "python": platform.python_version(),
    "status": "ready_for_physical_qualification",
}, indent=2, sort_keys=True))
PY
