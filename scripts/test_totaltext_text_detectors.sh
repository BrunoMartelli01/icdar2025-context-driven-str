#!/usr/bin/env bash
# One host command: isolated packages + checked weights + real smoke + complete TEST.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
main_python="$repo_dir/.venvs/context-str-main-py311/bin/python"
det_python="$repo_dir/.venvs/context-str-text-detectors-py311/bin/python"
uv="$repo_dir/.venvs/uv-0.12.5/bin/uv"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-text-detectors-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/totaltext-text-detectors-v1/matplotlib}"
export EASYOCR_MODULE_PATH="${EASYOCR_MODULE_PATH:-/tmp/totaltext-text-detectors-v1/easyocr}"
export USE_TORCH=1
export USE_TF=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Check user arguments and device before installing or downloading anything.
"$main_python" - "$@" <<'PY'
import argparse,sys,torch
p=argparse.ArgumentParser()
p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
p.add_argument('--config'); p.add_argument('--paths')
p.add_argument('--models',nargs='+',choices=['craft','db_resnet50','fast_tiny'])
p.add_argument('--limit',type=int)
a=p.parse_args(sys.argv[1:])
if a.device == 'cuda:0' and not torch.cuda.is_available():
 raise SystemExit('CUDA unavailable here. Run on the GPU host or explicitly add --device cpu.')
if a.limit is not None and a.limit < 1: raise SystemExit('--limit must be positive')
PY

if [[ ! -x "$det_python" ]]; then
    "$uv" venv "$repo_dir/.venvs/context-str-text-detectors-py311" --python "$main_python"
fi
main_site="$("$main_python" -c 'import site; print(site.getsitepackages()[0])')"
det_site="$("$det_python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$main_site" > "$det_site/context_str_main.pth"

if ! "$det_python" - 2>/dev/null <<'PY'
import importlib.metadata as m
from pathlib import Path
for line in Path('environments/text-detectors/requirements.txt').read_text().splitlines():
 if line and not line.startswith('#'):
  package,version=line.split('=='); assert m.version(package)==version
import cv2,easyocr,doctr,skimage
PY
then
    "$uv" pip install --python "$det_python" --no-deps -r environments/text-detectors/requirements.txt
fi

"$det_python" -c 'import easyocr,doctr,torch,cv2; assert torch.__version__ == "2.4.1+cu124"'
"$det_python" -m pytest -q tests/localize/test_external_text_detectors.py tests/scripts/test_text_detector_suite.py
# Download command gets only the applicable arguments; no shell interpolation of paths.
"$det_python" - "$@" <<'PY'
import argparse,sys
from pathlib import Path
from scripts.prepare_totaltext_text_detectors import prepare,DEFAULT_CONFIG,DEFAULT_PATHS
p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
p.add_argument('--paths',type=Path,default=DEFAULT_PATHS)
p.add_argument('--models',nargs='+')
a,_=p.parse_known_args(sys.argv[1:])
prepare(a.config,a.paths,a.models)
PY

# A requested partial run stays partial. Otherwise qualify a real one-image run first.
partial=false
for argument in "$@"; do
    if [[ "$argument" == --limit || "$argument" == --limit=* ]]; then partial=true; fi
done
if [[ "$partial" == false ]]; then
    "$det_python" -m scripts.evaluate_totaltext_text_detectors "$@" --limit 1
fi
"$det_python" -m scripts.evaluate_totaltext_text_detectors "$@"
