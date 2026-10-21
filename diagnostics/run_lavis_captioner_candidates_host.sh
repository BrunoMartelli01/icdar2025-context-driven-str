#!/usr/bin/env bash
# Five frozen RGB samples only. Models run sequentially in isolated environments.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
data_root="${1:-../../datasets/context-str/icdar13/raw/test_images}"
device="${2:-cpu}"
if [[ "$device" != cpu && "$device" != cuda:0 ]]; then
  echo 'Device must be cpu or cuda:0' >&2
  exit 2
fi
uv_exec=.venvs/uv-0.12.5/bin/uv
if [[ ! -x "$uv_exec" ]]; then
  echo 'Missing existing uv bootstrap: .venvs/uv-0.12.5/bin/uv' >&2
  exit 2
fi
export UV_PYTHON_INSTALL_DIR="$repo_root/.venvs/python"
export UV_CACHE_DIR="$repo_root/.venvs/uv-cache"
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
run_root=artifacts/diagnostics/lavis-captioner-candidates-20260914
export HF_HOME="$repo_root/$run_root/cache/huggingface-home"
export TORCH_HOME="$repo_root/$run_root/cache/torch"
# Do not inherit offline settings from the scientific production environments.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TRANSFORMERS_CACHE
legacy_env=.venvs/context-str-lavis
modern_python=.venvs/context-str-main-py311/bin/python
modern_deps="$repo_root/$run_root/runtime-deps"
if [[ ! -x "$modern_python" ]]; then
  echo "Missing existing main environment: $modern_python" >&2
  exit 2
fi
modern_run() {
  PYTHONPATH="$modern_deps${PYTHONPATH:+:$PYTHONPATH}" "$modern_python" "$@"
}
"$uv_exec" python install 3.10.16 --no-bin
if [[ -e "$legacy_env" && ! -x "$legacy_env/bin/python" ]]; then
  echo "Incomplete environment path; refusing overwrite: $legacy_env" >&2
  exit 2
fi
if [[ ! -e "$legacy_env" ]]; then
  "$uv_exec" venv --python 3.10.16 "$legacy_env"
fi
modern_run -u -m diagnostics.repair_decord_wheel --output-dir "$run_root/cache/packaging"
decord_wheel="$repo_root/$run_root/cache/packaging/decord-0.6.0-1tagfix-py3-none-manylinux2010_x86_64.whl"
index=https://download.pytorch.org/whl/cpu
legacy_torch=1.13.1+cpu legacy_vision=0.14.1+cpu
if [[ "$device" == cuda:0 ]]; then
  index=https://download.pytorch.org/whl/cu117
  legacy_torch=1.13.1+cu117 legacy_vision=0.14.1+cu117
fi
"$uv_exec" pip install --python "$legacy_env/bin/python" --extra-index-url "$index" --index-strategy unsafe-best-match \
  "torch==$legacy_torch" "torchvision==$legacy_vision" numpy==1.26.4 \
  transformers==4.26.1 huggingface-hub==0.25.2 timm==0.4.12 \
  spacy==3.7.5 pandas==2.2.3 "$decord_wheel" \
  'https://files.pythonhosted.org/packages/77/1b/112981873d44de113ecaab8084da82606871ad1e914e00c6ceb68f96b394/salesforce_lavis-1.0.2-py3-none-any.whl#sha256=489782efc2d98d8e878112b2758f82ac2e6c10dfcfa087db6656465c7784d49f'
"$uv_exec" pip check --python "$legacy_env/bin/python"
# The only missing pure-Python Florence dependency is kept in an isolated target;
# do not change the qualified main environment or duplicate PyTorch/CUDA wheels.
if ! modern_run -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("einops") else 1)'; then
  "$uv_exec" pip install --python "$modern_python" --target "$modern_deps" --no-deps einops==0.8.1
fi
"$uv_exec" pip check --python "$modern_python"
# Validate imports before spending time downloading the model checkpoints.
"$legacy_env/bin/python" -c 'import decord,torch,transformers,lavis; from lavis.models import model_zoo; print("LAVIS import PASS",torch.__version__,transformers.__version__,flush=True)'
modern_run -c 'import torch,transformers,einops,timm; print("Florence import PASS",torch.__version__,transformers.__version__,flush=True)'
modern_run -u -m diagnostics.lavis_captioner_candidates --resume-runtime-repair --prepare --data-root "$data_root"
failed=0
for candidate in blip-base-coco blip2-vitl florence2-base-ft; do
  if [[ "$candidate" == florence2-base-ft ]]; then
    if ! modern_run -u -m diagnostics.lavis_captioner_candidates \
      --resume-runtime-repair --candidate "$candidate" --data-root "$data_root" --device "$device"; then
      failed=1
      echo "[candidate failed] $candidate" >&2
    fi
  else
    if ! "$legacy_env/bin/python" -u -m diagnostics.lavis_captioner_candidates \
      --resume-runtime-repair --candidate "$candidate" --data-root "$data_root" --device "$device"; then
      failed=1
      echo "[candidate failed] $candidate; continuing with next candidate" >&2
    fi
  fi
done
modern_run -u -m diagnostics.lavis_captioner_candidates --resume-runtime-repair --compare --data-root "$data_root"
exit "$failed"
