#!/usr/bin/env bash
# One command for a GPU-host smoke followed by the complete Total-Text TEST diagnostic.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash scripts/run_totaltext_yolo_word_slots.sh CHECKPOINT.pt [OUTPUT_DIR]" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
checkpoint="$(realpath "$1")"
output_dir="${2:-/tmp/yolo-word-slots/evaluation}"
main_python="$repo_dir/.venvs/context-str-main-py311/bin/python"
yolo_python="$repo_dir/.venvs/context-str-yolo-py311/bin/python"
uv="$repo_dir/.venvs/uv-0.12.5/bin/uv"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-yolo-word-slots-cache}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/yolo-word-slots/ultralytics-config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/yolo-word-slots/matplotlib-config}"

"$main_python" -c 'import json,sys; from pathlib import Path; from scripts.evaluate_totaltext_yolo_word_slots import DEFAULT_CONFIG,validate_checkpoint,validate_config; c=json.loads(DEFAULT_CONFIG.read_text()); validate_config(c); validate_checkpoint(Path(sys.argv[1]), c["checkpoint_source"])' "$checkpoint"
"$main_python" -c 'import sys,torch; sys.exit(0 if torch.cuda.is_available() else "CUDA GPU unavailable on this host")'
if [[ ! -x "$yolo_python" ]]; then
    "$uv" venv "$repo_dir/.venvs/context-str-yolo-py311" --python "$main_python"
fi

# Inherit the already pinned main PyTorch/CUDA stack read-only. All new packages
# are installed solely into the YOLO environment.
main_site="$("$main_python" -c 'import site; print(site.getsitepackages()[0])')"
yolo_site="$("$yolo_python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$main_site" > "$yolo_site/context_str_main.pth"

if ! "$yolo_python" -c 'import importlib.metadata as m; assert m.version("ultralytics") == "8.1.0"' 2>/dev/null; then
    "$uv" pip install --python "$yolo_python" --no-deps \
        'ultralytics==8.1.0' 'matplotlib==3.9.2' 'contourpy==1.3.1' \
        'cycler==0.12.1' 'fonttools==4.55.3' 'kiwisolver==1.4.8' \
        'pyparsing==3.2.0' 'python-dateutil==2.9.0.post0' 'six==1.17.0' \
        'pandas==2.2.3' 'pytz==2024.2' 'tzdata==2024.2' \
        'seaborn==0.13.2' 'py-cpuinfo==9.0.0' \
        'thop==0.1.1.post2209072238' 'hub-sdk==0.0.2'
fi

"$yolo_python" -c 'from ultralytics import YOLO; import torch; assert torch.cuda.is_available()'
"$main_python" -m pytest -q tests/scripts/test_yolo_word_slots.py tests/localize/test_independent_word_slots.py
"$yolo_python" -m scripts.evaluate_totaltext_yolo_word_slots \
    --checkpoint "$checkpoint" --output "$output_dir/smoke" --limit 1
"$yolo_python" -m scripts.evaluate_totaltext_yolo_word_slots \
    --checkpoint "$checkpoint" --output "$output_dir/full"
