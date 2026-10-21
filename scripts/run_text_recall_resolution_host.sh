#!/usr/bin/env bash
# Run after separate offset training; one segmentation model at a time.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    printf 'Usage: bash scripts/run_text_recall_resolution_host.sh TARGET_ROOT BASE_TEXT_RUN NEW_OUTPUT_ROOT\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
cd "$repo"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
common=(--mode approximate --target-root "$1" --base-run "$2" --output-root "$3" --branch all)
"$py" -m scripts.run_text_recall_resolution "${common[@]}" --action preflight --device cpu
for action in smoke train evaluate; do
    "$py" -m scripts.run_text_recall_resolution "${common[@]}" --action "$action" --device cuda:0
done
"$py" -m scripts.run_text_recall_resolution "${common[@]}" --action compare --device cuda:0
