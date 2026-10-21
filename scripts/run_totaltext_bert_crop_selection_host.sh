#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
TASK_PYTHON="$REPO_ROOT/.venvs/context-str-main-py311/bin/python"
TASK_CONFIG="$REPO_ROOT/configs/approximate/totaltext_bert_crop_selection_v1.json"
TASK_SOURCE=""
TASK_CAPTIONS=""
TASK_MODEL=""
TASK_OUTPUT=""
TASK_DEVICE=cpu
TASK_LIMIT=""
TASK_REUSE=""
usage() {
    echo "Usage: bash scripts/run_totaltext_bert_crop_selection_host.sh --source-dir DIR --captions FILE --model-dir PINNED_SNAPSHOT --output-dir DIR [--device cpu|cuda:0] [--python FILE] [--config FILE] [--limit N] [--reuse-inference-dir DIR]" >&2
}
while (($#)); do
    if (($# < 2)); then usage; exit 2; fi
    case "$1" in
        --source-dir) TASK_SOURCE=$2 ;;
        --captions) TASK_CAPTIONS=$2 ;;
        --model-dir) TASK_MODEL=$2 ;;
        --output-dir) TASK_OUTPUT=$2 ;;
        --device) TASK_DEVICE=$2 ;;
        --python) TASK_PYTHON=$2 ;;
        --config) TASK_CONFIG=$2 ;;
        --limit) TASK_LIMIT=$2 ;;
        --reuse-inference-dir) TASK_REUSE=$2 ;;
        *) usage; exit 2 ;;
    esac
    shift 2
done
if [[ -z "$TASK_SOURCE" || -z "$TASK_CAPTIONS" || -z "$TASK_MODEL" || -z "$TASK_OUTPUT" ]]; then
    usage; exit 2
fi
TASK_PREPARE=(--source-dir "$TASK_SOURCE" --captions "$TASK_CAPTIONS" --config "$TASK_CONFIG")
if [[ -n "$TASK_LIMIT" ]]; then TASK_PREPARE+=(--limit "$TASK_LIMIT"); fi
"$TASK_PYTHON" -m scripts.evaluate_totaltext_bert_crop_selection prepare --output-dir "$TASK_OUTPUT" "${TASK_PREPARE[@]}"
TASK_INFER=(--model-dir "$TASK_MODEL" --device "$TASK_DEVICE")
if [[ -n "$TASK_REUSE" ]]; then TASK_INFER+=(--reuse-inference-dir "$TASK_REUSE"); fi
"$TASK_PYTHON" -m scripts.evaluate_totaltext_bert_crop_selection infer --output-dir "$TASK_OUTPUT" "${TASK_INFER[@]}"
"$TASK_PYTHON" -m scripts.evaluate_totaltext_bert_crop_selection evaluate --output-dir "$TASK_OUTPUT"
TASK_METHOD=$(
    "$TASK_PYTHON" - "$TASK_OUTPUT/results.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
points = [v for v in result['operating_points'].values() if v]
points.append('visual@0.00')
print(max(points, key=lambda n: (result['summary'][n]['precision_proxy'], result['summary'][n]['correct'], n)))
PY
)
"$TASK_PYTHON" -m scripts.evaluate_totaltext_bert_crop_selection export --output-dir "$TASK_OUTPUT" --method "$TASK_METHOD"
