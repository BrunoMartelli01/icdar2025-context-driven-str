#!/usr/bin/env bash
# Genuine BERT experiment in the existing environment; no package installation.
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bert_python="$project_root/.venvs/context-str-main-py311/bin/python"
bert_config="$project_root/configs/approximate/totaltext_v4_bert_correction_v1.json"
bert_source=""
bert_cache=""
bert_output=""
bert_device=""
bert_caption_args=()
bert_reuse_args=()

usage() {
  echo "Usage: bash scripts/run_totaltext_bert_correction_host.sh --source-dir DIR --hf-cache DIR --output-dir DIR --device cpu|cuda:0 [--reuse-ocr DIR] [--captions FILE] [--python FILE] [--config FILE]"
}

while (($#)); do
  case "$1" in
    --source-dir) bert_source="${2:?Missing source directory}"; shift 2 ;;
    --hf-cache) bert_cache="${2:?Missing Hugging Face cache}"; shift 2 ;;
    --output-dir) bert_output="${2:?Missing output directory}"; shift 2 ;;
    --device) bert_device="${2:?Missing explicit device}"; shift 2 ;;
    --reuse-ocr) bert_reuse_args+=(--reuse-ocr "${2:?Missing OCR directory}"); shift 2 ;;
    --captions) bert_caption_args=(--captions "${2:?Missing caption file}"); shift 2 ;;
    --python) bert_python="${2:?Missing Python executable}"; shift 2 ;;
    --config) bert_config="${2:?Missing approximate configuration}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$bert_source" || -z "$bert_cache" || -z "$bert_output" || -z "$bert_device" ]]; then
  usage >&2
  exit 2
fi

cd "$project_root"
export PYTHONUNBUFFERED=1
"$bert_python" -m scripts.evaluate_totaltext_bert_correction prepare \
  --config "$bert_config" --source-dir "$bert_source" --output-dir "$bert_output" \
  "${bert_reuse_args[@]}" "${bert_caption_args[@]}"
"$bert_python" -m scripts.evaluate_totaltext_bert_correction fetch-model \
  --config "$bert_config" --hf-cache "$bert_cache"
"$bert_python" -m scripts.evaluate_totaltext_bert_correction infer \
  --output-dir "$bert_output" --hf-cache "$bert_cache" --device "$bert_device"
"$bert_python" -m scripts.evaluate_totaltext_bert_correction evaluate --output-dir "$bert_output"
