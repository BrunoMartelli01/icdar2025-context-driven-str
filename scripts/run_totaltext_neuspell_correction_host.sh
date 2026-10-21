#!/usr/bin/env bash
# Frozen Total-Text and pinned spelling-trained BERT; existing environment only.
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
spell_python="$project_root/.venvs/context-str-main-py311/bin/python"
spell_config="$project_root/configs/approximate/totaltext_letterbox_neuspell_correction_v2.json"
spell_source=""
spell_cache=""
spell_assets=""
spell_output=""
spell_device=""
spell_caption_args=()
spell_reuse_args=()

usage() {
  echo "Usage: bash scripts/run_totaltext_neuspell_correction_host.sh --source-dir DIR --hf-cache DIR --asset-dir DIR --output-dir DIR --device cpu|cuda:0 [--reuse-ocr DIR] [--captions FILE] [--python FILE] [--config FILE]"
}

while (($#)); do
  case "$1" in
    --source-dir) spell_source="${2:?Missing source directory}"; shift 2 ;;
    --hf-cache) spell_cache="${2:?Missing Hugging Face cache}"; shift 2 ;;
    --asset-dir) spell_assets="${2:?Missing asset directory}"; shift 2 ;;
    --output-dir) spell_output="${2:?Missing output directory}"; shift 2 ;;
    --device) spell_device="${2:?Missing explicit device}"; shift 2 ;;
    --reuse-ocr) spell_reuse_args+=(--reuse-ocr "${2:?Missing OCR directory}"); shift 2 ;;
    --captions) spell_caption_args=(--captions "${2:?Missing caption file}"); shift 2 ;;
    --python) spell_python="${2:?Missing Python executable}"; shift 2 ;;
    --config) spell_config="${2:?Missing approximate configuration}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$spell_source" || -z "$spell_cache" || -z "$spell_assets" || -z "$spell_output" || -z "$spell_device" ]]; then
  usage >&2
  exit 2
fi

cd "$project_root"
export PYTHONUNBUFFERED=1
"$spell_python" -m scripts.evaluate_totaltext_neuspell prepare \
  --config "$spell_config" --source-dir "$spell_source" --output-dir "$spell_output" \
  "${spell_reuse_args[@]}" "${spell_caption_args[@]}"
"$spell_python" -m scripts.evaluate_totaltext_neuspell fetch-model \
  --config "$spell_config" --hf-cache "$spell_cache" --asset-dir "$spell_assets"
"$spell_python" -m scripts.evaluate_totaltext_neuspell infer \
  --output-dir "$spell_output" --hf-cache "$spell_cache" --asset-dir "$spell_assets" --device "$spell_device"
"$spell_python" -m scripts.evaluate_totaltext_neuspell evaluate --output-dir "$spell_output"
