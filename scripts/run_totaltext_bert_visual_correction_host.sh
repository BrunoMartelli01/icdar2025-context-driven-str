#!/usr/bin/env bash
# Uses existing pinned models/environments; never installs dependencies.
set -euo pipefail
visual_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
visual_python="$visual_root/.venvs/context-str-main-py311/bin/python"
visual_config="$visual_root/configs/approximate/totaltext_bert_visual_correction_v1.json"
visual_upstream="" visual_assets="" visual_manifest="" visual_data="" visual_output=""
visual_source="" visual_checkpoint="" visual_archive="" visual_deps="" visual_cache="" visual_device=""
usage() {
  echo "Usage: bash scripts/run_totaltext_bert_visual_correction_host.sh --upstream-dir DIR --asset-dir DIR --rgb-manifest FILE --data-root DIR --output-dir DIR --parseq-source DIR --parseq-checkpoint FILE --parseq-archive FILE --runtime-deps DIR --hf-cache DIR --device cpu|cuda:0 [--python FILE] [--config FILE]"
}
while (($#)); do
  case "$1" in
    --upstream-dir) visual_upstream="${2:?Missing upstream}"; shift 2 ;;
    --asset-dir) visual_assets="${2:?Missing assets}"; shift 2 ;;
    --rgb-manifest) visual_manifest="${2:?Missing RGB manifest}"; shift 2 ;;
    --data-root) visual_data="${2:?Missing data root}"; shift 2 ;;
    --output-dir) visual_output="${2:?Missing output}"; shift 2 ;;
    --parseq-source) visual_source="${2:?Missing PARSeq source}"; shift 2 ;;
    --parseq-checkpoint) visual_checkpoint="${2:?Missing checkpoint}"; shift 2 ;;
    --parseq-archive) visual_archive="${2:?Missing archive}"; shift 2 ;;
    --runtime-deps) visual_deps="${2:?Missing existing dependencies}"; shift 2 ;;
    --hf-cache) visual_cache="${2:?Missing HF cache}"; shift 2 ;;
    --device) visual_device="${2:?Missing device}"; shift 2 ;;
    --python) visual_python="${2:?Missing Python}"; shift 2 ;;
    --config) visual_config="${2:?Missing approximate config}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
for visual_arg in "$visual_upstream" "$visual_assets" "$visual_manifest" "$visual_data" "$visual_output" \
                  "$visual_source" "$visual_checkpoint" "$visual_archive" "$visual_deps" "$visual_cache" "$visual_device"; do
  if [[ -z "$visual_arg" ]]; then usage >&2; exit 2; fi
done
cd "$visual_root"
export PYTHONUNBUFFERED=1
"$visual_python" -m scripts.evaluate_totaltext_bert_visual_correction prepare --config "$visual_config" \
  --upstream-dir "$visual_upstream" --asset-dir "$visual_assets" --rgb-manifest "$visual_manifest" \
  --data-root "$visual_data" --output-dir "$visual_output"
"$visual_python" -m scripts.evaluate_totaltext_bert_visual_correction visual --output-dir "$visual_output" \
  --parseq-source "$visual_source" --parseq-checkpoint "$visual_checkpoint" --parseq-archive "$visual_archive" \
  --runtime-deps "$visual_deps" --device "$visual_device"
"$visual_python" -m scripts.evaluate_totaltext_bert_visual_correction bert --output-dir "$visual_output" \
  --hf-cache "$visual_cache" --device "$visual_device"
"$visual_python" -m scripts.evaluate_totaltext_bert_visual_correction evaluate --output-dir "$visual_output"
