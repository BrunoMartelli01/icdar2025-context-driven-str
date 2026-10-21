#!/usr/bin/env bash
# Existing assets only; the padded experiment uses a separate output directory.
set -euo pipefail
pad_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
pad_python="$pad_root/.venvs/context-str-main-py311/bin/python"
pad_config="$pad_root/configs/approximate/totaltext_bert_padded_evidence_v1.json"
pad_upstream="" pad_output="" pad_source="" pad_checkpoint="" pad_archive="" pad_deps="" pad_cache="" pad_device=""
usage() {
  echo "Usage: bash scripts/run_totaltext_bert_padded_evidence_host.sh --upstream-dir DIR --output-dir DIR --parseq-source DIR --parseq-checkpoint FILE --parseq-archive FILE --runtime-deps DIR --hf-cache DIR --device cpu|cuda:0 [--python FILE] [--config FILE]"
}
while (($#)); do
  case "$1" in
    --upstream-dir) pad_upstream="${2:?Missing upstream}"; shift 2 ;;
    --output-dir) pad_output="${2:?Missing output}"; shift 2 ;;
    --parseq-source) pad_source="${2:?Missing source}"; shift 2 ;;
    --parseq-checkpoint) pad_checkpoint="${2:?Missing checkpoint}"; shift 2 ;;
    --parseq-archive) pad_archive="${2:?Missing archive}"; shift 2 ;;
    --runtime-deps) pad_deps="${2:?Missing dependencies}"; shift 2 ;;
    --hf-cache) pad_cache="${2:?Missing cache}"; shift 2 ;;
    --device) pad_device="${2:?Missing device}"; shift 2 ;;
    --python) pad_python="${2:?Missing Python}"; shift 2 ;;
    --config) pad_config="${2:?Missing config}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
for pad_arg in "$pad_upstream" "$pad_output" "$pad_source" "$pad_checkpoint" "$pad_archive" "$pad_deps" "$pad_cache" "$pad_device"; do
  if [[ -z "$pad_arg" ]]; then usage >&2; exit 2; fi
done
cd "$pad_root"
export PYTHONUNBUFFERED=1
"$pad_python" -m scripts.evaluate_totaltext_bert_padded_evidence prepare --config "$pad_config" \
  --upstream-dir "$pad_upstream" --output-dir "$pad_output"
"$pad_python" -m scripts.evaluate_totaltext_bert_padded_evidence visual --output-dir "$pad_output" \
  --parseq-source "$pad_source" --parseq-checkpoint "$pad_checkpoint" --parseq-archive "$pad_archive" \
  --runtime-deps "$pad_deps" --device "$pad_device"
"$pad_python" -m scripts.evaluate_totaltext_bert_padded_evidence bert --output-dir "$pad_output" \
  --hf-cache "$pad_cache" --device "$pad_device"
"$pad_python" -m scripts.evaluate_totaltext_bert_padded_evidence evaluate --output-dir "$pad_output"
