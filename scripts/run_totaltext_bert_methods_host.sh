#!/usr/bin/env bash
# Existing environments/models only; dataset/cache/artifact locations are explicit.
set -euo pipefail
method_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
method_python="$method_root/.venvs/context-str-main-py311/bin/python"
method_config="$method_root/configs/approximate/totaltext_bert_methods_v1.json"
method_source="" method_cache="" method_assets="" method_manifest="" method_data=""
method_deps="" method_modules="" method_captions="" method_output="" method_device=""
method_reuse=()
usage() {
  echo "Usage: bash scripts/run_totaltext_bert_methods_host.sh --source-dir DIR --hf-cache DIR --asset-dir DIR --rgb-manifest FILE --data-root DIR --runtime-deps DIR --module-cache DIR --caption-dir DIR --output-dir DIR --device cpu|cuda:0 [--reuse-ocr DIR] [--config FILE] [--python FILE]"
}
while (($#)); do
  case "$1" in
    --source-dir) method_source="${2:?Missing source}"; shift 2 ;;
    --hf-cache) method_cache="${2:?Missing cache}"; shift 2 ;;
    --asset-dir) method_assets="${2:?Missing assets}"; shift 2 ;;
    --rgb-manifest) method_manifest="${2:?Missing RGB manifest}"; shift 2 ;;
    --data-root) method_data="${2:?Missing data root}"; shift 2 ;;
    --runtime-deps) method_deps="${2:?Missing existing isolated dependency target}"; shift 2 ;;
    --module-cache) method_modules="${2:?Missing module cache}"; shift 2 ;;
    --caption-dir) method_captions="${2:?Missing caption directory}"; shift 2 ;;
    --output-dir) method_output="${2:?Missing output directory}"; shift 2 ;;
    --device) method_device="${2:?Missing explicit device}"; shift 2 ;;
    --reuse-ocr) method_reuse+=(--reuse-ocr "${2:?Missing OCR cache}"); shift 2 ;;
    --config) method_config="${2:?Missing approximate config}"; shift 2 ;;
    --python) method_python="${2:?Missing Python}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
for method_value in "$method_source" "$method_cache" "$method_assets" "$method_manifest" "$method_data" \
                    "$method_deps" "$method_modules" "$method_captions" "$method_output" "$method_device"; do
  if [[ -z "$method_value" ]]; then usage >&2; exit 2; fi
done
cd "$method_root"
export PYTHONUNBUFFERED=1
"$method_python" -m scripts.evaluate_totaltext_bert_methods prepare --config "$method_config" \
  --source-dir "$method_source" --output-dir "$method_output/text-inputs" --conditions word_only local_ocr "${method_reuse[@]}"
if [[ ! -f "$method_captions/captions.json" ]]; then
  "$method_python" -m scripts.generate_totaltext_bert_captions --config "$method_config" \
    --inputs "$method_output/text-inputs/inputs.json" --manifest "$method_manifest" --data-root "$method_data" \
    --hf-cache "$method_cache" --runtime-deps "$method_deps" --module-cache "$method_modules" \
    --output-dir "$method_captions" --device "$method_device"
fi
# prepare verifies reused caption model provenance and exact RGB/sample coverage.
"$method_python" -m scripts.evaluate_totaltext_bert_methods prepare --config "$method_config" \
  --source-dir "$method_source" --output-dir "$method_output/full" --captions "$method_captions/captions.json" "${method_reuse[@]}"
"$method_python" -m scripts.evaluate_totaltext_bert_methods head --output-dir "$method_output/full" \
  --hf-cache "$method_cache" --asset-dir "$method_assets" --device "$method_device"
"$method_python" -m scripts.evaluate_totaltext_bert_methods pll --output-dir "$method_output/full" \
  --hf-cache "$method_cache" --device "$method_device"
"$method_python" -m scripts.evaluate_totaltext_bert_methods evaluate --output-dir "$method_output/full"
