#!/usr/bin/env bash
# Persistent, foreground PAPER runner. No installation or administrative changes.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != --check ) ]]; then
  echo 'Usage: bash scripts/run_icdar13_routing_suite_host.sh USER_STORAGE_ROOT [--check]' >&2
  exit 2
fi
storage="$(cd "$1" && pwd)"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifacts="$storage/context-str-artifacts"
data="$storage/datasets/context-str/icdar13"
runs="$storage/context-str-runs"
output="${CONTEXT_STR_ROUTING_SUITE_ROOT:-$runs/icdar13-routing-suite-v1}"
python="$repo/.venvs/context-str-main-py311/bin/python"
deep_python="$repo/.venvs/context-str-deepsolo-host/bin/python"
revision=1902db043c029a7e03a3818c616c06600af574be
parseq="$artifacts/parseq/parseq-$revision"
checkpoint="$runs/ap-full-20260901-174246/icdar13-author-protocol-approximate"
checkpoint="$checkpoint/checkpoints/epoch-0098.pt"
calibration="$runs/table5-icdar13-repository-approximate-traces"
calibration="$calibration/practical-remap-train-calibration-v1.json"
for executable in "$python" "$deep_python"; do
  [[ -x "$executable" ]] || { echo "Missing executable: $executable" >&2; exit 1; }
done
export HF_HOME="$artifacts/huggingface-home"
export HF_HUB_CACHE="$artifacts/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$repo"
args=(
  --segmentation-manifest manifests/icdar13_segmentation_test.yaml
  --recognition-manifest manifests/icdar13_recognition_test.yaml
  --data-root "$data"
  --evaluation-config configs/approximate/segmentation_icdar13_author_protocol_test.yaml
  --checkpoint "$checkpoint" --v1-calibration "$calibration"
  --parseq-source "$parseq" --parseq-source-archive "$parseq.tar.gz"
  --parseq-checkpoint "$artifacts/parseq/parseq-bb5792a6.pt"
  --deepsolo-python "$deep_python"
  --deepsolo-runtime-manifest "$artifacts/deepsolo-host/qualification/runtime.json"
  --deepsolo-source-root "$artifacts/deepsolo-host/DeepSolo-upstream/DeepSolo"
  --device cuda:0 --batch-size 64
  --artifact-dir "$output/artifacts" --output "$output/results.json"
)
"$python" -u -m scripts.evaluate_icdar13_routing_suite "${args[@]}" --preflight-only
if [[ "${2:-}" == --check ]]; then exit 0; fi
"$python" -u -m scripts.check_table5_assets \
  --component segmentation --component parseq --component blip2 --component mpnet --component python \
  --segmentation-config configs/approximate/segmentation_icdar13_author_protocol_test.yaml \
  --segmentation-checkpoint "$checkpoint" \
  --parseq-source "$parseq" --parseq-source-type github_commit_archive \
  --parseq-source-archive "$parseq.tar.gz" \
  --parseq-checkpoint "$artifacts/parseq/parseq-bb5792a6.pt" \
  --hf-cache "$HF_HUB_CACHE"
exec "$python" -u -m scripts.evaluate_icdar13_routing_suite "${args[@]}"
