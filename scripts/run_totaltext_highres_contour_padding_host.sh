#!/usr/bin/env bash
# Foreground, restartable GPU diagnostic; does not train or alter model weights.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/host-root" >&2
  exit 2
fi
HOST_ROOT="${1%/}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venvs/context-str-main-py311/bin/python"
DATA="$HOST_ROOT/datasets/context-str/scene-segmentation-v1"
RUNS="$HOST_ROOT/context-str-runs/totaltext-v4-highres-ablation-v1"
PARSEQ="$HOST_ROOT/context-str-artifacts/parseq"
REV=1902db043c029a7e03a3818c616c06600af574be
OUT="$HOST_ROOT/context-str-runs/totaltext-v4-highres-contour-padding-test-v1"

test -x "$PY" || { echo "Missing Python: $PY" >&2; exit 1; }
test -f "$DATA/manifests/total_text_test.yaml" || { echo "Missing qualified TEST manifest" >&2; exit 1; }
for variant in highres_baseline highres_separator highres_separator_residual; do
  test -f "$RUNS/$variant/run_manifest.json" || { echo "Missing run: $variant" >&2; exit 1; }
  test -f "$RUNS/$variant/best.pt" || { echo "Missing checkpoint: $variant" >&2; exit 1; }
done
test -f "$PARSEQ/parseq-bb5792a6.pt" || { echo "Missing PARSeq checkpoint" >&2; exit 1; }

cd "$REPO"
export PYTHONUNBUFFERED=1
exec "$PY" -u -m scripts.evaluate_totaltext_highres_contour_padding \
  --test-manifest "$DATA/manifests/total_text_test.yaml" \
  --data-root "$DATA" \
  --training-runs "$RUNS" \
  --parseq-source "$PARSEQ/parseq-$REV" \
  --parseq-source-archive "$PARSEQ/parseq-$REV.tar.gz" \
  --parseq-checkpoint "$PARSEQ/parseq-bb5792a6.pt" \
  --output-dir "$OUT" \
  --device cuda:0 \
  --parseq-batch-size 32
