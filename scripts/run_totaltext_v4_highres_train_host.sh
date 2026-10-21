#!/usr/bin/env bash
# Foreground GPU TRAIN-only practical ablation; no TEST or OCR is accessed.
set -euo pipefail

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
  shift
fi
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [--check-only] [host-root]" >&2
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${1:-$(dirname "$(dirname "$REPO")")}" # host root, outside repository
ROOT="${ROOT%/}"
PY="$REPO/.venvs/context-str-main-py311/bin/python"
TARGET="$REPO/artifacts/totaltext-curved-word-boxes-v4"
OUT="$ROOT/context-str-runs/totaltext-v4-highres-ablation-v1"

test -x "$PY" || { echo "Missing main Python: $PY" >&2; exit 1; }
test -f "$TARGET/manifests/total_text_train.yaml" || { echo "Missing qualified v4 TRAIN target" >&2; exit 1; }
test -d "$ROOT/context-str-runs" || { echo "Missing run root: $ROOT/context-str-runs" >&2; exit 1; }
available_kib="$(df -Pk "$ROOT/context-str-runs" | awk 'NR==2 {print $4}')"
if [[ -z "$available_kib" || "$available_kib" -lt 10485760 ]]; then
  echo "At least 10 GiB free disk is required before this ablation" >&2
  exit 1
fi

cd "$REPO"
export PYTHONUNBUFFERED=1
for variant in highres_baseline highres_separator highres_separator_residual; do
  echo "[highres] GPU full-batch preflight: $variant" >&2
  "$PY" -u -m scripts.train_totaltext_v4_highres_ablation --variant "$variant" --target-root "$TARGET" --device cuda:0 --check
done
if (( CHECK_ONLY )); then
  echo "[highres] all GPU preflights passed; no training started" >&2
  exit 0
fi
for variant in highres_baseline highres_separator highres_separator_residual; do
  echo "[highres] training: $variant" >&2
  "$PY" -u -m scripts.train_totaltext_v4_highres_ablation --variant "$variant" --target-root "$TARGET" --output-root "$OUT" --device cuda:0
done
echo "[highres] TRAIN/validation ablation complete: $OUT" >&2
