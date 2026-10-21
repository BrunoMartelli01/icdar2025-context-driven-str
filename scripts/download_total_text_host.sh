#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_DIR="${1:-/extra1/bmartelli/datasets/context-str/total_text/downloads}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_BIN="$REPO_ROOT/.venvs/context-str-main-py311/bin/hf"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

test -x "$HF_BIN" || { echo "Missing Hugging Face CLI: $HF_BIN" >&2; exit 1; }
find "$DOWNLOAD_DIR" -maxdepth 1 -type f -size 0 -name 'totaltext.zip' -delete

echo "[Total-Text] historical author host is unavailable (connection reset)."
echo "[Total-Text] using yunusserhat/Total-Text-Dataset, a public mirror of the official tree."
echo "[Total-Text] classification: MIRROR_CANDIDATE; physical contents require qualification."
HF_HUB_DISABLE_XET=1 "$HF_BIN" download yunusserhat/Total-Text-Dataset \
  --repo-type dataset --max-workers 1 \
  --local-dir "$DOWNLOAD_DIR/Total-Text-Dataset"

IMAGE_COUNT="$(find "$DOWNLOAD_DIR/Total-Text-Dataset/Images" -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l)"
test "$IMAGE_COUNT" -eq 1555 || {
  echo "Expected 1555 Total-Text images; found $IMAGE_COUNT" >&2
  exit 1
}
find "$DOWNLOAD_DIR/Total-Text-Dataset" -type f \
  ! -path '*/.cache/*' -print0 | sort -z | xargs -0 sha256sum \
  > SHA256SUMS.downloaded.txt
sha256sum SHA256SUMS.downloaded.txt | tee SHA256SUMS.downloaded.txt.sha256
echo "[Total-Text] downloads complete in $DOWNLOAD_DIR"
