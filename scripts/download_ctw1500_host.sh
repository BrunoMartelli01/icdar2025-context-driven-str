#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_DIR="${1:-/extra1/bmartelli/datasets/context-str/ctw1500/downloads}"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# Remove only zero-byte remnants created by the obsolete official-link script.
find "$DOWNLOAD_DIR" -maxdepth 1 -type f -size 0 \
  \( -name 'train_images.zip' -o -name 'train_labels.zip' \
     -o -name 'test_images.zip' -o -name 'test_labels.zip' \) -delete

echo "[CTW1500] official Box/CloudStor links are unavailable (HTTP 400/stale host)."
echo "[CTW1500] using the public HB16888 mirror prepared in MMOCR format."
echo "[CTW1500] classification: MIRROR_CANDIDATE; physical contents require qualification."
wget -c --progress=bar:force:noscroll -O ctw1500_mmocr.zip \
  'https://huggingface.co/datasets/HB16888/CTW1500/resolve/main/ctw1500_mmocr.zip?download=true'
unzip -tq ctw1500_mmocr.zip
sha256sum ctw1500_mmocr.zip | tee SHA256SUMS.downloaded.txt
echo "[CTW1500] downloads complete in $DOWNLOAD_DIR"
