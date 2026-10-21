#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_DIR="${1:-/extra1/bmartelli/datasets/context-str/ctw1500/downloads}"
OUTPUT="$DOWNLOAD_DIR/ctw1500_e2e_annos.zip"
URL='https://drive.usercontent.google.com/download?id=1tHf-n86z4zFRaRNEc02EO-hZINZqkQpz&export=download&confirm=t'

mkdir -p "$DOWNLOAD_DIR"
if [[ -s "$OUTPUT" ]] && ! unzip -tq "$OUTPUT"; then
  INVALID="$OUTPUT.invalid-$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$OUTPUT" "$INVALID"
  echo "[CTW1500] preserved invalid prior download as $INVALID" >&2
fi
if [[ ! -s "$OUTPUT" ]]; then
  curl -fL --retry 5 --retry-delay 2 --connect-timeout 30 \
    --progress-bar -o "$OUTPUT.part" "$URL"
  unzip -tq "$OUTPUT.part"
  mv "$OUTPUT.part" "$OUTPUT"
fi
unzip -tq "$OUTPUT"
sha256sum "$OUTPUT" | tee "$DOWNLOAD_DIR/SHA256SUMS.recognition.txt"
echo "[CTW1500] official recognition annotations: $OUTPUT"
