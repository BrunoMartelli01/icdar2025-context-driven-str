#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_DIR="${1:-/extra1/bmartelli/datasets/context-str/icdar2015/downloads}"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

download_zip() {
  local output="$1"
  local url="$2"
  local expected_md5="$3"
  local partial="${output}.part"

  if [[ -f "$output" ]] && printf '%s  %s\n' "$expected_md5" "$output" | md5sum --check --status; then
    echo "[ICDAR2015] verified reusable archive: $output"
    return
  fi

  # curl/wget may leave an empty destination after a transport failure.
  if [[ -f "$output" && ! -s "$output" ]]; then
    rm -- "$output"
  fi
  if [[ -e "$output" ]]; then
    echo "[ICDAR2015] existing $output does not match the published MD5; refusing to reuse it." >&2
    exit 1
  fi

  echo "[ICDAR2015] downloading $output"
  curl --fail --location --show-error --progress-bar \
    --retry 5 --retry-delay 3 --connect-timeout 30 \
    --continue-at - --output "$partial" "$url"

  if ! printf '%s  %s\n' "$expected_md5" "$partial" | md5sum --check --status; then
    echo "[ICDAR2015] checksum mismatch for $partial; leaving it for inspection." >&2
    echo "[ICDAR2015] expected MD5: $expected_md5" >&2
    md5sum "$partial" >&2
    exit 1
  fi
  if ! unzip -tq "$partial"; then
    echo "[ICDAR2015] $output is not a valid ZIP." >&2
    exit 1
  fi
  mv "$partial" "$output"
  echo "[ICDAR2015] verified published MD5: $output"
}

download_zip ch4_training_images.zip \
  'http://rrc.cvc.uab.es/downloads/ch4_training_images.zip' \
  'c51cbace155dcc4d98c8dd19d378f30d'
download_zip ch4_training_localization_transcription_gt.zip \
  'http://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip' \
  '3bfaf1988960909014f7987d2343060b'
download_zip ch4_test_images.zip \
  'http://rrc.cvc.uab.es/downloads/ch4_test_images.zip' \
  '97e4c1ddcf074ffcc75feff2b63c35dd'
# RRC Task 4 and Task 1 test GT archives contain the same annotations; MMOCR
# publishes the checksum under the Task 4 filename. Preserve the Task 1 name
# expected by this repository after byte-level qualification.
download_zip Challenge4_Test_Task1_GT.zip \
  'http://rrc.cvc.uab.es/downloads/Challenge4_Test_Task4_GT.zip' \
  '8bce173b06d164b98c357b0eb96ef430'
md5sum ch4_training_images.zip ch4_training_localization_transcription_gt.zip \
  ch4_test_images.zip Challenge4_Test_Task1_GT.zip \
  > MD5SUMS.verified.txt
sha256sum ch4_training_images.zip ch4_training_localization_transcription_gt.zip \
  ch4_test_images.zip Challenge4_Test_Task1_GT.zip \
  | tee SHA256SUMS.downloaded.txt
echo "[ICDAR2015] downloads complete in $DOWNLOAD_DIR"
