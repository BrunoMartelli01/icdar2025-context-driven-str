#!/usr/bin/env bash
# Download the optional original corpus, then reuse the qualified host launcher.
set -euo pipefail
if [[ $# -lt 5 ]]; then
    printf 'Usage: bash scripts/run_unet_detection_methods_with_download.sh WORD_ROOT V1_ROOT NEW_OUTPUT_ROOT PARSEQ_ROOT DATASET_PARENT [--subset-only] [--discard-archive]\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
word="$(realpath "$1")"
v1="$(realpath "$2")"
output="$(realpath -m "$3")"
parseq="$(realpath "$4")"
dataset_parent="$(realpath -m "$5")"
shift 5
subset_only=0
discard_archive=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --subset-only) subset_only=1; shift ;;
        --discard-archive) discard_archive=1; shift ;;
        *) printf 'Unknown acquisition option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
cd "$repo"
test -x "$py"
# Qualify the existing source artifacts before starting a40GiB transfer.
minimum_images="$("$py" - "$word" "$v1" "$parseq" "$output" "$dataset_parent" <<'PY'
import json
from pathlib import Path
import sys
from scripts.run_unet_detection_methods import read_config, DEFAULT_CONFIG
from scripts.prune_segmentation_checkpoints import sha256_file
from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256, PARSEQ_RUNTIME_REVISION
c, _ = read_config(DEFAULT_CONFIG, "approximate")
word, v1, parseq, output, parent = map(Path, sys.argv[1:])
expected = [(word/c["train_manifest"], c["word_manifest_sha256"]),
            (v1/"control/best.pt", c["source_checkpoint_sha256"]),
            (v1/"control/run_manifest.json", c["source_run_manifest_sha256"]),
            (parseq/"parseq-bb5792a6.pt", PARSEQ_CHECKPOINT_SHA256),
            (parseq/f"parseq-{PARSEQ_RUNTIME_REVISION}.tar.gz", PARSEQ_RUNTIME_ARCHIVE_SHA256)]
for path, digest in expected:
    if sha256_file(path) != digest:
        raise ValueError(f"Qualified artifact checksum differs: {path}")
if not (parseq/f"parseq-{PARSEQ_RUNTIME_REVISION}").is_dir():
    raise FileNotFoundError("Qualified local PARSeq runtime directory is missing")
plan = output/"host_plan.json"
if plan.exists() and json.loads(plan.read_text()) != {"config": c, "synthtext_root": str(parent/"SynthText")}:
    raise ValueError("Existing experiment plan differs; choose a NEW output directory")
print(c["synthtext"]["max_images"])
PY
)"
acquire_options=()
if [[ "$subset_only" -eq 1 ]]; then
    acquire_options+=(--subset-config "$repo/configs/approximate/unet_detection_methods_v1.json" --word-root "$word")
fi
if [[ "$discard_archive" -eq 1 ]]; then
    acquire_options+=(--discard-archive)
fi
"$py" -u -m scripts.download_synthtext --dataset-parent "$dataset_parent" --minimum-images "$minimum_images" "${acquire_options[@]}"
exec bash scripts/run_unet_detection_methods_host.sh \
    "$word" "$v1" "$output" "$parseq" --synthtext-root "$dataset_parent/SynthText"
