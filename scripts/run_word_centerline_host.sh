#!/usr/bin/env bash
# Qualified targets -> GPU smoke -> isolated head training -> geometry -> PARSeq.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    printf 'Usage: bash scripts/run_word_centerline_host.sh DATASET_ROOT SOURCE_CONTROL_RUN OUTPUT_ROOT PARSEQ_ARTIFACT_ROOT\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="${PYTHON:-$repo/.venvs/context-str-main-py311/bin/python}"
device="${DEVICE:-cuda:0}"
dataset="$(realpath "$1")"
source_run="$(realpath "$2")"
output="$(realpath -m "$3")"
parseq="$(realpath "$4")"
cd "$repo"
[[ -x "$py" ]] || { printf 'Python environment missing: %s\n' "$py" >&2; exit 2; }
[[ "$device" == cuda:* ]] || { printf 'DEVICE must explicitly select CUDA, e.g. cuda:0. Use the Python CLI for CPU smokes.\n' >&2; exit 2; }
[[ -f "$source_run/best.pt" && -f "$source_run/run_manifest.json" ]] || {
    printf 'SOURCE_CONTROL_RUN must be the qualified v1 control directory.\n' >&2; exit 2;
}
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$output"
exec 9>"$output/suite.lock"
flock -n 9 || { printf 'This centerline suite is already running.\n' >&2; exit 2; }
exec 8>"/tmp/context-str-single-gpu-$UID.lock"
flock -n 8 || { printf 'Another suite holds the single-GPU execution lock.\n' >&2; exit 2; }
revision="$("$py" -c 'from recognition.parseq import PARSEQ_RUNTIME_REVISION; print(PARSEQ_RUNTIME_REVISION)')"
"$py" -u - "$parseq" "$revision" "$output" "$device" <<'PY'
from pathlib import Path
import shutil
import sys
import torch
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    verify_parseq_runtime_source)
from scripts.prune_segmentation_checkpoints import sha256_file
p, revision, output, selected = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
verify_parseq_runtime_source(p / ("parseq-" + revision), source_type=PARSEQ_ARCHIVE_SOURCE,
                            source_archive=p / ("parseq-" + revision + ".tar.gz"))
if sha256_file(p / "parseq-bb5792a6.pt") != PARSEQ_CHECKPOINT_SHA256:
    raise ValueError("Qualified PARSeq checkpoint changed")
if shutil.disk_usage(output).free < 2_000_000_000:
    raise RuntimeError("Need at least 2GB free for checkpoints, maps and OCR receipts")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable on this host; no training started")
device = torch.device(selected)
print("[host]", torch.cuda.get_device_name(device), "device", selected, "PARSeq artifacts PASS", flush=True)
PY
common=(--mode approximate --dataset-root "$dataset" --source-run "$source_run" --cpu-threads 2)
"$py" -u -m scripts.run_word_centerline "${common[@]}" --action preflight --device cpu --output-root "$output"
# The smoke tests the same batch size as full training; no silent memory fallback.
"$py" -u -m scripts.run_word_centerline "${common[@]}" --action smoke --device "$device" --output-root "$output/gpu-smoke"
"$py" -u -m scripts.run_word_centerline "${common[@]}" --action train --device "$device" --output-root "$output"
"$py" -u -m scripts.run_word_centerline "${common[@]}" --action evaluate --device "$device" --output-root "$output"
"$py" -u -m scripts.evaluate_word_centerline_ocr --mode approximate --run-root "$output" --device "$device" \
    --cpu-threads 2 --parseq-source "$parseq/parseq-$revision" \
    --parseq-source-archive "$parseq/parseq-$revision.tar.gz" --parseq-checkpoint "$parseq/parseq-bb5792a6.pt"
printf 'Result: %s/centerline_ocr_comparison.md\n' "$output"
