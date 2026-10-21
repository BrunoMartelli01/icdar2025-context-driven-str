#!/usr/bin/env bash
# Frozen completed predictions -> CPU seeded/gated replay -> genuine GPU PARSeq.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    printf 'Usage: bash scripts/run_unet_seeded_selective_host.sh WORD_ROOT COMPLETED_CROP_ROOT NEW_OUTPUT_ROOT PARSEQ_ROOT\n' >&2
    exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$repo/.venvs/context-str-main-py311/bin/python"
word="$(realpath "$1")"
source_run="$(realpath "$2")"
output="$(realpath -m "$3")"
parseq="$(realpath "$4")"
cd "$repo"
export PYTHONUNBUFFERED=1
"$py" -c 'import sys; from pathlib import Path; out=Path(sys.argv[3]); roots=[Path(sys.argv[1]),Path(sys.argv[2])]; assert out!=Path.cwd() and all(out!=p and p not in out.parents and out not in p.parents for p in roots), "Output must be isolated from source artifacts"' "$word" "$source_run" "$output"
mkdir -p "$output"
exec 9>"$output/suite.lock"
flock -n 9 || { printf 'The seeded/selective suite is already running.\n' >&2; exit 2; }
if [[ -f "$source_run/suite.lock" ]]; then
    exec 8<"$source_run/suite.lock"
    flock -s 8
fi
revision="$("$py" -c 'from recognition.parseq import PARSEQ_RUNTIME_REVISION; print(PARSEQ_RUNTIME_REVISION)')"
"$py" -u -m scripts.run_unet_seeded_selective --mode approximate \
    --word-root "$word" --crop-root "$source_run" --output-root "$output"
"$py" -u -m scripts.evaluate_unet_seeded_selective_ocr --mode approximate \
    --word-root "$word" --run-root "$output" --parseq-source "$parseq/parseq-$revision" \
    --parseq-source-archive "$parseq/parseq-$revision.tar.gz" \
    --parseq-checkpoint "$parseq/parseq-bb5792a6.pt" --device cuda:0 \
    --cache-source "$source_run/corrective_ocr_cache"
"$py" -u -m scripts.summarize_unet_seeded_selective --run-root "$output"
