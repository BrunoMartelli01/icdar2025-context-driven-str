#!/usr/bin/env bash
# Qualified targets -> GPU smoke -> two matched GPU trainings -> frozen PARSeq.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    printf 'Usage: bash scripts/run_adaspot_gap_host.sh WORD_ROOT KERNEL_RUN NEW_OUTPUT_ROOT PARSEQ_ARTIFACT_ROOT\n' >&2
    exit 2
fi
adaspot_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
adaspot_python="${ADASPOT_PYTHON:-$adaspot_repo/.venvs/context-str-main-py311/bin/python}"
adaspot_device="${ADASPOT_DEVICE:-cuda:0}"
adaspot_word="$(realpath "$1")"
adaspot_kernel="$(realpath "$2")"
adaspot_output="$(realpath -m "$3")"
adaspot_parseq="$(realpath "$4")"
cd "$adaspot_repo"
[[ -f "$adaspot_python" ]] || { printf 'Missing existing Python environment: %s\n' "$adaspot_python" >&2; exit 2; }
case "$adaspot_device" in cuda|cuda:*) ;; *) printf 'Training requires CUDA; no CPU fallback.\n' >&2; exit 2 ;; esac
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$adaspot_output"
exec 9>"$adaspot_output/suite.lock"
flock -n 9 || { printf 'This AdaSpot suite is already running.\n' >&2; exit 2; }
adaspot_active_stage=preflight
adaspot_failure() {
    adaspot_exit=$?
    printf '[FAILED] stage=%s exit=%s; see stage failure.json and this log.\n' "$adaspot_active_stage" "$adaspot_exit" >&2
    "$adaspot_python" -c 'import json,sys; from pathlib import Path; from datetime import datetime,timezone; p=Path(sys.argv[1]); (p/"suite_failure.json").write_text(json.dumps({"status":"FAILED","stage":sys.argv[2],"exit_code":int(sys.argv[3]),"time_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")' "$adaspot_output" "$adaspot_active_stage" "$adaspot_exit"
    exit "$adaspot_exit"
}
trap adaspot_failure ERR
adaspot_revision="$("$adaspot_python" -c 'from recognition.parseq import PARSEQ_RUNTIME_REVISION; print(PARSEQ_RUNTIME_REVISION)')"
"$adaspot_python" -u -c 'import shutil,sys,torch; from pathlib import Path; from recognition.parseq import PARSEQ_CHECKPOINT_SHA256,PARSEQ_RUNTIME_ARCHIVE_SHA256; from scripts.prune_segmentation_checkpoints import sha256_file; p=Path(sys.argv[1]); assert torch.cuda.is_available(), "CUDA unavailable"; assert sha256_file(p/"parseq-bb5792a6.pt")==PARSEQ_CHECKPOINT_SHA256; assert sha256_file(p/("parseq-"+sys.argv[2]+".tar.gz"))==PARSEQ_RUNTIME_ARCHIVE_SHA256; assert (p/("parseq-"+sys.argv[2])).is_dir(); assert shutil.disk_usage(sys.argv[3]).free>=8_000_000_000, "Need8GB free for best/latest checkpoints and maps"; print("[preflight] GPU:",torch.cuda.get_device_name(torch.device(sys.argv[4])),"local PARSeq/disk PASS",flush=True)' "$adaspot_parseq" "$adaspot_revision" "$adaspot_output" "$adaspot_device"
adaspot_common=(--mode approximate --word-root "$adaspot_word" --kernel-run "$adaspot_kernel" --output-root "$adaspot_output")
"$adaspot_python" -u -m scripts.run_adaspot_gap "${adaspot_common[@]}" --action preflight --device cpu
for adaspot_branch in gap control; do
    adaspot_active_stage="$adaspot_branch:smoke"
    "$adaspot_python" -u -m scripts.run_adaspot_gap "${adaspot_common[@]}" --branch "$adaspot_branch" --action smoke --device "$adaspot_device"
    adaspot_active_stage="$adaspot_branch:train"
    "$adaspot_python" -u -m scripts.run_adaspot_gap "${adaspot_common[@]}" --branch "$adaspot_branch" --action train --device "$adaspot_device"
    adaspot_active_stage="$adaspot_branch:evaluate"
    "$adaspot_python" -u -m scripts.run_adaspot_gap "${adaspot_common[@]}" --branch "$adaspot_branch" --action evaluate --device "$adaspot_device"
done
adaspot_active_stage=PARSeq
"$adaspot_python" -u -m scripts.evaluate_adaspot_gap_ocr --mode approximate --word-root "$adaspot_word" --run-root "$adaspot_output" \
    --parseq-source "$adaspot_parseq/parseq-$adaspot_revision" --parseq-source-archive "$adaspot_parseq/parseq-$adaspot_revision.tar.gz" \
    --parseq-checkpoint "$adaspot_parseq/parseq-bb5792a6.pt" --device "$adaspot_device" --batch-size 8
"$adaspot_python" -c 'import json,sys; from pathlib import Path; from scripts.prune_segmentation_checkpoints import sha256_file; p=Path(sys.argv[1]); (p/"suite_success.json").write_text(json.dumps({"status":"SUCCESS","result_class":"approximate reproduction","ocr_comparison_sha256":sha256_file(p/"ocr_comparison.json")},indent=2)+"\n")' "$adaspot_output"
printf '[SUCCESS] Results: %s/comparison.md\n' "$adaspot_output"
