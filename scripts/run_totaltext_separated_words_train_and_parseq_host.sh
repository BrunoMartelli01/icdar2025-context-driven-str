#!/usr/bin/env bash
# Train new word masks, freeze TEST predictions/crops, then run PARSeq.
set -euo pipefail

CHECK_ONLY=0
RESUME=""
HOST_ROOT=""
while (( $# )); do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --resume)
      [[ $# -ge 2 ]] || { echo "--resume requires a checkpoint path" >&2; exit 2; }
      RESUME="$2"; shift 2 ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *)
      [[ -z "$HOST_ROOT" ]] || { echo "Only one host-root argument is allowed" >&2; exit 2; }
      HOST_ROOT="$1"; shift ;;
  esac
done
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${HOST_ROOT:-$(dirname "$(dirname "$REPO")")}"
PY="$REPO/.venvs/context-str-main-py311/bin/python"
VARIANT="${SEPARATED_WORD_VARIANT:-separated_v1}"
GT="${SEPARATED_WORD_GT_ROOT:-$REPO/artifacts/separated-word-gt-v1/v4-wider-gaps}"
BRIDGE="$GT/training"
CONFIG="${SEPARATED_WORD_CONFIG:-$REPO/configs/approximate/segmentation_total_text_separated_words_v1.yaml}"
DATA="$ROOT/datasets/context-str/scene-segmentation-v1"
RUN_ROOT="${SEPARATED_WORD_RUN_ROOT:-$ROOT/context-str-runs/totaltext-separated-words-v1-500ep}"
RUN_NAME="${SEPARATED_WORD_RUN_NAME:-total-text-separated-words-v1-500ep}"
RUN="$RUN_ROOT/$RUN_NAME"
RESULTS="${SEPARATED_WORD_RESULTS_ROOT:-$ROOT/context-str-runs/totaltext-separated-words-findcontours-parseq-test-v1}"
PARSEQ="$ROOT/context-str-artifacts/parseq"
REV=1902db043c029a7e03a3818c616c06600af574be

test -x "$PY" || { echo "Missing main Python: $PY" >&2; exit 1; }
test -f "$DATA/manifests/total_text_test.yaml" || { echo "Missing official TEST manifest" >&2; exit 1; }
test -d "$PARSEQ/parseq-$REV" || { echo "Missing qualified PARSeq source" >&2; exit 1; }
test -f "$PARSEQ/parseq-$REV.tar.gz" || { echo "Missing PARSeq source archive" >&2; exit 1; }
test -f "$PARSEQ/parseq-bb5792a6.pt" || { echo "Missing PARSeq checkpoint" >&2; exit 1; }

cd "$REPO"
export PYTHONUNBUFFERED=1
"$PY" -m scripts.prepare_separated_word_training \
  --mode approximate --variant "$VARIANT" --gt-root "$GT" --data-root "$ROOT" --output-root "$BRIDGE" \
  --base-config "$REPO/configs/approximate/segmentation_total_text_curved_word_boxes_v4.yaml" \
  --config-output "$CONFIG" --test-manifest "$DATA/manifests/total_text_test.yaml"
"$PY" - "$PARSEQ/parseq-bb5792a6.pt" "$PARSEQ/parseq-$REV.tar.gz" <<'PY'
from pathlib import Path
import sys
from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
from segmentation.data import file_sha256
for name, expected in zip(sys.argv[1:], [PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256]):
    if file_sha256(Path(name)) != expected:
        raise ValueError(f"PARSeq artifact checksum mismatch: {name}")
print('[preflight] Qualified PARSeq artifact hashes verified', flush=True)
PY
if (( CHECK_ONLY )); then
  echo "[preflight] Passed; GPU training and TEST inference have not started"
  exit 0
fi

if [[ ! -f "$RUN/run_manifest.json" || -n "$RESUME" ]]; then
  COMMAND=("$PY" -u -m segmentation.train --config "$CONFIG" --data-root "$ROOT"
           --manifest-root "$BRIDGE" --output-root "$RUN_ROOT" --device cuda)
  if [[ -n "$RESUME" ]]; then
    COMMAND+=(--resume "$RESUME")
  fi
  "${COMMAND[@]}"
fi
RUN_MANIFEST="$RUN/run_manifest.json"
CHECKPOINT="$("$PY" - "$RUN_MANIFEST" <<'PY'
import json,sys
run=json.load(open(sys.argv[1]))
if run.get('status') != 'SUCCESS' or run.get('success') is not True or len(run.get('epochs', [])) != 500:
    raise ValueError('Expected successful complete 500-epoch training; resume incomplete runs explicitly')
print(run['checkpoint_output_location'])
PY
)"
test -f "$CHECKPOINT" || { echo "Selected checkpoint missing: $CHECKPOINT" >&2; exit 1; }

"$PY" -u -m scripts.evaluate_totaltext_v1_findcontours_parseq \
  --test-manifest "$DATA/manifests/total_text_test.yaml" --data-root "$DATA" \
  --segmentation-config "$CONFIG" \
  --localization-config "$REPO/configs/approximate/localization_repository_contours_native128_v1.json" \
  --training-run-manifest "$RUN_MANIFEST" --checkpoint "$CHECKPOINT" \
  --parseq-source "$PARSEQ/parseq-$REV" --parseq-source-archive "$PARSEQ/parseq-$REV.tar.gz" \
  --parseq-checkpoint "$PARSEQ/parseq-bb5792a6.pt" --device cuda:0 \
  --parseq-batch-size 32 --output-dir "$RESULTS"

"$PY" - "$RESULTS/results.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))['summary']
r=s['recognition']; g=s['geometry']
print(f"Accuracy end-to-end (casefold, IoU>=0.5): {100*r['exact_casefold_rate_all_gt']:.2f}%")
print(f"OCR accuracy on available word matches: {100*r['exact_casefold_rate_geometry_available']:.2f}%")
print(f"GT words: {g['gt_instances']}; crop: {g['retained_proposals']}")
print('Report:',sys.argv[1])
PY
