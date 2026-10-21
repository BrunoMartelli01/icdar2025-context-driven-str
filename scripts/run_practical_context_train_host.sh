#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 ARTIFACT_ROOT ICDAR13_ROOT WORK_ROOT SEGMENTATION_CHECKPOINT" >&2
  exit 2
fi

ARTIFACT_ROOT=$1
DATA_ROOT=$2
WORK_ROOT=$3
CHECKPOINT=$4
mkdir -p "$WORK_ROOT"
WORK_ROOT=$(cd "$WORK_ROOT" && pwd -P)
case "$WORK_ROOT" in
  /tmp|/tmp/*)
    echo "WORK_ROOT must be persistent shared storage, not /tmp: $WORK_ROOT" >&2
    exit 2
    ;;
esac
PY=.venvs/context-str-main-py311/bin/python
CAPTION_DIR="$WORK_ROOT/icdar13-train-blip2-captions-practical-antirepetition-v1"
V1_INPUT="$WORK_ROOT/train-practical-remap-v1-parseq-compact-v1"
REPORT="$WORK_ROOT/v1-local-context-train-antirepetition-v1.json"
PARSEQ_REVISION=1902db043c029a7e03a3818c616c06600af574be

[[ -x "$PY" ]] || { echo "missing Python environment: $PY" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "missing checkpoint file: $CHECKPOINT" >&2; exit 1; }

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="$ARTIFACT_ROOT/huggingface"
export HF_HOME="$WORK_ROOT/huggingface-home"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME"

"$PY" -u -m scripts.export_train_blip2_captions \
  --train-manifest manifests/icdar13_segmentation_author_protocol.yaml \
  --data-root "$DATA_ROOT" \
  --hf-cache "$HF_HUB_CACHE" \
  --device cuda:0 \
  --generation-profile practical-antirepetition-v1 \
  --output-dir "$CAPTION_DIR"

"$PY" -u -m scripts.export_train_v1_parseq_inputs \
  --train-manifest manifests/icdar13_segmentation_author_protocol.yaml \
  --data-root "$DATA_ROOT" \
  --evaluation-config configs/approximate/segmentation_icdar13_author_protocol_test.yaml \
  --localizer-config configs/approximate/pipeline_icdar13_practical_remap_v1.yaml \
  --checkpoint "$CHECKPOINT" \
  --parseq-source "$ARTIFACT_ROOT/parseq/parseq-$PARSEQ_REVISION" \
  --parseq-source-archive "$ARTIFACT_ROOT/parseq/parseq-$PARSEQ_REVISION.tar.gz" \
  --parseq-checkpoint "$ARTIFACT_ROOT/parseq/parseq-bb5792a6.pt" \
  --device cuda:0 \
  --output-dir "$V1_INPUT"

CUDA_VISIBLE_DEVICES="" "$PY" -u -m scripts.evaluate_v1_local_context_train \
  --train-manifest manifests/icdar13_segmentation_author_protocol.yaml \
  --data-root "$DATA_ROOT" \
  --input-manifest "$V1_INPUT/manifest.json" \
  --caption-manifest "$CAPTION_DIR/manifest.json" \
  --hf-cache "$HF_HUB_CACHE" \
  --device cpu \
  --output "$REPORT"
