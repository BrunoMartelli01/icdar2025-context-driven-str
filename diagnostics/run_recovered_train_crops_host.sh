#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="${1:?Usage: bash diagnostics/run_recovered_train_crops_host.sh USER_DATA_ROOT}"
A="$ROOT/context-str-artifacts/parseq"
REV=1902db043c029a7e03a3818c616c06600af574be
REPORT=artifacts/diagnostics/train-crop-completeness-v1/results.json
OUT=artifacts/diagnostics/train-recovered-crops-parseq-v1
PY=.venvs/context-str-main-py311/bin/python
export PYTHONUNBUFFERED=1
"$PY" -u -m diagnostics.recognize_recovered_train_crops \
  --report "$REPORT" \
  --expected-report-sha256 f49d244e1e6c439bb9ec72e7fdb004ef2029bfe5ee405877f7d5f7e32a269cf9 \
  --data-root "$ROOT/datasets/context-str/icdar13" \
  --parseq-source "$A/parseq-$REV" \
  --parseq-source-archive "$A/parseq-$REV.tar.gz" \
  --parseq-checkpoint "$A/parseq-bb5792a6.pt" \
  --device cuda:0 --output-dir "$OUT"
