#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/extra1/bmartelli/context-str-runs"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venvs/context-str-main-py311/bin/python"
DUPLICATE_RUN="$RUN_ROOT/ap-full-20260901-160732"
ACTIVE_RUN="$RUN_ROOT/ap-full-20260901-174246"
OLD_FULL="$RUN_ROOT/author-protocol-full"
QUALIFICATION="$RUN_ROOT/author-protocol-qualification"

for required in \
  "$PYTHON_BIN" \
  "$DUPLICATE_RUN/icdar13-author-protocol-approximate/run_manifest.json" \
  "$ACTIVE_RUN/icdar13-author-protocol-approximate/run_manifest.json" \
  "$OLD_FULL/icdar13-author-protocol-approximate/checkpoints/epoch-0098.pt" \
  "$QUALIFICATION/icdar13-author-protocol-approximate/checkpoints/epoch-0099.pt"
do
  test -e "$required" || { echo "Missing required path: $required" >&2; exit 1; }
done

"$PYTHON_BIN" - \
  "$DUPLICATE_RUN/icdar13-author-protocol-approximate/run_manifest.json" \
  "$ACTIVE_RUN/icdar13-author-protocol-approximate/run_manifest.json" <<'PY'
import json
import sys

left = json.load(open(sys.argv[1], encoding="utf-8"))
right = json.load(open(sys.argv[2], encoding="utf-8"))
left_hashes = [item["checkpoint_sha256"] for item in left["epochs"]]
right_hashes = [item["checkpoint_sha256"] for item in right["epochs"]]
if len(left_hashes) != 100 or left_hashes != right_hashes:
    raise SystemExit("Refusing cleanup: the two full runs are not exact 100-checkpoint duplicates")
if left["checkpoint"]["sha256"] != right["checkpoint"]["sha256"]:
    raise SystemExit("Refusing cleanup: selected checkpoint identities differ")
print("Duplicate verification PASS: 100/100 checkpoint SHA256 values match")
PY

echo "Before cleanup:"
du -sh "$DUPLICATE_RUN" "$OLD_FULL" "$QUALIFICATION"

find "$DUPLICATE_RUN" -depth -delete

find "$OLD_FULL/icdar13-author-protocol-approximate/checkpoints" \
  -type f ! -name 'epoch-0098.pt' ! -name 'epoch-0098.pt.sha256' -delete

find "$QUALIFICATION/icdar13-author-protocol-approximate/checkpoints" \
  -type f ! -name 'epoch-0099.pt' ! -name 'epoch-0099.pt.sha256' -delete

echo "Cleanup complete. Preserved old selected checkpoints and all metadata/logs."
du -sh "$OLD_FULL" "$QUALIFICATION" "$ACTIVE_RUN"
df -h "$RUN_ROOT"
