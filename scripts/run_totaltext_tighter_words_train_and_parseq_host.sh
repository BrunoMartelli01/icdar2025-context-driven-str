#!/usr/bin/env bash
# Isolated v2 compact-word experiment; production trainer/evaluator stay shared.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname "$(dirname "$REPO")")"
export SEPARATED_WORD_VARIANT=compact_v2
export SEPARATED_WORD_GT_ROOT="${COMPACT_WORD_GT_ROOT:-$REPO/artifacts/separated-word-gt-v2/compact-cores}"
export SEPARATED_WORD_CONFIG="$REPO/configs/approximate/segmentation_total_text_tighter_words_v2.yaml"
export SEPARATED_WORD_RUN_NAME=total-text-tighter-words-v2-500ep
export SEPARATED_WORD_RUN_ROOT="${COMPACT_WORD_RUN_ROOT:-$ROOT/context-str-runs/totaltext-tighter-words-v2-500ep}"
export SEPARATED_WORD_RESULTS_ROOT="${COMPACT_WORD_RESULTS_ROOT:-$ROOT/context-str-runs/totaltext-tighter-words-v2-findcontours-parseq-test}"
exec bash "$REPO/scripts/run_totaltext_separated_words_train_and_parseq_host.sh" "$@"
