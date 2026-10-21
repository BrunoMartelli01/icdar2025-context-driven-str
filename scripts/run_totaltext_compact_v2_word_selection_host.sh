#!/usr/bin/env bash
# Fresh compact-v2 run with held-out metrics and at most four completed weights.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    printf 'Usage: bash scripts/run_totaltext_compact_v2_word_selection_host.sh /extra1/bmartelli\n' >&2
    exit 2
fi
host_root="${1%/}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$repo/.venvs/context-str-main-py311/bin/python"
output="$host_root/context-str-runs/totaltext-tighter-words-v2-word-selection-v1"
encoder="$host_root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"
mkdir -p "$output"
# Protect the preflight and training together against a duplicate launcher.
exec 9>"$output/training.lock"
flock --nonblock 9 || { printf 'This new compact-v2 run is already active.\n' >&2; exit 1; }
cd "$repo"
export PYTHONUNBUFFERED=1
command=("$python" -u -m scripts.train_totaltext_compact_v2_word_selection
         --mode approximate --device cuda:0 --data-root "$host_root"
         --encoder-checkpoint "$encoder" --output-root "$output")
"${command[@]}" --check --check-report "$output/gpu-preflight.json"
"${command[@]}" --check-report "$output/training-failure.json"
