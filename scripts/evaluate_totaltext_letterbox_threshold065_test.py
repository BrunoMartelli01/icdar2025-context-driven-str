"""Frozen TRAIN-selected 0.65 threshold on Total-Text TEST; no parameter search."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.evaluate_totaltext_letterbox_pad0_test import run


CONFIG_PATH = Path(__file__).resolve().parents[1] / (
    "configs/approximate/totaltext_letterbox_threshold065_test_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("test-manifest", "data-root", "target-root", "training-runs",
                 "parseq-source", "parseq-source-archive", "parseq-checkpoint",
                 "threshold-selection-report", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--parseq-batch-size", type=int, default=32)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.parseq_batch_size < 1:
        parser.error("--parseq-batch-size must be positive")
    run(args, config_path=CONFIG_PATH, expected_threshold=0.65)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
