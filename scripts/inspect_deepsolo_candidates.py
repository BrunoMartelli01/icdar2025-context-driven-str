#!/usr/bin/env python3
"""CPU-safe provenance/config identity check; does not qualify model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepsolo.candidates import CANDIDATES, inspect_candidate_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), action="append")
    args = parser.parse_args()
    selected = args.candidate or list(CANDIDATES)
    result = {
        "result_class": "APPROXIMATE candidate metadata inspection",
        "candidates": [
            inspect_candidate_source(args.source_root, artifact_id)
            for artifact_id in selected
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
