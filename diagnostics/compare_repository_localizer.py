"""CPU-only saved-mask comparison; no model inference or GT-dependent selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from localize.mask_to_blocks import BlockParam, localize_mask
from localize.repository_contours import (
    SOURCE_COMMIT, SOURCE_SHA256, RepositoryContourParams, crop_repository_blocks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True, help="binary PNG in original image size")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    config = json.loads(args.config.read_text())
    if (config.get("reproduction_mode") != "approximate"
            or config.get("variant") != "repository-contours-v1"
            or config.get("source_commit") != SOURCE_COMMIT
            or config.get("source_sha256") != SOURCE_SHA256):
        raise ValueError("require the explicit approximate repository-contours-v1 config")
    params = RepositoryContourParams(**config["parameters"])
    with Image.open(args.mask) as source:
        mask = np.array(source)
    with Image.open(args.image) as source:
        image = np.array(source.convert("RGB"))
    upstream, crops = crop_repository_blocks(mask, image, params)
    baseline = localize_mask(mask, image, BlockParam(
        min_area=params.min_area, pad=params.pad, nmax=params.nmax,
        connectivity=8, tie_break="top_to_bottom_then_left_to_right",
    ))
    report = {
        "result_class": "approximate reproduction saved-mask diagnostic",
        "resolved_configuration": config,
        "inputs": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in (("mask", args.mask), ("image", args.image), ("config", args.config))},
        "repository": upstream,
        "repository_crop_shapes": [list(crop.shape) for crop in crops],
        "current_component_area": baseline.as_dict(),
        "same_ordered_boxes": ([item["padded_bbox"] for item in upstream["retained"]]
                               == [list(item.padded_bbox) for item in baseline.retained]),
        "recognition_accuracy": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(f"CURRENT {len(baseline.retained)} | REPOSITORY {len(crops)} crops", flush=True)
    print(f"JSON {args.output} SHA256 {hashlib.sha256(payload).hexdigest()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
