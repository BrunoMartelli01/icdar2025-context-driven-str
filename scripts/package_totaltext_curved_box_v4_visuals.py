"""One composite JPEG per TRAIN image: RGB, GT, v1, rejected v3, curved v4."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image
import numpy as np
import yaml

from scripts.package_totaltext_adaptive_boxes_visuals import ExitStackImages, _panel
from segmentation.scene_text_preparation import file_sha256


LABELS = ("RGB TRAIN", "ORIGINAL PIXEL GT", "FILLED V1", "REJECTED STRIP V3", "CURVED FILLED V4")


def package(
    base_root: Path, strip_root: Path, pixel_root: Path, curve_root: Path,
    output_zip: Path, count: int,
) -> None:
    qualification = json.loads((curve_root / "qualification.json").read_text())
    if qualification.get("images") != 1255:
        raise ValueError("v4 qualification must cover the full official TRAIN split")
    base = yaml.safe_load((base_root / "manifests/total_text_train.yaml").read_text())
    pixel = yaml.safe_load((pixel_root / "manifests/total_text_train.yaml").read_text())
    source = {row["id"]: row for row in pixel["samples"]}
    def changed_pixels(row: dict) -> int:
        with Image.open(base_root / row["mask"]) as before:
            old_mask = np.asarray(before.convert("L")) > 0
        with Image.open(curve_root / row["mask"]) as after:
            new_mask = np.asarray(after.convert("L")) > 0
        return int(np.count_nonzero(old_mask ^ new_mask))

    ranked = sorted(base["samples"], key=changed_pixels, reverse=True)
    selected = ranked[:count]
    for sample_id in ("total_text:147", "total_text:1240", "total_text:1378"):
        special = next((row for row in base["samples"] if row["id"] == sample_id), None)
        if special is not None and all(row["id"] != sample_id for row in selected):
            selected.append(special)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(
            "README.txt",
            "TRAIN-only visual diagnostic; one composite per image. "
            "The rejected strip v3 is shown for comparison, not for training v4.\n",
        )
        for offset, row in enumerate(selected, 1):
            other = source[row["id"]]
            if file_sha256(pixel_root / other["mask"]) != other["mask_sha256"]:
                raise ValueError(f"Pixel GT checksum mismatch: {row['id']}")
            paths = (
                base_root / row["image"], pixel_root / other["mask"],
                base_root / row["mask"], strip_root / row["mask"],
                curve_root / row["mask"],
            )
            with ExitStackImages(paths) as images:
                panels = [_panel(image, name, (430, 330)) for image, name in zip(images, LABELS)]
            board = Image.new("RGB", (430 * 5, 360), "#242424")
            for column, panel in enumerate(panels):
                board.paste(panel, (column * 430, 0))
            stream = BytesIO()
            board.save(stream, format="JPEG", quality=92, optimize=True)
            archive.writestr(f"{row['id'].split(':', 1)[1]}.jpg", stream.getvalue())
            print(f"[curved box visual] {offset}/{len(selected)} {row['id']}", flush=True)
    print("ZIP", output_zip, file_sha256(output_zip), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--strip-root", type=Path, required=True)
    parser.add_argument("--pixel-root", type=Path, required=True)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    package(args.base_root, args.strip_root, args.pixel_root, args.curve_root, args.output_zip, args.count)


if __name__ == "__main__":
    main()
