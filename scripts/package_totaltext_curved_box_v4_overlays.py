"""Package TRAIN-only pixel-GT/v4 overlays as one composite per image.

The selected examples are strong *visual changes*, not accuracy winners. Pixel
ground truth is shown only for visual audit and never used to select a model.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageDraw
import yaml

from scripts.package_totaltext_adaptive_boxes_visuals import _panel
from segmentation.scene_text_preparation import file_sha256


def overlap_panel(original: np.ndarray, curved: np.ndarray) -> Image.Image:
    """White=both, red=original only, cyan=v4 only, black=neither."""

    if original.shape != curved.shape or original.ndim != 2:
        raise ValueError("Original/v4 masks must have matching 2D geometry")
    result = np.zeros((*original.shape, 3), dtype=np.uint8)
    old, new = original > 0, curved > 0
    result[old & new] = (255, 255, 255)
    result[old & ~new] = (255, 80, 80)
    result[~old & new] = (0, 210, 230)
    return Image.fromarray(result, mode="RGB")


def _selected_rows(rows: list[dict], audit: dict, count: int) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")
    eligible = []
    for row in rows:
        item = audit[row["id"]]
        if (
            item["changed_words"] < 2
            or item["geometric_fallbacks"]
            or item["coverage_or_topology_fallbacks"]
            or item["full_image_fallback"]
            or item["max_uniform_buffer_pixels"]
            or item["retained_v1_source_seed_pixels"] != item["v1_source_seed_pixels"]
        ):
            continue
        delta = abs(item["final_foreground_pixels"] - item["base_foreground_pixels"])
        strength = item["changed_words"] * delta / max(1, item["base_foreground_pixels"])
        eligible.append((strength, item["changed_words"], row["id"], row))
    ranked = [item[-1] for item in sorted(eligible, key=lambda item: (-item[0], -item[1], item[2]))]
    selected = ranked[:count]
    example = next((row for row in rows if row["id"] == "total_text:147"), None)
    if example is not None and not any(row["id"] == example["id"] for row in selected):
        selected.append(example)
    return selected


def package(base_root: Path, pixel_root: Path, curved_root: Path, output_zip: Path, count: int) -> dict:
    base = yaml.safe_load((base_root / "manifests/total_text_train.yaml").read_text(encoding="utf-8"))
    pixels = yaml.safe_load((pixel_root / "manifests/total_text_train.yaml").read_text(encoding="utf-8"))
    curved = yaml.safe_load((curved_root / "manifests/total_text_train.yaml").read_text(encoding="utf-8"))
    qualification = json.loads((curved_root / "qualification.json").read_text(encoding="utf-8"))
    if any(len(manifest["samples"]) != 1255 for manifest in (base, pixels, curved)):
        raise ValueError("Expected 1255 Total-Text TRAIN images in every manifest")
    pixel_rows = {row["id"]: row for row in pixels["samples"]}
    curved_rows = {row["id"]: row for row in curved["samples"]}
    if set(pixel_rows) != set(curved_rows) or set(pixel_rows) != {row["id"] for row in base["samples"]}:
        raise ValueError("TRAIN manifests do not align by sample ID")
    visually_changed = _selected_rows(
        base["samples"], qualification["per_image"], len(base["samples"])
    )
    selected = []
    for row in visually_changed:
        pixel_path = pixel_root / pixel_rows[row["id"]]["mask"]
        curved_path = curved_root / curved_rows[row["id"]]["mask"]
        with Image.open(pixel_path) as old_image, Image.open(curved_path) as new_image:
            old = np.asarray(old_image.convert("L")) > 0
            new = np.asarray(new_image.convert("L")) > 0
        if old.shape != new.shape:
            raise ValueError(f"Mask dimensions differ: {row['id']}")
        coverage = np.count_nonzero(old & new) / max(1, np.count_nonzero(old))
        if coverage >= 0.98:
            selected.append(row)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} high-coverage visual examples found")
    example = next((row for row in base["samples"] if row["id"] == "total_text:147"), None)
    if example is not None and not any(row["id"] == example["id"] for row in selected):
        selected.append(example)
    report = {
        "classification": "PRACTICAL_APPROXIMATION",
        "result_class": "TRAIN-only visual target audit; not model accuracy",
        "selection": "largest changed_words times relative foreground-area change, restricted to no-fallback/no-buffer cases with >=0.98 original-pixel-GT coverage; sample 147 also included",
        "legend": {"white": "pixel GT and v4", "red": "pixel GT only", "cyan": "v4 only", "black": "neither"},
        "samples": [],
    }
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("README.txt", "TRAIN target visual audit only; one composite JPG per image.\nWhite=original pixel GT AND v4; red=GT only; cyan=v4 only; black=neither.\nSelected for strong visible target changes, NOT model performance.\n")
        for index, row in enumerate(selected, 1):
            pixel, newer = pixel_rows[row["id"]], curved_rows[row["id"]]
            paths = (base_root / row["image"], pixel_root / pixel["mask"], curved_root / newer["mask"])
            for path, expected in zip(paths, (row["image_sha256"], pixel["mask_sha256"], newer["mask_sha256"])):
                if file_sha256(path) != expected:
                    raise ValueError(f"Input checksum mismatch: {path}")
            with Image.open(paths[0]) as image, Image.open(paths[1]) as old_image, Image.open(paths[2]) as new_image:
                rgb = image.convert("RGB")
                original = np.asarray(old_image.convert("L")) > 0
                new = np.asarray(new_image.convert("L")) > 0
                if original.shape != new.shape or original.shape != (rgb.height, rgb.width):
                    raise ValueError(f"Mask/RGB dimensions differ: {row['id']}")
                overlap = overlap_panel(original, new)
                parts = (
                    _panel(rgb, "ORIGINAL RGB", (430, 330)),
                    _panel(old_image, "ORIGINAL PIXEL GT", (430, 330)),
                    _panel(new_image, "FILLED CURVED V4", (430, 330)),
                    _panel(overlap, "OVERLAP: WHITE BOTH / RED GT / CYAN V4", (430, 330)),
                )
                board = Image.new("RGB", (1720, 380), "#242424")
                for column, panel in enumerate(parts):
                    board.paste(panel, (430 * column, 0))
                ImageDraw.Draw(board).text((10, 365), row["id"], fill="white")
                stream = BytesIO()
                board.save(stream, format="JPEG", quality=94, optimize=True)
                image_name = f"{row['id'].split(':', 1)[1]}.jpg"
                archive.writestr(image_name, stream.getvalue())
                report["samples"].append({
                    "sample_id": row["id"], "image": image_name,
                    "changed_words": qualification["per_image"][row["id"]]["changed_words"],
                    "original_pixel_gt_pixels": int(np.count_nonzero(original)),
                    "v4_pixels": int(np.count_nonzero(new)),
                    "intersection_pixels": int(np.count_nonzero(original & new)),
                    "pixel_gt_coverage_by_v4": float(np.count_nonzero(original & new) / max(1, np.count_nonzero(original))),
                })
            print(f"[v4 overlay] {index}/{len(selected)} {row['id']}", flush=True)
        archive.writestr("selection.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("ZIP", output_zip, "SHA256", file_sha256(output_zip), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--pixel-root", type=Path, required=True)
    parser.add_argument("--curved-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()
    package(args.base_root, args.pixel_root, args.curved_root, args.output_zip, args.count)


if __name__ == "__main__":
    main()
