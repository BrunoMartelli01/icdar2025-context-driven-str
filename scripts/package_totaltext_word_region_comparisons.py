"""Lossless mask pairs plus compact per-image visual comparisons in one ZIP."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import yaml

from segmentation.scene_text_preparation import file_sha256


PANEL_WIDTH = 384
PANEL_HEIGHT = 384
HEADER_HEIGHT = 28


def _panel(source: Image.Image, title: str) -> Image.Image:
    panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT + HEADER_HEIGHT), (25, 25, 25))
    fitted = ImageOps.contain(source.convert("RGB"), (PANEL_WIDTH, PANEL_HEIGHT))
    panel.paste(fitted, ((PANEL_WIDTH - fitted.width) // 2, HEADER_HEIGHT + (PANEL_HEIGHT - fitted.height) // 2))
    ImageDraw.Draw(panel).text((8, 8), title, fill=(255, 255, 255))
    return panel


def _preview(image_path: Path, old_path: Path, new_path: Path, sample_id: str) -> bytes:
    with Image.open(image_path) as opened:
        rgb = opened.convert("RGB")
    with Image.open(old_path) as opened:
        old = opened.convert("L")
    with Image.open(new_path) as opened:
        new = opened.convert("L")
    if rgb.size != old.size or old.size != new.size:
        raise ValueError(f"Image/mask geometry mismatch: {sample_id}")
    original = np.asarray(old) > 0
    curved = np.asarray(new) > 0
    difference = np.zeros((*original.shape, 3), dtype=np.uint8)
    difference[original & curved] = (255, 255, 255)
    difference[original & ~curved] = (255, 70, 70)
    difference[~original & curved] = (60, 230, 240)
    diff_image = Image.fromarray(difference, mode="RGB")
    titles = (f"{sample_id} RGB", "ORIGINAL pixel GT", "NEW curved-word GT", "DIFF")
    pictures = (rgb, old, new, diff_image)
    canvas = Image.new("RGB", (4 * PANEL_WIDTH, PANEL_HEIGHT + HEADER_HEIGHT))
    for index, (picture, title) in enumerate(zip(pictures, titles)):
        canvas.paste(_panel(picture, title), (index * PANEL_WIDTH, 0))
    stream = BytesIO()
    canvas.save(stream, format="JPEG", quality=85, subsampling=0)
    return stream.getvalue()


def package(source_root: Path, new_root: Path, output: Path, *, worst_count: int) -> tuple[int, str]:
    source_root, new_root, output = source_root.resolve(), new_root.resolve(), output.resolve()
    old_manifest_path = source_root / "manifests/total_text_train.yaml"
    new_manifest_path = new_root / "manifests/total_text_train.yaml"
    old_manifest = yaml.safe_load(old_manifest_path.read_text(encoding="utf-8"))
    new_manifest = yaml.safe_load(new_manifest_path.read_text(encoding="utf-8"))
    old_records = {item["id"]: item for item in old_manifest["samples"]}
    new_records = {item["id"]: item for item in new_manifest["samples"]}
    if old_records.keys() != new_records.keys() or len(new_records) != 1255:
        raise ValueError("Original/new TRAIN manifests do not share exactly 1255 sample IDs")
    if not 1 <= worst_count <= len(new_records):
        raise ValueError("--worst must be between 1 and 1255")
    qualification = json.loads((new_root / "qualification.json").read_text(encoding="utf-8"))
    if qualification["images"] != len(new_records):
        raise ValueError("Target qualification count differs from manifest")
    scores = []
    for sample_id, new in new_records.items():
        old = old_records[sample_id]
        with Image.open(source_root / old["mask"]) as opened:
            old_mask = np.asarray(opened.convert("L")) > 0
        with Image.open(new_root / new["mask"]) as opened:
            new_mask = np.asarray(opened.convert("L")) > 0
        with Image.open(new_root / new["validity_annotation"]) as opened:
            valid = np.asarray(opened.convert("L")) > 0
        source_fg = old_mask & valid
        coverage = float((source_fg & new_mask).sum() / source_fg.sum()) if source_fg.any() else 1.0
        retained = float(qualification["per_image"][sample_id]["retained_polygon_fraction"])
        scores.append((sample_id, coverage, retained))
    coverage_order = sorted(scores, key=lambda row: (row[1], row[0]))
    retained_order = sorted(scores, key=lambda row: (row[2], row[0]))
    chosen: dict[str, tuple[str, float, float]] = {}
    for row in coverage_order[: worst_count // 2] + retained_order[: worst_count - worst_count // 2]:
        chosen.setdefault(row[0], row)
    for row in coverage_order:
        if len(chosen) >= worst_count:
            break
        chosen.setdefault(row[0], row)
    selected = sorted(chosen.values(), key=lambda row: (row[1], row[2], row[0]))
    print(f"[comparison ZIP] selected {len(selected)} worst TRAIN images from 1255", flush=True)
    if output.exists():
        raise FileExistsError(f"Comparison ZIP already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    rows = []
    try:
        with ZipFile(partial, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(
                "README.txt",
                f"Total-Text TRAIN: {len(selected)} worst cases selected from 1255.\n"
                "Selection: lower original-pixel-foreground coverage and lower "
                "retained word-polygon area after disjointness; details in manifest.json.\n"
                "masks_original/ and masks_curved/ contain exact lossless PNGs; "
                "previews/ are visual JPEGs only.\n"
                "DIFF: white=both foreground, red=original only, cyan=new only.\n"
                "separators/ show valid background pixels emphasized in the fusion-penalty run.\n"
                "PRACTICAL_APPROXIMATION; no model predictions or TEST images are included.\n",
            )
            for index, (sample_id, coverage, retained) in enumerate(selected, 1):
                old, new = old_records[sample_id], new_records[sample_id]
                if old["image_sha256"] != new["image_sha256"]:
                    raise ValueError(f"Source RGB bytes differ for {sample_id}")
                image_path = new_root / new["image"]
                old_path = source_root / old["mask"]
                new_path = new_root / new["mask"]
                separator_path = new_root / new["separation_annotation"]
                for path, expected in (
                    (image_path, new["image_sha256"]),
                    (old_path, old["mask_sha256"]),
                    (new_path, new["mask_sha256"]),
                    (separator_path, new["separation_annotation_sha256"]),
                ):
                    if file_sha256(path) != expected:
                        raise ValueError(f"Input checksum mismatch: {path}")
                stem = sample_id.split(":", 1)[1]
                archive.write(old_path, f"masks_original/{stem}.png")
                archive.write(new_path, f"masks_curved/{stem}.png")
                archive.write(separator_path, f"separators/{stem}.png")
                archive.writestr(
                    f"previews/{stem}.jpg",
                    _preview(image_path, old_path, new_path, sample_id),
                )
                rows.append({
                    "sample_id": sample_id,
                    "image_sha256": new["image_sha256"],
                    "original_mask_sha256": old["mask_sha256"],
                    "curved_word_mask_sha256": new["mask_sha256"],
                    "separator_sha256": new["separation_annotation_sha256"],
                    "source_pixel_foreground_coverage": coverage,
                    "retained_polygon_fraction": retained,
                })
                if index % 10 == 0 or index == len(selected):
                    print(f"[comparison ZIP] {index}/{len(selected)}", flush=True)
            archive.writestr(
                "manifest.json",
                json.dumps({
                    "classification": "PRACTICAL_APPROXIMATION",
                    "split": "Total-Text TRAIN",
                    "total_train_images": len(new_records),
                    "selected_images": len(selected),
                    "selection_rule": "half-lowest-source-pixel-foreground-coverage; half-lowest-retained-polygon-fraction; fill duplicates from lowest coverage",
                    "source_manifest_sha256": file_sha256(old_manifest_path),
                    "new_manifest_sha256": file_sha256(new_manifest_path),
                    "samples": rows,
                }, indent=2, sort_keys=True) + "\n",
            )
        with ZipFile(partial) as archive:
            if archive.testzip() is not None:
                raise ValueError("Comparison ZIP integrity check failed")
            if len(archive.namelist()) != 2 + 4 * len(selected):
                raise ValueError("Comparison ZIP member count mismatch")
        partial.replace(output)
    except Exception:
        # Keep the .part file for recovery/inspection; never expose it as the
        # completed requested ZIP.
        raise
    digest = file_sha256(output)
    print("ZIP", output, "SHA256", digest, "BYTES", output.stat().st_size)
    return len(rows), digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worst", type=int, default=50)
    args = parser.parse_args()
    package(args.source_root, args.new_root, args.output, worst_count=args.worst)


if __name__ == "__main__":
    main()
