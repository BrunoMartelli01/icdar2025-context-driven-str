"""Package the largest TRAIN v1-to-refined changes for visual target inspection."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
import yaml

from scripts.package_totaltext_word_region_comparisons import _panel, PANEL_HEIGHT, PANEL_WIDTH, HEADER_HEIGHT
from segmentation.scene_text_preparation import file_sha256


def package(base_root: Path, refined_root: Path, pixel_root: Path, output: Path, count: int) -> str:
    if not 1 <= count <= 1255:
        raise ValueError("count must be between 1 and 1255")
    base_path = base_root / "manifests/total_text_train.yaml"
    refined_path = refined_root / "manifests/total_text_train.yaml"
    pixel_path = pixel_root / "manifests/total_text_train.yaml"
    base = {r["id"]: r for r in yaml.safe_load(base_path.read_text())["samples"]}
    refined = {r["id"]: r for r in yaml.safe_load(refined_path.read_text())["samples"]}
    pixel = {r["id"]: r for r in yaml.safe_load(pixel_path.read_text())["samples"]}
    if len(base) != len(refined) or len(base) != len(pixel) or len(base) != 1255:
        raise ValueError("TRAIN manifests must contain the same 1255 samples")
    audit = json.loads((refined_root / "qualification.json").read_text())["per_image"]
    selected = sorted(
        base,
        key=lambda key: (
            audit[key]["refined_owner_pixels"] / audit[key]["original_owner_pixels"], key
        ),
    )[:count]
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing ZIP: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    rows = []
    with ZipFile(partial, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(
            "README.txt",
            "Total-Text TRAIN: largest relative v1-to-refined foreground contraction.\n"
            "RGB | official pixel GT | filled word-v1 | refined word-v2 | v1-only pixels.\n"
            "Exact masks are in separate lossless PNGs; previews are JPEG visual aids.\n"
            "PRACTICAL_APPROXIMATION; no model prediction or TEST image.\n",
        )
        for offset, sample_id in enumerate(selected, 1):
            before, after, source = base[sample_id], refined[sample_id], pixel[sample_id]
            if not (before["image_sha256"] == after["image_sha256"] == source["image_sha256"]):
                raise ValueError(f"RGB provenance mismatch: {sample_id}")
            inputs = (
                (refined_root, after, "image"),
                (pixel_root, source, "mask"),
                (base_root, before, "mask"),
                (refined_root, after, "mask"),
            )
            pictures = []
            for root, record, key in inputs:
                path = root / record[key]
                if file_sha256(path) != record[f"{key}_sha256"]:
                    raise ValueError(f"Checksum mismatch: {path}")
                with Image.open(path) as opened:
                    pictures.append(opened.convert("RGB" if key == "image" else "L"))
            if len({picture.size for picture in pictures}) != 1:
                raise ValueError(f"Mask geometry mismatch: {sample_id}")
            before_mask = np.asarray(pictures[2]) > 0
            after_mask = np.asarray(pictures[3]) > 0
            if np.any(after_mask & ~before_mask):
                raise ValueError(f"Refined target escapes v1: {sample_id}")
            removed = Image.fromarray(((before_mask & ~after_mask) * 255).astype(np.uint8), mode="L")
            titles = ("RGB", "OFFICIAL pixel GT", "WORD v1", "REFINED v2", "REMOVED from v1")
            canvas = Image.new("RGB", (5 * PANEL_WIDTH, PANEL_HEIGHT + HEADER_HEIGHT))
            for index, (picture, title) in enumerate(zip((*pictures, removed), titles)):
                canvas.paste(_panel(picture, f"{sample_id}: {title}"), (index * PANEL_WIDTH, 0))
            stream = BytesIO()
            canvas.save(stream, format="JPEG", quality=86, subsampling=0)
            stem = sample_id.split(":", 1)[1]
            archive.writestr(f"previews/{stem}.jpg", stream.getvalue())
            for name, (root, record, key) in zip(
                ("source_pixel", "word_v1", "refined_v2"), inputs[1:]
            ):
                archive.write(root / record[key], f"masks_{name}/{stem}.png")
            rows.append({
                "sample_id": sample_id,
                "refined_area_fraction_of_v1": audit[sample_id]["refined_owner_pixels"]
                / audit[sample_id]["original_owner_pixels"],
                "v1_mask_sha256": before["mask_sha256"],
                "refined_mask_sha256": after["mask_sha256"],
                "pixel_mask_sha256": source["mask_sha256"],
            })
            if offset % 10 == 0 or offset == count:
                print(f"[refinement visual ZIP] {offset}/{count}", flush=True)
        archive.writestr("manifest.json", json.dumps({
            "classification": "PRACTICAL_APPROXIMATION",
            "split": "Total-Text TRAIN",
            "selection": "lowest refined_owner_pixels / v1_owner_pixels",
            "base_manifest_sha256": file_sha256(base_path),
            "refined_manifest_sha256": file_sha256(refined_path),
            "pixel_manifest_sha256": file_sha256(pixel_path),
            "samples": rows,
        }, sort_keys=True, indent=2) + "\n")
    with ZipFile(partial) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != 2 + 4 * count:
            raise ValueError("Comparison ZIP failed integrity verification")
    partial.replace(output)
    digest = file_sha256(output)
    print("ZIP", output, "SHA256", digest, "BYTES", output.stat().st_size)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--refined-root", type=Path, required=True)
    parser.add_argument("--pixel-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    package(args.base_root, args.refined_root, args.pixel_root, args.output, args.count)


if __name__ == "__main__":
    main()
