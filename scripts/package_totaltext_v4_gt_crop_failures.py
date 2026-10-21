"""Package GT-v4 oracle crops where both PARSeq crop shapes are wrong.

One PNG per word shows original RGB/GT box, native GT mask, the exact tight
rectangle and curved cutout passed to PARSeq, both outputs, and the GT text.
This is a read-only TEST visual audit; it does not select scientific settings.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from localize.curved_component_crop import native_component_curved_crops
from localize.repository_contours import RepositoryContourParams
from recognition.scene_text_v1_evaluation import box_iou
from segmentation.data import file_sha256, load_segmentation_manifest


EXPECTED_REPORT_SHA256 = "272613b094777b9836a1ffde2fd9265fcec17c84e7ccb60661f9a87ac0e29989"
EXPECTED_TEST_MANIFEST_SHA256 = "da2f87e7cae765f426af39c9641535138d6657d2e9332e6fdf9e80f2ba6b69bd"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-side", type=int, default=6)
    return parser.parse_args()


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _sha_rgb(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _select(records: list[dict], per_side: int) -> list[dict]:
    if per_side < 1:
        raise ValueError("--per-side must be positive")
    selected, seen = [], set()
    for ordered in (
        sorted(records, key=lambda row: (-row["iou"], row["sample_id"], row["rank"])),
        sorted(records, key=lambda row: (row["iou"], row["sample_id"], row["rank"])),
    ):
        count = 0
        for row in ordered:
            if row["sample_id"] in seen:
                continue
            selected.append(row)
            seen.add(row["sample_id"])
            count += 1
            if count == per_side:
                break
    if len(selected) != 2 * per_side:
        raise ValueError("Not enough distinct-image failures for the requested visual sample")
    return selected


def _panel_crop(canvas: Image.Image, crop: np.ndarray, x: int, y: int) -> None:
    image = Image.fromarray(crop, mode="RGB")
    image.thumbnail((550, 190), Image.Resampling.LANCZOS)
    canvas.paste(image, (x + (550 - image.width) // 2, y + (190 - image.height) // 2))


def _render(row: dict, original: np.ndarray, native: np.ndarray, rect: np.ndarray, curved: np.ndarray) -> Image.Image:
    canvas = Image.new("RGB", (1200, 910), "white")
    draw = ImageDraw.Draw(canvas)
    title, body = _font(22), _font(18)
    draw.text((20, 15), f"GT-v4 oracle | {row['sample_id']} | rank {row['rank']} | IoU {row['iou']:.3f}", fill="black", font=title)
    draw.text((20, 50), f"GT: {row['gt_text']}", fill=(160, 0, 0), font=body)
    draw.text((20, 78), "Red: GT word box     Blue: component crop box", fill="black", font=body)
    original_image = Image.fromarray(original, mode="RGB")
    original_image.thumbnail((870, 490), Image.Resampling.LANCZOS)
    original_x = 20 + (870 - original_image.width) // 2
    original_y = 125 + (490 - original_image.height) // 2
    canvas.paste(original_image, (original_x, original_y))
    source_h, source_w = original.shape[:2]
    for box, color in ((row["gt_bbox"], (220, 20, 20)), (row["crop_bbox"], (20, 60, 220))):
        x1, y1, x2, y2 = box
        mapped = [original_x + x1 * original_image.width / source_w,
                  original_y + y1 * original_image.height / source_h,
                  original_x + x2 * original_image.width / source_w,
                  original_y + y2 * original_image.height / source_h]
        draw.rectangle(mapped, outline=color, width=3)
    native_image = Image.fromarray(native * 255, mode="L").resize((256, 256), Image.Resampling.NEAREST)
    canvas.paste(native_image.convert("RGB"), (930, 170))
    draw.text((915, 140), "GT v4 native 128", fill="black", font=body)
    draw.line((20, 640, 1180, 640), fill=(150, 150, 150), width=2)
    draw.text((20, 655), f"TIGHT RECTANGLE: {row['rect_text']}", fill="black", font=body)
    draw.text((610, 655), f"CURVED CUTOUT: {row['curved_text']}", fill="black", font=body)
    _panel_crop(canvas, rect, 20, 700)
    _panel_crop(canvas, curved, 610, 700)
    return canvas


def main() -> int:
    args = _args()
    root = args.experiment_dir.resolve()
    if file_sha256(root / "results.json") != EXPECTED_REPORT_SHA256:
        raise ValueError("Recognition report SHA256 differs from the qualified GT-v4 experiment")
    if file_sha256(args.test_manifest) != EXPECTED_TEST_MANIFEST_SHA256:
        raise ValueError("Official TEST manifest SHA256 mismatch")
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    samples = {sample.sample_id: sample for sample in manifest.samples}
    if len(samples) != 300:
        raise ValueError("Expected 300 official Total-Text TEST samples")
    records = []
    for path in sorted((root / "samples").glob("*.json")):
        sample = json.loads(path.read_text(encoding="utf-8"))
        if sample["sample_id"] not in samples:
            raise ValueError("A diagnostic sample is absent from official TEST")
        key = path.stem
        rectangle = json.loads((root / "ocr/gt_v4_oracle/tight_rectangular" / f"{key}.json").read_text(encoding="utf-8"))["predictions"]
        curve = json.loads((root / "ocr/gt_v4_oracle/curved_cutout" / f"{key}.json").read_text(encoding="utf-8"))["predictions"]
        proposals = sample["branches"]["gt_v4_oracle"]["proposals"]
        if len(rectangle) != len(curve) or len(rectangle) != len(proposals):
            raise ValueError(f"OCR/proposal cardinality mismatch: {sample['sample_id']}")
        for proposal, first, second in zip(proposals, rectangle, curve):
            best = max(sample["ground_truth"], key=lambda word: box_iou(proposal["bbox"], word["bbox"]))
            iou = box_iou(proposal["bbox"], best["bbox"])
            if (iou < 0.5 or first["raw_text"].casefold() == best["text"].casefold()
                or second["raw_text"].casefold() == best["text"].casefold()):
                continue
            records.append({
                "sample_id": sample["sample_id"], "rank": proposal["rank"],
                "iou": iou, "gt_text": best["text"], "gt_bbox": best["bbox"],
                "crop_bbox": proposal["bbox"], "rect_text": first["raw_text"],
                "curved_text": second["raw_text"],
            })
    selected = _select(records, args.per_side)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing visual output: {output}")
    output.mkdir(parents=True)
    files = []
    for number, row in enumerate(selected, 1):
        sample = samples[row["sample_id"]]
        key = row["sample_id"].replace(":", "__")
        shard = json.loads((root / "samples" / f"{key}.json").read_text(encoding="utf-8"))
        image_path = args.data_root / sample.image_path
        if file_sha256(image_path) != sample.image_sha256:
            raise ValueError(f"Source RGB checksum mismatch: {row['sample_id']}")
        with Image.open(image_path) as opened:
            original = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
        native_path = root / shard["gt_native_mask"]["file"]
        if file_sha256(native_path) != shard["gt_native_mask"]["sha256"]:
            raise ValueError(f"Oracle native mask checksum mismatch: {row['sample_id']}")
        with Image.open(native_path) as opened:
            native = np.ascontiguousarray((np.asarray(opened.convert("L")) > 0).astype(np.uint8))
        _, crops = native_component_curved_crops(native, original, RepositoryContourParams(150, 0, 10))
        proposal = next(item for item in crops if item.rank == row["rank"])
        expected = next(item for item in shard["branches"]["gt_v4_oracle"]["proposals"] if item["rank"] == row["rank"])
        if (_sha_rgb(proposal.rectangular_rgb) != expected["tight_rgb_sha256"]
            or _sha_rgb(proposal.curved_rgb) != expected["curved_rgb_sha256"]):
            raise ValueError(f"Recreated PARSeq crop differs: {row['sample_id']} rank {row['rank']}")
        image = _render(row, original, native, proposal.rectangular_rgb, proposal.curved_rgb)
        filename = f"{number:02d}-{key}-rank{row['rank']:02d}.png"
        path = output / filename
        image.save(path, optimize=True)
        files.append({**row, "file": filename, "sha256": file_sha256(path)})
        print(f"[GT-v4 crop failures] {number}/{len(selected)} {row['sample_id']} rank={row['rank']}", flush=True)
    report = {"schema_version": 1, "result_class": "TEST GT-v4 visual diagnostic only",
              "source_report_sha256": EXPECTED_REPORT_SHA256,
              "selection": "Both crop shapes wrong, IoU>=0.5; six highest and six lowest IoU, distinct images, deterministic ties",
              "eligible_both_wrong": len(records), "selected": files}
    report_path = output / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    archive = output.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        bundle.write(report_path, "manifest.json")
        for row in files:
            bundle.write(output / row["file"], row["file"])
    print("ZIP", archive, "SHA256", file_sha256(archive), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
