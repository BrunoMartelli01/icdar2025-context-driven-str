"""Package comparable Total-Text TEST words missed by three GT-oracle OCR paths.

One composite PNG per word compares original pixel-GT contours reconstructed
with PARSeq, the tight GT-v4 rectangle, and the GT-v4 curved cutout. GT is used
only for diagnostic association/evaluation. No inference is performed here.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import unicodedata
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from localize.curved_component_crop import native_component_curved_crops
from localize.repository_contours import RepositoryContourParams
from recognition.scene_text_v1_evaluation import box_iou
from scripts.package_totaltext_v4_gt_crop_failures import _font, _render
from segmentation.data import file_sha256, load_segmentation_manifest


EXPECTED_COMPONENT_REPORT = "7e892a86d4cb19ab1c6e9613c4dfb073105fa057414fa631ea464e1124310dfb"
EXPECTED_V4_REPORT = "272613b094777b9836a1ffde2fd9265fcec17c84e7ccb60661f9a87ac0e29989"
EXPECTED_MANIFEST = "da2f87e7cae765f426af39c9641535138d6657d2e9332e6fdf9e80f2ba6b69bd"


def normalized(text: str) -> str:
    return "".join(char for char in text.casefold()
                   if unicodedata.category(char)[0] not in "PS")


def is_correct(prediction: str, truth: str) -> bool:
    return bool(normalized(truth)) and normalized(prediction) == normalized(truth)


def select_failures(components_report: dict, v4_root: Path) -> list[dict]:
    words = {(word["sample_id"], word["word_id"]): word for word in components_report["words"]}
    selected = []
    for path in sorted((v4_root / "samples").glob("*.json")):
        sample = json.loads(path.read_text(encoding="utf-8"))
        key = path.stem
        ocr = {
            mode: {row["rank"]: row["raw_text"] for row in json.loads(
                (v4_root / "ocr/gt_v4_oracle" / mode / f"{key}.json").read_text(encoding="utf-8")
            )["predictions"]}
            for mode in ("tight_rectangular", "curved_cutout")
        }
        for word_index, truth in enumerate(sample["ground_truth"], 1):
            component_word = words[(sample["sample_id"], word_index)]
            if is_correct(component_word["reconstruction"], truth["text"]):
                continue
            matching = [proposal for proposal in sample["branches"]["gt_v4_oracle"]["proposals"]
                        if box_iou(proposal["bbox"], truth["bbox"]) >= 0.5]
            # To compare actual OCR failures, both v4 crop inputs must exist.
            if len(matching) != 1:
                continue
            proposal = matching[0]
            rectangle_text = ocr["tight_rectangular"][proposal["rank"]]
            curved_text = ocr["curved_cutout"][proposal["rank"]]
            if is_correct(rectangle_text, truth["text"]) or is_correct(curved_text, truth["text"]):
                continue
            selected.append({
                "sample_id": sample["sample_id"], "word_id": word_index,
                "rank": proposal["rank"], "iou": box_iou(proposal["bbox"], truth["bbox"]),
                "gt_text": truth["text"], "gt_bbox": truth["bbox"],
                "crop_bbox": proposal["bbox"], "rect_text": rectangle_text,
                "curved_text": curved_text,
                "component_reconstruction": component_word["reconstruction"],
                "component_texts": component_word["component_texts"],
                "contour_indices": component_word["contour_indices"],
            })
    return sorted(selected, key=lambda row: (row["sample_id"], row["word_id"]))


def _sha_rgb(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _draw_wrapped(draw, text: str, xy: tuple[int, int], *, width: int, lines: int = 2):
    font = _font(17)
    remaining = text
    for line in range(lines):
        if not remaining:
            break
        take = min(width, len(remaining))
        fragment, remaining = remaining[:take], remaining[take:]
        if line == lines - 1 and remaining:
            fragment = fragment[:-3] + "..."
        draw.text((xy[0], xy[1] + line * 25), fragment, fill="black", font=font)


def _composite(row: dict, original: np.ndarray, native: np.ndarray,
               pixel_mask: np.ndarray, rectangle: np.ndarray, curved: np.ndarray,
               source_components: list[dict], source_ocr: dict) -> Image.Image:
    base = _render(row, original, native, rectangle, curved)
    canvas = Image.new("RGB", (1200, 1260), "white")
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)
    body = _font(18)
    draw.line((20, 915, 1180, 915), fill=(150, 150, 150), width=2)
    draw.text((20, 930), "Original pixel GT (word box)", fill="black", font=body)
    x1, y1, x2, y2 = row["gt_bbox"]
    word_mask = Image.fromarray(pixel_mask[y1:y2, x1:x2] * 255, mode="L")
    word_mask.thumbnail((290, 205), Image.Resampling.NEAREST)
    canvas.paste(word_mask.convert("RGB"), (20, 970))
    draw.text((330, 935), "PARSeq contours joined by GT word (oracle)", fill="black", font=body)
    reconstruction = row["component_reconstruction"] or "[none: no uniquely assigned component]"
    component_outputs = " | ".join(row["component_texts"]) or "[none]"
    _draw_wrapped(draw, f"RECONSTRUCTION: {reconstruction}", (330, 970), width=92)
    _draw_wrapped(draw, "COMPONENT OUTPUTS: " + component_outputs, (330, 1030), width=92)
    draw.text((330, 1088), f"{len(row['contour_indices'])} associated components; first 12 RGB crops below", fill="black", font=body)
    for place, index in enumerate(row["contour_indices"][:12]):
        component = source_components[index]
        x1, y1, x2, y2 = component["bbox"]
        crop = Image.fromarray(original[y1:y2, x1:x2], mode="RGB")
        crop.thumbnail((73, 82), Image.Resampling.LANCZOS)
        x = 20 + place * 97
        canvas.paste(crop, (x + (75 - crop.width) // 2, 1140 + (82 - crop.height) // 2))
        label = source_ocr[index]["text"]
        draw.text((x, 1228), f"{place + 1}:{label[:7]}", fill="black", font=_font(14))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if file_sha256(args.component_root / "results.json") != EXPECTED_COMPONENT_REPORT:
        raise ValueError("Original-GT component report SHA256 mismatch")
    if file_sha256(args.v4_root / "results.json") != EXPECTED_V4_REPORT:
        raise ValueError("GT-v4 curved crop report SHA256 mismatch")
    if file_sha256(args.manifest) != EXPECTED_MANIFEST:
        raise ValueError("Total-Text TEST manifest SHA256 mismatch")
    manifest = load_segmentation_manifest(args.manifest, require_checksums=True)
    samples = {sample.sample_id: sample for sample in manifest.samples}
    if len(samples) != 300:
        raise ValueError("Expected official TEST-300")
    component_report = json.loads((args.component_root / "results.json").read_text(encoding="utf-8"))
    selected = select_failures(component_report, args.v4_root)
    if len(selected) != 60:
        raise ValueError(f"Qualified triple-failure population changed: {len(selected)} instead of 60")
    if args.output_dir.exists() or args.output_dir.with_suffix(".zip").exists():
        raise FileExistsError("Refusing to overwrite an existing visual package")
    args.output_dir.mkdir(parents=True)
    manifest_rows = []
    for number, row in enumerate(selected, 1):
        sample = samples[row["sample_id"]]
        key = sample.sample_id.replace(":", "__")
        paths = [args.data_root / sample.image_path, args.data_root / sample.mask_path]
        for path, expected in zip(paths, (sample.image_sha256, sample.mask_sha256)):
            if file_sha256(path) != expected:
                raise ValueError(f"Source checksum mismatch: {path}")
        with Image.open(paths[0]) as opened:
            original = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
        with Image.open(paths[1]) as opened:
            pixel_mask = np.ascontiguousarray((np.asarray(opened.convert("L")) >= 128).astype(np.uint8))
        v4_shard = json.loads((args.v4_root / "samples" / f"{key}.json").read_text(encoding="utf-8"))
        native_path = args.v4_root / v4_shard["gt_native_mask"]["file"]
        if file_sha256(native_path) != v4_shard["gt_native_mask"]["sha256"]:
            raise ValueError(f"GT-v4 native mask checksum mismatch: {sample.sample_id}")
        with Image.open(native_path) as opened:
            native = np.ascontiguousarray((np.asarray(opened.convert("L")) > 0).astype(np.uint8))
        _, proposals = native_component_curved_crops(native, original, RepositoryContourParams(150, 0, 10))
        proposal = next(item for item in proposals if item.rank == row["rank"])
        prepared = next(item for item in v4_shard["branches"]["gt_v4_oracle"]["proposals"] if item["rank"] == row["rank"])
        if (_sha_rgb(proposal.rectangular_rgb) != prepared["tight_rgb_sha256"]
            or _sha_rgb(proposal.curved_rgb) != prepared["curved_rgb_sha256"]):
            raise ValueError(f"GT-v4 crop hashes differ: {sample.sample_id}")
        source_shard = json.loads((args.component_root / "samples" / f"{key}.json").read_text(encoding="utf-8"))
        ocr_shard = json.loads((args.component_root / "ocr" / f"{key}.json").read_text(encoding="utf-8"))
        source_components = {item["contour_index"]: item for item in source_shard["components"]}
        source_ocr = {item["contour_index"]: item for item in ocr_shard["predictions"]}
        if (len(source_components) != len(source_ocr)
            or [source_ocr[index]["text"] for index in row["contour_indices"]] != row["component_texts"]):
            raise ValueError(f"Original-GT OCR association differs: {sample.sample_id}")
        image = _composite(row, original, native, pixel_mask, proposal.rectangular_rgb,
                           proposal.curved_rgb, source_components, source_ocr)
        filename = f"{number:02d}-{key}-word{row['word_id']:03d}.png"
        target = args.output_dir / filename
        image.save(target, optimize=True)
        manifest_rows.append({**row, "file": filename, "sha256": file_sha256(target)})
        if number % 10 == 0 or number == len(selected):
            print(f"[triple-failure composites] {number}/{len(selected)}", flush=True)
    report = {"schema_version": 1, "result_class": "TEST GT-oracle visual diagnostic; not Table 5",
              "selection": "all three OCR outputs wrong after symmetric casefold and Unicode punctuation/symbol removal; actual GT-v4 crop IoU>=0.5 required",
              "eligible": len(selected), "component_report_sha256": EXPECTED_COMPONENT_REPORT,
              "v4_report_sha256": EXPECTED_V4_REPORT, "images": manifest_rows}
    report_path = args.output_dir / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    archive = args.output_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        bundle.write(report_path, "manifest.json")
        for row in manifest_rows:
            bundle.write(args.output_dir / row["file"], row["file"])
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None or len(bundle.namelist()) != 61:
            raise ValueError("ZIP integrity or file count failed")
    print("ZIP", archive, "SHA256", file_sha256(archive), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
