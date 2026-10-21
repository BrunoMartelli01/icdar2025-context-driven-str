"""Package exactly one visual audit image per Total-Text TEST sample.

Uses only existing no-area evaluation shards/crops/OCR. The curved word mask is
regenerated from test polygons solely as a labeled visual diagnostic; no model
is run and no inference decision uses ground truth.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import textwrap
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml

from recognition.scene_text_datasets import parse_totaltext_annotation
from segmentation.totaltext_word_regions import make_word_regions


PANEL_W, PANEL_H, HEADER = 360, 240, 28
TILE_W, TILE_IMAGE_H = 360, 132
EXPECTED_EVALUATION_SHA256 = "b934f04129d04ae084f990f6de76aa1ce371eac3f69ef9c6085dc0a9526e2b6e"


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        return ImageFont.load_default()


FONT = _font()


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _fit_panel(image: Image.Image, title: str) -> Image.Image:
    panel = Image.new("RGB", (PANEL_W, PANEL_H + HEADER), (28, 28, 28))
    ImageDraw.Draw(panel).text((8, 7), title, fill="white", font=FONT)
    fit = ImageOps.contain(image.convert("RGB"), (PANEL_W - 8, PANEL_H - 8))
    panel.paste(fit, ((PANEL_W - fit.width) // 2, HEADER + (PANEL_H - fit.height) // 2))
    return panel


def _tile(crop_path: Path, proposal: dict, ocr: dict | None, gt: list[dict]) -> Image.Image:
    pred = ocr["raw_text"]
    box = proposal["bbox"]
    overlaps = sorted(
        ((float(_iou(box, row["bbox"])), row["text"]) for row in gt), reverse=True
    )
    best_iou, best_gt = overlaps[0] if overlaps else (0.0, "")
    lines = [f"crop #{proposal['rank']}  IoU={best_iou:.2f}"]
    lines.extend(textwrap.wrap(f"PARSeq: {pred}", width=46, break_long_words=True) or ["PARSeq:"])
    lines.extend(textwrap.wrap(f"best GT: {best_gt if best_iou > 0 else '(none)'}", width=46, break_long_words=True) or ["best GT:"])
    header_height = 6 + 18 * len(lines)
    tile = Image.new("RGB", (TILE_W, TILE_IMAGE_H + header_height), (245, 245, 245))
    draw = ImageDraw.Draw(tile)
    for index, line in enumerate(lines):
        draw.text((6, 4 + index * 18), line, fill="black", font=FONT)
    with Image.open(crop_path) as opened:
        crop = opened.convert("RGB")
    fit = ImageOps.contain(crop, (TILE_W - 8, TILE_IMAGE_H - 8))
    tile.paste(fit, ((TILE_W - fit.width) // 2, header_height + (TILE_IMAGE_H - fit.height) // 2))
    return tile


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _save_jpeg(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=84, optimize=True)
    return stream.getvalue()


def package(data_root: Path, evaluation_root: Path, output: Path) -> str:
    data_root, evaluation_root, output = data_root.resolve(), evaluation_root.resolve(), output.resolve()
    report_path = evaluation_root / "results.json"
    if digest(report_path) != EXPECTED_EVALUATION_SHA256:
        raise ValueError("No-area evaluation report checksum differs from the supplied frozen result")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("summary", {}).get("images") != 300:
        raise ValueError("Expected completed Total-Text TEST-300 no-area evaluation")
    manifest_path = data_root / "manifests/total_text_test.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    if len(samples) != 300:
        raise ValueError("Expected official Total-Text TEST-300 manifest")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    with ZipFile(partial, "w", ZIP_DEFLATED, compresslevel=5) as archive:
        for offset, sample in enumerate(samples, 1):
            sample_id = sample["id"]
            key = sample_id.replace(":", "__")
            stem = sample_id.split(":", 1)[1]
            shard_path = evaluation_root / "samples" / f"{key}.json"
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            if shard["sample_id"] != sample_id:
                raise ValueError(f"Shard/sample mismatch: {sample_id}")
            image_path = data_root / sample["image"]
            pixel_mask_path = data_root / sample["mask"]
            if digest(image_path) != sample["image_sha256"] or digest(pixel_mask_path) != sample["mask_sha256"]:
                raise ValueError(f"Source checksum mismatch: {sample_id}")
            annotation_path = data_root / "datasets/total_text/test/annotations" / f"{stem}.txt"
            with Image.open(image_path) as opened:
                rgb = opened.convert("RGB")
            with Image.open(pixel_mask_path) as opened:
                pixel_mask = opened.convert("L")
            curved, _, _, _, _ = make_word_regions(
                rgb.size, parse_totaltext_annotation(annotation_path)
            )
            native_path = evaluation_root / shard["native_mask"]["file"]
            if digest(native_path) != shard["native_mask"]["sha256"]:
                raise ValueError(f"Native prediction checksum mismatch: {sample_id}")
            with Image.open(native_path) as opened:
                native = opened.convert("L").resize(rgb.size, Image.Resampling.NEAREST)
            titles = (
                (rgb, f"{sample_id} RGB"),
                (pixel_mask, "OFFICIAL PIXEL GT"),
                (curved, "DERIVED CURVED WORD TARGET (visual only)"),
                (native, "U-NET PREDICTION (native128, enlarged)"),
            )
            proposal_tiles = []
            for proposal in shard["proposals"]:
                ocr_path = evaluation_root / "ocr" / f"{key}__{proposal['rank']:02d}.json"
                crop_path = evaluation_root / proposal["file"]
                if digest(crop_path) != proposal["sha256"] or not ocr_path.is_file():
                    raise ValueError(f"Missing/corrupt crop or OCR: {sample_id} rank {proposal['rank']}")
                ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
                if ocr["crop_sha256"] != proposal["sha256"]:
                    raise ValueError(f"OCR/crop mismatch: {sample_id} rank {proposal['rank']}")
                proposal_tiles.append(_tile(crop_path, proposal, ocr, shard["ground_truth"]))
            gt_text = "  |  ".join(f"{i + 1}: {row['text']}" for i, row in enumerate(shard["ground_truth"]))
            gt_lines = textwrap.wrap("ALL GT WORDS (annotation order): " + gt_text, width=90, break_long_words=True)
            row_heights = [max(tile.height for tile in proposal_tiles[i:i + 2]) for i in range(0, len(proposal_tiles), 2)]
            top_height = 2 * (PANEL_H + HEADER)
            footer_height = 12 + 18 * max(1, len(gt_lines))
            canvas = Image.new("RGB", (2 * PANEL_W, top_height + sum(row_heights) + footer_height), "white")
            for i, (picture, title) in enumerate(titles):
                canvas.paste(_fit_panel(picture, title), ((i % 2) * PANEL_W, (i // 2) * (PANEL_H + HEADER)))
            y = top_height
            for row_index, row_height in enumerate(row_heights):
                for col, tile in enumerate(proposal_tiles[2 * row_index:2 * row_index + 2]):
                    canvas.paste(tile, (col * TILE_W, y))
                y += row_height
            draw = ImageDraw.Draw(canvas)
            for line in gt_lines:
                draw.text((6, y + 4), line, fill="black", font=FONT)
                y += 18
            archive.writestr(f"{stem}.jpg", _save_jpeg(canvas))
            if offset % 25 == 0 or offset == len(samples):
                print(f"[Total-Text visual ZIP] {offset}/300", flush=True)
    with ZipFile(partial) as check:
        if check.testzip() is not None or len(check.namelist()) != 300 or any(not name.endswith(".jpg") for name in check.namelist()):
            raise ValueError("ZIP integrity test failed")
    partial.replace(output)
    print("ZIP", output, "SHA256", digest(output), "BYTES", output.stat().st_size)
    return digest(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.data_root, args.evaluation_root, args.output)


if __name__ == "__main__":
    main()
