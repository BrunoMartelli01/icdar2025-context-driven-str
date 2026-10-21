"""One composite per TRAIN image: RGB, pixel GT, v1, v2 and filled v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageOps
import yaml


LABELS = ("RGB TRAIN", "ORIGINAL PIXEL GT", "FILLED V1", "LETTER-LIKE V2", "FILLED BOX V3")


def _panel(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    width, height = size
    panel = Image.new("RGB", (width, height + 30), "#242424")
    fitted = ImageOps.contain(image.convert("RGB"), (width, height))
    panel.paste(fitted, ((width - fitted.width) // 2, 30 + (height - fitted.height) // 2))
    ImageDraw.Draw(panel).text((8, 8), label, fill="white")
    return panel


def package(
    base_root: Path, refined_root: Path, pixel_root: Path, box_root: Path,
    output_zip: Path, count: int,
) -> None:
    report = json.loads((box_root / "qualification.json").read_text())
    base = yaml.safe_load((base_root / "manifests/total_text_train.yaml").read_text())
    pixel = yaml.safe_load((pixel_root / "manifests/total_text_train.yaml").read_text())
    pixels = {row["id"]: row for row in pixel["samples"]}
    rows = sorted(
        base["samples"],
        key=lambda row: (
            report["per_image"][row["id"]]["base_foreground_pixels"]
            - report["per_image"][row["id"]]["final_box_foreground_pixels"]
        ) / max(1, report["per_image"][row["id"]]["base_foreground_pixels"]),
        reverse=True,
    )[:count]
    special = next((row for row in base["samples"] if row["id"] == "total_text:1378"), None)
    if special is not None and all(row["id"] != special["id"] for row in rows):
        rows.append(special)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(
            "README.txt",
            "TRAIN-only visual diagnostic. One composite JPEG per image. "
            "Masks are displayed without geometric warping; no prediction or OCR is shown.\n",
        )
        for index, row in enumerate(rows, 1):
            source = pixels[row["id"]]
            paths = (
                base_root / row["image"], pixel_root / source["mask"],
                base_root / row["mask"], refined_root / row["mask"],
                box_root / row["mask"],
            )
            with ExitStackImages(paths) as images:
                parts = [_panel(image, label, (430, 330)) for image, label in zip(images, LABELS)]
            board = Image.new("RGB", (430 * 5, 360), "#242424")
            for column, part in enumerate(parts):
                board.paste(part, (430 * column, 0))
            from io import BytesIO
            stream = BytesIO()
            board.save(stream, format="JPEG", quality=92, optimize=True)
            archive.writestr(f"{row['id'].split(':', 1)[1]}.jpg", stream.getvalue())
            print(f"[adaptive boxes visual] {index}/{len(rows)} {row['id']}", flush=True)
    print("ZIP", output_zip)


class ExitStackImages:
    def __init__(self, paths: tuple[Path, ...]):
        self.paths = paths
        self.images: list[Image.Image] = []

    def __enter__(self) -> list[Image.Image]:
        for path in self.paths:
            image = Image.open(path)
            image.load()
            self.images.append(image)
        return self.images

    def __exit__(self, *args: object) -> None:
        for image in self.images:
            image.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--refined-root", type=Path, required=True)
    parser.add_argument("--pixel-root", type=Path, required=True)
    parser.add_argument("--box-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    package(args.base_root, args.refined_root, args.pixel_root, args.box_root, args.output_zip, args.count)


if __name__ == "__main__":
    main()
