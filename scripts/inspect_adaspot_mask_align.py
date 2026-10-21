"""Posthoc geometry coverage and fixed-first-eight visualizations, no tuning."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from models.adaspot_mask_align import MaskAlignOptions, mask_align_geometry
from scripts.audit_adaspot_phase0 import digest
from scripts.diagnose_adaspot_phase0 import write_json


def run(root):
    result = json.loads((root / "results.json").read_text())
    audit = json.loads((root / "independent_audit.json").read_text())
    assert audit["status"] == "PASS" and audit["result_sha256"] == digest(root / "results.json")
    protocol = json.loads((root / "protocol_final.json").read_text())
    source = Path(protocol["source_run"])
    arms = {key: value["ocr"]["per_image"] for key, value in result["arms"].items()}
    receipts = json.loads((root / "feature_receipts.json").read_text())
    options = MaskAlignOptions(**protocol["resolved_config"]["geometry"])
    rows, outside, inside, fallback_equal = [], Counter(), Counter(), 0
    preview = Image.new("RGB", (1120, 8 * 148 + 50), "white")
    draw = ImageDraw.Draw(preview)
    headings = ["Original RGB / OCR", "Predicted mask + sample grid", "Rotated RGB / OCR", "Mask geometry RGB / OCR", "Actual F64: mean abs (8x32)"]
    for j, text in enumerate(headings):
        draw.text((j * 224 + 4, 5), text, fill="black")
    shown = 0
    rgb_shapes = []
    for i, receipt in enumerate(receipts):
        sid = receipt["sample_id"]
        with Image.open(receipt["image_path"]) as image:
            rgb_shapes.append([1, 3, image.height, image.width])
        with np.load(source / "test_predictions" / sid.replace(":", "__") / "full_selected_labels.npz") as values:
            labels = values["labels"]
        with np.load(receipt["feature_path"]) as values:
            feature_light = values["light"]
        for j, original in enumerate(arms["original_axis"][i]["predictions"]):
            predicted = {key: values[i]["predictions"][j] for key, values in arms.items()}
            own = labels == original["instance_id"]
            geometry = mask_align_geometry(own, view="mask", options=options)
            method = geometry.audit["method"]
            if method == "rotated_rectangle":
                assert predicted["mask_align"]["crop_rgb_sha256"] == predicted["rotated"]["crop_rgb_sha256"]
                fallback_equal += 1
            # Approximate continuous footprint rasterized at native centers.
            probe = geometry.grid((2, 256))
            center, half = probe.mean(0), probe[1] - probe[0]
            boundary = np.concatenate((center - half, (center + half)[::-1]))
            footprint = np.zeros_like(labels, dtype=np.uint8)
            cv2.fillPoly(footprint, [np.rint(boundary).astype(np.int32)], 1)
            covered = int((own & (footprint > 0)).sum())
            coverage = covered / int(own.sum())
            group = "cubic" if method == "cubic_centerline" else "fallback"
            inside[group] += covered
            outside[group] += int(own.sum()) - covered
            rows.append({"sample_id": sid, "instance_id": original["instance_id"], "method": method,
                "predicted_mask_pixels": int(own.sum()), "rasterized_footprint_covered_pixels": covered,
                "predicted_support_coverage": coverage})
            if shown >= 8:
                continue
            y = shown * 148 + 40
            draw.text((4, y), f"{sid}, instance {original['instance_id']}", fill="black")
            for column, arm in ((0, "original_axis"), (2, "rotated"), (3, "mask_align")):
                p = predicted[arm]
                with Image.open(root / "crops" / (p["crop_rgb_sha256"] + ".png")) as crop:
                    crop.thumbnail((216, 95))
                    preview.paste(crop, (column * 224 + 4, y + 15))
                text = p["text"].encode("ascii", "replace").decode()[:33]
                draw.text((column * 224 + 4, y + 116), text, fill="black")
            xx0, yy0, xx1, yy1 = original["bbox_native"]
            grid = geometry.grid((8, 32)) - [xx0, yy0]
            mask = Image.fromarray((own[yy0:yy1, xx0:xx1] * 210).astype(np.uint8)).convert("RGB")
            md = ImageDraw.Draw(mask)
            for col in range(0, 32, 4):
                md.line([tuple(grid[0, col]), tuple(grid[-1, col])], fill=(255, 40, 40), width=1)
            centerline = grid.mean(0)
            md.line([tuple(p) for p in centerline], fill=(20, 255, 60), width=1)
            mask.thumbnail((216, 95))
            preview.paste(mask, (228, y + 15))
            draw.text((228, y + 116), method, fill="black")
            heat = np.abs(feature_light[j]).mean(0)
            heat = (255 * (heat - heat.min()) / max(float(np.ptp(heat)), 1e-12)).astype(np.uint8)
            preview.paste(Image.fromarray(heat).resize((216, 64), Image.Resampling.NEAREST).convert("RGB"), (900, y + 15))
            draw.text((900, y + 116), "Per-instance display scaling", fill="black")
            shown += 1
    path = root / "fixed_first8_preview.png"
    preview.save(path)
    coverage = {key: {"mask_pixels": inside[key] + outside[key], "covered_pixels": inside[key],
        "pixel_weighted_coverage": inside[key] / (inside[key] + outside[key]),
        "median_instance_coverage": float(np.median([r["predicted_support_coverage"] for r in rows if (r["method"] == "cubic_centerline") == (key == "cubic")]))}
        for key in inside}
    write_json(root / "geometry_inspection.json", {"scope": "posthoc diagnostics only; approximate rasterized footprint, not GT character coverage",
        "result_sha256": digest(root / "results.json"), "analysis_source_sha256": digest(Path(__file__)),
        "coverage": coverage, "fallback_crops_identical_to_rotated": fallback_equal,
        "rgb_tensor_shapes": rgb_shapes, "preview_selection": "first8 care instances in frozen source order, regardless of OCR outcome",
        "preview_sha256": digest(path), "per_instance": rows})
    print(json.dumps({"coverage": coverage, "fallback_crops_identical_to_rotated": fallback_equal, "preview": str(path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    run(args.output_root.resolve())
