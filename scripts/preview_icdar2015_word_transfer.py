"""Render a scientific target comparison; panels are GT, not model predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from segmentation.data import load_segmentation_manifest
from segmentation.icdar2015_transfer import load_config, transform
from segmentation.totaltext_image_frequency_augment import letterbox_geometry


def preview(data_root, config_path, output):
    config = load_config(config_path)
    manifest = load_segmentation_manifest(data_root / config["source_manifest"])
    qualification = json.loads((data_root / "qualification.json").read_text())
    by_id = {sample.sample_id: sample for sample in manifest.samples}
    audits = qualification["per_image"]
    # One strongest fusion case and one fragmentation case, declared by GT topology.
    selected = sorted(audits, key=lambda name: (
        -audits[name]["original_native128_stretch"]["words_in_fused_components"], name))[:1]
    selected += sorted((name for name in audits if name not in selected), key=lambda name: (
        -audits[name]["transferred_native256_letterbox"]["fragmented_words"], name))[:1]
    paired = transform(config, augment=False)
    canvas = Image.new("RGB", (1200, 520 * len(selected) + 60), "#f4f5f7")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), "ICDAR2015 TRAIN target audit / approximate reproduction / no model predictions", fill="black")
    for row, sample_id in enumerate(selected):
        sample = by_id[sample_id]
        with Image.open(data_root / sample.image_path) as source:
            rgb = source.convert("RGB")
        with Image.open(data_root / sample.mask_path) as source:
            target = source.convert("L")
        with Image.open(data_root / sample.validity_annotation_path) as source:
            validity = source.convert("L")
        with Image.open(data_root / sample.separation_annotation_path) as source:
            separator = source.convert("L")
        with Image.open(data_root / sample.word_owner_annotation_path) as source:
            owners = source.convert("I")
        _, new_target, valid, _ = paired.transform_with_validity_and_separation(
            rgb, target, validity, separator, owners, 0)
        # Reconstruct the old union of official quadrilaterals from immutable annotations.
        from recognition.scene_text_datasets import parse_icdar2015_annotation
        from segmentation.scene_text_preparation import rasterize_instances
        annotation = Path(sample.image_path).parent.parent / "annotations" / (sample_id.split(":", 1)[1] + ".txt")
        old_target, old_valid = rasterize_instances(rgb.size, parse_icdar2015_annotation(data_root / annotation))
        old_native = (np.asarray(old_target.resize((128, 128), Image.Resampling.BILINEAR)) >= 128)
        old_native &= np.asarray(old_valid.resize((128, 128), Image.Resampling.NEAREST)) > 0
        new_mask = Image.fromarray(new_target[0].numpy().astype(np.uint8) * 255)
        new_mask_rgb = new_mask.convert("RGB")
        new_array = np.array(new_mask_rgb)
        new_array[~valid[0].numpy().astype(bool)] = (90, 95, 105)
        panels = (letterbox_geometry(rgb, fill=(124, 116, 104)),
                  Image.fromarray(old_native.astype(np.uint8) * 255).convert("RGB"),
                  Image.fromarray(new_array))
        headings = ("RGB (aspect preserved)", "Original 128x128 / stretch", "Transferred 256x256 / letterbox")
        y = row * 520 + 60
        draw.text((15, y), sample_id, fill="black")
        for col, (panel, label) in enumerate(zip(panels, headings)):
            draw.text((col * 400 + 10, y + 20), label, fill="black")
            canvas.paste(panel.resize((380, 380), Image.Resampling.NEAREST), (col * 400 + 10, y + 45))
        old_a, new_a = audits[sample_id]["original_native128_stretch"], audits[sample_id]["transferred_native256_letterbox"]
        draw.text((15, y + 440), f"GT words: {old_a['care_words']} | fused words: {old_a['words_in_fused_components']} -> {new_a['words_in_fused_components']} | "
                  f"lost: {old_a['lost_words']} -> {new_a['lost_words']} | fragmented: {old_a['fragmented_words']} -> {new_a['fragmented_words']}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"Preview already exists: {output}")
    canvas.save(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(preview(args.data_root, args.config, args.output))


if __name__ == "__main__":
    main()
