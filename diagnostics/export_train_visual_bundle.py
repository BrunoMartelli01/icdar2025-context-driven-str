"""Export a portable, read-only visual audit of frozen TRAIN artifacts.

No inference is performed. ``paper_faithful`` means the repository's frozen
CURRENT_ALG1 approximate diagnostic, not a claim of strict paper reproduction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from diagnostics.contextual_ocr_correction import verify
from diagnostics.gt_mask_localization_oracle import checked_native
from diagnostics.lavis_captioner_candidates import sha, write_json
from diagnostics.ocr_target_bound_battery_train import load as load_battery
from pipeline.frozen_segmentation import backproject_binary_mask
from segmentation.data import (
    icdar13_task22_rgb_to_target_and_validity,
    load_segmentation_manifest,
)
from segmentation.evaluate import resize_binary_target_to_native_logits


EXPECTED = {
    "battery": "53dc86cca57ff11ff0f03d1b1ed729db90a151ddfad5765e58375198614c0746",
    "source": "38afc5fbb93bf6c6cc9970ad90454509f1089d6249a56447e6f60aecfbf2cc1b",
    "checkpoint": "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2",
}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_mask(path: Path, array: np.ndarray) -> None:
    Image.fromarray((np.asarray(array, dtype=np.uint8) * 255), mode="L").save(path, optimize=True)


def draw_boxes(rgb: Image.Image, groups: list[tuple[str, list[int], str]]) -> Image.Image:
    shown = rgb.copy()
    draw = ImageDraw.Draw(shown)
    for label, box, color in groups:
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color, stroke_width=2, stroke_fill="black")
    return shown


def crop_rows(rows: list[dict], rgb: Image.Image, root: Path, method: str) -> list[dict]:
    target = root / "crops" / method
    target.mkdir(parents=True)
    exported = []
    for row in sorted(rows, key=lambda item: item["rank"]):
        box = [int(value) for value in row["bbox"]]
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= rgb.width and 0 <= y1 < y2 <= rgb.height):
            raise RuntimeError(f"invalid {method} crop bbox: {box}")
        name = f"{int(row['rank']):02d}.png"
        crop = rgb.crop((x1, y1, x2, y2))
        crop.save(target / name, optimize=True)
        exported.append({
            "rank": int(row["rank"]), "file": f"crops/{method}/{name}",
            "bbox_xyxy_exclusive": box, "parseq_text": row["text"],
            "best_gt_index": row["best_gt_index"], "best_gt_iou": row["best_gt_iou"],
            "wordness": row["category"],
        })
    return exported


def cards(rows: list[dict]) -> str:
    result = []
    for row in rows:
        result.append(
            '<article class="crop"><img src="{}" loading="lazy"><div><b>#{:02d}</b> '
            '<code>{}</code><br>bbox={}<br>{}, best GT={} IoU={:.3f}</div></article>'.format(
                html.escape(row["file"]), row["rank"], html.escape(str(row["parseq_text"])),
                row["bbox_xyxy_exclusive"], html.escape(row["wordness"]), row["best_gt_index"],
                float(row["best_gt_iou"])))
    return "\n".join(result) or "<p>No retained crops.</p>"


STYLE = """
body{font-family:Arial,sans-serif;margin:20px;color:#202020}a{color:#0645ad}code{font-size:1.1em}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.panels figure{margin:0}
.panels img{max-width:100%;max-height:420px;border:1px solid #888}.caption{padding:10px;background:#f5f5f5;margin:8px 0}
.crops{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}.crop{border:1px solid #aaa;padding:7px}
.crop img{width:100%;height:100px;object-fit:contain;background:#eee}.nav{position:sticky;top:0;background:white;padding:8px;border-bottom:1px solid #aaa}
"""


def sample_html(sid: str, captions: dict, gt_words: list[dict], v1: list[dict], faithful: list[dict]) -> str:
    gt = ", ".join(f"[{w['gt_index']}] {html.escape(w['text'])}" for w in gt_words)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>TRAIN {sid}</title><style>{STYLE}</style></head><body>
<div class="nav"><a href="../../index.html">All TRAIN images</a> · sample {sid}</div>
<h1>ICDAR13 TRAIN sample {sid}</h1>
<div class="panels">
<figure><img src="image.jpg"><figcaption>Original RGB</figcaption></figure>
<figure><img src="image_gt_boxes.jpg"><figcaption>GT word boxes</figcaption></figure>
<figure><img src="image_localization_boxes.jpg"><figcaption>V1 cyan; paper-faithful Algorithm1 magenta</figcaption></figure>
<figure><img src="gt_mask_original_binary.png"><figcaption>GT binary mask, original resolution</figcaption></figure>
<figure><img src="gt_mask_native_128.png"><figcaption>GT target, native 128×128</figcaption></figure>
<figure><img src="predicted_mask_native_128.png"><figcaption>Prediction sigmoid≥0.5, native 128×128</figcaption></figure>
<figure><img src="predicted_mask_original.png"><figcaption>Frozen nearest-neighbor backprojection</figcaption></figure>
<figure><img src="gt_vs_prediction_native.png"><figcaption>Native error map: white TP, green FN, red FP</figcaption></figure>
</div>
<h2>Ground-truth words</h2><p>{gt}</p>
<h2>Captions</h2>
<div class="caption"><b>InstructBLIP:</b> {html.escape(captions['instructblip'])}</div>
<div class="caption"><b>Florence-2-base-ft:</b> {html.escape(captions['florence2-base-ft'])}</div>
<h2>Practical-remap V1 crops + PARSeq ({len(v1)})</h2><div class="crops">{cards(v1)}</div>
<h2>Paper-faithful Algorithm1 approximate crops + PARSeq ({len(faithful)})</h2><div class="crops">{cards(faithful)}</div>
<p>Machine-readable details: <a href="metadata.json">metadata.json</a>.</p></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--battery-root", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-output", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() or args.zip_output.exists():
        raise FileExistsError("refusing to replace an existing bundle or ZIP")
    verify(args.battery_root / "results.json", EXPECTED["battery"])
    _, _, battery, _, _ = load_battery(args.battery_root)
    verify(args.source_report, EXPECTED["source"])
    source_report = json.loads(args.source_report.read_text())
    if source_report["protocol"]["scope"] != {"split":"TRAIN","images":229,"gt_words":849}:
        raise RuntimeError("expected frozen TRAIN229 source")
    files = source_report["protocol"]["files"]
    # Source report points to the exact manifest used by the run.
    manifest_candidates = [Path(path) for path in files if path.endswith("/manifests/icdar13_segmentation_author_protocol.yaml")]
    if len(manifest_candidates) != 1:
        raise RuntimeError("cannot resolve the frozen TRAIN manifest")
    manifest_path = manifest_candidates[0]
    verify(manifest_path, files[str(manifest_path)])
    manifest = load_segmentation_manifest(manifest_path, require_checksums=False)
    if manifest.split_role != "train" or len(manifest.samples) != 229:
        raise RuntimeError("expected official TRAIN229 manifest")

    samples = {s["sample_id"]: s for s in battery["samples"]}
    words = defaultdict(list)
    for row in source_report["per_word"]: words[row["sample_id"]].append(row)
    crops = defaultdict(lambda: defaultdict(list))
    for row in source_report["per_crop"]:
        if row["method"] in ("PRACTICAL_REMAP_V1", "CURRENT_ALG1"):
            crops[row["sample_id"]][row["method"]].append(row)
    if (len(samples), sum(map(len, words.values())),
            sum(len(v["PRACTICAL_REMAP_V1"]) for v in crops.values()),
            sum(len(v["CURRENT_ALG1"]) for v in crops.values())) != (229,849,1203,1761):
        raise RuntimeError("frozen population mismatch")

    args.output_dir.mkdir(parents=True)
    index_rows = []
    all_metadata = []
    for number, sample in enumerate(manifest.samples, 1):
        sid = sample.sample_id
        info = samples[sid]
        if not all(key in info["captions"] for key in ("instructblip", "florence2-base-ft")):
            raise RuntimeError(f"missing requested captions: {sid}")
        image_path = args.data_root / sample.image_path
        verify(image_path, info["image_sha256"])
        with Image.open(image_path) as opened: rgb = opened.convert("RGB")
        mask_path = args.data_root / sample.mask_path
        annotation_path = args.data_root / sample.validity_annotation_path
        dataset_files = source_report["protocol"]["dataset_files"]
        verify(mask_path, dataset_files[str(sample.mask_path)]["sha256"])
        verify(annotation_path, dataset_files[str(sample.validity_annotation_path)]["sha256"])
        with Image.open(mask_path) as opened:
            target, validity = icdar13_task22_rgb_to_target_and_validity(
                opened, annotation_path.read_text(encoding="utf-8-sig"))
        gt_native = resize_binary_target_to_native_logits(target).numpy().astype(np.uint8)
        native_path = args.source_report.parent / "masks" / f"{sid}.json"
        pred_native, shard = checked_native(native_path, source_report["contract_sha256"], sid,
            info["image_sha256"], EXPECTED["checkpoint"])
        predicted = backproject_binary_mask(pred_native, (rgb.height, rgb.width))
        if digest_bytes(np.ascontiguousarray(predicted).tobytes()) != shard["segmentation_metadata"]["output_mask_sha256"]:
            raise RuntimeError(f"projected prediction mismatch: {sid}")

        sample_dir = args.output_dir / "samples" / sid
        sample_dir.mkdir(parents=True)
        shutil.copyfile(image_path, sample_dir / "image.jpg")
        save_mask(sample_dir / "gt_mask_original_binary.png", np.asarray(target) > 0)
        save_mask(sample_dir / "validity_mask_original.png", np.asarray(validity) > 0)
        save_mask(sample_dir / "gt_mask_native_128.png", gt_native)
        save_mask(sample_dir / "predicted_mask_native_128.png", pred_native)
        save_mask(sample_dir / "predicted_mask_original.png", predicted)
        error = np.zeros((128,128,3), dtype=np.uint8)
        error[(gt_native == 1) & (pred_native == 1)] = (255,255,255)
        error[(gt_native == 1) & (pred_native == 0)] = (0,255,0)
        error[(gt_native == 0) & (pred_native == 1)] = (255,0,0)
        Image.fromarray(error, mode="RGB").save(sample_dir / "gt_vs_prediction_native.png", optimize=True)

        gt_words = sorted(words[sid], key=lambda item:item["gt_index"])
        Image.fromarray(np.asarray(draw_boxes(rgb, [(f"GT{w['gt_index']}:{w['text']}",w['bbox'],'lime') for w in gt_words])))\
            .save(sample_dir / "image_gt_boxes.jpg", quality=92)
        localization = ([(f"V{r['rank']}",r['bbox'],'cyan') for r in crops[sid]["PRACTICAL_REMAP_V1"]] +
                        [(f"A{r['rank']}",r['bbox'],'magenta') for r in crops[sid]["CURRENT_ALG1"]])
        draw_boxes(rgb, localization).save(sample_dir / "image_localization_boxes.jpg", quality=92)
        v1 = crop_rows(crops[sid]["PRACTICAL_REMAP_V1"], rgb, sample_dir, "v1")
        faithful = crop_rows(crops[sid]["CURRENT_ALG1"], rgb, sample_dir, "paper_faithful_algorithm1")
        metadata = {
            "sample_id": sid, "image_source_path": str(sample.image_path), "image_sha256": info["image_sha256"],
            "ground_truth_words": [{"index":w["gt_index"],"text":w["text"],"bbox_xyxy_exclusive":w["bbox"]} for w in gt_words],
            "captions": {k:info["captions"][k] for k in ("instructblip","florence2-base-ft")},
            "v1": v1, "paper_faithful_algorithm1_approximate": faithful,
            "segmentation": {"checkpoint_sha256":EXPECTED["checkpoint"],"native_shape":[128,128],
                "threshold":0.5,"comparison":"greater_than_or_equal","backprojection":"nearest_binary"},
        }
        write_json(sample_dir / "metadata.json", metadata)
        (sample_dir / "index.html").write_text(sample_html(sid,metadata["captions"],gt_words,v1,faithful), encoding="utf-8")
        all_metadata.append(metadata)
        index_rows.append(f'<li><a href="samples/{sid}/index.html">sample {sid}</a> — GT {len(gt_words)}, V1 {len(v1)}, Algorithm1 {len(faithful)}</li>')
        print(f"[visual bundle] {number}/229 sample={sid}", flush=True)

    readme = """# ICDAR13 TRAIN visual audit bundle

This bundle contains all 229 TRAIN images. Open `index.html` in a browser.

- `ground_truth_words`: official recognition text and boxes used only for evaluation.
- `gt_mask_*`: Task 2.2 target; `validity_mask_original.png` marks care pixels.
- `predicted_mask_native_128.png`: frozen epoch-0098 sigmoid >= 0.5 prediction.
- `predicted_mask_original.png`: frozen nearest-neighbor binary backprojection.
- `v1`: frozen practical-remap-v1 crops and cached PARSeq outputs.
- `paper_faithful_algorithm1`: frozen CURRENT_ALG1 connected-component crops and cached PARSeq outputs.

The Algorithm1 path is an approximate diagnostic under unresolved paper ambiguities; it is not labeled strict reproduction.
No inference was executed to build this archive. In the native error map: white=TP, green=FN, red=FP.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    (args.output_dir / "index.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>ICDAR13 TRAIN audit</title><style>{STYLE}</style></head>'
        f'<body><h1>ICDAR13 TRAIN: V1 vs Algorithm1</h1><p>229 images; 849 GT words; 1203 V1 crops; 1761 Algorithm1 crops.</p>'
        f'<p><a href="README.md">Protocol notes</a> · <a href="manifest.json">Machine-readable manifest</a></p><ol>{"".join(index_rows)}</ol></body></html>', encoding="utf-8")
    protocol = {
        "classification":"APPROXIMATE_DIAGNOSTIC", "split":"TRAIN", "images":229, "gt_words":849,
        "v1_crops":1203, "paper_faithful_algorithm1_crops":1761, "new_inference":False,
        "source_sha256":{"battery_results":EXPECTED["battery"],"repository_contours_results":EXPECTED["source"],
            "segmentation_checkpoint":EXPECTED["checkpoint"],"exporter":sha(__file__)},
        "samples":all_metadata,
    }
    write_json(args.output_dir / "manifest.json", protocol)

    args.zip_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.zip_output.with_suffix(args.zip_output.suffix + ".partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file(): archive.write(path, Path(args.output_dir.name) / path.relative_to(args.output_dir))
    temporary.replace(args.zip_output)
    with zipfile.ZipFile(args.zip_output) as archive:
        bad = archive.testzip()
        if bad is not None: raise RuntimeError(f"ZIP CRC failure: {bad}")
        entries = len(archive.infolist())
    print("BUNDLE", args.output_dir.resolve(), flush=True)
    print("ZIP", args.zip_output.resolve(), flush=True)
    print("ZIP_SHA256", sha(args.zip_output), flush=True)
    print("ZIP_ENTRIES", entries, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
