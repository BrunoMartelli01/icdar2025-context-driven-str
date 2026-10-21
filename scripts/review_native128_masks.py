"""Exact data-derived GT/prediction cards and full-TRAIN target-size statistics."""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from scripts.review_native128_ocr import inference_config
from scripts.evaluate_totaltext_compact_v2_selected_ocr import distribution
from scripts.evaluate_totaltext_v1_epoch447_ocr_padding import write
from segmentation.data import SegmentationDataset, file_sha256, load_segmentation_manifest
from segmentation.evaluate import _model_from_config, _preprocess, validate_checkpoint
from segmentation.train import _transform
from segmentation.training import seed_everything


def font(size):
    return ImageFont.truetype("DejaVuSans.ttf", size)


def binary_image(mask):
    return Image.fromarray((np.asarray(mask)>0).astype(np.uint8)*255)


def save_binary(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    binary_image(mask).save(path)


def source_checks(dataset, i):
    s = dataset.manifest.samples[i]
    fields = [s.image_sha256, s.mask_sha256, s.validity_annotation_sha256,
              s.separation_annotation_sha256, s.word_owner_annotation_sha256]
    for path, digest in zip(dataset.paths[i], fields):
        if path is not None and digest is not None and file_sha256(path) != digest:
            raise ValueError(f"GT source changed: {s.sample_id} {path}")


def dataset_for(c, version):
    model_id = {"pixel": "pixel_tt", "curved": "curved", "v4": "v4", "separated": "sep_447", "compact": "compact_361"}[version["id"]]
    model = next(m for m in c["models"] if m["id"] == model_id)
    cfg = inference_config(Path(model["config"]))
    if file_sha256(Path(version["manifest"])) != version["manifest_sha256"]:
        raise ValueError("GT manifest changed")
    manifest = load_segmentation_manifest(Path(version["manifest"]), require_checksums=True)
    transform = _transform(cfg.transforms, seed=c["runtime"]["seed"], augmentation_enabled=False)
    return SegmentationDataset(manifest, Path(version["data_root"]), transform=transform, verify_checksums=False)


def targets(c, output):
    torch.set_num_threads(c["runtime"]["cpu_threads"])
    reports, common = {}, None
    for version in c["gt_versions"]:
        dataset = dataset_for(c, version)
        rows, gallery = [], {}
        for i, sample in enumerate(dataset.manifest.samples):
            source_checks(dataset, i)
            item = dataset[i]
            target = item["mask"][0].numpy().astype(np.uint8)
            if target.shape != (128, 128):
                raise ValueError("Actual TRAIN target is not native128")
            with Image.open(dataset.paths[i][1]) as im:
                raw = np.asarray(im.convert("L"))>0
            n, _, stats, _ = cv2.connectedComponentsWithStats(target, connectivity=8)
            rows.append({"sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                "raw_foreground_pixels": int(raw.sum()), "raw_mask_pixels": raw.size,
                "native_foreground_pixels": int(target.sum()), "native_foreground_fraction": float(target.mean()),
                "native_components": n-1, "native_component_areas": stats[1:, cv2.CC_STAT_AREA].tolist(),
                "source_mask_sha256": sample.mask_sha256, "native_target_array_sha256": sha256(target.tobytes()).hexdigest()})
            if sample.sample_id in c["gallery_train_ids"]:
                key = sample.sample_id.replace(":", "__")
                save_binary(output/"gt"/version["id"]/(key+"-native.png"), target)
                with Image.open(dataset.paths[i][1]) as im:
                    im.convert("RGB").resize((256,256), Image.Resampling.NEAREST).save(output/"gt"/version["id"]/(key+"-raw.png"))
                if version["id"] == "pixel":
                    rgb_path = output/"rgb"/(key+".png")
                    rgb_path.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(dataset.paths[i][0]) as im:
                        im.convert("RGB").resize((256,256), Image.Resampling.BILINEAR).save(rgb_path)
                gallery[sample.sample_id] = {"dataset_index": i, "image": str(dataset.paths[i][0]),
                    "target": str(dataset.paths[i][1]), "target_sha256": sample.mask_sha256,
                    "native_target_array_sha256": rows[-1]["native_target_array_sha256"]}
            if (i+1)%250 == 0 or i+1 == len(dataset):
                print(f"[GT] {version['id']} {i+1}/{len(dataset)}", flush=True)
        identities = {r["sample_id"]:r["image_sha256"] for r in rows}
        if common is not None and common != identities:
            raise ValueError("GT families do not use identical TRAIN image population")
        common = identities
        reports[version["id"]] = {"source": version, "images": len(rows), "per_image": rows, "gallery": gallery,
            "mean_native_foreground_pixels": float(np.mean([r["native_foreground_pixels"] for r in rows])),
            "native_foreground_fraction": distribution([r["native_foreground_fraction"] for r in rows]),
            "raw_foreground_pixels_total": sum(r["raw_foreground_pixels"] for r in rows),
            "native_foreground_pixels_total": sum(r["native_foreground_pixels"] for r in rows),
            "native_components_per_image": distribution([r["native_components"] for r in rows]),
            "native_component_area": distribution([v for r in rows for v in r["native_component_areas"]])}
    write(output/"gt_statistics.json", {"status": "SUCCESS", "claim_label": "approximate reproduction",
        "scope": "all1255 official TRAIN images; actual unaugmented native transform; not segmentation evaluation",
        "versions": reports})


def render_overview(c, output, sid):
    key = sid.replace(":", "__")
    labels = [("pixel", "pixel_tt"), ("curved", "curved"), ("v4", "v4"),
              ("separated", "sep_447"), ("compact", "compact_361")]
    canvas = Image.new("RGB", (1180, 90+len(labels)*300), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 10), f"{sid} — actual GT and predictions at128; approximate reproduction", font=font(22), fill="black")
    for col, title in enumerate(["RGB input256", "GT before resize", "GT128 actually supervised", "Prediction128"]):
        draw.text((150+col*256, 52), title, font=font(17), fill="black")
    for row, (family, model) in enumerate(labels):
        top = 90+row*300
        draw.text((12, top+20), family, font=font(20), fill="black")
        record = next(m for m in c["models"] if m["id"] == model)
        draw.text((12, top+52), f"epoch{record['epoch']}", font=font(15), fill="black")
        paths = [output/"rgb"/(key+".png"), output/"gt"/family/(key+"-raw.png"),
                 output/"gt"/family/(key+"-native.png"), output/"predictions"/model/(key+".png")]
        for col, path in enumerate(paths):
            with Image.open(path) as im:
                image = im.convert("RGB").resize((248,248), Image.Resampling.NEAREST)
            canvas.paste(image, (150+col*256, top))
        draw.line((10,top+280,1170,top+280), fill=(180,180,180))
    canvas.save(output/(key+"-versions.png"))


def predictions(c, output, icdar13_data_root=None):
    rt = c["runtime"]
    torch.set_num_threads(rt["cpu_threads"])
    seed_everything(rt["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    gt = json.loads((output/"gt_statistics.json").read_text())
    common_gallery = gt["versions"]["pixel"]["gallery"]
    records = []
    for record in c["models"]:
        cfg = inference_config(Path(record["config"]))
        model = _model_from_config(cfg).eval()
        validate_checkpoint(Path(record["checkpoint"]), model, record["checkpoint_sha256"], cfg.architecture.identity)
        model.cuda()
        for sid in c["gallery_train_ids"]:
            image_path = Path(common_gallery[sid]["image"])
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
            with torch.inference_mode():
                native = (model(_preprocess(rgb).unsqueeze(0).cuda()).sigmoid()[0,0]>=c["probability_threshold"]).cpu().numpy().astype(np.uint8)
            save_binary(output/"predictions"/record["id"]/(sid.replace(":","__")+".png"), native)
            records.append({"model":record["id"], "sample_id":sid, "split":"TRAIN_visual_only",
                "image_path":str(image_path), "image_sha256":file_sha256(image_path),
                "native_prediction_array_sha256":sha256(native.tobytes()).hexdigest()})
        # One actual sample from each model's own training manifest, including
        # other-domain models. Never label a Total-Text reference as their GT.
        if record["family"] in ["compact", "separated"]:
            version = next(v for v in c["gt_versions"] if v["id"] == record["family"])
            manifest_path, data_root = Path(version["manifest"]), Path(version["data_root"])
        else:
            manifest_path = Path(cfg.runtime_overrides.manifest_root)/str(cfg.data.train.manifest)
            data_root = Path(cfg.runtime_overrides.manifest_root)
            if "icdar13" in record["id"]:
                if icdar13_data_root is None:
                    raise ValueError("Supply the separate ICDAR13 raw-data root for own-training cards")
                data_root = icdar13_data_root.resolve()
        manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
        dataset = SegmentationDataset(manifest, data_root,
            transform=_transform(cfg.transforms, seed=rt["seed"], augmentation_enabled=False), verify_checksums=False)
        source_checks(dataset, 0)
        item = dataset[0]
        with torch.inference_mode():
            native = (model(item["image"].unsqueeze(0).cuda()).sigmoid()[0,0]>=c["probability_threshold"]).cpu().numpy().astype(np.uint8)
        canvas = Image.new("RGB", (1000, 340), "white"); draw = ImageDraw.Draw(canvas)
        draw.text((10,8), f"{record['id']} epoch{record['epoch']} | actual TRAIN example {item['sample_id']}", font=font(20), fill="black")
        with Image.open(dataset.paths[0][0]) as im:
            panels = [im.convert("RGB").resize((256,256),Image.Resampling.BILINEAR),
                binary_image(item["mask"][0].numpy()).resize((256,256),Image.Resampling.NEAREST),
                binary_image(native).resize((256,256),Image.Resampling.NEAREST)]
        for col, (label, image) in enumerate(zip(["RGB256", "GT128 actually used", "Prediction128"], panels)):
            canvas.paste(image,(20+col*330,65)); draw.text((20+col*330,38),label,font=font(16),fill="black")
        own=output/"own_training"/(record["id"]+".png"); own.parent.mkdir(exist_ok=True)
        canvas.save(own)
        records.append({"model":record["id"],"role":"actual_own_TRAIN_example","sample_id":item["sample_id"],
            "manifest":str(manifest_path),"manifest_sha256":file_sha256(manifest_path),"data_root":str(data_root),
            "source_paths":[str(p) if p else None for p in dataset.paths[0]],
            "source_sha256":[file_sha256(p) if p else None for p in dataset.paths[0]]})
        del model, item, native, dataset
        gc.collect(); torch.cuda.empty_cache()
        print(f"[visual] {record['id']} complete",flush=True)
    for sid in c["gallery_train_ids"]:
        render_overview(c,output,sid)
    write(output/"visual_provenance.json",{"status":"SUCCESS","claim_label":"approximate reproduction",
        "examples":records,"threshold":c["probability_threshold"],"prediction_shape":[128,128],
        "precision":"float32","device":"cuda:0","batch_size":1,"input_shape":[1,3,256,256],
        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved(),
        "interpretation":"Unaugmented TRAIN examples for visual inspection; not independent performance evaluation"})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=["targets","predictions"],required=True)
    parser.add_argument("--icdar13-data-root",type=Path)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=="targets": targets(cfg,args.output)
    else: predictions(cfg,args.output,args.icdar13_data_root)
