"""TRAIN-only CPU geometry counterfactual; never instantiate a learned model.

Use the actual frozen localizers on cached predictions and qualified GT masks.
GT word annotations enter only evaluation, not proposal generation. Results are
not OCR accuracy or a rigorous upper bound (grouping need not be monotonic).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from diagnostics.repository_contours_train import digest, read_shard, unpack_mask
from localize.grouped_word_blocks import group_word_blocks
from localize.mask_to_blocks import BlockParam, localize_mask
from localize.native_repository_contours import localize_native_repository_contours, remap_native_box
from localize.practical_remap_protocol import _box_iou, proposal_geometry, train_task22_word_boxes
from localize.repository_contours import RepositoryContourParams, audit_repository_contours
from localize.v1_local_rescue import LocalRescueParams, local_rescue
from pipeline.frozen_segmentation import backproject_binary_mask
from scripts.diagnose_train_gtbox_vs_v1_parseq import _wordness
from scripts.export_train_v1_parseq_inputs import _params
from segmentation.data import load_segmentation_manifest, icdar13_task22_rgb_to_target_and_validity
from segmentation.evaluate import resize_binary_target_to_native_logits


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path, expected):
    value = sha(path)
    if value != expected:
        raise RuntimeError(f"checksum mismatch: {path}: {value} != {expected}")
    return value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_native(cache_path, contract, sample_id, image_sha, checkpoint_sha):
    value = read_shard(cache_path, contract)
    if value["sample_id"] != sample_id or value["image_sha256"] != image_sha:
        raise RuntimeError(f"cache sample/image mismatch: {sample_id}")
    meta = value["segmentation_metadata"]
    expected = {"native_mask_shape": [128, 128], "threshold": .5,
                "threshold_comparison": "greater_than_or_equal",
                "probability_activation": "sigmoid"}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"cache segmentation protocol mismatch: {sample_id}")
    if meta["checkpoint"]["sha256"] != checkpoint_sha:
        raise RuntimeError("cache checkpoint mismatch")
    if meta["native_mask_sha256"] != value["array_sha256"]:
        raise RuntimeError("cache native mask metadata mismatch")
    return unpack_mask(value), value


def record(boxes, before_cap, raw_count, merges=0, multi=0):
    return {"boxes": [list(b) for b in boxes],
            "before_cap_boxes": [list(b) for b in before_cap],
            "raw_count": raw_count, "merges": merges, "multi": multi}


def original_proposals(mask, image, contour_params, v1_params, rescue_params):
    """No word GT argument: all decisions use only input mask and frozen params."""
    current = localize_mask(mask, image, BlockParam(
        contour_params.min_area, contour_params.pad, v1_params.connectivity,
        v1_params.tie_break, contour_params.nmax))
    kept = [b.padded_bbox for b in current.retained]
    before = kept + [b.padded_bbox for b in current.rejected if b.reason == "top_10_limit"]
    raw = len(current.retained) + len(current.rejected)
    out = {"CURRENT_ALG1": record(kept, before, raw)}
    repo = audit_repository_contours(mask, contour_params)
    kept = [b["padded_bbox"] for b in repo["retained"]]
    before = kept + [b["padded_bbox"] for b in repo["rejected"] if b["reason"] == "top_n_limit"]
    out["REPOSITORY_CONTOURS"] = record(kept, before, repo["raw_external_contours"])
    v1 = group_word_blocks(mask, image, v1_params)
    groups = v1.groups_before_cap
    out["PRACTICAL_REMAP_V1"] = record(
        [b.padded_bbox for b in v1.retained], [g["padded_bbox"] for g in groups], raw,
        sum(len(g["raw_component_ids"]) - 1 for g in groups),
        sum(len(g["raw_component_ids"]) > 1 for g in groups))
    rescued = local_rescue(mask, image, v1_params, rescue_params,
                          {r.component_id: r.bbox for r in v1.raw_components})
    groups = rescued.groups_before_cap
    out["V1_1_RESCUE"] = record(
        [b.padded_bbox for b in rescued.retained], [g["padded_bbox"] for g in groups], raw,
        sum(len(g["raw_component_ids"]) - 1 for g in groups),
        sum(len(g["raw_component_ids"]) > 1 for g in groups))
    return out


def native_proposals(mask, image_hw, params):
    result = localize_native_repository_contours(mask, image_hw, params)
    native = result["native_localization"]
    kept = [b["padded_bbox_original"] for b in result["retained"]]
    before = kept + [remap_native_box(b["padded_bbox"], image_hw)
                     for b in native["rejected"] if b["reason"] == "top_n_limit"]
    return record(kept, before, native["raw_external_contours"])


def topology(mask, original_hw, gt_boxes):
    """Pixel-intersection counts in mapped half-open word rectangles, raw CC8."""
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    h, w = mask.shape
    oh, ow = original_hw
    words = []
    for x1, y1, x2, y2 in gt_boxes:
        x1, x2 = x1 * w // ow, (x2 * w + ow - 1) // ow
        y1, y2 = y1 * h // oh, (y2 * h + oh - 1) // oh
        ids, counts = np.unique(labels[y1:y2, x1:x2], return_counts=True)
        counts = counts[ids > 0]
        words.append({"components_intersecting_word": len(counts),
                      "foreground_pixels_in_word": int(counts.sum()),
                      "largest_fraction": float(counts.max() / counts.sum()) if len(counts) else None})
    return {"raw_components": n - 1, "words": words}


def compare_success(before, after):
    if len(before) != len(after):
        raise ValueError("unaligned word population")
    return {"recovered": sum(a < .5 <= b for a, b in zip(before, after)),
            "lost": sum(b < .5 <= a for a, b in zip(before, after)),
            "both_success": sum(min(a, b) >= .5 for a, b in zip(before, after)),
            "both_failure": sum(max(a, b) < .5 for a, b in zip(before, after))}


def summarize(records, gt_by_sample, methods, conditions):
    summaries = {}
    per_word = []
    for method in methods:
        summaries[method] = {}
        for condition in conditions:
            if method not in records[0]["conditions"][condition]:
                continue
            geometry = []
            categories = Counter()
            before25 = before50 = before75 = cap_losses = 0
            for sample in records:
                words = gt_by_sample[sample["sample_id"]]
                gt = [w["bbox"] for w in words]
                result = sample["conditions"][condition][method]
                boxes, before = result["boxes"], result["before_cap_boxes"]
                geometry.append((boxes, gt, len(before), result["merges"], result["multi"]))
                for box in boxes:
                    categories[_wordness(box, words)[0]] += 1
                for index, word in enumerate(words):
                    best = max((_box_iou(b, word["bbox"]) for b in boxes), default=0.)
                    precap = max((_box_iou(b, word["bbox"]) for b in before), default=0.)
                    before25 += precap >= .25
                    before50 += precap >= .5
                    before75 += precap >= .75
                    cap_losses += best < .5 <= precap
                    per_word.append({"sample_id": sample["sample_id"], "gt_index": index,
                                     "text": word["text"], "gt_bbox": word["bbox"],
                                     "condition": condition, "method": method,
                                     "best_iou": best, "best_iou_before_cap": precap})
            summaries[method][condition] = {
                "geometry": proposal_geometry(geometry).as_dict(),
                "wordness": {k: categories[k] for k in
                             ("WORD_GOOD", "WORD_PARTIAL", "MULTIWORD_MERGE", "UNMATCHED")},
                "before_cap_gt_iou_ge_0_25": int(before25),
                "before_cap_gt_iou_ge_0_50": int(before50),
                "before_cap_gt_iou_ge_0_75": int(before75),
                "gt_losses_due_to_top10_at_0_50": int(cap_losses)}
    return summaries, per_word


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source-report", "train-manifest", "data-root", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(f"Refusing to replace diagnostic: {a.output_dir}")
    config = json.loads(a.config.read_text())
    if config["scope"] != {"split": "TRAIN", "images": 229, "gt_words": 849}:
        raise RuntimeError("requires frozen TRAIN229/849 scope")
    verify_file(a.source_report, config["source_report_sha256"])
    verify_file(a.train_manifest, config["train_manifest_sha256"])
    source = json.loads(a.source_report.read_text())
    if source["protocol"]["scope"] != config["scope"]:
        raise RuntimeError("source report is not TRAIN229/849")
    manifest = load_segmentation_manifest(a.train_manifest, require_checksums=False)
    if manifest.split_role != "train" or len(manifest.samples) != 229:
        raise RuntimeError("official TRAIN229 only")
    files = source["protocol"]["files"]
    # Check the implementations and configs responsible for frozen controls.
    checks = ["localize/mask_to_blocks.py", "localize/grouped_word_blocks.py",
              "localize/repository_contours.py", "localize/practical_remap_protocol.py",
              "pipeline/frozen_segmentation.py", "scripts/diagnose_train_gtbox_vs_v1_parseq.py",
              "scripts/export_train_v1_parseq_inputs.py", "diagnostics/repository_contours_train.py",
              config["v1_config"], config["repository_config"]]
    for path in checks:
        verify_file(path, files[str(Path(path).resolve())])
    contour_params = RepositoryContourParams(**json.loads(Path(config["repository_config"]).read_text())["parameters"])
    native_params = RepositoryContourParams(**json.loads(Path(config["native_repository_config"]).read_text())["parameters"])
    v1_params = _params(Path(config["v1_config"]))
    rescue_params = LocalRescueParams(**config["rescue_parameters"])
    gt_by_sample = defaultdict(list)
    historical_boxes = defaultdict(list)
    for word in source["per_word"]:
        if word["gt_index"] != len(gt_by_sample[word["sample_id"]]):
            raise RuntimeError("GT word ordering mismatch")
        gt_by_sample[word["sample_id"]].append({"bbox": word["bbox"], "text": word["text"]})
    for crop in source["per_crop"]:
        historical_boxes[(crop["sample_id"], crop["method"])].append(crop["bbox"])
    if len(source["per_word"]) != 849 or set(gt_by_sample) != {s.sample_id for s in manifest.samples}:
        raise RuntimeError("TRAIN GT population mismatch")
    protocol = {**config, "config_sha256": sha(a.config),
                "parameters": {"contours": asdict(contour_params), "native_contours": asdict(native_params),
                               "v1": asdict(v1_params), "rescue": asdict(rescue_params)},
                "device": "cpu", "models_loaded": False, "ocr_accuracy": None,
                "metric_policy": "exclusive xyxy; best-IoU coverage plus existing greedy one-to-one proposal precision/recall/F1; wordness MULTIWORD first if two GT boxes each covered >=.25",
                "interpretation": "counterfactual localization with perfect pixel targets, not guaranteed upper bound nor word-level masks",
                "source_paths": {"source_report": str(a.source_report.resolve()), "train_manifest": str(a.train_manifest.resolve()), "data_root": str(a.data_root.resolve())},
                "code_sha256": {path: sha(path) for path in checks + [__file__,
                    "localize/native_repository_contours.py", "localize/v1_local_rescue.py",
                    "segmentation/data.py", "segmentation/evaluate.py", config["native_repository_config"]]},
                "versions": {"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__}}
    a.output_dir.mkdir(parents=True)
    # Predeclare the complete protocol before reading/evaluating any mask.
    save_json(a.output_dir / "protocol.json", protocol)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    records = []
    for index, sample in enumerate(manifest.samples, 1):
        if index == 1 or index % 10 == 0:
            print(f"[GT-mask oracle CPU] {index}/229 sample={sample.sample_id}", flush=True)
        for rel in (sample.image_path, sample.mask_path, sample.validity_annotation_path):
            verify_file(a.data_root / rel, source["protocol"]["dataset_files"][str(rel)]["sha256"])
        image_sha = source["protocol"]["dataset_files"][str(sample.image_path)]["sha256"]
        pred, cache = checked_native(a.source_report.parent / "masks" / f"{sample.sample_id}.json",
            source["contract_sha256"], sample.sample_id, image_sha, config["checkpoint_sha256"])
        with Image.open(a.data_root / sample.image_path) as rgb:
            image = np.asarray(rgb.convert("RGB"))
        with Image.open(a.data_root / sample.mask_path) as rgb_mask:
            target, validity = icdar13_task22_rgb_to_target_and_validity(
                rgb_mask, (a.data_root / sample.validity_annotation_path).read_text(encoding="utf-8-sig"))
        full = (np.asarray(target) > 0).astype(np.uint8)
        if full.shape != image.shape[:2]:
            raise RuntimeError("GT/RGB shape mismatch")
        native = resize_binary_target_to_native_logits(target).numpy().astype(np.uint8)
        conditions = {}
        for name, binary in (("PRED128", pred), ("GT128", native), ("GT_FULLRES", full)):
            mask = full if name == "GT_FULLRES" else backproject_binary_mask(binary, full.shape)
            if name == "PRED128" and hashlib.sha256(mask.tobytes()).hexdigest() != cache["segmentation_metadata"]["output_mask_sha256"]:
                raise RuntimeError("backprojected prediction differs from actual pipeline")
            methods = original_proposals(mask, image, contour_params, v1_params, rescue_params)
            if name != "GT_FULLRES":
                methods["REPOSITORY_NATIVE128"] = native_proposals(binary, full.shape, native_params)
            conditions[name] = methods
        for method in ("CURRENT_ALG1", "REPOSITORY_CONTOURS", "PRACTICAL_REMAP_V1"):
            if conditions["PRED128"][method]["boxes"] != historical_boxes[(sample.sample_id, method)]:
                raise RuntimeError(f"frozen proposal mismatch {sample.sample_id}/{method}")
        # Word GT is consulted only AFTER every proposal has been generated.
        gt = [w.bbox for w in train_task22_word_boxes(a.data_root / sample.mask_path,
                                                    a.data_root / sample.validity_annotation_path)]
        if [list(b) for b in gt] != [w["bbox"] for w in gt_by_sample[sample.sample_id]]:
            raise RuntimeError("TRAIN word GT changed")
        records.append({"sample_id": sample.sample_id, "image_hw": list(full.shape),
                        "prediction_native_sha256": cache["array_sha256"],
                        "gt_native_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
                        "gt_fullres_sha256": hashlib.sha256(full.tobytes()).hexdigest(),
                        "dont_care_pixels_source": int(np.count_nonzero(np.asarray(validity) == 0)),
                        "conditions": conditions,
                        "topology": {k: topology(v, full.shape, gt) for k, v in
                                     (("PRED128", pred), ("GT128", native), ("GT_FULLRES", full))}})
    summary, rows = summarize(records, gt_by_sample, config["methods"], config["conditions"])
    for method in ("CURRENT_ALG1", "REPOSITORY_CONTOURS", "PRACTICAL_REMAP_V1"):
        for key in ("geometry", "wordness"):
            if summary[method]["PRED128"][key] != source["summary"][method][key]:
                raise RuntimeError(f"historical {key} control mismatch: {method}")
    transitions = {}
    for method in config["methods"]:
        values = {c: [r["best_iou"] for r in rows if r["method"] == method and r["condition"] == c]
                  for c in summary[method]}
        transitions[method] = {f"{before}_to_{after}": compare_success(values[before], values[after])
                              for before, after in (("PRED128", "GT128"), ("GT128", "GT_FULLRES"))
                              if before in values and after in values}
    topology_summary = {}
    for condition in config["conditions"]:
        counts = [w["components_intersecting_word"] for r in records for w in r["topology"][condition]["words"]]
        topology_summary[condition] = {"raw_components": sum(r["topology"][condition]["raw_components"] for r in records),
            "words_with_0_components": counts.count(0), "words_with_1_component": counts.count(1),
            "words_with_2_components": counts.count(2), "words_with_ge_3_components": sum(c >= 3 for c in counts),
            "median_components_per_word": float(np.median(counts))}
    with (a.output_dir / "per-word.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol": protocol, "summary": summary, "transitions": transitions,
              "topology": topology_summary, "samples": records,
              "validation": {"images": len(records), "gt_words": 849,
                  "all_dataset_files_hash_verified": True, "cached_checkpoint_identity_verified": True,
                  "all_229_masks_and_3_historical_proposal_controls_exact": True},
              "csv_sha256": sha(a.output_dir / "per-word.csv")}
    save_json(a.output_dir / "results.json", result)
    print(json.dumps({"summary": summary, "transitions": transitions, "topology": topology_summary}, indent=2))
    print(f"JSON {a.output_dir / 'results.json'} SHA256 {sha(a.output_dir / 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
