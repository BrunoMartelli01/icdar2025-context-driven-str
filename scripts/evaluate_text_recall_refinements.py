"""Frozen Text+Kernel, native384 and crop-refinement ablations with real PARSeq."""

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from localize.word_crop_refinements import METHODS, prepare_recipes, resize_probability
from scripts.evaluate_text_recall_crops import REPO, completed_source, prepare_ocr_provider, summarize
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.evaluate_totaltext_word_kernel_parseq_val import _save_labels
from scripts.evaluate_word_multitask_crops import recognize, ratios, score_predictions, verify_geometry_exports, write_json
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from recognition.scene_text_v1_evaluation import box_iou
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.text_recall import TextRecallDataset, make_transform
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.word_multitask_evaluation import aggregate_word_metrics


CONFIG = REPO / "configs/approximate/text_recall_refinements_v1.json"


def save_instance_labels(path, labels):
    """Reuse native256 writer; explicitly extend lossless export to native384."""
    if labels.shape == (256, 256):
        return _save_labels(path, labels)
    if (labels.shape != (384, 384) or labels.dtype.kind not in "iu"
            or labels.min() < 0 or labels.max() > 65535):
        raise ValueError("Expected losslessly encodable native384 instance IDs")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as opened:
            if not np.array_equal(np.asarray(opened), labels):
                raise ValueError("Cached native384 instance map differs")
    else:
        Image.fromarray(labels.astype(np.uint16)).save(path)
    return sha256_file(path)


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO / "configs/approximate" not in path.parents:
        raise ValueError("Explicit APPROXIMATE namespace required")
    cfg = json.loads(path.read_text())
    if (cfg["schema_version"] != 1 or cfg["reproduction_mode"] != mode
            or cfg["result_class"] != "approximate reproduction" or tuple(cfg["methods"]) != METHODS
            or cfg["sources"] != ["reference512", "hybrid256", "hybrid384"]
            or cfg["canonical_size"] != 256 or cfg["fresh_reference_ocr"] is not True
            or cfg["crop"] != {"padding": 0, "min_area": 0, "max_instances": None}
            or cfg["recognition"] != {"decode_ar": True, "refine_iters": 1, "batch_size": 8,
                                      "casefold": True, "symbol_normalized_secondary_metric": True}
            or not 0 < cfg["growth"]["low_threshold"] <= min(cfg["reference_threshold"], cfg["text_threshold"])
            or not 0 < cfg["fallback_kernel_threshold"] <= cfg["postprocessing"]["kernel_threshold"] < 1):
        raise ValueError("Unsupported frozen crop-refinement protocol")
    source = REPO / cfg["source_config"]
    if sha256_file(source) != cfg["source_config_sha256"]:
        raise ValueError("Qualified dataset/preprocessing source configuration changed")
    prepare_recipes(np.zeros((2, 16, 16), np.float32), np.ones((16, 16), bool),
                    (16, 16), cfg, text_threshold=cfg["text_threshold"])
    return cfg, json.loads(source.read_text())


def qualify_sources(args, cfg, source):
    reports = {}
    for name in ("reference", "text", "kernel"):
        root = getattr(args, name + "_run")
        report, validation = completed_source(root)
        if (report["checkpoint"]["sha256"] != cfg[name + "_checkpoint_sha256"]
                or report["run_contract"]["validation_ids_sha256"] != cfg["validation_ids_sha256"]):
            raise ValueError("Frozen checkpoint/split differs: " + name)
        reports[name] = {"root": root, "report": report, "validation": validation}
    contract = reports["text"]["report"]["run_contract"]
    if contract["branch"] != "tversky768" or reports["text"]["validation"]["operating_point"]["threshold"] != cfg["text_threshold"]:
        raise ValueError("Expected unchanged selected Tversky768 operating point")
    reference = reports["reference"]["report"]["run_contract"]
    if (reference["branch"] != "control" or reference["resolved_config"]["postprocessing"]["text_threshold"] != cfg["reference_threshold"]
            or reports["kernel"]["report"]["run_contract"]["variant"] != "text_kernel"):
        raise ValueError("Expected selected joint Text+Kernel control and original two-head kernels")
    target = Path(contract["target_root"])
    dataset = contract["config"]["dataset"]
    manifest_path = target / dataset["manifest"]
    if sha256_file(manifest_path) != dataset["manifest_sha256"]:
        raise ValueError("Qualified TRAIN manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    _, indices, split_sha = _split_indices(manifest, source["training"]["seed"], dataset["validation_images"])
    ids = [manifest.samples[i].sample_id for i in indices]
    if manifest.dataset != "total_text" or manifest.split_role != "train" or len(ids) != 125 or split_sha != cfg["validation_ids_sha256"]:
        raise ValueError("Complete frozen TRAIN holdout required")
    for name, item in reports.items():
        key = "exports" if name == "text" else "per_image"
        if [row["sample_id"] for row in item["validation"][key]] != ids:
            raise ValueError("Source sample ordering differs: " + name)
        item["rows"] = {row["sample_id"]: row for row in item["validation"][key]}
    return reports, target, manifest, indices


def frozen_map(item, sample_id, kind):
    row = item["rows"][sample_id]
    expected = row["files"]["maps.npz"] if kind == "text" else row["maps_sha256"] if kind == "reference" else row["exports"]["maps.npz"]
    path = item["root"] / "validation_exports" / sample_id.replace(":", "__") / "maps.npz"
    if sha256_file(path) != expected:
        raise ValueError("Frozen source map changed: " + str(path))
    with np.load(path, allow_pickle=False) as value:
        return {k: value[k].copy() for k in value.files}, expected


def build_geometry(args, cfg, source, reports, target, manifest, indices, protocol):
    output = args.output_root
    cached = output / "geometry_results.json"
    if cached.exists():
        value = json.loads(cached.read_text())
        if value["status"] != "SUCCESS" or value["protocol"] != protocol:
            raise ValueError("Existing geometry contract differs; select a NEW output directory")
        verify_geometry_exports(value, output)
        return value
    branches = {name + "__selected": {method: {"per_image": [], "native_totals": Counter(), "original_totals": Counter()}
                 for method in METHODS} for name in cfg["sources"]}
    branches["oracle__diagnostic"] = {"gt_boxes": {"per_image": [], "native": {"merge_rate": None, "split_rate": None}, "original_totals": Counter()}}
    native_dataset = TextRecallDataset(SegmentationDataset(manifest, target,
        transform=make_transform(source, input_size=768, train=False)))
    replays = Counter()
    for position, index in enumerate(indices, 1):
        sample = manifest.samples[index]; sid = sample.sample_id; key = sid.replace(":", "__")
        image_path = target / sample.image_path
        if sha256_file(image_path) != sample.image_sha256:
            raise ValueError("RGB input changed")
        with Image.open(image_path) as image:
            rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
        image_wh = (rgb.shape[1], rgb.shape[0])
        maps = {}; digests = {}
        for name, item in reports.items():
            maps[name], digests[name] = frozen_map(item, sid, name)
        for name in ("reference", "kernel"):
            if any(not np.array_equal(maps[name][k], maps["text"][k]) for k in ("owners", "validity", "canvas", "expected_owner_ids")):
                raise ValueError("Source reference alignment differs: " + sid)
        canvas384 = letterbox_canvas_mask(image_wh, native_size=384).numpy().astype(bool)
        inputs = {
            "reference512": (maps["reference"]["probabilities"], maps["reference"]["canvas"], cfg["reference_threshold"]),
            "hybrid256": (np.stack([maps["text"]["probabilities"][0], maps["kernel"]["probabilities"][1]]), maps["text"]["canvas"], cfg["text_threshold"]),
            "hybrid384": (np.stack([maps["text"]["native_probabilities"][0], resize_probability(maps["kernel"]["probabilities"][1], 384)]), canvas384, cfg["text_threshold"])}
        frozen = []
        # All inference recipes are complete before loading original GT or native targets.
        for name, (p, canvas, threshold) in inputs.items():
            recipes = prepare_recipes(p, canvas, image_wh, cfg, text_threshold=threshold)
            if name == "reference512":
                if not np.array_equal(recipes["baseline"]["labels"], maps["reference"]["labels"]):
                    raise RuntimeError("Joint reference instance replay differs")
                replays["reference_masks"] += 1
            for method, recipe in recipes.items():
                base = Path(name + "__selected") / method / "samples" / key
                save_instance_labels(output / base / "instances.png", recipe["labels"])
                proposals = []
                for prop in recipe["proposals"]:
                    x1, y1, x2, y2 = prop["bbox"]
                    proposals.append({**prop, "crop_rgb_sha256": crop_array_hash(rgb[y1:y2, x1:x2])})
                content = {"sample_id": sid, "result_class": "approximate reproduction", "image_path": str(image_path),
                    "image_sha256": sample.image_sha256, "source_maps_sha256": digests, "native_size": canvas.shape[0],
                    "threshold": threshold, "proposals": proposals, "audit": recipe["audit"]}
                write_json(output / base / "proposals.json", content)
                files = {str(base / n): sha256_file(output / base / n) for n in ("instances.png", "proposals.json")}
                frozen.append((name, method, recipe["labels"], content, files))
        gt, annotation_sha = _train_gt(sample, target, image_wh)
        native_batch = native_dataset[index]
        if native_batch["expected_words"] != len(maps["text"]["expected_owner_ids"]):
            raise ValueError("Original owner population changed")
        references = {256: (maps["text"]["owners"], maps["text"]["validity"]),
                      384: (native_batch["owners_native"].numpy(), native_batch["validity"][0].numpy().astype(bool))}
        annotation_path = target / "datasets/total_text/train/annotations" / (sid.split(":", 1)[1] + ".txt")
        for name, method, labels, content, files in frozen:
            owners, valid = references[content["native_size"]]
            ignored = [int(v) for v in np.unique(labels[labels > 0]) if not np.any((labels == v) & valid)]
            care = labels.copy(); care[np.isin(care, ignored)] = 0
            native = instance_geometry(care, owners, maps["text"]["expected_owner_ids"])
            original = score_predictions([p for p in content["proposals"] if p["instance_id"] not in ignored], gt, text_available=False)
            if name == "reference512" and method == "baseline":
                old = reports["reference"]["rows"][sid]
                if any(old[k] != v for k, v in native.items()):
                    raise RuntimeError("Reference native geometry replay differs")
                replays["reference_geometry"] += 1
            part = branches[name + "__selected"][method]
            part["per_image"].append({**content, "ground_truth": gt, "annotation_path": str(annotation_path),
                "annotation_sha256": annotation_sha, "ignored_prediction_ids": ignored,
                "native_geometry": native, "original_geometry": original, "exports": files})
            part["native_totals"].update(native); part["original_totals"].update(original)
        # Oracle crops are a distinct diagnostic; never included in inference recipe inputs.
        proposals = []
        for gi, g in enumerate(gt):
            x1, y1, x2, y2 = [int(v) for v in g["bbox"]]
            if g["bbox"] != [x1, y1, x2, y2]:
                raise ValueError("GT bbox requires an explicit fractional-coordinate crop protocol")
            proposals.append({"instance_id": gi + 1, "bbox": [x1, y1, x2, y2],
                "crop_rgb_sha256": crop_array_hash(rgb[y1:y2, x1:x2]), "gt_crop_index": gi})
        base = Path("oracle__diagnostic/gt_boxes/samples") / key
        content = {"sample_id": sid, "result_class": "approximate reproduction", "oracle": True,
            "image_path": str(image_path), "image_sha256": sample.image_sha256, "proposals": proposals}
        write_json(output / base / "proposals.json", content)
        original = score_predictions(proposals, gt, text_available=False)
        part = branches["oracle__diagnostic"]["gt_boxes"]
        part["per_image"].append({**content, "ground_truth": gt, "annotation_path": str(annotation_path),
            "annotation_sha256": annotation_sha, "ignored_prediction_ids": [], "original_geometry": original,
            "exports": {str(base / "proposals.json"): sha256_file(output / base / "proposals.json")}})
        part["original_totals"].update(original)
        if position % 25 == 0:
            print(f"[refinements] {position}/125 images", flush=True)
    for methods in branches.values():
        for part in methods.values():
            if len(part["per_image"]) != 125 or part["original_totals"]["gt_words"] != 983:
                raise RuntimeError("Complete population/denominator differs")
            if "native_totals" in part:
                part["native"] = aggregate_word_metrics(dict(part.pop("native_totals")))
            part["original"] = ratios(dict(part.pop("original_totals")))
    thresholds = {name + "__selected": cfg["reference_threshold"] if name == "reference512" else cfg["text_threshold"] for name in cfg["sources"]}
    thresholds["oracle__diagnostic"] = 0.0
    result = {"status": "SUCCESS", "result_class": "approximate reproduction", "reproduction_mode": "approximate",
        "protocol": protocol, "branches": branches, "thresholds": thresholds, "reference_replays": dict(replays),
        "segmentation_inference": "none; frozen selected maps reused"}
    write_json(cached, result)
    return result


def diagnostic(recognition):
    oracle = {r["sample_id"]: r for r in recognition["branches"]["oracle__diagnostic"]["gt_boxes"]["per_image"]}
    result = {}
    for name, methods in recognition["branches"].items():
        if name == "oracle__diagnostic":
            continue
        result[name] = {}
        for method, part in methods.items():
            counts = Counter(); words = []
            for row in part["per_image"]:
                gt = row["ground_truth"]; ok = set(row["correct_casefold_gt_indices"])
                preds = [p for p in row["predictions_all"] if p["instance_id"] not in row["ignored_prediction_ids"]]
                gt_texts = {p["gt_crop_index"]: p["text"] for p in oracle[row["sample_id"]]["predictions_all"]}
                if set(gt_texts) != set(range(len(gt))):
                    raise RuntimeError("GT crop identity population differs")
                for gi, g in enumerate(gt):
                    viable = [p for p in preds if box_iou(p["bbox"], g["bbox"]) >= .5]
                    exact = any(bool(g["text"]) and p["text"].casefold() == g["text"].casefold() for p in viable)
                    category = ("correct" if gi in ok else "no_iou50_candidate" if not viable
                                else "one_to_one_competition" if exact else "viable_crop_wrong_text")
                    counts[category] += 1
                    gt_correct = bool(g["text"]) and gt_texts[gi].casefold() == g["text"].casefold()
                    if gi not in ok:
                        counts[category + ("_gt_crop_correct" if gt_correct else "_gt_crop_wrong")] += 1
                    words.append({"sample_id": row["sample_id"], "gt_index": gi, "gt_text": g["text"],
                        "gt_bbox": g["bbox"], "category": category, "gt_crop_text": gt_texts[gi],
                        "gt_crop_correct": gt_correct, "viable_prediction_texts": [p["text"] for p in viable]})
            result[name][method] = {"counts": dict(counts), "per_word": words}
    fixed_gt_correct = sum(bool(g["text"]) and p["text"].casefold() == g["text"].casefold()
        for row in oracle.values() for g, p in zip(row["ground_truth"], row["predictions_all"]))
    return {"result_class": "approximate reproduction", "protocol": "GT bbox crops indexed by annotation identity; diagnostic only, not detector output",
        "gt_box_fixed_pair_correct": fixed_gt_correct,
        "gt_words": sum(len(row["ground_truth"]) for row in oracle.values()), "branches": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["approximate"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    for name in ("reference-run", "text-run", "kernel-run", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--action", required=True, choices=["preflight", "geometry", "recognize", "all"])
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    for name in ("parseq-source", "parseq-source-archive", "parseq-checkpoint"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--reuse-run-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    for name in ("config", "reference_run", "text_run", "kernel_run", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.cpu_threads < 1:
        raise ValueError("CPU thread count must be positive")
    torch.set_num_threads(args.cpu_threads)
    cfg, source = read_config(args.config, args.mode)
    reports, target, manifest, indices = qualify_sources(args, cfg, source)
    inputs = [target, args.reference_run, args.text_run, args.kernel_run, *args.reuse_run_root]
    if args.output_root == REPO or any(args.output_root == p.resolve() or p.resolve() in args.output_root.parents or args.output_root in p.resolve().parents for p in inputs):
        raise ValueError("Use an isolated output directory")
    files = ["scripts/evaluate_text_recall_refinements.py", "localize/word_crop_refinements.py",
        "scripts/evaluate_text_recall_crops.py", "scripts/evaluate_word_multitask_crops.py",
        "localize/word_guided_contours.py", "localize/word_multitask.py", "localize/crop_recovery.py",
        "segmentation/text_recall.py", "segmentation/totaltext_word_kernel.py", "segmentation/data.py",
        "segmentation/totaltext_image_frequency_augment.py", "recognition/parseq.py",
        "recognition/scene_text_v1_evaluation.py", "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py",
        "scripts/evaluate_totaltext_highres_contour_padding.py", "scripts/evaluate_totaltext_word_kernel_parseq_val.py",
        "segmentation/word_multitask_evaluation.py"]
    protocol = {"schema_version": 1, "result_class": "approximate reproduction", "reproduction_mode": "approximate",
        "source_seed": source["training"]["seed"], "resolved_config": cfg, "config_sha256": sha256_file(args.config),
        "source_sha256": {n: sha256_file(REPO / n) for n in files}, "target_root": str(target),
        "manifest_sha256": reports["text"]["report"]["run_contract"]["config"]["dataset"]["manifest_sha256"],
        "sources": {n: {"root": str(v["root"]), "report_sha256": sha256_file(v["root"] / "run_manifest.json"),
            "validation_sha256": sha256_file(v["root"] / "validation.json"), "checkpoint": v["report"]["checkpoint"]} for n, v in reports.items()},
        "matching": "original RGB exclusive bbox maximum one-to-one IoU>=.50 AND exact casefold, punctuation retained",
        "ignore": "exclude wholly invalid predicted instances after inference; keep partial boxes complete",
        "native_metrics": "384 native topology has a different raster; original-coordinate OCR/geometry are common primary metrics",
        "oracle": "separate GT bbox diagnostic, never an inference output"}
    if args.action == "preflight":
        print(json.dumps({"status": "PASS", "images": 125, "gt_words": 983, "pipelines": 15, "gt_box_diagnostic": True})); return
    output = args.output_root; output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(output / "geometry_protocol.json", protocol)
        try:
            geometry = build_geometry(args, cfg, source, reports, target, manifest, indices, protocol)
            if args.action in ("all", "recognize"):
                # Every reference crop is rerun on CPU, even when it appears in another arm/cache.
                fresh = {p["crop_rgb_sha256"] for part in geometry["branches"]["reference512__selected"].values()
                    for row in part["per_image"] for p in row["proposals"]}
                reuse_geometry = {"branches": {"reuse": {"reuse": {"per_image": [{"proposals": [p
                    for methods in geometry["branches"].values() for part in methods.values()
                    for row in part["per_image"] for p in row["proposals"] if p["crop_rgb_sha256"] not in fresh]}]}}}}
                args.reuse_ocr = [prepare_ocr_provider(output, args.reuse_run_root, reuse_geometry)] if args.reuse_run_root else []
                recognition = recognize(args, cfg, geometry)
                result = summarize(output, geometry, recognition)
                diagnosis = diagnostic(recognition)
                if diagnosis["gt_words"] != 983:
                    raise RuntimeError("GT crop diagnostic denominator differs")
                write_json(output / "gt_crop_diagnostic.json", diagnosis)
            else:
                result = {"status": "SUCCESS", "pipelines": 15, "gt_box_diagnostic": True}
        except Exception as error:
            write_json(output / "failure.json", {"status": "FAILED", "result_class": "approximate reproduction",
                "type": type(error).__name__, "message": str(error)})
            raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
