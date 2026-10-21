"""Reuse trained kernels/coherent votes to separate the same frozen text masks."""

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from localize.center_guided_word_crops import center_guided_masks
from localize.word_multitask import parseq_word_proposals, word_instances
from scripts.evaluate_text_recall_crops import (
    BRANCHES, REPO, completed_source, geometry_protocol, load_sources, prepare_ocr_provider,
    read_config as read_base_config, summarize,
)
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.evaluate_totaltext_word_kernel_parseq_val import _save_labels
from scripts.evaluate_word_multitask_crops import recognize, ratios, score_predictions, verify_geometry_exports, write_json
from scripts.prune_segmentation_checkpoints import sha256_file
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.word_multitask_evaluation import aggregate_word_metrics


CONFIG = REPO / "configs/approximate/text_recall_learned_crops_v1.json"
METHODS = ("kernel_watershed", "offset_votes_v2")


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO / "configs/approximate" not in path.parents:
        raise ValueError("Explicit APPROXIMATE namespace required")
    added = json.loads(path.read_text())
    if (added["schema_version"] != 1 or added["reproduction_mode"] != mode
        or added["result_class"] != "approximate reproduction" or tuple(added["methods"]) != METHODS):
        raise ValueError("Unsupported learned-cue transfer protocol")
    source = REPO / added["source_config"]
    if sha256_file(source) != added["source_config_sha256"]:
        raise ValueError("Original comparison configuration changed")
    base = read_base_config(source, mode)
    cfg = {**base, **added, "scientific_choices": base["scientific_choices"] + added["scientific_choices"]}
    prepare_recipes(np.zeros((16, 16)), np.zeros((16, 16)), np.zeros((2, 16, 16)),
                    np.ones((16, 16), bool), (16, 16), cfg, threshold=cfg["common_threshold"])
    return cfg


def prepare_recipes(probability, kernel, offsets, canvas, image_wh, config, *, threshold):
    """Only inference signals enter; neither GT, care validity nor OCR can seed words."""
    post = {**config["kernel_postprocessing"], "text_threshold": float(threshold)}
    labels, audit, core = word_instances(np.stack([probability, kernel]), canvas, post)
    joint = center_guided_masks(probability, offsets, canvas, {**config["offset"], "text_threshold": float(threshold)})
    if not np.array_equal(joint["preserved_labels"] > 0, core):
        raise RuntimeError("Learned cue methods differ in foreground support")
    maps = {"kernel_watershed": (labels, audit), "offset_votes_v2": (joint["preserved_labels"], joint["audit"])}
    return {method: {"labels": value, "audit": meta,
                    "proposals": parseq_word_proposals(value, image_wh, **config["crop"])}
            for method, (value, meta) in maps.items()}


def qualify_kernel(args, cfg, sources):
    report, validation = completed_source(args.kernel_run)
    c = report["run_contract"]
    if (c["variant"] != "text_kernel" or c["validation_ids_sha256"] != sources["common"]["validation_ids_sha256"]
        or c["resolved_preprocessing"] != sources["common"]["resolved_preprocessing"]
        or c["config"]["postprocessing"] != cfg["kernel_postprocessing"]
        or report["validation_word_geometry"] != validation["aggregate"]
        or validation["checkpoint_sha256"] != report["checkpoint"]["sha256"]
        or sources["ids"] != [r["sample_id"] for r in validation["per_image"]]
        or sources["offset"]["run_contract"]["resolved_config"]["loss"]["name"] != "word_balanced_coherent_center_votes"):
        raise ValueError("Kernel/coherent-vote sources are incomplete or misaligned")
    return report, validation


def build_geometry(args, cfg, sources, kernel_validation, protocol):
    output = args.output_root
    if (output / "geometry_results.json").exists():
        g = json.loads((output / "geometry_results.json").read_text())
        if g["status"] != "SUCCESS" or g["protocol"] != protocol:
            raise ValueError("Completed learned geometry differs")
        verify_geometry_exports(g, output); return g
    branches = {b+"__"+policy: {m: {"per_image": [], "native_totals": Counter(), "original_totals": Counter()}
        for m in METHODS} for b in BRANCHES for policy in cfg["threshold_policies"]}
    by_id = {s.sample_id: s for s in sources["manifest"].samples}
    krows = {r["sample_id"]: r for r in kernel_validation["per_image"]}
    orows = {r["sample_id"]: r for r in sources["offset_validation"]["per_image"]}
    thresholds = {}
    for position, sid in enumerate(sources["ids"], 1):
        key = sid.replace(":", "__"); sample = by_id[sid]
        image_path = sources["target"] / sample.image_path
        if sha256_file(image_path) != sample.image_sha256: raise ValueError("RGB changed")
        with Image.open(image_path) as image: rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
        kpath = args.kernel_run / "validation_exports" / key / "maps.npz"
        opath = args.offset_run / "validation_exports" / key / "offsets.npz"
        if sha256_file(kpath) != krows[sid]["exports"]["maps.npz"] or sha256_file(opath) != orows[sid]["export_sha256"]:
            raise ValueError("Frozen learned-cue export changed")
        with np.load(kpath, allow_pickle=False) as k, np.load(opath, allow_pickle=False) as o:
            kernel, vector = k["probabilities"][1].copy(), o["normalized_offsets"].copy()
            kowner, kvalid, kcanvas = k["owners"].copy(), k["validity"].copy(), k["canvas"].copy()
            if not np.array_equal(kowner, o["owners"]) or not np.array_equal(kvalid, o["validity"]):
                raise ValueError("Learned source raster alignment differs")
        cache = {}; frozen = []
        for branch in BRANCHES:
            source = sources["branches"][branch]
            row = next(r for r in source["validation"]["exports"] if r["sample_id"] == sid)
            path = source["root"] / "validation_exports" / key / "maps.npz"
            if sha256_file(path) != row["files"]["maps.npz"]: raise ValueError("Text export changed")
            with np.load(path, allow_pickle=False) as z:
                p, owners, valid, canvas, expected = [z[x].copy() for x in ("probabilities", "owners", "validity", "canvas", "expected_owner_ids")]
            if (p.shape != (1, 256, 256) or kernel.shape != (256, 256) or vector.shape != (2, 256, 256)
                or not np.array_equal(owners, kowner) or not np.array_equal(valid, kvalid) or not np.array_equal(canvas, kcanvas)):
                raise ValueError("Transferred cues are not on the same canonical grid")
            for policy in cfg["threshold_policies"]:
                threshold = source["validation"]["operating_point"]["threshold"] if policy == "selected" else cfg["common_threshold"]
                name = branch+"__"+policy; thresholds[name] = threshold
                signature = (p.tobytes(), float(threshold))
                if signature not in cache: cache[signature] = prepare_recipes(p[0], kernel, vector, canvas, (rgb.shape[1], rgb.shape[0]), cfg, threshold=threshold)
                for method, recipe in cache[signature].items():
                    base = Path(name) / method / "samples" / key
                    _save_labels(output / base / "instances.png", recipe["labels"])
                    proposals = []
                    for proposal in recipe["proposals"]:
                        x1, y1, x2, y2 = proposal["bbox"]
                        proposals.append({**proposal, "crop_rgb_sha256": crop_array_hash(rgb[y1:y2, x1:x2])})
                    content = {"sample_id": sid, "result_class": "approximate reproduction", "threshold": threshold,
                        "image_path": str(image_path), "image_sha256": sample.image_sha256,
                        "source_maps_sha256": row["files"]["maps.npz"], "kernel_maps_sha256": krows[sid]["exports"]["maps.npz"],
                        "offset_maps_sha256": orows[sid]["export_sha256"], "proposals": proposals, "audit": recipe["audit"]}
                    write_json(output / base / "proposals.json", content)
                    files = {str(base / n): sha256_file(output / base / n) for n in ("instances.png", "proposals.json")}
                    frozen.append((name, method, content, recipe["labels"], owners, valid, expected, files))
        gt, annotation_sha = _train_gt(sample, sources["target"], (rgb.shape[1], rgb.shape[0]))
        for name, method, content, labels, owners, valid, expected, files in frozen:
            ignored = [int(i) for i in np.unique(labels[labels > 0]) if not np.any((labels == i) & valid)]
            care_labels = labels.copy(); care_labels[np.isin(labels, ignored)] = 0
            native = instance_geometry(care_labels, owners, expected)
            original = score_predictions([p for p in content["proposals"] if p["instance_id"] not in ignored], gt, text_available=False)
            part = branches[name][method]
            part["per_image"].append({**content, "ignored_prediction_ids": ignored, "ground_truth": gt,
                "annotation_path": str(sources["target"] / "datasets/total_text/train/annotations" / (sid.split(":", 1)[1]+".txt")),
                "annotation_sha256": annotation_sha, "native_geometry": native, "original_geometry": original, "exports": files})
            part["native_totals"].update(native); part["original_totals"].update(original)
        if position % 25 == 0: print(f"[learned crops] {position}/125", flush=True)
    for methods in branches.values():
        for part in methods.values():
            if len(part["per_image"]) != 125 or part["original_totals"]["gt_words"] != 983:
                raise RuntimeError("Complete learned-cue denominator differs")
            part["native"] = aggregate_word_metrics(dict(part.pop("native_totals")))
            part["original"] = ratios(dict(part.pop("original_totals")))
    g = {"status": "SUCCESS", "result_class": "approximate reproduction", "reproduction_mode": "approximate",
        "protocol": protocol, "branches": branches, "thresholds": thresholds, "segmentation_inference": "none; frozen maps reused"}
    write_json(output / "geometry_results.json", g); return g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["approximate"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    for name in ("run-root", "offset-run", "kernel-run", "output-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--action", required=True, choices=["preflight", "geometry", "recognize", "all"])
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    for name in ("parseq-source", "parseq-source-archive", "parseq-checkpoint"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--reuse-run-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    for name in ("config", "run_root", "offset_run", "kernel_run", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.cpu_threads < 1: raise ValueError("CPU threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    cfg = read_config(args.config, args.mode); sources = load_sources(args, cfg)
    kr, kv = qualify_kernel(args, cfg, sources)
    if args.output_root == REPO or any(args.output_root == p or p in args.output_root.parents or args.output_root in p.parents
        for p in [args.run_root, args.offset_run, args.kernel_run, sources["target"], *[p.resolve() for p in args.reuse_run_root]]):
        raise ValueError("Use an isolated output directory")
    protocol = geometry_protocol(args, cfg, sources)
    protocol["source_sha256"]["scripts/evaluate_text_recall_learned_crops.py"] = sha256_file(Path(__file__))
    protocol["kernel_run"] = str(args.kernel_run)
    protocol["kernel_report_sha256"] = sha256_file(args.kernel_run / "run_manifest.json")
    protocol["kernel_validation_sha256"] = sha256_file(args.kernel_run / "validation.json")
    protocol["kernel_checkpoint_sha256"] = kr["checkpoint"]["sha256"]
    if args.action == "preflight": print(json.dumps({"status": "PASS", "pipelines": 16, "images": 125, "gt_words": 983})); return 0
    output = args.output_root; output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(output / "geometry_protocol.json", protocol)
        try:
            g = build_geometry(args, cfg, sources, kv, protocol)
            if args.action in ("recognize", "all"):
                args.reuse_ocr = [prepare_ocr_provider(output, args.reuse_run_root, g)] if args.reuse_run_root else []
                result = summarize(output, g, recognize(args, cfg, g))
            else: result = {"status": "SUCCESS", "pipelines": 16}
        except Exception as error:
            write_json(output / "failure.json", {"status": "FAILED", "result_class": "approximate reproduction", "type": type(error).__name__, "message": str(error)})
            raise
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": main()
