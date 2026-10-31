"""Freeze ICDAR15 calibration policies, then evaluate authenticated full TEST outputs."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from PIL import Image

from recognition.clip_router_evaluation import deepsolo_predictions, native_mask
from recognition.deepsolo_word_f1 import recognition_confidence
from recognition.deepsolo_icdar2015_gap import evaluate_scenes, paired_interval, select_policy
from recognition.scene_text_datasets import parse_icdar2015_annotation
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_scenes(config, split, variant, pins):
    root = Path(config["study_root"])
    protocol = read(root / "predeclared_protocol.json")
    if sha(root / "predeclared_protocol.json") != config["predeclared_protocol_sha256"]:
        raise ValueError("Changed predeclared protocol")
    source = Path(config["source_study"])
    input_pins = read(config["evaluation_input_manifest"])
    if variant == "tt_native1024":
        predictions = source / ("fallback" if split == "test" else "fallback_validation")
    else:
        predictions = root / variant / split
    scenes = {}
    for sample in protocol["population"][split]:
        sid = sample["sample_id"]
        if sha(sample["image_path"]) != sample["image_sha256"]:
            raise ValueError("Changed input image: " + sid)
        identity = sid.split(":")[-1]
        data_split = "test" if split == "test" else "train"
        base = Path(config["data_root"]) / "datasets/icdar2015" / data_split
        annotation = base / "annotations" / (identity + ".txt")
        validity = base / "validity" / (identity + ".png")
        for path in (annotation, validity):
            pins[str(path)] = sha(path)
            if pins[str(path)] != input_pins[str(path)]:
                raise ValueError("Changed sealed evaluation input: " + str(path))
        if "annotation_sha256" in sample and sha(annotation) != sample["annotation_sha256"]:
            raise ValueError("Changed annotation: " + sid)
        instances, _ = clip_instances_to_pixel_extent(
            parse_icdar2015_annotation(annotation), tuple(sample["image_wh"]))
        gt = [dict(text=w.transcription,
                   bbox=[w.bbox_xyxy[0], w.bbox_xyxy[1], w.bbox_xyxy[2]+1, w.bbox_xyxy[3]+1])
              for w in instances if not w.ignore]
        with Image.open(validity) as image:
            valid = native_mask(image.convert("L"), config["ignore_resolution"])
        path = predictions / (sid.replace(":", "__") + ".json")
        pins[str(path)] = sha(path)
        raw = read(path)
        content = {k: v for k, v in raw.items() if k != "payload_sha256"}
        if (raw["payload_sha256"] != hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
                or raw["contract"]["image_sha256"] != sample["image_sha256"]
                or raw["contract"]["sample_id"] != sid or not raw["response"]["success"]):
            raise ValueError("Mismatched or changed prediction: " + sid)
        rows = deepsolo_predictions(raw["response"], sample["image_wh"], sample["orientation"],
                                   valid, protocol["selection"]["inference_floor"])
        by_id = {d["source_index"]: d for d in raw["response"]["detections"]}
        for row in rows:
            row["recognition_confidence"] = recognition_confidence(
                by_id[row["instance_id"]]["raw"]["recognition_class_probabilities"])
        scenes[sid] = rows, gt
    return scenes


def run(args):
    c = read(args.config)
    if c["mode"] != "approximate" or c["claim_label"] != "approximate reproduction":
        raise ValueError("Explicit approximate configuration required")
    root = Path(c["study_root"])
    protocol = read(root / "predeclared_protocol.json")
    for name, expected in ((c["evaluation_input_manifest"], c["evaluation_input_manifest_sha256"]),
                           (Path(c["source_study"]) / "resolved_configuration.json", c["source_study_plan_sha256"]),
                           (Path(c["source_study"]) / "supervised_validation_plan.json", c["source_validation_plan_sha256"])):
        if sha(name) != expected:
            raise ValueError("Changed pinned study input: " + str(name))
    selection = protocol["selection"]
    pins = {str(args.config): sha(args.config)}
    variants = list(c["profiles"])
    if args.stage == "calibrate":
        selected, grids, observations = {}, {}, {}
        for variant in variants:
            scenes = load_scenes(c, "validation", variant, pins)
            observations[variant] = scenes
            best, grid = select_policy(scenes, selection["score_thresholds"], selection["recognition_thresholds"])
            selected[variant], grids[variant] = best, grid
            print(variant, "CAL", best["metrics"]["f1"], best["policy"], flush=True)
        winner = max(variants, key=lambda name: (Fraction(2*selected[name]["metrics"]["tp"],
                                                selected[name]["metrics"]["detected_words"]+selected[name]["metrics"]["gt_words"]),
                                                selected[name]["metrics"]["tp"], -variants.index(name)))
        save(root / "calibration_grid.json", grids)
        save(root / "calibration_observations.json", observations)
        save(root / "frozen_calibration_policies.json", dict(
            status="FROZEN", mode="approximate", claim_label="approximate reproduction",
            selected=selected, selected_model=winner, input_sha256=pins,
            protocol_sha256=sha(root / "predeclared_protocol.json"),
            config_sha256=sha(args.config),
            observations_sha256=sha(root / "calibration_observations.json"),
            grid_sha256=sha(root / "calibration_grid.json"),
            test_used=False, code_sha256=sha(__file__),
            upstream_exposure="Calibration images may have been used in official checkpoint pretraining/fine-tuning."))
        return
    frozen = read(root / "frozen_calibration_policies.json")
    if frozen["test_used"] is not False or frozen["protocol_sha256"] != sha(root / "predeclared_protocol.json"):
        raise ValueError("Unsealed or changed calibration choice")
    reports = {}
    for variant in variants:
        scenes = load_scenes(c, "test", variant, pins)
        previous = read(c["profiles"]["tt_native1024"])["selected_policy"]
        upstream = dict(previous, score_threshold=0.3, recognition_threshold=0.0)
        policies = dict(previous_tt_filter=previous, upstream_ic15_filter=upstream,
                        calibration_selected=frozen["selected"][variant]["policy"])
        for name, policy in policies.items():
            key = variant + "__" + name
            report = evaluate_scenes(scenes, policy, iou_threshold=c["diagnostic_bbox_iou"],
                                     height_breaks=c["diagnostic_word_height_breaks"])
            report["policy"] = policy
            save(root / (key + "_per_image.json"), report["per_image"])
            reports[key] = {k: v for k, v in report.items() if k != "per_image"}
            reports[key]["per_image_sha256"] = sha(root / (key + "_per_image.json"))
            print(key, report["metrics"]["f1"], report["metrics"]["tp"], flush=True)
    reference_key = "tt_native1024__previous_tt_filter"
    reference = read(root / (reference_key + "_per_image.json"))
    expected = read(Path(c["source_study"]) / "threshold_analysis_v1/results.json")["datasets"]["icdar2015"]["unet"]["all_deepsolo"]
    for key in ("tp", "fp", "fn", "detected_words", "gt_words", "f1"):
        if reports[reference_key]["metrics"][key] != expected[key]:
            raise ValueError("Historical baseline replay failed: " + key)
    for name, report in reports.items():
        candidate = read(root / (name + "_per_image.json"))
        report["gain95_vs_reference"] = paired_interval(reference, candidate,
            repetitions=c["bootstrap_repetitions"], seed=c["seed"])
    report = dict(status="PASS", mode="approximate", claim_label="approximate reproduction",
                  selected_model=frozen["selected_model"],
                  frozen_calibration_sha256=sha(root / "frozen_calibration_policies.json"),
                  datasets={"icdar2015": reports}, input_sha256=pins,
                  code_sha256=sha(__file__),
                  metrics="Unchanged symbol-free lexical microF1. Geometry counts diagnose errors only; no replacement metric. Not verified as paper82 accuracy.",
                  selection="All new thresholds and selected_model frozen on TRAIN calibration100 before new TEST analysis.")
    save(root / "results.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibrate", "evaluate"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
