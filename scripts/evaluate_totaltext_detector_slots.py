#!/usr/bin/env python3
"""Independent DeepSolo word-slot test over frozen Total-Text PARSeq outputs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import subprocess
import sys

from PIL import Image
import torch

from deepsolo.adapter import DeepSoloSubprocessAdapter
from deepsolo.contracts import DeepSoloResponse
from localize.independent_word_slots import (
    EmptyDetectorBoundaryError, assign_to_slots, boundary_bbox,
    detector_boundary_in_original, overlap_iou,
)
from recognition.scene_text_v1_evaluation import maximum_matching
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.crop_clip_runtime import sha
from scripts.run_crop_group_word_selection import evaluate_predictions
from segmentation.data import load_segmentation_manifest


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/totaltext_independent_detector_slots_v1.json"
DEFAULT_PATHS = REPO / "configs/approximate/totaltext_independent_detector_slots_local_paths_v1.json"
CODE = ("scripts/evaluate_totaltext_detector_slots.py", "localize/independent_word_slots.py",
        "recognition/group_word_selection.py", "scripts/run_crop_group_word_selection.py",
        "recognition/scene_text_v1_evaluation.py", "deepsolo/adapter.py",
        "deepsolo/worker.py", "deepsolo/contracts.py", "deepsolo/candidates.py",
        "deepsolo/reduction.py")
ANALYSIS_CODE = ("scripts/evaluate_totaltext_detector_slots.py", "localize/independent_word_slots.py")
GEOMETRY_POLICY = {
    "classification": "ASSUMPTION",
    "ambiguity": "TT-INDEPENDENT-DETECTOR-SLOTS-EXIF-V1",
    "coordinate_frame": "stored untransposed raster, as used by PIL crops and annotations",
    "mapping": "inverse EXIF transform of OpenCV polygon pixel coordinates, with W-1/H-1 offsets",
    "unspecified_orientation": "EXIF 0 or absent: identity; unsupported values fail",
    "empty_envelope": "record and exclude finite boundaries with no positive clipped bbox",
    "invalid_geometry": "fail on malformed/nonfinite boundaries; never catch generic ValueError",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _local_paths(profile: dict) -> dict[str, Path]:
    keys = ("frozen_refinement_run", "test_manifest", "deepsolo_python",
            "deepsolo_runtime_manifest", "deepsolo_source_root", "output_root")
    paths = {key: Path(profile[key]).resolve() for key in keys if key != "deepsolo_python"}
    # The host venv's bin/python is a symlink. Following it to /usr/bin/python3.8
    # bypasses pyvenv.cfg and hides the qualified DeepSolo site-packages.
    paths["deepsolo_python"] = Path(profile["deepsolo_python"]).absolute()
    return paths


def _probe_deepsolo_python(executable: Path) -> dict:
    script = ("import cv2, detectron2, json, sys, torch; "
              "print(json.dumps({'executable': sys.executable, 'prefix': sys.prefix, "
              "'cv2': cv2.__version__, 'detectron2': getattr(detectron2, '__version__', None), "
              "'torch': torch.__version__}))")
    try:
        completed = subprocess.run([str(executable), "-c", script],
                                   capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"DeepSolo Python preflight failed for {executable}: {exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"DeepSolo Python preflight failed for {executable}: "
                           f"{completed.stderr.strip()}")
    try:
        runtime = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DeepSolo Python preflight returned invalid JSON: "
                           f"{completed.stdout!r}") from exc
    if Path(runtime["executable"]).absolute() != executable.absolute():
        raise RuntimeError(f"DeepSolo Python changed its entry point: {runtime['executable']}")
    return runtime


def _controls(config: dict) -> list[tuple[float, float]]:
    if (config.get("schema_version") != 1 or config.get("reproduction_mode") != "approximate"
        or config.get("table_5_claim") is not False
        or config.get("detector_candidate") != "ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE"
        or config.get("detector_device") != "cuda:0"
        or config.get("source_selection") != "guarded_consolidated"
        or config.get("evaluation_matching_iou") != .5):
        raise ValueError("invalid isolated approximate detector-slot config")
    inference = config.get("detector_inference_score_threshold")
    scores = config.get("detector_score_controls")
    ious = config.get("crop_slot_iou_controls")
    if (not isinstance(inference, (int, float)) or not 0 <= inference <= 1
        or not isinstance(scores, list) or not scores
        or not isinstance(ious, list) or not ious
        or any(not isinstance(value, (int, float)) or not inference <= value <= 1
               for value in scores)
        or any(not isinstance(value, (int, float)) or not 0 < value <= 1
               for value in ious)
        or len(set(scores)) != len(scores) or len(set(ious)) != len(ious)):
        raise ValueError("invalid predeclared detector/association controls")
    return [(float(score), float(iou)) for score in scores for iou in ious]


def _name(score: float, iou: float) -> str:
    return f"det{round(score * 1000):03d}_iou{round(iou * 1000):03d}"


def _image_frame(image_path: Path) -> dict:
    with Image.open(image_path) as image:
        width, height = image.size
        orientation = image.getexif().get(274, 1)
    if (not isinstance(orientation, int) or isinstance(orientation, bool)
        or orientation not in range(9)):
        raise ValueError(f"unsupported EXIF orientation {orientation!r}: {image_path}")
    return {"original_wh": [width, height], "exif_orientation": orientation,
            "detector_decoded_wh": [height, width] if orientation >= 5 else [width, height],
            "mapping": "inverse_exif_pixel_coordinates_v1"}


def _slots(response: dict, image_wh: tuple[int, int], *, exif_orientation: int = 1,
           rejected: list[dict] | None = None) -> list[dict]:
    if not response.get("success"):
        raise RuntimeError("frozen DeepSolo inference is unsuccessful")
    output = []
    seen = set()
    for detection in response["detections"]:
        source_index = detection["source_index"]
        if (not isinstance(source_index, int) or isinstance(source_index, bool)
            or source_index < 0 or source_index in seen):
            raise ValueError("invalid or duplicate DeepSolo detection ID")
        seen.add(source_index)
        score = detection["score"]
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
            raise ValueError("invalid DeepSolo score")
        points = detector_boundary_in_original(
            detection["geometry"]["boundary_points"], image_wh, exif_orientation)
        try:
            bbox = boundary_bbox(points, image_wh)
        except EmptyDetectorBoundaryError as exc:
            if rejected is None:
                raise  # Exclusion requires a caller that persists the rejection record.
            rejected.append({"source_index": source_index, "score": score,
                             "reason": "no_positive_in_image_bbox",
                             "unclipped_bbox": exc.unclipped_bbox,
                             "clipped_bbox": exc.clipped_bbox})
            continue
        output.append({"source_index": source_index, "score": score, "bbox": bbox})
    return output


def _validate_detector_plan(previous: dict, current: dict) -> None:
    """Only analysis code may change when explicitly reusing frozen inference."""
    fields = ("schema_version", "reproduction_mode", "table_5_claim", "config", "config_sha256",
              "frozen_run", "frozen_run_plan_sha256", "source", "source_plan_sha256",
              "manifest", "manifest_sha256", "runtime_manifest", "runtime_manifest_sha256",
              "deepsolo_python_entrypoint", "checkpoint_sha256", "device",
              "detector_inference_threshold", "execution", "controls", "ground_truth_use")
    for field in fields:
        if field not in previous or previous[field] != current[field]:
            raise RuntimeError(f"frozen detector inference contract differs: {field}")
    old_ids = previous.get("sample_ids")
    if (not isinstance(old_ids, list) or len(set(old_ids)) != len(old_ids)
        or not set(current["sample_ids"]).issubset(old_ids)):
        raise RuntimeError("frozen detector sample IDs differ")
    old_code, new_code = previous.get("code_sha256", {}), current["code_sha256"]
    if set(old_code) != set(new_code):
        raise RuntimeError("frozen detector code inventory differs")
    for name in CODE:
        if name not in ANALYSIS_CODE and old_code[name] != new_code[name]:
            raise RuntimeError(f"frozen detector dependency code differs: {name}")


def _validate_detector_shard(cached: dict, sid: str, plan_hash: str,
                             image_path: Path, image_wh: tuple[int, int], plan: dict) -> None:
    response = DeepSoloResponse.from_dict(cached.get("response", {})).to_dict()
    artifact = response["artifact"]
    if (cached.get("plan_sha256") != plan_hash or cached.get("sample_id") != sid
        or cached.get("image_sha256") != sha(image_path)
        or cached.get("image_wh") != list(image_wh) or cached.get("ground_truth_use") != "none"
        or not response["success"]
        or artifact.get("checkpoint_sha256") != plan["checkpoint_sha256"]
        or artifact.get("candidate", {}).get("artifact_id") != plan["config"]["detector_candidate"]
        or artifact.get("inference_options") != {
            "device": plan["device"], "score_threshold": plan["detector_inference_threshold"],
            "lexicon_type": None}
        or response["reduction_policy"] != "highest_score"
        or response["placement"].get("device") != plan["device"]):
        raise RuntimeError(f"cached detector shard differs: {sid}")


def _detector_reuse(source_run: Path, plan: dict, images: dict) -> dict:
    previous_path = source_run / "resolved_configuration.json"
    previous = _read_json(previous_path)
    _validate_detector_plan(previous, plan)
    previous_hash = sha(previous_path)
    shards = {}
    for sid in plan["sample_ids"]:
        path = source_run / "detector" / f"{sid.replace(':', '__')}.json"
        _validate_detector_shard(_read_json(path), sid, previous_hash, *images[sid], plan)
        shards[sid] = sha(path)
    return {"source_run": str(source_run), "source_plan_sha256": previous_hash,
            "source_code_sha256": previous["code_sha256"], "shard_sha256": shards,
            "validation": "same frozen inputs, inference controls and dependency code; analysis code may differ"}


def _inputs(args):
    config_path, paths_path = args.config.resolve(), args.paths.resolve()
    if (config_path.parent != REPO / "configs/approximate"
        or paths_path.parent != REPO / "configs/approximate"):
        raise ValueError("slot configs must remain in configs/approximate")
    config, paths = _read_json(config_path), _read_json(paths_path)
    controls = _controls(config)
    if paths.get("schema_version") != 1 or paths.get("reproduction_mode") != "approximate":
        raise ValueError("invalid local path profile")
    paths = _local_paths(paths)
    run = paths["frozen_refinement_run"]
    run_plan = _read_json(run / "resolved_configuration.json")
    if (run_plan.get("reproduction_mode") != "approximate"
        or run_plan.get("result_class") != "APPROXIMATE_TEST_DIAGNOSTIC"
        or not isinstance(run_plan.get("sample_ids"), list)
        or len(set(run_plan["sample_ids"])) != len(run_plan["sample_ids"])
        or sha(paths["test_manifest"]) != run_plan["manifest_sha256"]):
        raise RuntimeError("frozen TEST selector or manifest contract differs")
    source = Path(run_plan["source"])
    source_plan = _read_json(source / "resolved_configuration.json")
    if source_plan.get("manifest_sha256") != run_plan["manifest_sha256"]:
        raise RuntimeError("frozen source manifest differs")
    root = Path(source_plan["paths"]["data_root"]).resolve()
    samples = {sample.sample_id: sample for sample in
               load_segmentation_manifest(paths["test_manifest"], require_checksums=True).samples}
    ids = run_plan["sample_ids"][:args.limit] if args.limit else run_plan["sample_ids"]
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if len({_name(score, iou) for score, iou in controls}) != len(controls):
        raise ValueError("control labels collide after serialization")
    images = {}
    for sid in ids:
        if sid not in samples:
            raise RuntimeError(f"missing TEST sample {sid}")
        sample = samples[sid]
        image_path = (root / Path(sample.image_path)).resolve()
        if root not in image_path.parents or sha(image_path) != sample.image_sha256:
            raise RuntimeError(f"TEST RGB identity differs: {sid}")
        with Image.open(image_path) as opened:
            image_wh = opened.size
        images[sid] = (image_path, image_wh)
    manifest = _read_json(paths["deepsolo_runtime_manifest"])
    candidate = manifest.get("candidates", {}).get(config["detector_candidate"])
    if manifest.get("runtime", {}).get("kind") != "host_subprocess" or not candidate:
        raise RuntimeError("expected qualified host DeepSolo candidate manifest")
    if sha(Path(candidate["checkpoint_path"])) != candidate["checkpoint_sha256"]:
        raise RuntimeError("DeepSolo checkpoint SHA256 differs")
    if not paths["deepsolo_python"].is_file() or not paths["deepsolo_source_root"].is_dir():
        raise FileNotFoundError("qualified DeepSolo Python or source is missing")
    return config, paths, controls, run, run_plan, source, ids, images


def run(args) -> Path | None:
    if args.detector_run and not args.analyze_only:
        raise ValueError("--detector-run requires --analyze-only; frozen inference is read-only")
    config, paths, controls, frozen_run, frozen_plan, source, ids, images = _inputs(args)
    runtime = None if args.analyze_only else _probe_deepsolo_python(paths["deepsolo_python"])
    if args.check:
        print(json.dumps({"inputs": "verified", "images": len(ids),
                          "deepsolo_runtime": runtime,
                          "cuda_available": torch.cuda.is_available(),
                          "next_command": "bash scripts/test_totaltext_detector_slots.sh"}))
        return None
    if not args.analyze_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable here; run bash scripts/test_totaltext_detector_slots.sh on the GPU host")
    output_root = args.output_root.resolve() if args.output_root else paths["output_root"]
    output = output_root / (f"smoke-{len(ids)}" if args.limit else "full")
    if output == REPO or REPO in output.parents:
        raise ValueError("generated detector results must stay outside the repository")
    detector_run = args.detector_run.resolve() if args.detector_run else None
    if detector_run and (output == detector_run or detector_run in output.parents):
        raise ValueError("choose a new --output-root; frozen detector runs must remain untouched")
    plan = {"schema_version": 1, "reproduction_mode": "approximate",
            "result_class": "SMOKE_TEST" if args.limit else "APPROXIMATE_TEST_DIAGNOSTIC",
            "table_5_claim": False, "claim_label": config["claim_label"],
            "config": config, "config_sha256": sha(args.config.resolve()),
            "local_paths_sha256": sha(args.paths.resolve()),
            "frozen_run": str(frozen_run),
            "frozen_run_plan_sha256": sha(frozen_run / "resolved_configuration.json"),
            "source": str(source), "source_plan_sha256": sha(source / "resolved_configuration.json"),
            "manifest": str(paths["test_manifest"]), "manifest_sha256": sha(paths["test_manifest"]),
            "runtime_manifest": str(paths["deepsolo_runtime_manifest"]),
            "runtime_manifest_sha256": sha(paths["deepsolo_runtime_manifest"]),
            "deepsolo_python_entrypoint": str(paths["deepsolo_python"]),
            "checkpoint_sha256": _read_json(paths["deepsolo_runtime_manifest"])["candidates"][
                config["detector_candidate"]]["checkpoint_sha256"],
            "sample_ids": ids, "code_sha256": {name: sha(REPO / name) for name in CODE},
            "device": config["detector_device"], "detector_inference_threshold":
            config["detector_inference_score_threshold"],
            "execution": {"batch_size": 1, "precision": "qualified DeepSolo runtime float32",
                          "lifecycle": "one isolated subprocess per image; sequential",
                          "image_shape": "original per-image RGB, saved in detector shard"},
            "controls": [{"detector_min_score": s, "crop_slot_min_iou": i} for s, i in controls],
            "python": sys.version, "platform": platform.platform(),
            "ground_truth_use": "after all detector and assignment shards are frozen",
            "geometry_policy": GEOMETRY_POLICY,
            "image_frames": {sid: _image_frame(images[sid][0]) for sid in ids}}
    reuse = _detector_reuse(detector_run, plan, images) if detector_run else None
    if reuse:
        plan["detector_reuse"] = reuse
    freeze(output / "resolved_configuration.json", plan)
    plan_hash = sha(output / "resolved_configuration.json")
    adapter = None if args.analyze_only else DeepSoloSubprocessAdapter(
        paths["deepsolo_python"], paths["deepsolo_runtime_manifest"],
        host_source_root=paths["deepsolo_source_root"])
    for position, sid in enumerate(ids, 1):
        key = sid.replace(":", "__")
        image_path, image_wh = images[sid]
        shard = (detector_run or output) / "detector" / f"{key}.json"
        if reuse:
            if sha(shard) != reuse["shard_sha256"][sid]:
                raise RuntimeError(f"source detector shard changed after validation: {sid}")
        if shard.is_file():
            cached = _read_json(shard)
            _validate_detector_shard(cached, sid,
                                     reuse["source_plan_sha256"] if reuse else plan_hash,
                                     image_path, image_wh, plan)
        else:
            if adapter is None:
                raise FileNotFoundError(f"missing frozen DeepSolo shard for --analyze-only: {sid}")
            response = adapter.infer(
                image_path=image_path, candidate_id=config["detector_candidate"],
                inference_options={"device": config["detector_device"],
                                   "score_threshold": config["detector_inference_score_threshold"],
                                   "lexicon_type": None}, reduction_policy="highest_score").to_dict()
            if (not response.get("success") or response["artifact"]["checkpoint_sha256"] !=
                _read_json(paths["deepsolo_runtime_manifest"])["candidates"][
                    config["detector_candidate"]]["checkpoint_sha256"]):
                raise RuntimeError(f"DeepSolo inference/checkpoint failed: {sid}")
            freeze(shard, {"sample_id": sid, "plan_sha256": plan_hash,
                           "image_sha256": sha(image_path), "image_wh": image_wh,
                           "response": response, "ground_truth_use": "none"})
        if position % 25 == 0 or position == len(ids):
            print(f"[DeepSolo] {position}/{len(ids)}", flush=True)
    for position, sid in enumerate(ids, 1):
        key = sid.replace(":", "__")
        frozen = _read_json(frozen_run / "selection" / f"{key}.json")
        predictions = frozen["methods"][config["source_selection"]]
        detector_shard = (detector_run or output) / "detector" / f"{key}.json"
        detector = _read_json(detector_shard)
        rejected = []
        frame = plan["image_frames"][sid]
        slots = _slots(detector["response"], images[sid][1],
                       exif_orientation=frame["exif_orientation"], rejected=rejected)
        geometry = {"coordinate_frame": frame,
                    "raw_detections": len(detector["response"]["detections"]),
                    "valid_slots": len(slots), "rejected_detections": rejected}
        arms = {}
        for score, iou in controls:
            label = _name(score, iou)
            retained_slots = [s for s in slots if s["score"] >= score]
            selected, assignments, memberships = assign_to_slots(
                predictions, retained_slots, min_iou=iou)
            covered_ids = {row["candidate_id"] for group in memberships
                           for row in group["eligible_candidates"]}
            covered = [p for p in predictions if p["candidate_id"] in covered_ids]
            arms[label] = {"slots": retained_slots, "covered": covered,
                           "selected": selected, "assignments": assignments,
                           "memberships": memberships}
        freeze(output / "selection" / f"{key}.json", {
            "sample_id": sid, "plan_sha256": plan_hash,
            "detector_sha256": sha(detector_shard),
            "frozen_selection_sha256": sha(frozen_run / "selection" / f"{key}.json"),
            "original_predictions": predictions, "arms": arms, "detector_geometry": geometry,
            "ground_truth_use": "none"})
        if position % 25 == 0 or position == len(ids):
            print(f"[slots] {position}/{len(ids)}", flush=True)
    totals = {name: Counter() for name in ("original", *(_name(s, i) for s, i in controls))}
    memory = Counter()
    geometry_totals, rejection_reasons = Counter(), Counter()
    denominator = 0
    for position, sid in enumerate(ids, 1):
        key = sid.replace(":", "__")
        selected = _read_json(output / "selection" / f"{key}.json")
        geometry = selected["detector_geometry"]
        geometry_totals.update(raw_detections=geometry["raw_detections"],
                               valid_slots=geometry["valid_slots"],
                               rejected_detections=len(geometry["rejected_detections"]),
                               reoriented_images=int(geometry["coordinate_frame"]["exif_orientation"] >= 2))
        rejection_reasons.update(row["reason"] for row in geometry["rejected_detections"])
        detector = _read_json((detector_run or output) / "detector" / f"{key}.json")
        gt = _read_json(source / "evaluation" / f"{key}.json")["ground_truth"]
        base = evaluate_predictions(selected["original_predictions"], gt,
                                    config["evaluation_matching_iou"])
        old = _read_json(frozen_run / "evaluation" / f"{key}.json")[
            "methods"][config["source_selection"]]
        if (base["correct"] != old["correct"] or base["predictions"] != old["predictions"]
            or base["correct_gt_indices"] != old["correct_gt_indices"]):
            raise RuntimeError(f"frozen original evaluation differs: {sid}")
        scores = {"original": base}
        for score, iou in controls:
            name = _name(score, iou)
            arm = selected["arms"][name]
            covered = evaluate_predictions(arm["covered"], gt, config["evaluation_matching_iou"])
            assigned = evaluate_predictions(arm["selected"], gt, config["evaluation_matching_iou"])
            edges = [(si, gi) for si, slot in enumerate(arm["slots"])
                     for gi, word in enumerate(gt)
                     if overlap_iou(slot["bbox"], word["bbox"]) >=
                     config["evaluation_matching_iou"]]
            slot_gt = maximum_matching(len(arm["slots"]), len(gt), edges)
            scores[name] = {"slots": len(arm["slots"]), "covered": covered,
                            "selected": assigned, "detector_gt_box_matches": len(slot_gt),
                            "detector_gt_box_indices": sorted(slot_gt)}
            totals[name].update(slots=len(arm["slots"]), outputs=assigned["predictions"],
                                correct=assigned["correct"],
                                covered_outputs=covered["predictions"],
                                covered_correct=covered["correct"],
                                detector_gt_box_matches=len(slot_gt))
        totals["original"].update(outputs=base["predictions"], correct=base["correct"])
        denominator += len(gt)
        record = detector["response"]["memory"]
        for field in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
            value = record[field]
            if value is not None:
                memory[field] = max(memory[field], value)
        freeze(output / "evaluation" / f"{key}.json", {
            "sample_id": sid, "selection_sha256": sha(output / "selection" / f"{key}.json"),
            "source_evaluation_sha256": sha(source / "evaluation" / f"{key}.json"),
            "gt_words": len(gt), "arms": scores})
        if position % 25 == 0 or position == len(ids):
            print(f"[evaluation] {position}/{len(ids)}", flush=True)
    summary = {}
    base_correct = totals["original"]["correct"]
    for name, values in totals.items():
        precision = values["correct"] / values["outputs"] if values["outputs"] else 0.
        recall = values["correct"] / denominator if denominator else 0.
        summary[name] = {**dict(values), "precision_proxy": precision,
                         "recall_proxy": recall,
                         "hmean_proxy": 2 * precision * recall / (precision + recall)
                         if precision + recall else 0.}
        if name != "original":
            summary[name]["retained_original_correct_fraction"] = (
                values["covered_correct"] / base_correct if base_correct else 0.)
    result = {"schema_version": 1, "reproduction_mode": "approximate",
              "result_class": plan["result_class"], "table_5_claim": False,
              "plan_sha256": plan_hash, "images": len(ids), "gt_words": denominator,
              "summary": summary,
              "detector_geometry": {**dict(geometry_totals), "rejection_reasons": dict(rejection_reasons)},
              "gpu_peak": dict(memory),
              "metric": "casefold/P-S-stripped exact reading; original bbox IoU>=0.5; one-to-one",
              "limitation": "Exploratory independent detector slots, not official polygon spotting Hmean"}
    freeze(output / "results.json", result)
    print(json.dumps({"report": str(output / "results.json"),
                      "summary": result["summary"], "gpu_peak": result["gpu_peak"]}), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--check", action="store_true", help="verify paths/hashes without CUDA inference")
    parser.add_argument("--analyze-only", action="store_true",
                        help="evaluate already frozen detector shards without CUDA")
    parser.add_argument("--detector-run", type=Path,
                        help="reuse verified detector shards from a previous run, with --analyze-only")
    parser.add_argument("--output-root", type=Path,
                        help="explicit new output root for a revised analysis; preserve the old run")
    parser.add_argument("--limit", type=int, help="smoke test on first N images")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
