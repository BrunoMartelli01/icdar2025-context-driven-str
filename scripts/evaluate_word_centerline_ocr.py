"""Qualify a completed centerline experiment and compare its crops with real PARSeq."""

import argparse
from collections import Counter
import fcntl
from hashlib import sha256
import json
from pathlib import Path

import torch

from recognition.parseq import PARSEQ_ARCHIVE_SOURCE, verify_parseq_runtime_source
from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _correct
from scripts.evaluate_word_multitask_crops import (
    ratios, recognize, score_predictions, verify_geometry_exports, write_json,
)
from scripts.prune_segmentation_checkpoints import sha256_file


REPO = Path(__file__).resolve().parents[1]
BRANCH = "centerline_radius"
METHODS = ("baseline", "line_markers", "line_radius")
OCR_SOURCES = (
    "scripts/evaluate_word_centerline_ocr.py",
    "scripts/evaluate_word_multitask_crops.py",
    "scripts/evaluate_totaltext_highres_contour_padding.py",
    "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py",
    "scripts/evaluate_totaltext_word_kernel_parseq_val.py",
    "scripts/rescore_totaltext_highres_padding_symbols.py",
    "recognition/parseq.py",
    "recognition/scene_text_v1_evaluation.py",
)


def qualify_run(root, mode):
    """No model loading: fail on incomplete training, altered inputs or missing words."""
    root = Path(root).resolve()
    if mode != "approximate":
        raise ValueError("Explicit APPROXIMATE mode is required")
    report = json.loads((root / "run_manifest.json").read_text())
    geometry = json.loads((root / "geometry_results.json").read_text())
    contract = report["run_contract"]
    cfg = contract["resolved_config"]
    if (report.get("status") != "SUCCESS" or report.get("training_complete") is not True
            or report.get("completed_epochs") != cfg["epochs"]
            or type(report.get("completed_epochs")) is not int
            or cfg["epochs"] < 1):
        raise ValueError("Incomplete centerline training; finish the configured epoch budget first")
    if (report.get("result_class") != "approximate reproduction"
            or contract.get("reproduction_mode") != "approximate"
            or contract.get("result_class") != "approximate reproduction"
            or geometry.get("status") != "SUCCESS"
            or geometry.get("reproduction_mode") != "approximate"
            or geometry.get("result_class") != "approximate reproduction"):
        raise ValueError("Completed approximate reproduction artifacts are required")
    checkpoint = report["checkpoint"]
    if checkpoint["path"] != "best.pt" or sha256_file(root / "best.pt") != checkpoint["sha256"]:
        raise ValueError("Selected centerline checkpoint changed")
    sources = contract["source_sha256"]
    if not sources:
        raise ValueError("Scientific source hashes are missing")
    for name, expected in sources.items():
        path = (REPO / name).resolve()
        if REPO not in path.parents or sha256_file(path) != expected:
            raise ValueError("Frozen scientific source changed: " + name)
    protocol = geometry["protocol"]
    if (protocol["run_contract"] != contract
            or protocol["checkpoint_sha256"] != checkpoint["sha256"]
            or protocol["source_seed"] != cfg["seed"]
            or protocol["expected_images"] != cfg["validation_count"]
            or protocol["expected_care_words"] != cfg["expected_validation_words"]
            or contract["validation_ids_sha256"] != cfg["validation_ids_sha256"]):
        raise ValueError("Geometry and selected training provenance differ")
    if set(geometry["branches"]) != {BRANCH} or set(geometry["branches"][BRANCH]) != set(METHODS):
        raise ValueError("Expected baseline, line_markers and line_radius crop methods")
    reference = None
    for method in METHODS:
        rows = geometry["branches"][BRANCH][method]["per_image"]
        ids = [r["sample_id"] for r in rows]
        fingerprint = sha256("".join(f"{sid}\n" for sid in ids).encode()).hexdigest()
        if (len(ids) != cfg["validation_count"] or len(set(ids)) != len(ids)
                or fingerprint != contract["validation_ids_sha256"]
                or sum(len(r["ground_truth"]) for r in rows) != cfg["expected_validation_words"]):
            raise ValueError("Incomplete or changed held-out population: " + method)
        population = [(r["sample_id"], r["ground_truth"], r["image_path"], r["image_sha256"],
                       r["annotation_path"], r["annotation_sha256"]) for r in rows]
        if reference is not None and population != reference:
            raise ValueError("Ground-truth identities or image ordering differ between methods")
        reference = population
        for row in rows:
            proposals = row["proposals"]
            instance_ids = [p["instance_id"] for p in proposals]
            if (len(instance_ids) != len(set(instance_ids))
                    or not set(row["ignored_prediction_ids"]).issubset(instance_ids)):
                raise ValueError("Duplicate or unknown crop instance identities")
            for name in row["exports"]:
                if root not in (root / name).resolve().parents:
                    raise ValueError("Geometry export must remain inside its experiment directory")
    verify_geometry_exports(geometry, root)
    return report, geometry


def compare_recognition(geometry, recognition):
    """Recompute scores and preserve individual gained/lost GT identities."""
    if (recognition.get("status") != "SUCCESS"
            or recognition.get("result_class") != "approximate reproduction"
            or set(recognition["branches"]) != {BRANCH}
            or set(recognition["branches"][BRANCH]) != set(METHODS)):
        raise ValueError("Complete approximate centerline OCR results required")
    methods = {}
    correct_sets = {}
    for method in METHODS:
        part = recognition["branches"][BRANCH][method]
        expected_rows = geometry["branches"][BRANCH][method]["per_image"]
        rows = part["per_image"]
        if [r["sample_id"] for r in rows] != [r["sample_id"] for r in expected_rows]:
            raise ValueError("Recognition population differs from frozen geometry")
        totals = Counter()
        identities = set()
        for row, source in zip(rows, expected_rows):
            predictions = row["predictions_all"]
            if (row["ground_truth"] != source["ground_truth"]
                    or set(row["ignored_prediction_ids"]) != set(source["ignored_prediction_ids"])
                    or [{k: v for k, v in p.items() if k != "text"} for p in predictions] != source["proposals"]):
                raise ValueError("OCR changed frozen crop or GT identities")
            ignored = set(row["ignored_prediction_ids"])
            care = [p for p in predictions if p["instance_id"] not in ignored]
            counts = score_predictions(care, row["ground_truth"], text_available=True)
            correct = sorted(_correct(care, row["ground_truth"], normalize=False))
            if counts != row["counts"] or correct != row["correct_casefold_gt_indices"]:
                raise ValueError("Recognition row scores do not reproduce one-to-one matching")
            totals.update(counts)
            identities.update((row["sample_id"], index) for index in correct)
        aggregate = ratios(dict(totals))
        if aggregate != part["aggregate"]:
            raise ValueError("Recognition aggregate differs from its complete per-image rows")
        methods[method] = aggregate
        correct_sets[method] = identities
    reference = correct_sets["baseline"]
    gt_by_sample = {row["sample_id"]: row["ground_truth"]
                    for row in geometry["branches"][BRANCH]["baseline"]["per_image"]}

    def word_rows(identities):
        return [{"sample_id": sid, "gt_index": index, "gt_text": gt_by_sample[sid][index]["text"],
                 "gt_bbox": gt_by_sample[sid][index]["bbox"]} for sid, index in sorted(identities)]

    paired = {}
    for method in METHODS[1:]:
        gained, lost = correct_sets[method] - reference, reference - correct_sets[method]
        paired[method] = {"gained_count": len(gained), "lost_count": len(lost),
                          "net_correct_words": len(gained) - len(lost),
                          "gained": word_rows(gained), "lost": word_rows(lost)}
    return {"status": "SUCCESS", "reproduction_mode": "approximate",
            "result_class": "approximate reproduction", "methods": methods, "paired_vs_baseline": paired,
            "metric_protocol": "Original bbox IoU>=0.50 and exact Unicode casefold; maximum-cardinality "
                "one-to-one matching, punctuation retained, all held-out care words in recall denominator; "
                "only wholly ignored predictions excluded after GT-blind crop generation.",
            "limitations": "Exploratory reused TRAIN holdout, not official TEST or Table5 accuracy. "
                "Gained/lost identities use deterministic maximum matching; alternative equally-sized "
                "matchings may assign different duplicate-word identities."}


def markdown_report(result):
    lines = ["# Centerline crop comparison", "", "**Approximate reproduction**, reused TRAIN holdout.", "",
             "| Method | Correct words | Care proposals | Precision | Recall | F1 | Gain / loss |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in METHODS:
        row = result["methods"][method]
        paired = result["paired_vs_baseline"].get(method)
        change = "reference" if paired is None else f"+{paired['gained_count']} / -{paired['lost_count']}"
        lines.append(f"| {method} | {row['exact_casefold_iou_ge_50']}/{row['gt_words']} | "
                     f"{row['proposals']} | {row['exact_casefold_precision']:.2%} | "
                     f"{row['exact_casefold_recall_all_gt']:.2%} | {row['exact_casefold_f1']:.2%} | {change} |")
    return "\n".join(lines + ["", result["metric_protocol"], "", result["limitations"], ""])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["approximate"])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--action", choices=["preflight", "recognize"], default="recognize")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    for name in ("parseq-source", "parseq-source-archive", "parseq-checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1 or args.batch_size < 1:
        raise ValueError("CPU threads and PARSeq batch size must be positive")
    args.output_root = args.run_root.resolve()
    args.reuse_ocr = []
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Explicit CPU or CUDA execution is required")
    torch.set_num_threads(args.cpu_threads)
    # Opening an existing experiment does not create an incomplete result directory.
    with (args.output_root / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process holds the centerline experiment lock") from error
        try:
            report, geometry = qualify_run(args.output_root, args.mode)
            runtime = verify_parseq_runtime_source(args.parseq_source, source_type=PARSEQ_ARCHIVE_SOURCE,
                                                   source_archive=args.parseq_source_archive)
            from recognition.parseq import PARSEQ_CHECKPOINT_SHA256
            if sha256_file(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256:
                raise ValueError("Pinned PARSeq checkpoint changed")
            cfg = {"recognition": {"decode_ar": True, "refine_iters": 1, "batch_size": args.batch_size,
                    "casefold": True, "symbol_normalized_secondary_metric": True}}
            contract = {"result_class": "approximate reproduction", "reproduction_mode": "approximate",
                "run_manifest_sha256": sha256_file(args.output_root / "run_manifest.json"),
                "geometry_results_sha256": sha256_file(args.output_root / "geometry_results.json"),
                "recognition": cfg["recognition"], "runtime": runtime,
                "source_sha256": {name: sha256_file(REPO / name) for name in OCR_SOURCES},
                "device": str(device), "cpu_threads": args.cpu_threads,
                "hardware_deviation": "PARSeq executes separately from segmentation in FP32; device and "
                    "batch are recorded. CPU execution and sequential loading differ from the paper environment."}
            if args.action == "preflight":
                print(json.dumps({"status": "PREFLIGHT_PASS", "result_class": "approximate reproduction",
                    "completed_epochs": report["completed_epochs"], "images": geometry["protocol"]["expected_images"],
                    "care_words": geometry["protocol"]["expected_care_words"]}))
                return 0
            write_json(args.output_root / "centerline_ocr_protocol.json", contract)
            recognition = recognize(args, cfg, geometry)
            comparison = compare_recognition(geometry, recognition)
            comparison["recognition_results_sha256"] = sha256_file(args.output_root / "recognition_results.json")
            comparison["protocol"] = contract
            write_json(args.output_root / "centerline_ocr_comparison.json", comparison)
            markdown = args.output_root / "centerline_ocr_comparison.md"
            rendered = markdown_report(comparison)
            if markdown.exists() and markdown.read_text() != rendered:
                raise ValueError("Existing OCR Markdown differs from verified results")
            if not markdown.exists():
                markdown.write_text(rendered)
        except Exception as error:
            cuda = device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_initialized()
            _atomic_json(args.output_root / "centerline_ocr_failure.json", {
                "status": "FAILED", "result_class": "approximate reproduction", "action": args.action,
                "type": type(error).__name__, "message": str(error),
                "oom": isinstance(error, torch.cuda.OutOfMemoryError), "device": str(device),
                "dtype": "float32", "model_set": "PARSeq only; frozen segmentation exports",
                "batch_size": args.batch_size, "input_shape": [args.batch_size, 3, 32, 128],
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None})
            raise
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
