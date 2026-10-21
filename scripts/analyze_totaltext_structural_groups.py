#!/usr/bin/env python3
"""Four GT-blind structural grouping ablations on frozen Total-Text TEST crops."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import numpy as np

from localize.crop_recovery import add_tile_candidates, mask_candidates
from localize.structural_word_groups import TraceCollector, structural_variants
from recognition.crop_word_group_metric import grouped_oracle
from recognition.group_word_selection import (ProbabilityMass, aligned_probability,
                                              group_evidence, select_groups)
from scripts.analyze_totaltext_crop_word_groups import audit_geometry, digest, freeze
from scripts import run_crop_group_word_selection as selector_runner
from scripts.evaluate_totaltext_letterbox_threshold_sweep_train import _load_probability
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _image
from scripts.run_totaltext_crop_recovery_suite import read_tile
from segmentation.data import load_segmentation_manifest


REPO = Path(__file__).resolve().parents[1]
PATHS = REPO / "configs/approximate/totaltext_crop_group_test_local_paths_v1.json"
BASE_VARIANT = "complete_link_iou0.55"
METHODS = ("baseline", "lineage", "lineage_footprint",
           "lineage_footprint_boundaries", "lineage_footprint_boundaries_admission")
CODE = ("localize/crop_recovery.py", "localize/structural_word_groups.py",
        "scripts/analyze_totaltext_structural_groups.py",
        "recognition/group_word_selection.py", "recognition/crop_word_group_metric.py",
        "recognition/scene_text_v1_evaluation.py")


def _source(paths: Path) -> Path:
    local = json.loads(paths.read_text())
    if local["path_base"] != "repository_root":
        raise ValueError("invalid local path profile")
    source = Path(local["reusable_source"])
    return source if source.is_absolute() else (REPO / source).resolve()


def _base_grouping(source: Path) -> Path:
    possibilities = []
    for report in (source / "word_groups").glob("*/results.json"):
        value = json.loads(report.read_text())
        if (BASE_VARIANT in value.get("summary", {}) and value.get("images") == 300
            and value.get("reference", {}).get("correct") == 1823):
            possibilities.append(report.parent)
    if len(possibilities) != 1:
        raise RuntimeError(f"expected one qualified full TEST baseline grouping, found {possibilities}")
    return possibilities[0]


def _ocr_rows(prepared: dict, ocr: dict, source_plan: dict) -> tuple[list[dict], dict[str, str]]:
    frozen = {row["candidate_id"]: row for row in ocr["rows"]}
    caches = {}
    output = []
    for row in prepared["candidates"]:
        cid, crop_hash = row["candidate_id"], row["crop_rgb_sha256"]
        if crop_hash not in caches:
            record = ocr["crop_cache_sources"][crop_hash]
            cache_path = Path(record["path"])
            if digest(cache_path) != record["file_sha256"]:
                raise RuntimeError(f"frozen OCR cache differs: {cache_path}")
            caches[crop_hash] = json.loads(cache_path.read_text())
        reading = caches[crop_hash]
        if (reading["raw_text"] != frozen[cid]["raw_text"]
            or reading["parseq_checkpoint_sha256"] != source_plan["parseq_checkpoint_sha256"]):
            raise RuntimeError("frozen PARSeq reading or checkpoint differs")
        output.append({"candidate_id": cid, "bbox": row["bbox"], "methods": row["methods"],
                       "origins": row["origins"], "ocr": {field: reading[field] for field in
                       ("raw_text", "token_probabilities", "token_log_probabilities")}})
    return output, {row["candidate_id"]: row["ocr"]["raw_text"] for row in output}


def _score(groups: list[dict], rows: list[dict], readings: list[dict], gt: list[dict],
           witnesses: dict[int, list[str]], mass: ProbabilityMass) -> dict:
    grouping = {"groups": groups}
    audit = audit_geometry(rows, grouping, gt, .5)
    matched = grouped_oracle(groups, witnesses, len(gt))
    evidence = group_evidence(readings, groups, allowed_methods={"baseline", "threshold_tree",
                             "tiled_segmentation"}, mass=mass)
    predictions = select_groups(evidence, "mask_completeness_confidence")
    selected = selector_runner.evaluate_predictions(predictions, gt, .5)
    witness_groups = {gi: sum(bool(set(group["candidate_ids"]) & set(cids))
                          for group in groups) for gi, cids in witnesses.items()}
    return {"groups": len(groups), "predictions": selected["predictions"],
            "selected_correct": selected["correct"], "oracle_correct": len(matched),
            "oracle_gt_indices": sorted(matched), "selected_gt_indices": selected["correct_gt_indices"],
            "words_in_multiple_groups": sum(count > 1 for count in witness_groups.values()),
            "total_correct_word_groups": sum(witness_groups.values()),
            "parent_only_groups": sum(group.get("parent_only", False) for group in groups), **audit}


def run(args) -> Path:
    source = args.source.resolve() if args.source else _source(args.paths)
    baseline = args.baseline.resolve() if args.baseline else _base_grouping(source)
    source_plan_path = source / "resolved_configuration.json"
    base_plan_path = baseline / "resolved_configuration.json"
    source_plan = json.loads(source_plan_path.read_text())
    base_plan = json.loads(base_plan_path.read_text())
    base_result = json.loads((baseline / "results.json").read_text())
    if (source_plan["reproduction_mode"] != "approximate"
        or source_plan["config"]["split"] != "official_TEST_300"
        or base_plan["source_plan_sha256"] != digest(source_plan_path)
        or base_result["plan_sha256"] != digest(base_plan_path)
        or BASE_VARIANT not in base_result["summary"]
        or source_plan["config"]["source_methods"] !=
            ["baseline", "threshold_tree", "tiled_segmentation"]):
        raise RuntimeError("source/base grouping contracts differ")
    manifest_path = Path(args.manifest) if args.manifest else Path(json.loads(args.paths.read_text())["test_manifest"])
    manifest_path = manifest_path if manifest_path.is_absolute() else REPO / manifest_path
    if digest(manifest_path) != source_plan["manifest_sha256"]:
        raise RuntimeError("TEST manifest differs from frozen source")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    samples = {sample.sample_id: sample for sample in manifest.samples}
    ids = source_plan["sample_ids"][:args.limit] if args.limit else source_plan["sample_ids"]
    if args.limit is not None and not 1 <= args.limit <= 300:
        raise ValueError("--limit must be 1..300")
    if set(ids) - set(samples):
        raise RuntimeError("sample IDs missing from official TEST manifest")
    plan = {"schema_version": 1, "reproduction_mode": "approximate",
            "claim_label": "approximate reproduction; structural Total-Text grouping diagnostic",
            "result_class": "SMOKE_TEST" if args.limit else "APPROXIMATE_TEST_DIAGNOSTIC",
            "table_5_claim": False, "source": str(source), "baseline": str(baseline),
            "source_plan_sha256": digest(source_plan_path), "baseline_plan_sha256": digest(base_plan_path),
            "source_results_sha256": digest(source / "results.json"),
            "baseline_results_sha256": digest(baseline / "results.json"),
            "manifest_sha256": digest(manifest_path), "sample_ids": ids,
            "code_sha256": {name: digest(REPO / name) for name in CODE},
            "parameters": {"footprint_grid": 256, "lineage_containment": .85,
                           "lineage_area_ratio": .5, "tile_containment": .8,
                           "tile_area_ratio": .5, "tile_border_containment": .9,
                           "tile_border_area_ratio": .25, "seed_gap_per_height": .35,
                           "seed_transverse_overlap": .4,
                           "admission_mass_reference": "image median of group mean probabilities"},
            "scientific_choices": [{"classification": "ASSUMPTION", "choice":
              "GT-blind lineage, original-image mask footprints, local seed-gap boundaries and image-relative admission; no parameter fitting on TEST."},
              {"classification": "HARDWARE_DEVIATION", "choice":
              "Frozen float32 model outputs; CPU-only downstream recomputation; no new model inference."}],
            "execution": {"device": "cpu", "new_model_inference": False,
                          "python": sys.version, "numpy": np.__version__, "platform": platform.platform()}}
    identity = sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output.resolve() if args.output else source / "structural_groups" / identity
    freeze(output / "resolved_configuration.json", plan)
    totals = {name: Counter() for name in METHODS}
    reference = Counter()
    for position, sid in enumerate(ids, 1):
        key = sid.replace(":", "__")
        prepared_path = source / "prepared" / f"{key}.json"
        ocr_path = source / "ocr" / f"{key}.json"
        eval_path = source / "evaluation" / f"{key}.json"
        group_path = baseline / "groups" / f"{key}.json"
        prepared = json.loads(prepared_path.read_text())
        ocr = json.loads(ocr_path.read_text())
        evaluation = json.loads(eval_path.read_text())
        original = json.loads(group_path.read_text())
        if (evaluation["prepared_sha256"] != digest(prepared_path)
            or evaluation["ocr_sha256"] != digest(ocr_path)
            or original["prepared_sha256"] != digest(prepared_path)
            or ocr["contract"]["prepared_sha256"] != digest(prepared_path)):
            raise RuntimeError(f"source shard hashes differ for {sid}")
        sample = samples[sid]
        image = _image(sample, Path(source_plan["paths"]["data_root"]))
        if prepared["contract"]["image_sha256"] != sample.image_sha256:
            raise RuntimeError(f"image SHA differs for {sid}")
        sidecar = source / "probability_cache" / "probabilities" / f"{key}.json"
        if digest(sidecar) != prepared["contract"]["probability_sidecar_sha256"]:
            raise RuntimeError(f"probability sidecar differs for {sid}")
        whole = _load_probability(sidecar.with_suffix(".npy"), json.loads(sidecar.read_text()),
                                  sample=sample, protocol_sha=source_plan["probability_sidecar_contract_sha256"])
        tile_path = source / "tiles" / f"{key}.json"
        if digest(tile_path) != prepared["contract"]["tile_shard_sha256"]:
            raise RuntimeError(f"tile shard differs for {sid}")
        tiles = json.loads(tile_path.read_text())["tiles"]
        collector = TraceCollector((image.shape[1], image.shape[0]), key)
        pool = mask_candidates(whole, image, source_plan["config"],
                               include_auxiliary=False, trace=collector)
        tile_arrays = []
        for tile in tiles:
            probability = read_tile(source, tile)
            tile_arrays.append((probability, tile["window"]))
            add_tile_candidates(pool, probability, tile["window"],
                                source_plan["config"]["tiled_segmentation"]["thresholds"],
                                tile["tile_id"], trace=collector)
        recreated = {f"{key}:{cid}": row for cid, row in pool.rows.items()}
        rows = prepared["candidates"]
        if set(recreated) != {row["candidate_id"] for row in rows}:
            raise RuntimeError(f"regenerated crop IDs differ for {sid}")
        for row in rows:
            check = recreated[row["candidate_id"]]
            if any(row[field] != check[field] for field in ("bbox", "methods", "origins")):
                raise RuntimeError(f"regenerated crop recipe differs for {sid}")
        frozen_rows, raw_text = _ocr_rows(prepared, ocr, source_plan)
        groups = original["variants"][BASE_VARIANT]["groups"]
        probability = aligned_probability(whole, tile_arrays, (image.shape[1], image.shape[0]))
        mass = ProbabilityMass(probability, (image.shape[1], image.shape[0]))
        preliminary = structural_variants(groups, rows, collector, raw_text)
        mean_probability = {}
        for group in preliminary["lineage_footprint_boundaries"]:
            box = group["reference_bbox"]
            area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            mean_probability[group["group_id"]] = mass(box) / (area * (probability.shape[0] /
                                                           max(image.shape[1], image.shape[0])) ** 2)
        variants = structural_variants(groups, rows, collector, raw_text, mean_probability)
        variants = {"baseline": groups, **variants}
        grouping_shard = {"sample_id": sid, "source_prepared_sha256": digest(prepared_path),
                          "source_group_sha256": digest(group_path), "ground_truth_use": "none",
                          "trace_events": sum(map(len, collector.events.values())), "variants": variants}
        frozen_group_path = output / "groups" / f"{key}.json"
        freeze(frozen_group_path, grouping_shard)
        gt = evaluation["ground_truth"]
        witnesses = {int(index): ids for index, ids in
                     evaluation["arms"]["all_methods"]["witnesses"].items()}
        image_scores = {}
        for name, variant in variants.items():
            scored = _score(variant, rows, frozen_rows, gt, witnesses, mass)
            image_scores[name] = scored
            totals[name].update({field: value for field, value in scored.items()
                                 if isinstance(value, int)})
        reference.update(candidates=len(rows), gt_words=len(gt),
                         oracle_correct=len(evaluation["arms"]["all_methods"]["correct_gt_indices"]))
        freeze(output / "evaluation" / f"{key}.json", {"sample_id": sid,
               "ground_truth_use": "post-hoc only", "source_evaluation_sha256": digest(eval_path),
               "grouping_sha256": digest(frozen_group_path), "variants": image_scores})
        print(f"[structural groups] {position}/{len(ids)} {sid} " +
              " ".join(f"{name}={image_scores[name]['groups']}/{image_scores[name]['oracle_correct']}"
                       for name in METHODS), flush=True)
    for name, total in totals.items():
        total["oracle_lost"] = reference["oracle_correct"] - total["oracle_correct"]
        total["mean_groups_per_recoverable_word"] = (total["total_correct_word_groups"] /
            reference["oracle_correct"] if reference["oracle_correct"] else None)
        total["core_pair_precision"] = (total["same_word_pairs"] / total["labelled_pairs"]
            if total["labelled_pairs"] else None)
    if not args.limit and (reference["gt_words"] != 2204 or reference["oracle_correct"] != 1823):
        raise RuntimeError("full frozen TEST denominator or crop oracle changed")
    result = {"schema_version": 1, "reproduction_mode": "approximate",
              "result_class": plan["result_class"], "table_5_claim": False,
              "plan_sha256": digest(output / "resolved_configuration.json"),
              "images": len(ids), "reference": dict(reference),
              "summary": {name: dict(total) for name, total in totals.items()},
              "metric": "Unicode casefold/P-S removal; one original crop per group; bbox IoU>=0.5; care GT denominator",
              "limitation": "Approximate TEST diagnostic; source model and OCR outputs frozen; not official polygon hmean."}
    freeze(output / "results.json", result)
    lines = ["# Structural grouping battery", "", plan["claim_label"], "",
             f"{len(ids)} TEST images; {reference['candidates']} frozen crop readings; "
             f"source oracle {reference['oracle_correct']}/{reference['gt_words']}.", "",
             "| Method | Groups | Oracle correct | Lost | Selected correct | Outputs | Duplicated recoverable words | Mixed core groups |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, total in totals.items():
        lines.append(f"| {name} | {total['groups']} | {total['oracle_correct']} | {total['oracle_lost']} | "
                     f"{total['selected_correct']} | {total['predictions']} | "
                     f"{total['words_in_multiple_groups']} | {total['mixed_core_groups']} |")
    lines += ["", "Selected correct uses the unchanged mask-completeness/PARSeq-confidence rule.",
              "No TEST labels enter group construction; labels are read after per-image groups are frozen."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(output / "results.json"), "summary": result["summary"]}), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=PATHS)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="smoke test only; omit for full TEST-300")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
