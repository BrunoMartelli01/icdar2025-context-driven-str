"""Frozen Total-Text crop selection: prepare -> BERT infer -> evaluate -> export.

Inference opens neither annotations nor evaluation labels. No new OCR or crops.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import re
import time

from recognition.bert_correction import BertWordScorer
from recognition.bert_crop_selection import (CONDITIONS, FAMILIES, CropPolicy,
    apply_filter, observable_sample, score_requests, selections)
from recognition.crop_group_refinements import robust_group_evidence
from scripts.evaluate_totaltext_bert_correction import (checked_file, digest,
    load_captions, payload_digest, read_json, seal_json)
from scripts.evaluate_totaltext_bert_methods import setup
from scripts.run_crop_group_postfilters import load_mass
from scripts.run_crop_group_word_selection import evaluate_predictions
from segmentation.data import load_segmentation_manifest


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/totaltext_bert_crop_selection_v1.json"
CODE = ("recognition/bert_crop_selection.py", "recognition/bert_correction.py",
        "recognition/group_word_selection.py", "recognition/crop_group_refinements.py",
        "recognition/scene_text_v1_evaluation.py", "scripts/evaluate_totaltext_bert_crop_selection.py",
        "scripts/evaluate_totaltext_bert_correction.py", "scripts/evaluate_totaltext_bert_methods.py",
        "scripts/run_crop_group_postfilters.py", "scripts/run_crop_group_word_selection.py",
        "scripts/crop_clip_runtime.py", "scripts/run_totaltext_crop_recovery_suite.py",
        "scripts/evaluate_totaltext_letterbox_threshold_sweep_train.py", "segmentation/data.py",
        "scripts/replay_totaltext_bert_crop_selection.py")


def hashes():
    return {name: digest(REPO / name) for name in CODE}


def load_config(path):
    if path.resolve().parent != REPO / "configs/approximate":
        raise ValueError("crop selection requires isolated configs/approximate/")
    cfg = read_json(path)
    if (cfg.get("schema_version") != 1 or cfg.get("reproduction_mode") != "approximate"
            or cfg.get("dataset") != "total_text" or cfg.get("table_5_claim") is not False
            or cfg.get("claim_label") != "approximate reproduction"
            or cfg.get("experiment_kind") != "bert_existing_crop_selection"
            or cfg.get("baseline_method") != "guarded_consolidated"
            or tuple(cfg.get("families", [])) != FAMILIES
            or tuple(cfg.get("conditions", [])) != CONDITIONS
            or cfg.get("evaluation", {}).get("matching_iou") != .5):
        raise ValueError("invalid approximate Total-Text crop-selection configuration")
    if set(cfg["policy"]) != set(CropPolicy.__dataclass_fields__):
        raise ValueError("every crop-selection parameter must be explicit; no fallback defaults")
    CropPolicy(**cfg["policy"])
    grid = cfg["admission_thresholds"]
    if (not grid or grid != sorted(set(grid)) or grid[0] != 0
            or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in grid)
            or len({f"{v:.2f}" for v in grid}) != len(grid)
            or not 0 < cfg["operating_point"]["minimum_baseline_correct_retention"] <= 1):
        raise ValueError("invalid predeclared threshold/retention policy")
    model = cfg["model"]
    if (not re.fullmatch(r"[0-9a-f]{40}", model["revision"])
            or model["precision"] != "float32"
            or model["scoring"] != "target_only_mean_PLL_word_l2r"
            or any(type(model[k]) is not int or model[k] < 1 for k in
                   ("batch_size", "cpu_threads", "max_input_tokens"))
            or type(model["seed"]) is not int or not 0 <= model["seed"] < 2**32
            or not model["files_sha256"] or any(not re.fullmatch(r"[0-9a-f]{64}", v)
                                               for v in model["files_sha256"].values())):
        raise ValueError("invalid pinned BERT identity/runtime")
    return cfg


def path_for(root, stage, sid):
    if not re.fullmatch(r"total_text:[0-9]+", sid):
        raise ValueError("invalid Total-Text sample ID")
    return root / stage / f"{sid.replace(':', '__')}.json"


def prepare(args):
    cfg = load_config(args.config)
    run = args.source_dir.resolve()
    refined_path = run / "resolved_configuration.json"
    refined = read_json(refined_path)
    source = Path(refined["source"])
    raw = Path(refined["raw_selector"])
    source_path = source / "resolved_configuration.json"
    source_plan = checked_file(source_path, refined["source_plan_sha256"])
    raw_path = raw / "resolved_configuration.json"
    raw_plan = read_json(raw_path)
    if (refined.get("reproduction_mode") != "approximate"
            or refined.get("table_5_claim") is not False
            or refined.get("result_class") != "APPROXIMATE_TRAIN_DIAGNOSTIC"
            or source_plan.get("reproduction_mode") != "approximate"
            or raw_plan.get("reproduction_mode") != "approximate"
            or Path(raw_plan["source"]) != source
            or raw_plan["source_plan_sha256"] != digest(source_path)):
        raise ValueError("expected frozen approximate TRAIN crop-group refinements")
    manifest = Path(refined["manifest"])
    if digest(manifest) != refined["manifest_sha256"]:
        raise ValueError("source image manifest changed")
    samples = {s.sample_id: s for s in load_segmentation_manifest(manifest, require_checksums=True).samples}
    all_ids = refined["sample_ids"]
    if (len(set(all_ids)) != len(all_ids) or any(sid not in samples for sid in all_ids)
            or len(all_ids) != cfg["source"]["expected_images"]):
        raise ValueError("source population differs from frozen TRAIN protocol")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    ids = all_ids[:args.limit] if args.limit else all_ids
    # Validate complete caption coverage even for a smoke subset.
    original = [read_json(path_for(raw, "inputs", sid)) for sid in all_ids]
    captions = load_captions(args.captions, original)
    caption_meta = read_json(args.captions)
    if (caption_meta["model_id"] != cfg["caption"]["model_id"]
            or caption_meta["revision"] != cfg["caption"]["revision"]
            or caption_meta["generation"] != cfg["caption"]["generation"]
            or digest(args.captions) != cfg["caption"]["artifact_sha256"]):
        raise ValueError("Florence caption protocol/identity changed")
    observable, labels, files = [], [], {}
    policy = CropPolicy(**cfg["policy"])
    for position, sid in enumerate(ids, 1):
        context_path = path_for(raw, "inputs", sid)
        context = read_json(context_path)
        selected_path = path_for(run, "selection", sid)
        selected = read_json(selected_path)
        if (context["sample_id"] != sid or selected["sample_id"] != sid
                or selected["context_sha256"] != digest(context_path)
                or selected["raw_selector_input_sha256"] != digest(context_path)
                or selected["frozen_selection_sha256"] != digest(path_for(raw, "selection", sid))):
            raise ValueError("refined selection/source identity changed")
        mass = load_mass(sid, selector_input=context, source=source,
                         source_plan=source_plan, sample=samples[sid])
        groups = context["groups"]
        if isinstance(groups, dict):
            groups = groups["groups"]
        evidence, _ = robust_group_evidence(context["evidence"], groups, context["rows"], mass,
                                            quantile=refined["config"]["reference_quantile"])
        observations = observable_sample(selected["methods"][cfg["baseline_method"]], evidence,
                                          context["rows"], mass, captions[sid], policy)
        observable.append({"sample_id": sid, "image_sha256": context["image_sha256"],
                           "image_wh": [mass.width, mass.height], **observations})
        source_eval = path_for(source, "evaluation", sid)
        ref_eval = read_json(path_for(run, "evaluation", sid))
        if (ref_eval["selection_sha256"] != digest(selected_path)
                or ref_eval["source_evaluation_sha256"] != digest(source_eval)):
            raise ValueError("frozen evaluation/source identity changed")
        labels.append({"sample_id": sid, "ground_truth": read_json(source_eval)["ground_truth"]})
        for file in (context_path, selected_path, source_eval, path_for(run, "evaluation", sid)):
            files[str(file)] = digest(file)
        # Authenticate each original OCR record, including caches shared by runs.
        for file_info in context["contract"]["ocr_sources"].values():
            file = Path(file_info["path"])
            expected = file_info["sha256"]
            if str(file) not in files:
                if digest(file) != expected:
                    raise ValueError(f"source PARSeq record changed: {file}")
                files[str(file)] = expected
            elif files[str(file)] != expected:
                raise ValueError("source OCR file has inconsistent contracts")
        if position % 25 == 0 or position == len(ids):
            print(f"[prepare] {position}/{len(ids)} {sid}", flush=True)
    protocol = {"schema_version": 1, "reproduction_mode": "approximate",
                "claim_label": cfg["claim_label"], "table_5_claim": False,
                "result_class": "SMOKE_TEST" if args.limit else "APPROXIMATE_TRAIN_DEVELOPMENT",
                "config": cfg, "config_sha256": digest(args.config), "code_sha256": hashes(),
                "source_dir": str(run), "source_plan_sha256": digest(refined_path),
                "generator_plan_sha256": digest(source_path), "raw_plan_sha256": digest(raw_path),
                "source_results_sha256": digest(run / "results.json"),
                "manifest_sha256": digest(manifest), "source_files_sha256": files,
                "captions_sha256": digest(args.captions),
                "caption_provenance": {k: caption_meta[k] for k in
                    ("model_id", "revision", "generation", "model_files_sha256")},
                "sample_ids": ids, "ground_truth_use": "prepare/evaluate only; no annotations in infer"}
    out = args.output_dir
    seal_json(out / "protocol.json", protocol)
    protocol_sha = digest(out / "protocol.json")
    seal_json(out / "inputs.json", {"protocol_sha256": protocol_sha, "samples": observable})
    seal_json(out / "evaluation_labels.json", {"protocol_sha256": protocol_sha, "samples": labels})
    seal_json(out / "prepared.json", {"protocol_sha256": protocol_sha,
        "inputs_sha256": digest(out / "inputs.json"),
        "evaluation_labels_sha256": digest(out / "evaluation_labels.json")})
    print(f"Prepared {len(ids)} images, {sum(len(s['groups']) for s in observable)} groups", flush=True)


def prepared(out):
    record = read_json(out / "prepared.json")
    protocol = checked_file(out / "protocol.json", record["protocol_sha256"])
    if protocol["code_sha256"] != hashes():
        raise ValueError("implementation changed; prepare a new output version")
    inputs = checked_file(out / "inputs.json", record["inputs_sha256"])
    if inputs["protocol_sha256"] != record["protocol_sha256"]:
        raise ValueError("input/protocol identity differs")
    return protocol, record, inputs


def check_payload(path, record, runtime_sha, sid):
    value = read_json(path)
    if (value.get("payload_sha256") != payload_digest({k: v for k, v in value.items()
                                                       if k != "payload_sha256"})
            or value["inputs_sha256"] != record["inputs_sha256"]
            or value["runtime_sha256"] != runtime_sha or value["sample_id"] != sid):
        raise ValueError("BERT crop inference payload/identity changed")
    return value


def _infer(args):
    protocol, record, inputs = prepared(args.output_dir)
    cfg = protocol["config"]
    torch = setup(cfg, args.device)
    if getattr(args, "reuse_inference_dir", None) is not None:
        from scripts.replay_totaltext_bert_crop_selection import replay
        return replay(args, protocol, record, inputs)
    import numpy as np
    import transformers
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    model_dir = args.model_dir.resolve()
    model_cfg = cfg["model"]
    if model_dir.name != model_cfg["revision"]:
        raise ValueError("model directory must be named by its pinned commit SHA")
    for name, expected in model_cfg["files_sha256"].items():
        if digest(model_dir / name) != expected:
            raise ValueError(f"BERT model file changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    model, loading = AutoModelForMaskedLM.from_pretrained(model_dir, local_files_only=True,
                                                        output_loading_info=True, use_safetensors=True)
    if (any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs"))
            or set(loading.get("unexpected_keys", [])) != set(model_cfg["allowed_unused_checkpoint_keys"])):
        raise RuntimeError(f"BERT learned-tensor loading failed: {loading}")
    model = model.eval().float().to(args.device)
    scorer = BertWordScorer(tokenizer, model, device=args.device,
                           batch_size=model_cfg["batch_size"], max_tokens=model_cfg["max_input_tokens"])
    runtime = {"inputs_sha256": record["inputs_sha256"], "device": args.device,
               "precision": "float32", "model_files_sha256": model_cfg["files_sha256"],
               "model_set": [model_cfg["model_id"]], "loading_audit": loading,
               "seed": model_cfg["seed"], "cpu_threads": model_cfg["cpu_threads"],
               "deterministic_algorithms": True, "tf32": False,
               "batch_size": model_cfg["batch_size"], "max_input_tokens": model_cfg["max_input_tokens"],
               "input_shape": "variable text tokens; bounds recorded in batch_size and max_input_tokens",
               "device_placement": "one complete BERT masked language model on the selected device",
               "python": platform.python_version(), "torch": torch.__version__,
               "transformers": transformers.__version__, "numpy": np.__version__,
               "reproduction_mode": "approximate", "claim_label": "approximate reproduction"}
    seal_json(args.output_dir / "runtime.json", runtime)
    runtime_sha = digest(args.output_dir / "runtime.json")
    started, status = time.monotonic(), Counter()
    for position, sample in enumerate(inputs["samples"], 1):
        sid = sample["sample_id"]
        target = path_for(args.output_dir, "inference", sid)
        if target.exists():
            value = check_payload(target, record, runtime_sha, sid)
            if value["image_sha256"] != sample["image_sha256"]:
                raise ValueError("BERT image identity changed")
        else:
            requests, links, lookup = [], [], {}
            for group in sample["groups"]:
                for condition in CONDITIONS:
                    for candidate in group["candidates"]:
                        request = {"text": candidate["raw_text"], **group["context"],
                                   "caption": sample["caption"] if condition == "scene" else None}
                        key = payload_digest(request)
                        if key not in lookup:
                            lookup[key] = len(requests)
                            requests.append(request)
                        links.append((group["group_id"], condition, lookup[key]))
            values = score_requests(scorer, requests)
            scores = {g["group_id"]: {condition: [] for condition in CONDITIONS} for g in sample["groups"]}
            for gid, condition, index in links:
                scores[gid][condition].append(values[index])
            selected = selections(sample, scores, CropPolicy(**cfg["policy"]))
            value = {"inputs_sha256": record["inputs_sha256"], "runtime_sha256": runtime_sha,
                     "sample_id": sid, "image_sha256": sample["image_sha256"],
                     "scores": scores, "unique_requests": len(requests), **selected,
                     "max_input_tokens_seen": scorer.max_input_tokens_seen,
                     "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device != "cpu" else None,
                     "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device != "cpu" else None}
            value["payload_sha256"] = payload_digest(value)
            seal_json(target, value)
        for group in value["scores"].values():
            status.update(row["status"] for rows in group.values() for row in rows)
        print(f"[BERT crops] {position}/{len(inputs['samples'])} {sid} "
              f"groups={len(sample['groups'])} requests={value['unique_requests']}", flush=True)
    receipt = {"inputs_sha256": record["inputs_sha256"], "runtime_sha256": runtime_sha,
               "images": len(inputs["samples"]), "statuses": dict(status),
               "inference_sha256": {s["sample_id"]: digest(path_for(args.output_dir, "inference", s["sample_id"]))
                                    for s in inputs["samples"]},
               "ground_truth_use": "none; only protocol, observable inputs and pinned BERT files opened"}
    seal_json(args.output_dir / "inference_complete.json", receipt)
    print(f"BERT complete in {time.monotonic() - started:.1f}s; {dict(status)}", flush=True)


def infer(args):
    import torch
    try:
        return _infer(args)
    except torch.cuda.OutOfMemoryError as error:
        seal_json(args.output_dir / "failure.json", {
            "stage": "infer", "device": args.device, "error_type": type(error).__name__,
            "message": str(error), "automatic_fallback": False,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        raise


def metrics(total, denominator):
    precision = total["correct"] / total["predictions"] if total["predictions"] else 0.
    recall = total["correct"] / denominator if denominator else 0.
    return {**dict(total), "precision_proxy": precision, "recall_proxy": recall,
            "hmean_proxy": 2 * precision * recall / (precision + recall) if precision + recall else 0.}


def method_name(family, threshold):
    return f"{family}@{threshold:.2f}"


def evaluate(args):
    protocol, record, inputs = prepared(args.output_dir)
    receipt = read_json(args.output_dir / "inference_complete.json")
    runtime_sha = digest(args.output_dir / "runtime.json")
    ids = protocol["sample_ids"]
    if (receipt["inputs_sha256"] != record["inputs_sha256"]
            or receipt["runtime_sha256"] != runtime_sha or receipt["images"] != len(ids)
            or set(receipt["inference_sha256"]) != set(ids)):
        raise ValueError("incomplete BERT inference population/runtime")
    # Check all prediction artifacts before opening any evaluation labels.
    predictions = {}
    for sample in inputs["samples"]:
        sid = sample["sample_id"]
        path = path_for(args.output_dir, "inference", sid)
        if digest(path) != receipt["inference_sha256"][sid]:
            raise ValueError("frozen inference changed")
        predictions[sid] = check_payload(path, record, runtime_sha, sid)
    labels = checked_file(args.output_dir / "evaluation_labels.json", record["evaluation_labels_sha256"])
    if (labels["protocol_sha256"] != record["protocol_sha256"]
            or [s["sample_id"] for s in labels["samples"]] != ids):
        raise ValueError("label/protocol population differs")
    cfg = protocol["config"]
    totals, denominator = {}, 0
    for label in labels["samples"]:
        sid, gt = label["sample_id"], label["ground_truth"]
        value = predictions[sid]
        evaluated = {name: evaluate_predictions(rows, gt, cfg["evaluation"]["matching_iou"])
                     for name, rows in value["choices"].items()}
        for family in FAMILIES:
            for threshold in cfg["admission_thresholds"]:
                evaluated[method_name(family, threshold)] = evaluate_predictions(
                    apply_filter(value, family, threshold), gt, cfg["evaluation"]["matching_iou"])
        base_set = set(evaluated["baseline"]["correct_gt_indices"])
        for name, result in evaluated.items():
            correct = set(result["correct_gt_indices"])
            counts = {"predictions": result["predictions"], "correct": result["correct"],
                      "recovered": len(correct - base_set), "lost": len(base_set - correct),
                      "retained_baseline_correct": len(correct & base_set)}
            totals.setdefault(name, Counter()).update(counts)
        denominator += len(gt)
        seal_json(path_for(args.output_dir, "evaluation", sid), {
            "sample_id": sid, "inference_sha256": receipt["inference_sha256"][sid],
            "gt_words": len(gt), "methods": evaluated})
    summary = {name: metrics(total, denominator) for name, total in totals.items()}
    if protocol["result_class"] != "SMOKE_TEST":
        upstream = checked_file(Path(protocol["source_dir"]) / "results.json",
                                protocol["source_results_sha256"])
        expected = upstream["summary"][cfg["baseline_method"]]
        if (summary["baseline"]["correct"] != expected["correct"]
                or summary["baseline"]["predictions"] != expected["predictions"]
                or denominator != cfg["source"]["expected_care_words"]):
            raise ValueError("source baseline/denominator not reproduced")
    retention = cfg["operating_point"]["minimum_baseline_correct_retention"]
    selected = {}
    for family in FAMILIES:
        eligible = [method_name(family, t) for t in cfg["admission_thresholds"]
                    if summary[method_name(family, t)]["retained_baseline_correct"] >=
                    retention * summary["baseline"]["correct"]]
        if not eligible:
            selected[family] = None
        else:
            selected[family] = max(eligible, key=lambda name: (
                summary[name]["precision_proxy"], summary[name]["correct"],
                summary[name]["retained_baseline_correct"], -float(name.split("@")[1])))
    result = {"schema_version": 1, "reproduction_mode": "approximate",
              "claim_label": "approximate reproduction", "table_5_claim": False,
              "result_class": protocol["result_class"], "images": len(ids), "gt_words": denominator,
              "protocol_sha256": record["protocol_sha256"],
              "inference_receipt_sha256": digest(args.output_dir / "inference_complete.json"),
              "summary": summary, "operating_points": selected,
              "operating_point_selection": f"TRAIN development only; maximize precision with >={100 * retention:g}% of baseline correct GT retained; fixed grid",
              "metric": cfg["evaluation"],
              "limitations": ["Local bbox spotting proxy, not official polygon evaluation or Table 5",
                              "Reused TRAIN development, not independent validation",
                              "Language plausibility is not visual text detection",
                              "CPU execution does not qualify RTX3070 GPU memory"]}
    seal_json(args.output_dir / "results.json", result)
    print(json.dumps({"report": str(args.output_dir / "results.json"),
                      "baseline": summary["baseline"],
                      "operating_points": {k: {"method": v, **summary[v]} if v else None
                                           for k, v in selected.items()}}, indent=2), flush=True)


def export(args):
    protocol, record, inputs = prepared(args.output_dir)
    result = read_json(args.output_dir / "results.json")
    receipt = checked_file(args.output_dir / "inference_complete.json", result["inference_receipt_sha256"])
    if result["protocol_sha256"] != record["protocol_sha256"]:
        raise ValueError("result/protocol mismatch")
    method = args.method
    if method not in result["summary"] or "@" not in method:
        raise ValueError("export requires an evaluated family@threshold method")
    family, threshold = method.split("@")
    samples = []
    for sample in inputs["samples"]:
        sid = sample["sample_id"]
        value = checked_file(path_for(args.output_dir, "inference", sid), receipt["inference_sha256"][sid])
        selected = apply_filter(value, family, float(threshold))
        by_group = {g["group_id"]: g for g in sample["groups"]}
        blocks = []
        for row in selected:
            candidate = next(c for c in by_group[row["group_id"]]["candidates"]
                             if c["candidate_id"] == row["candidate_id"])
            blocks.append({**row, "text": row["raw_text"],
                           "crop_rgb_sha256": candidate["crop_rgb_sha256"],
                           "selection_score": value["admission"][family][row["group_id"]]})
        samples.append({"sample_id": sid, "image_sha256": sample["image_sha256"], "blocks": blocks})
    seal_json(args.output_dir / f"selected-crops-{method}.json", {
        "schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
        "protocol_sha256": record["protocol_sha256"], "result_class": protocol["result_class"],
        "method": method, "selection_rule": "TRAIN-selected operating point; frozen for transfer",
        "samples": samples, "ground_truth_use": "no labels in export; threshold chosen on TRAIN"})
    print(f"Exported {sum(len(s['blocks']) for s in samples)} original crops", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "evaluate", "export"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--reuse-inference-dir", type=Path,
                        help="authenticated BERT cache with exactly identical observable samples")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--method")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output_dir.resolve() == REPO or REPO in args.output_dir.resolve().parents:
        raise ValueError("keep generated experiment artifacts outside the repository")
    required = {"prepare": ("source_dir", "captions"), "infer": ("model_dir",),
                "evaluate": (), "export": ("method",)}[args.stage]
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"{args.stage} requires --{name.replace('_', '-')}")
    {"prepare": prepare, "infer": infer, "evaluate": evaluate, "export": export}[args.stage](args)


if __name__ == "__main__":
    main()
