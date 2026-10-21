"""Transfer frozen CLIP and paper context routers to cached high-resolution OCR.

APPROXIMATE only. Never trains, downloads, writes images or invokes DeepSolo.
Routing sees original RGB/PARSeq outputs and frozen context, never annotations.
Qualified full-image DeepSolo responses are reused for actual combined metrics.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from recognition.clip_image_router import accepted
from recognition.clip_router_evaluation import (
    aggregate_split, deepsolo_predictions, evaluate_predictions, fixed_image_score,
)
from recognition.group_word_selection import reading_confidence
from recognition.paper_image_router_evaluation import word_metrics
from scripts.crop_clip_runtime import sha, write_json
from scripts.evaluate_totaltext_highres_contour_padding import _annotation_path, contour_proposals
from scripts.run_clip_high_recall_admission import payload_digest, seal
from scripts.run_clip_image_router_word_advantage import WordAdvantageScorer
from scripts import run_clip_router_e2e as original
from scripts import run_paper_image_router_baseline as paper
from segmentation.training import seed_everything

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/approximate/totaltext_highres_routing_v1.json"
PATHS = REPO / "configs/approximate/totaltext_highres_routing_local_paths_v1.json"
CODE = tuple(sorted(set((*original.CODE, *paper.CODE, __file__))))


def checked(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError(f"Frozen input SHA256 changed: {path}")
    return json.loads(Path(path).read_text())


def crop_rows(proposals, ocr_by_hash):
    """Keep every original proposal, including empty/low-confidence readings."""
    ranks = [p["rank"] for p in proposals]
    if ranks != list(range(1, len(proposals) + 1)):
        raise ValueError("Noncontiguous or duplicate source crop ranks")
    rows = []
    for p in proposals:
        ocr = ocr_by_hash[p["crop_rgb_sha256"]]
        if ocr["crop_rgb_sha256"] != p["crop_rgb_sha256"]:
            raise ValueError("OCR/crop hash mismatch")
        if (ocr["parseq_checkpoint_sha256"] != original.PARSEQ_CHECKPOINT_SHA256
                or ocr["parseq_source_archive_sha256"] != original.PARSEQ_RUNTIME_ARCHIVE_SHA256):
            raise ValueError("Unqualified PARSeq cache")
        native = reading_confidence(ocr)
        rows.append({"instance_id": p["rank"], "bbox": p["bbox"],
            "raw_text": ocr["raw_text"], "crop_rgb_sha256": p["crop_rgb_sha256"],
            "native": native, "confidence": native["confidence_balanced"]})
    return rows


def predictions(rows):
    return [{"instance_id": r["instance_id"], "bbox": r["bbox"],
             "text": r["raw_text"], "ignored": False} for r in rows]


def summarize_methods(samples, routes, fallback_predictions, paper_routes, *, iou_threshold, paper_threshold):
    """Evaluate frozen decisions; annotations enter only this evaluation step."""
    ids = [s["observed"]["sample_id"] for s in samples]
    expected_words = sum(len(s["ground_truth"]) for s in samples)
    if set(routes) != set(ids) or set(fallback_predictions) != set(ids):
        raise ValueError("Incomplete routing/fallback population")
    methods = {"parseq_highres": [], "clip_then_deepsolo": [], "deepsolo_only": []}
    methods.update({"paper_" + p + "_then_deepsolo": [] for p in paper_routes})
    per_image = []
    for s in samples:
        sid = s["observed"]["sample_id"]
        gt = s["ground_truth"]
        ps, ds = predictions(routes[sid]["rows"]), fallback_predictions[sid]
        selection = {
            "parseq_highres": ("accept_parseq", "T3", ps),
            "deepsolo_only": ("deepsolo", "DeepSolo", ds),
            "clip_then_deepsolo": (routes[sid]["action"],
                "T3" if routes[sid]["action"] == "accept_parseq" else "DeepSolo",
                ps if routes[sid]["action"] == "accept_parseq" else ds),
        }
        for profile, values in paper_routes.items():
            f = values[sid]
            use = accepted(f["scores"]["paper_selected"], paper_threshold)
            source = f["selected_source"] if use else "DeepSolo"
            output = ds if not use else (ps if source == "T3" else [
                {"text": f["T1_raw_text"], "bbox": None, "ignored": False, "unlocalized": True}])
            selection["paper_" + profile + "_then_deepsolo"] = (
                "accept_parseq" if use else "deepsolo", source, output)
        row = {"sample_id": sid, "ground_truth": gt, "methods": {}}
        for method, (action, source, outputs) in selection.items():
            spatial = None if source == "T1" else evaluate_predictions(outputs, gt, iou_threshold=iou_threshold)
            m = {"status": "SUCCESS", "sample_id": sid, "action": action, "source": source,
                 "word_metrics": word_metrics(outputs, gt), "spatial_metrics": spatial}
            methods[method].append(m)
            row["methods"][method] = m
        per_image.append(row)
    reports = {}
    for name, rows in methods.items():
        lexical_rows = [{**r, "metrics": r["word_metrics"]} for r in rows]
        spatial_rows = [{**r, "metrics": r["spatial_metrics"]} for r in rows]
        report = {"lexical": aggregate_split(lexical_rows, ids, expected_words),
            "spatial": aggregate_split(spatial_rows, ids, expected_words)
                if all(r["metrics"] is not None for r in spatial_rows) else None,
            "accepted_t1_images": sum(r["source"] == "T1" for r in rows),
            "accepted_t3_images": sum(r["source"] == "T3" for r in rows)}
        if report["spatial"] is None:
            report["spatial_unavailable_reason"] = "T1 has no localization; no invented bbox"
        reports[name] = report
    return reports, per_image


def qualify(args):
    cfg, raw_paths = original.read_approximate(args.config), original.read_approximate(args.paths)
    if (args.mode != "approximate" or cfg["reproduction_mode"] != "approximate"
            or cfg["claim_label"] != "approximate reproduction" or cfg["table_5_claim"] is not False
            or cfg["padding_native_pixels"] != 0 or cfg["matching_iou"] != .5
            or cfg["paper_router"] != {"alpha": .6, "beta": .4, "tau": .8,
                "primary_gate": "paper_selected", "lexical_mode": "raw"}
            or raw_paths["path_base"] != "repository_root"):
        raise ValueError("Unsupported isolated approximate transfer configuration")
    paths = {k: (REPO / v).resolve() for k, v in raw_paths.items()
             if k not in ("schema_version", "path_base")}
    if args.output_root:
        paths["output_root"] = args.output_root.resolve()
    if any(paths["output_root"].is_relative_to(paths[k]) for k in
           ("source_run", "source_training_run", "paper_run")):
        raise ValueError("Output must not mutate an input run")
    q = original.qualify(SimpleNamespace(mode="approximate", output_root=None,
        config=REPO / cfg["clip_e2e_config"], paths=REPO / cfg["clip_e2e_paths"]))
    old_cfg, _, router_cfg, _, _, spec, identity, fallback, _, reference, inputs, splits = q
    if old_cfg["threshold"] != cfg["clip_threshold"]:
        raise ValueError("Frozen TRAIN router threshold changed")
    root = paths["source_run"]
    for path, digest in ((root / "protocol.json", cfg["source_protocol_sha256"]),
                         (root / "results.json", cfg["source_results_sha256"]),
                         (paths["source_training_run"] / "run_manifest.json", cfg["source_run_manifest_sha256"])):
        inputs[str(path)] = digest
        checked(path, digest)
    manifest = json.loads((paths["source_training_run"] / "run_manifest.json").read_text())
    cp = manifest["checkpoint"]
    if (not manifest["training_complete"] or cp["epoch"] != cfg["checkpoint_epoch"]
            or cp["sha256"] != cfg["checkpoint_sha256"]):
        raise ValueError("Best high-resolution checkpoint differs")
    checkpoint = paths["source_training_run"] / cp["path"]
    if sha(checkpoint) != cp["sha256"]:
        raise ValueError("High-resolution checkpoint bytes changed")
    inputs[str(checkpoint)] = cp["sha256"]
    runtime_path = root / cfg["source_variant"] / "inference_runtime.json"
    inputs[str(runtime_path)] = sha(runtime_path)
    samples = splits["test"]
    if (len(samples) != cfg["expected_images"]
            or sum(len(s["ground_truth"]) for s in samples) != cfg["expected_gt_words"]):
        raise ValueError("Expected complete TEST denominator changed")
    if args.limit:
        samples = samples[:args.limit]
    ocr = {}
    for s in samples:
        sid = s["observed"]["sample_id"]
        key = original.key(sid)
        path = root / cfg["source_variant"] / "samples" / (key + ".json")
        mask_path = root / cfg["source_variant"] / "masks" / (key + ".png")
        v = json.loads(path.read_text())
        if (v["sample_id"] != sid or v["protocol_sha256"] != cfg["source_protocol_sha256"]
                or v["image_sha256"] != s["observed"]["image_sha256"]
                or v["ground_truth"] != s["ground_truth"]
                or v["annotation_sha256"] != sha(_annotation_path(s["sample"], s["data_root"]))
                or sha(mask_path) != v["native_mask_file_sha256"]):
            raise ValueError("High-resolution sample/GT/mask identity changed: " + sid)
        with Image.open(mask_path) as m:
            native = (np.asarray(m) > 0).astype(np.uint8)
        if hashlib.sha256(native.tobytes()).hexdigest() != v["native_mask_array_sha256"]:
            raise ValueError("Native mask array changed")
        regenerated = contour_proposals(native, (s["image_wh"][1], s["image_wh"][0]), [0])["0"]
        old = v["paddings"]["0"]
        if ({**old, "proposals": [{k: x for k, x in p.items() if k != "crop_rgb_sha256"}
                for p in old["proposals"]]} != regenerated):
            raise ValueError("Original native256 contour replay differs: " + sid)
        for p in old["proposals"]:
            digest = p["crop_rgb_sha256"]
            op = root / "ocr" / (digest + ".json")
            if digest not in ocr:
                ocr[digest] = json.loads(op.read_text())
                inputs[str(op)] = sha(op)
        s["observed"]["rows"] = crop_rows(old["proposals"], ocr)
        inputs[str(path)], inputs[str(mask_path)] = sha(path), sha(mask_path)
    baseline, _ = summarize_methods(samples, {s["observed"]["sample_id"]: {
        "rows": s["observed"]["rows"], "action": "accept_parseq"} for s in samples},
        {s["observed"]["sample_id"]: [] for s in samples}, {},
        iou_threshold=cfg["matching_iou"], paper_threshold=cfg["paper_router"]["tau"])
    if not args.limit and (baseline["parseq_highres"]["spatial"]["tp"] != cfg["expected_parseq_spatial_tp"]
            or baseline["parseq_highres"]["lexical"]["tp"] != cfg["expected_parseq_lexical_tp"]
            or baseline["parseq_highres"]["spatial"]["detected_words"] != cfg["expected_parseq_outputs"]):
        raise ValueError("Full high-resolution OCR baseline replay differs")
    paper_plan_path = paths["paper_run"] / "resolved_configuration.json"
    paper_plan = checked(paper_plan_path, cfg["paper_plan_sha256"])
    checked(paths["paper_run"] / "per_image_scores.json", cfg["paper_scores_sha256"])
    t1_manifest = checked(paths["paper_t1_manifest"], cfg["paper_t1_manifest_sha256"])
    for p in (paper_plan_path, paths["paper_run"] / "per_image_scores.json", paths["paper_t1_manifest"]):
        inputs[str(p)] = sha(p)
    for n in ("scoring/semantic.py", "scoring/paper_image_router.py", "scoring/lexical.py", "recognition/parseq.py"):
        if sha(REPO / n) != paper_plan["code_sha256"][n]:
            raise ValueError("Frozen paper scorer/recognizer implementation changed: " + n)
    if (not cfg["paper_profiles"] or not set(cfg["paper_profiles"]).issubset(paper_plan["profiles"])
            or paper_plan["config"]["router"] != cfg["paper_router"]
            or paper_plan["software"] != paper.software()):
        raise ValueError("Qualified paper scoring recipe/software changed")
    asset = paper_plan["assets"]["mpnet"]
    mpnet_root = Path(asset["snapshot"])
    for n, digest in {**asset["configuration_file_sha256"], "model.safetensors": asset["weight_sha256"]}.items():
        if sha(mpnet_root / n) != digest:
            raise ValueError("Pinned MPNet asset changed: " + n)
        inputs[str(mpnet_root / n)] = digest
    source_t1_plan = checked(t1_manifest["source_plan_path"], t1_manifest["source_plan_sha256"])
    by_id = {e["sample_id"]: e for e in t1_manifest["samples"]}
    frozen_t1 = {}
    if len(by_id) != len(t1_manifest["samples"]):
        raise ValueError("Duplicate T1 image")
    for s in samples:
        obs = s["observed"]
        entry = by_id[obs["sample_id"]]
        value = checked(entry["path"], entry["sha256"])
        c = value["contract"]
        if (value["payload_sha256"] != payload_digest(value) or c["stage"] != "t1"
                or c["sample_id"] != obs["sample_id"] or c["image_sha256"] != obs["image_sha256"]
                or c["ground_truth_used"] or c["asset"] != source_t1_plan["assets"]["parseq"]
                or value["result"]["provenance"]["checkpoint_sha256"] != original.PARSEQ_CHECKPOINT_SHA256
                or not value["result"]["provenance"]["strict_state_dict_load"]):
            raise ValueError("Frozen T1 source changed")
        frozen_t1[obs["sample_id"]] = value
        inputs[entry["path"]] = entry["sha256"]
        for profile in cfg["paper_profiles"]:
            contract = paper.cache_contract(paper_plan, obs, "captions", profile)
            cap = paper.read_cache(Path(paper_plan["paths"]["output_root"]), contract)
            if cap is None or cap["result"]["ground_truth_in_caption"] or cap["result"]["parseq_in_prompt"]:
                raise ValueError("Missing/invalid original-image caption")
            cp = paper.cache_path(Path(paper_plan["paths"]["output_root"]), contract)
            inputs[str(cp)] = sha(cp)
    paper_plan["_frozen_t1"] = frozen_t1
    return cfg, paths, router_cfg, spec, identity, fallback, reference, inputs, samples, paper_plan


def prepare_clip(samples, root, plan_sha, cfg, router_cfg, identity, spec, reference):
    model, scorer = None, None
    result = {}
    start = time.monotonic()
    try:
        for i, sample in enumerate(samples, 1):
            obs = sample["observed"]
            path = root / "routes" / (original.key(obs["sample_id"]) + ".json")
            contract = {"plan_sha256": plan_sha, "sample_id": obs["sample_id"],
                        "image_sha256": obs["image_sha256"]}
            if path.exists():
                route = original.cached(path, contract)
                if (route["payload_sha256"] != payload_digest(route)
                        or [{k: v for k, v in r.items() if k != "clip_word_advantage"}
                            for r in route["rows"]] != obs["rows"]):
                    raise ValueError("Cached high-resolution route changed")
            else:
                rows = obs["rows"]
                t0 = time.monotonic()
                if rows and scorer is None:
                    from transformers import CLIPModel, CLIPProcessor
                    model, loading = CLIPModel.from_pretrained(identity["local_path"], local_files_only=True,
                        use_safetensors=spec["weight_file"].endswith("safetensors"), output_loading_info=True)
                    if any(loading.values()):
                        raise ValueError("CLIP checkpoint key mismatch: " + str(loading))
                    model = model.eval().float().to("cpu")
                    scorer = WordAdvantageScorer(model, CLIPProcessor.from_pretrained(
                        identity["local_path"], local_files_only=True), router_cfg, "cpu")
                values = scorer.score(rows, original.original_crops(obs, rows)) if rows else []
                rows = [{**r, "clip_word_advantage": value} for r, value in zip(rows, values)]
                score = fixed_image_score(rows, reference["rank_reference"])
                route = {"contract": contract, "rows": rows, "score": score,
                    "threshold": cfg["clip_threshold"],
                    "action": "accept_parseq" if accepted(score, cfg["clip_threshold"]) else "deepsolo",
                    "ground_truth_in_routing": False,
                    "runtime": {"seconds": time.monotonic() - t0, "device": "cpu", "precision": "float32",
                        "batch_size": router_cfg["execution"]["batch_size"], "input_shape": [3, 224, 224],
                        "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None,
                        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}}
                route["payload_sha256"] = payload_digest(route)
                seal(path, route)
            # Recheck every decision against the frozen rule, including resume.
            score = fixed_image_score(route["rows"], reference["rank_reference"])
            if route["score"] != score or route["action"] != (
                    "accept_parseq" if accepted(score, cfg["clip_threshold"]) else "deepsolo"):
                raise ValueError("Cached routing decision differs")
            result[obs["sample_id"]] = route
            if i % 25 == 0 or i == len(samples):
                print(f"[highres CLIP] {i}/{len(samples)}", flush=True)
    finally:
        del scorer, model
        gc.collect()
    seal(root / "routing_requests.json", {"plan_sha256": plan_sha, "ground_truth_in_routing": False,
        "threshold": cfg["clip_threshold"], "test_fitting": False,
        "images": [{"sample_id": sid, "score": r["score"], "action": r["action"]}
                   for sid, r in result.items()]})
    return result, time.monotonic() - start


def run(args):
    cfg, paths, router_cfg, spec, identity, fallback, reference, inputs, samples, pp = qualify(args)
    if args.check:
        print(json.dumps({"status": "PREFLIGHT_PASS", "images": len(samples),
            "checkpoint_epoch": cfg["checkpoint_epoch"], "new_inference_device": "cpu",
            "saved_crop_files": 0, "complete_deepsolo_cache": True}), flush=True)
        return
    seed_everything(cfg["seed"], deterministic_algorithms=True)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = paths["output_root"] / ("full" if not args.limit else f"smoke-{args.limit}")
    if shutil.disk_usage(root.parent if root.parent.exists() else REPO).free < 64 * 1024**2:
        raise RuntimeError("Less than64MiB free; refusing diagnostic writes")
    plan = {"schema_version": 1, "claim_label": "approximate reproduction",
        "reproduction_mode": "approximate", "table_5_claim": False, "config": cfg,
        "config_sha256": sha(args.config), "paths": {k: str(v) for k, v in paths.items()},
        "paths_sha256": sha(args.paths), "device": "cpu", "precision": "float32",
        "code_sha256": {str(Path(n).resolve().relative_to(REPO)): sha(Path(n)) for n in CODE},
        "software": paper.software(), "seed": cfg["seed"], "deterministic_algorithms": True,
        "tf32": False, "new_gpu_inference": False, "clip_model": identity,
        "input_sha256": inputs, "sample_ids": [s["observed"]["sample_id"] for s in samples],
        "result_class": "smoke-test result" if args.limit else "Total-Text TEST exploratory transfer diagnostic",
        "test_fitting": False, "reference_fit_split": "previous Text+Kernel TRAIN80",
        "saved_crop_files": 0, "new_mask_files": 0, "new_checkpoints": 0}
    seal(root / "resolved_configuration.json", plan)
    plan_sha = sha(root / "resolved_configuration.json")
    start = time.monotonic()
    try:
        routes, clip_seconds = prepare_clip(samples, root, plan_sha, cfg, router_cfg, identity, spec, reference)
        score_path = root / "paper_scores.json"
        if score_path.exists():
            cached = original.cached(score_path, {"plan_sha256": plan_sha})
            if cached["payload_sha256"] != payload_digest(cached):
                raise ValueError("Cached paper scores changed")
            features = cached["profiles"]
            paper_seconds = 0.
        else:
            print("[highres paper] scoring cached full-image BLIP-2 and new T3 with MPNet", flush=True)
            pp["profiles"] = cfg["paper_profiles"]
            t0 = time.monotonic()
            features = paper.score_samples(pp["config"], {"output_root": Path(pp["paths"]["output_root"])}, pp,
                {"test": [s["observed"] for s in samples]})
            paper_seconds = time.monotonic() - t0
            value = {"contract": {"plan_sha256": plan_sha}, "profiles": features,
                     "ground_truth_in_routing": False, "test_fitting": False,
                     "runtime": {"device": "cpu", "precision": "float32", "seconds": paper_seconds,
                         "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}}
            value["payload_sha256"] = payload_digest(value)
            seal(score_path, value)
        ds = {}
        for s in samples:
            sid = s["observed"]["sample_id"]
            if sha(s["ds_cache_path"]) != inputs[str(s["ds_cache_path"])]:
                raise ValueError("DeepSolo source changed after preflight: " + sid)
            raw = json.loads(s["ds_cache_path"].read_text())
            ds[sid] = deepsolo_predictions(raw["response"], s["image_wh"], s["orientation"],
                np.ones((256, 256), dtype=np.bool_), fallback["score_threshold"])
            if any(p["ignored"] for p in ds[sid]):
                raise ValueError("Prediction ignore policy must retain every output")
        reports, per_image = summarize_methods(samples, routes, ds, features,
            iou_threshold=cfg["matching_iou"], paper_threshold=cfg["paper_router"]["tau"])
        result = {"schema_version": 1, "status": "SUCCESS", "claim_label": "approximate reproduction",
            "reproduction_mode": "approximate", "table_5_claim": False, "result_class": plan["result_class"],
            "plan_sha256": plan_sha, "checkpoint_epoch": cfg["checkpoint_epoch"],
            "methods": reports, "per_image": per_image, "fallback_predictions": ds,
            "protocol": {"spatial": "micro maximum-cardinality one-to-one bboxIoU>=.5 and Unicode casefold exact",
                "lexical": "micro per-image casefold-exact word multiset; punctuation retained; no localization",
                "ignore": cfg["prediction_ignore_policy"], "gt": "all care GT;2204 words in complete TEST300",
                "empty_predictions": "all care words become FN", "failures": "abort complete aggregate",
                "clip_threshold": cfg["clip_threshold"], "threshold_refitted": False,
                "paper_router": cfg["paper_router"], "saved_crop_files": 0},
            "runtime": {"device": "cpu", "precision": "float32", "clip_seconds_this_invocation": clip_seconds,
                "mpnet_seconds_this_invocation": paper_seconds, "elapsed_seconds": time.monotonic() - start,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None, "oom": False}}
        write_json(root / "results.json", result)
        print(json.dumps({"report": str(root / "results.json"), "methods": reports}), flush=True)
        return root
    except Exception as error:
        write_json(root / "failure.json", {"status": "FAILED", "error_type": type(error).__name__,
            "error": str(error), "out_of_memory": isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError))})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=["approximate"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--paths", type=Path, default=PATHS)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--check", action="store_true")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.limit is not None and not 1 <= args.limit < 300:
        p.error("--limit must be between1 and299; omit for complete TEST300")
    run(args)


if __name__ == "__main__":
    main()
