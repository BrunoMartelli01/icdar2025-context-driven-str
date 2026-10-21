"""Persistent TRAIN caption/routing comparison; staged GPU export, CPU replay.

Reads frozen practical-v1 crop OCR. Never imports segmentation inference, changes
boxes, loads TEST images, or uses GT to generate captions/order/route text.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace

from diagnostics.lavis_captioner_candidates import CANDIDATES, digest, sha, write_json, atomic


TRAIN_SHA = "3aeecf40411326f0721e5d4eea1bf74a9357eab1c9da54a77d0624b6e24f7c14"
CHECKPOINT_SHA = "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2"
ANTI_SHA = "d427d18e435b314c4407906d86d1558c4908a6c57a55d670937262cdc54a1497"
COMPACT_SCOPE = {"split": "TRAIN", "images": 229, "test_loaded": False, "gt_loaded": False}


def read(path):
    return json.loads(Path(path).read_text())


def contained(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise RuntimeError("Artifact path escapes its root")
    return path


def seal_write(path, value):
    payload = {**value, "payload_sha256": digest(value)}
    if path.exists():
        if read(path) != payload:
            raise RuntimeError(f"Refusing to replace a frozen artifact: {path}")
    else:
        write_json(path, payload)


def checked(path, contract=None):
    obj = read(path)
    payload = {k: v for k, v in obj.items() if k != "payload_sha256"}
    if obj.get("payload_sha256") != digest(payload):
        raise RuntimeError(f"Artifact integrity failure: {path}")
    if contract is not None and obj.get("contract_sha256") != contract:
        raise RuntimeError(f"Artifact contract failure: {path}")
    return obj


def dependencies():
    names = [__file__, "diagnostics/run_caption_routing_train_host.sh", "diagnostics/lavis_captioner_candidates.py",
             "diagnostics/joined_t3.py", "scoring/local_context.py", "scoring/strategies.py",
             "scoring/semantic.py", "recognition/parseq.py", "caption/blip2_captioner.py",
             "localize/practical_candidate_selection_v2.py"]
    return {str(Path(p)): sha(p) for p in names}


def prepare(args):
    from segmentation.data import load_segmentation_manifest
    from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_REVISION, PARSEQ_RUNTIME_ARCHIVE_SHA256
    from diagnostics.joined_t3 import join_crop_transcriptions
    from scripts.diagnose_local_context_matching_train import _load_captions
    from scripts.export_train_blip2_captions import _validate_shard

    config = read(args.config)
    if (config["split"] != "TRAIN" or config["images"] != 229 or
            (config["alpha"], config["beta"], config["tau"]) != (.6, .4, .8)):
        raise RuntimeError("Wrong frozen TRAIN/scoring configuration")
    if sha(args.train_manifest) != TRAIN_SHA:
        raise RuntimeError("Expected official author-protocol TRAIN-229 manifest")
    manifest = load_segmentation_manifest(args.train_manifest)
    if manifest.split_role != "train" or len(manifest.samples) != 229:
        raise RuntimeError("TRAIN-only population required")
    compact_path = args.source_root / "train-practical-remap-v1-parseq-compact-v1/manifest.json"
    compact = read(compact_path)
    if compact["scope"] != COMPACT_SCOPE or compact["candidate_count"] != 1203:
        raise RuntimeError("Frozen v1 population mismatch")
    if (compact["segmentation"]["checkpoint_sha256"] != CHECKPOINT_SHA or
            compact["parseq"]["checkpoint_sha256"] != PARSEQ_CHECKPOINT_SHA256):
        raise RuntimeError("Frozen model provenance mismatch")
    refs = {str(row["sample_id"]): row for row in compact["samples"]}
    if len(refs) != 229 or set(refs) != {s.sample_id for s in manifest.samples}:
        raise RuntimeError("Compact cache sample population mismatch")
    inputs = []
    for sample in manifest.samples:
        ref = refs[sample.sample_id]
        cp = contained(compact_path.parent, ref["candidate_shard"])
        op = contained(compact_path.parent, ref["ocr_shard"])
        if sha(cp) != ref["candidate_shard_sha256"] or sha(op) != ref["ocr_shard_sha256"]:
            raise RuntimeError(f"Cached OCR/crop checksum mismatch: {sample.sample_id}")
        boxes, ocr = read(cp), read(op)
        if (boxes["sample_id"] != sample.sample_id or boxes["image_sha256"] != sample.image_sha256 or
                ocr["parseq_checkpoint_sha256"] != PARSEQ_CHECKPOINT_SHA256 or
                ocr["parseq_runtime_revision"] != PARSEQ_RUNTIME_REVISION or
                ocr["parseq_runtime_archive_sha256"] != PARSEQ_RUNTIME_ARCHIVE_SHA256):
            raise RuntimeError("Cached image/PARSeq provenance mismatch")
        texts = {r["candidate_id"]: r["raw_text"] for r in ocr["results"]}
        if len(texts) != ref["candidate_count"] or set(texts) != {b["candidate_id"] for b in boxes["candidates"]}:
            raise RuntimeError("Cached candidate IDs mismatch")
        proposals = [{"candidate_id": b["candidate_id"], "bbox": b["bbox"], "text": texts[b["candidate_id"]]}
                     for b in boxes["candidates"]]
        image = contained(args.data_root, sample.image_path)
        if sha(image) != sample.image_sha256:
            raise RuntimeError("TRAIN image checksum mismatch")
        inputs.append({"sample_id": sample.sample_id, "image_path": str(image), "image_sha256": sample.image_sha256,
                       "proposals": proposals, "joined": join_crop_transcriptions(proposals)})
    if sum(len(s["proposals"]) for s in inputs) != 1203:
        raise RuntimeError("Expected 1203 frozen v1 blocks")
    anti_path = args.source_root / "icdar13-train-blip2-captions-practical-antirepetition-v1/manifest.json"
    if sha(anti_path) != ANTI_SHA:
        raise RuntimeError("Frozen anti-repetition caption manifest changed")
    captions, anti, profile = _load_captions(anti_path)
    anti_by_id = {r["sample_id"]: r for r in anti["samples"]}
    if set(anti_by_id) != set(refs):
        raise RuntimeError("TRAIN caption sample IDs mismatch")
    for sample in inputs:
        entry = anti_by_id[sample["sample_id"]]
        shard = read(contained(anti_path.parent, entry["shard_path"]))
        _validate_shard(shard, sample_id=sample["sample_id"], image_sha256=sample["image_sha256"], profile=profile)
        if shard["result"]["raw_caption"] != captions[sample["sample_id"]] or entry["image_sha256"] != sample["image_sha256"]:
            raise RuntimeError("Caption manifest and sealed shard disagree")
    specs = copy.deepcopy(CANDIDATES)
    specs["florence2-base-ft"]["tasks"] = ["<MORE_DETAILED_CAPTION>"]
    specs["florence2-large-ft"] = config["florence_large"]
    specs["instructblip"] = config["instructblip"]
    protocol = {"config": config, "specs": specs, "dependencies": dependencies(),
                "config_sha256": sha(args.config), "train_manifest_sha256": TRAIN_SHA,
                "compact_manifest_sha256": sha(compact_path), "anti_manifest_sha256": ANTI_SHA,
                "inputs_sha256": digest(inputs), "data_root": str(args.data_root.resolve()),
                "asset_root": str(args.asset_root.resolve()), "visual_cache": str(args.visual_cache.resolve()),
                "parseq_runtime_revision": PARSEQ_RUNTIME_REVISION, "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                "mask_checkpoint_sha256": CHECKPOINT_SHA, "gt_used_for_inference": False}
    contract = digest(protocol)
    seal_write(args.output_dir / "protocol.json", protocol)
    seal_write(args.output_dir / "inputs.json", {"contract_sha256": contract, "samples": inputs})
    source_copy = args.output_dir / "source"
    for source in protocol["dependencies"]:
        target = source_copy / Path(source).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(args.config, source_copy / args.config.name)
    for sample in inputs:
        seal_write(shard_path(args, "flan-antirepetition", sample), {
            "contract_sha256": contract, "sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"],
            "text": captions[sample["sample_id"]], "source_manifest_sha256": ANTI_SHA,
            "source_shard_sha256": anti_by_id[sample["sample_id"]]["shard_sha256"]})
    print("[preflight] TRAIN 229; frozen v1 blocks 1203; cached Flan anti-repetition 229; integrity PASS", flush=True)
    for stage in [*config["caption_profiles"], "t1"]:
        count = sum(shard_path(args, stage, s).exists() for s in inputs)
        print(f"[preflight] {stage}: {count}/229 cached", flush=True)


def shard_path(args, stage, sample):
    return args.output_dir / "cache" / stage / f"{int(sample['sample_id']):04d}.json"


def load_inputs(args):
    protocol = checked(args.output_dir / "protocol.json")
    plain = {k: v for k, v in protocol.items() if k != "payload_sha256"}
    if protocol["dependencies"] != dependencies():
        raise RuntimeError("Frozen runtime source changed; use a new output directory")
    contract = digest(plain)
    inputs = checked(args.output_dir / "inputs.json", contract)["samples"]
    if len(inputs) != 229 or digest(inputs) != protocol["inputs_sha256"]:
        raise RuntimeError("Frozen TRAIN input manifest mismatch")
    return protocol, contract, inputs


def export(args):
    protocol, contract, samples = load_inputs(args)
    if args.stage not in [*protocol["config"]["caption_profiles"], "t1"]:
        raise ValueError("Unknown stage")
    pending = []
    for sample in samples:
        path = shard_path(args, args.stage, sample)
        if path.exists():
            value = checked(path, contract)
            if value["sample_id"] != sample["sample_id"] or value["image_sha256"] != sample["image_sha256"]:
                raise RuntimeError("Saved output image mismatch")
        else:
            pending.append(sample)
    if not pending:
        print(f"[reuse] {args.stage}: 229/229", flush=True)
        return
    if args.stage == "flan-antirepetition":
        raise RuntimeError("Missing frozen anti-repetition captions; rerun preflight")
    if args.device != "cuda:0":
        raise ValueError("Run export explicitly on PAPER cuda:0; CPU replay uses --evaluate")
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("Requested PAPER GPU unavailable")
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
               "cuda": torch.version.cuda, "seed": 0, "batch_size": 1, "device": args.device}
    load_start = time.perf_counter()
    if args.stage == "t1":
        from recognition.parseq import ParseqApproxConfig, ParseqApproxRecognizer, PARSEQ_RUNTIME_REVISION
        base = Path(protocol["asset_root"]) / "parseq"
        model = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=base / f"parseq-{PARSEQ_RUNTIME_REVISION}",
            checkpoint=base / "parseq-bb5792a6.pt", device=args.device, runtime_source_type="github_commit_archive",
            runtime_source_archive=base / f"parseq-{PARSEQ_RUNTIME_REVISION}.tar.gz"))
        def infer(image):
            result = model.recognize(np.ascontiguousarray(np.asarray(image)), input_role="T1_full_image")
            return result.raw_text
        evidence = dict(model._provenance)
        runtime["dtype"] = "float32"
    elif args.stage == "instructblip":
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        spec = protocol["specs"][args.stage]
        snapshot = Path(protocol["asset_root"]) / "huggingface" / ("models--" + spec["model"].replace("/", "--")) / "snapshots" / spec["revision"]
        for name, expected in spec["weights_sha256"].items():
            if sha(snapshot / name) != expected:
                raise RuntimeError(f"InstructBLIP checkpoint checksum mismatch: {name}")
        processor = InstructBlipProcessor.from_pretrained(str(snapshot), local_files_only=True)
        model, loading = InstructBlipForConditionalGeneration.from_pretrained(
            str(snapshot), local_files_only=True, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, output_loading_info=True)
        if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"InstructBLIP checkpoint compatibility failure: {loading}")
        model = model.to(args.device).eval()
        def infer(image):
            encoded = processor(images=image, text=spec["prompt"], return_tensors="pt")
            encoded = {k: v.to(args.device, dtype=torch.float16) if k == "pixel_values" else v.to(args.device)
                       for k, v in encoded.items()}
            generated = model.generate(**encoded, **spec["generation"])
            return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        evidence = {"model": spec["model"], "revision": spec["revision"], "weights_sha256": spec["weights_sha256"],
                    "generation": spec["generation"], "prompt": spec["prompt"], "loading_info": loading,
                    "config_sha256": sha(snapshot / "config.json")}
        runtime["dtype"] = "float16"
        runtime["parameters"] = sum(p.numel() for p in model.parameters())
    elif args.stage == "flan-repository":
        from caption.blip2_captioner import BLIP2Captioner, BLIP2ApproxConfig
        from scripts.export_train_blip2_captions import _caption_contract
        from scripts.check_table5_assets import check_blip2
        asset = check_blip2(SimpleNamespace(hf_cache=Path(protocol["asset_root"]) / "huggingface"))
        model = BLIP2Captioner(BLIP2ApproxConfig(device=args.device, dtype="float16"))
        model._load()
        def infer(image):
            return model.caption(np.ascontiguousarray(np.asarray(image))).raw_caption
        evidence = {"caption_contract": _caption_contract(), "asset": asset}
        runtime["dtype"] = "float16"
    else:
        from diagnostics.lavis_captioner_candidates import load_lavis, load_florence
        spec = protocol["specs"][args.stage]
        cache = Path(protocol["asset_root"]) if args.stage == "florence2-large-ft" else Path(protocol["visual_cache"])
        loader_args = SimpleNamespace(candidate=args.stage, cache_root=cache, device=args.device)
        loader = load_lavis if spec["backend"] == "lavis" else load_florence
        model, caption, evidence, _ = loader(spec, loader_args, torch)
        def infer(image):
            return caption(image, "<MORE_DETAILED_CAPTION>")
        runtime["dtype"] = "float32"
        runtime["parameters"] = sum(p.numel() for p in model.parameters())
    # Load/verify model on restart, but preserve its first-run evidence and timing.
    metadata_path = args.output_dir / "cache" / args.stage / "model.json"
    metadata = {"contract_sha256": contract, "runtime": runtime, "evidence": evidence}
    if metadata_path.exists():
        old = checked(metadata_path, contract)
        # repr(processor) contains a process address and is not a scientific identity.
        def stable(value):
            return {k: v for k, v in value.items() if k not in ("visual_preprocessor",)}
        if stable(old["evidence"]) != stable(evidence) or old["runtime"] != runtime:
            raise RuntimeError("Model/runtime changed during resumed export")
    else:
        seal_write(metadata_path, {**metadata, "load_seconds": time.perf_counter() - load_start})
    with torch.inference_mode():
        for index, sample in enumerate(pending, 229 - len(pending) + 1):
            path = Path(sample["image_path"])
            if sha(path) != sample["image_sha256"]:
                raise RuntimeError("TRAIN image changed during export")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with Image.open(path) as image:
                text = infer(image.convert("RGB"))
            torch.cuda.synchronize()
            if not isinstance(text, str) or (args.stage != "t1" and not text.strip()):
                raise RuntimeError("Invalid generated text")
            seal_write(shard_path(args, args.stage, sample), {"contract_sha256": contract,
                "sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"], "text": text,
                "model_metadata_sha256": sha(metadata_path), "latency_seconds": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            print(f"[{args.stage}] {index}/229 sample={sample['sample_id']}: {text}", flush=True)


def route_scores(t1, joined, proposals, caption, scorer, matching):
    import numpy as np
    from rapidfuzz.fuzz import ratio
    from scoring.local_context import score_local_candidates
    texts = [t1, joined["text"], *[p["text"] for p in proposals]]
    if matching == "whole_caption":
        embeddings = np.asarray(scorer.embeddings([*texts, caption]), dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) + 1 or not np.isfinite(embeddings).all():
            raise RuntimeError("Invalid embedding matrix")
        scores = []
        for text, vector in zip(texts, embeddings[:-1]):
            denominator = float(np.linalg.norm(vector) * np.linalg.norm(embeddings[-1]))
            semantic = float(np.clip(np.dot(vector, embeddings[-1]) / denominator, -1, 1)) if denominator else 0.0
            lexical = ratio(text, caption) / 100.0
            scores.append({"window": caption, "semantic_similarity": semantic,
                           "lexical_similarity": lexical, "confidence": .6 * semantic + .4 * lexical})
        return scores
    if matching != "neighbor_lengths":
        raise ValueError("Unknown frozen matching rule")
    return score_local_candidates(texts, caption, scorer, variant=matching)


def routes(t1, joined, proposals, caption, scores, strategies):
    from scoring.strategies import decide
    if len(scores) != len(proposals) + 2:
        raise RuntimeError("Routing score/candidate cardinality mismatch")
    a, b = scores[:2]
    image_routes = {}
    for strategy in strategies:
        decision = decide(strategy, T1=t1, T2=caption, T3=joined["text"],
                          S1=a["semantic_similarity"], S3=b["semantic_similarity"],
                          L1=a["lexical_similarity"], L3=b["lexical_similarity"]).to_dict()
        decision["empty_output_guard"] = not decision["selected_output"].strip()
        decision["needs_fallback"] |= decision["empty_output_guard"]
        image_routes[strategy] = decision
    return {"T1": t1, "T2": caption, "T3": joined, "T1_score": a, "T3_score": b,
            "image_routes": image_routes,
            "blocks": [{**p, **score, "needs_fallback": not p["text"].strip() or score["confidence"] < .8}
                       for p, score in zip(proposals, scores[2:])]}


def evaluate(args):
    import numpy as np
    import torch
    from rapidfuzz.fuzz import ratio
    from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
    from localize.multihypothesis_word_proposals import box_iou
    from localize.practical_candidate_selection import intersection_area
    from segmentation.data import load_segmentation_manifest
    from localize.practical_remap_protocol import train_task22_word_boxes
    from scripts.diagnose_parseq_candidate_ranking_train import _task22_word_texts
    from scripts.check_table5_assets import check_mpnet
    protocol, contract, samples = load_inputs(args)
    config = protocol["config"]
    torch.set_num_threads(4)
    mpnet_asset = check_mpnet(SimpleNamespace(hf_cache=Path(protocol["asset_root"]) / "huggingface"))
    runtime = {"python": platform.python_version(), "device": "cpu", "threads": 4,
               "precision": "float32", "asset": mpnet_asset,
               "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "sentence-transformers", "rapidfuzz")}}
    # Verify all caption/T1 shards before loading the CPU embedder or reading GT.
    cached = {}
    input_hashes = {}
    for stage in [*config["caption_profiles"], "t1"]:
        cached[stage] = {}
        for sample in samples:
            path = shard_path(args, stage, sample)
            value = checked(path, contract)
            if (value["sample_id"] != sample["sample_id"] or value["image_sha256"] != sample["image_sha256"] or
                    not isinstance(value["text"], str)):
                raise RuntimeError("Frozen caption/T1 sample identity mismatch")
            if "model_metadata_sha256" in value:
                if sha(path.parent / "model.json") != value["model_metadata_sha256"]:
                    raise RuntimeError("Model metadata changed after export")
                checked(path.parent / "model.json", contract)
            cached[stage][sample["sample_id"]] = value["text"]
            input_hashes[str(path.relative_to(args.output_dir))] = sha(path)
    scorer = None
    runtime_hash = digest(runtime)
    results = {}
    try:
        for profile in config["caption_profiles"]:
            for matching in config["context_matching"]:
                key = f"{profile}/{matching}"
                rows = []
                for number, sample in enumerate(samples, 1):
                    path = args.output_dir / "routes" / profile / matching / f"{int(sample['sample_id']):04d}.json"
                    if path.exists():
                        row = checked(path, contract)
                        if row.get("scoring_runtime_sha256") != runtime_hash:
                            raise RuntimeError("CPU scoring runtime changed during replay")
                    else:
                        if scorer is None:
                            scorer = MPNetSemanticSimilarity(MPNetApproxConfig(device="cpu", precision="float32"))
                        sid = sample["sample_id"]
                        caption, t1 = cached[profile][sid], cached["t1"][sid]
                        scores = route_scores(t1, sample["joined"], sample["proposals"], caption, scorer, matching)
                        row = {"contract_sha256": contract, "sample_id": sid, "scoring_runtime_sha256": runtime_hash,
                               **routes(t1, sample["joined"], sample["proposals"], caption, scores, config["joined_strategies"])}
                        seal_write(path, row)
                    rows.append(row)
                    print(f"[routing CPU] {profile}/{matching} {number}/229 sample={sample['sample_id']}", flush=True)
                results[key] = rows
    finally:
        if scorer is not None:
            scorer.release()
    # First GT access: all scoring/ordering/routing above has been frozen already.
    if sha(args.train_manifest) != protocol["train_manifest_sha256"]:
        raise RuntimeError("TRAIN manifest changed")
    manifest = load_segmentation_manifest(args.train_manifest)
    gt = {}
    for sample in manifest.samples:
        image_gt = contained(Path(protocol["data_root"]), sample.mask_path)
        annotation = contained(Path(protocol["data_root"]), sample.validity_annotation_path)
        boxes = train_task22_word_boxes(image_gt, annotation)
        texts = _task22_word_texts(annotation)
        if len(boxes) != len(texts):
            raise RuntimeError("GT geometry/transcription mismatch")
        gt[sample.sample_id] = [{"bbox": list(b.bbox), "text": t} for b, t in zip(boxes, texts)]
    if sum(map(len, gt.values())) != 849:
        raise RuntimeError("Expected 849 TRAIN words")
    control = {"exact": 0, "fuzzy": 0, "available": 0}
    for sample in samples:
        for word in gt[sample["sample_id"]]:
            best = max(sample["proposals"], key=lambda p: box_iou(p["bbox"], word["bbox"]), default=None)
            if best and intersection_area(best["bbox"], word["bbox"]) > 0:
                control["available"] += 1
                control["exact"] += best["text"].casefold() == word["text"].casefold()
                control["fuzzy"] += ratio(best["text"].casefold(), word["text"].casefold()) >= 80
    if control != {"exact": 502, "fuzzy": 578, "available": 745}:
        raise RuntimeError(f"Frozen PARSeq baseline mismatch: {control}")
    summaries = {}
    for key, rows in results.items():
        blocks = [b for r in rows for b in r["blocks"]]
        accepted = [b for b in blocks if not b["needs_fallback"]]
        correct_spatial = sum(not b["needs_fallback"] and any(
            box_iou(b["bbox"], word["bbox"]) >= .5 and b["text"].casefold() == word["text"].casefold()
            for word in gt[row["sample_id"]]) for row in rows for b in row["blocks"])
        rejected_correct = sum(b["needs_fallback"] and any(
            box_iou(b["bbox"], word["bbox"]) >= .5 and b["text"].casefold() == word["text"].casefold()
            for word in gt[row["sample_id"]]) for row in rows for b in row["blocks"])
        accepted_fuzzy = sum(not b["needs_fallback"] and any(
            box_iou(b["bbox"], word["bbox"]) >= .5 and ratio(b["text"].casefold(), word["text"].casefold()) >= 80
            for word in gt[row["sample_id"]]) for row in rows for b in row["blocks"])
        image_summaries = {}
        for strategy in config["joined_strategies"]:
            chosen = [r["image_routes"][strategy] for r in rows]
            image_summaries[strategy] = {
                "images": 229, "fallback_images": sum(d["needs_fallback"] for d in chosen),
                "accepted_images": sum(not d["needs_fallback"] for d in chosen),
                "fallback_image_rate": sum(d["needs_fallback"] for d in chosen) / 229,
                "selected_T1": sum(d["semantic_winner"] == "T1" for d in chosen),
                "selected_T3": sum(d["semantic_winner"] == "T3" for d in chosen),
                "median_C": float(np.median([d["C"] for d in chosen])),
                **word_summary(rows, gt, lambda r: r["image_routes"][strategy]["selected_output"],
                               lambda r: not r["image_routes"][strategy]["needs_fallback"])}
        summaries[key] = {"blocks": {
            "total_blocks": len(blocks), "accepted_blocks": len(accepted), "fallback_blocks": len(blocks) - len(accepted),
            "images_requiring_any_fallback": sum(not r["blocks"] or any(b["needs_fallback"] for b in r["blocks"]) for r in rows),
            "accepted_spatial_exact_iou_ge_0_5": correct_spatial,
            "accepted_spatial_fuzzy_ge_0_8_iou_ge_0_5": accepted_fuzzy,
            "accepted_not_spatial_exact": len(accepted) - correct_spatial,
            "rejected_but_parseq_spatial_exact": rejected_correct,
            "fallback_block_rate": (len(blocks) - len(accepted)) / len(blocks) if blocks else None,
            "accepted_spatial_exact_precision": correct_spatial / len(accepted) if accepted else None,
            "median_C": float(np.median([b["confidence"] for b in blocks])),
            **word_summary(rows, gt, lambda r: " ".join(b["text"] for b in r["blocks"] if not b["needs_fallback"]), lambda r: True)},
            "joined_T1_vs_T3": image_summaries}
    from diagnostics.lavis_captioner_candidates import text_metrics
    quality = {}
    for profile in config["caption_profiles"]:
        values = [text_metrics(t) for t in cached[profile].values()]
        quality[profile] = {"mean_word_count": sum(v["word_count"] for v in values)/229,
                            "captions_with_repeated_trigrams": sum(v["repeated_trigram_fraction"] > 0 for v in values)}
    report = {"contract_sha256": contract, "mode": "APPROXIMATE", "classification": "PRACTICAL_APPROXIMATION",
              "scope": {"split": "TRAIN", "images": 229, "gt_words": 849}, "baseline_v1": control,
              "summary": summaries, "caption_quality": quality, "scoring_runtime": runtime, "input_shards_sha256": input_hashes,
              "metric_definition": "Spatial exact: crop IoU>=.5 and whole transcription casefold equality. Word summary: per-image whitespace-token casefold multiset intersection, duplicates capped by GT multiplicity; no spatial matching. Accepted-only metrics exclude requested fallback outputs, denominator remains 849.",
              "fallback_executed": False, "post_fallback_accuracy": None, "table5_accuracy": None,
              "note": "No qualified TRAIN DeepSolo outputs supplied. This comparison measures acceptance correctness and fallback requests; it does not simulate successful fallback."}
    seal_write(args.output_dir / "results.json", report)
    lines = ["# TRAIN caption/routing comparison", "", report["metric_definition"], "", report["note"], "",
             "| Caption/matching | Routing | Accepted units | Fallback units | Accepted word matches /849 | Accepted word precision |", "|---|---|---:|---:|---:|---:|"]
    for key, value in summaries.items():
        b = value["blocks"]
        lines.append(f"| {key} | blocks | {b['accepted_blocks']} | {b['fallback_blocks']} | {b['accepted_word_matches']} | {b['accepted_word_precision']} |")
        for strategy, j in value["joined_T1_vs_T3"].items():
            lines.append(f"| {key} | joined/{strategy} | {j['accepted_images']} | {j['fallback_images']} | {j['accepted_word_matches']} | {j['accepted_word_precision']} |")
    atomic(args.output_dir / "results.md", "\n".join(lines) + "\n")
    atomic(args.output_dir / "RESULTS_SHA256SUMS", "".join(f"{sha(args.output_dir / name)}  {name}\n" for name in ("results.json", "results.md", "protocol.json", "inputs.json")))
    print(f"JSON {args.output_dir / 'results.json'} SHA256 {sha(args.output_dir / 'results.json')}", flush=True)


def word_summary(rows, gt, text_getter, accepted_getter):
    total_gt = accepted_matches = accepted_tokens = before_matches = accepted_any = accepted_all = 0
    for row in rows:
        expected = Counter(w["text"].casefold() for w in gt[row["sample_id"]])
        predicted = Counter(text_getter(row).casefold().split())
        matches = sum((expected & predicted).values())
        before_matches += matches
        total_gt += sum(expected.values())
        if accepted_getter(row):
            accepted_matches += matches
            accepted_tokens += sum(predicted.values())
            accepted_any += matches > 0
            accepted_all += predicted == expected
    return {"gt_words": total_gt, "pre_fallback_word_matches": before_matches,
            "accepted_word_matches": accepted_matches, "accepted_predicted_tokens": accepted_tokens,
            "accepted_word_precision": accepted_matches / accepted_tokens if accepted_tokens else None,
            "accepted_images_with_any_word_match": accepted_any,
            "accepted_images_with_exact_word_multiset": accepted_all,
            "accepted_word_recall": accepted_matches / total_gt if total_gt else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--stage")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/approximate/caption_routing_five_models_train_v1.json"))
    parser.add_argument("--train-manifest", type=Path, default=Path("manifests/icdar13_segmentation_author_protocol.yaml"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--visual-cache", type=Path, default=Path("artifacts/diagnostics/lavis-captioner-candidates-20260914/cache"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    args = parser.parse_args()
    if args.prepare:
        if any(getattr(args, k) is None for k in ("source_root", "data_root", "asset_root")):
            parser.error("preflight requires source-root, data-root and asset-root")
        prepare(args)
    elif args.evaluate:
        if args.device != "cpu":
            parser.error("routing replay is CPU only")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        evaluate(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
