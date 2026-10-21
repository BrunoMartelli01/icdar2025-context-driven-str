"""Persistent TEST-233 comparison of frozen localization/context alternatives.

Run on PAPER only. Model outputs are shared across branches; GT is opened after
all inference/routing. Metrics are approximate diagnostics, not Table 5 accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from caption.blip2_captioner import (
    BLIP2ApproxConfig, BLIP2Captioner, PRACTICAL_ANTI_REPETITION_GENERATION,
)
from deepsolo.adapter import DeepSoloSubprocessAdapter
from deepsolo.candidates import QUALIFIED_CANDIDATE_ID
from localize.grouped_word_blocks import GroupedWordBlockParams
from pipeline.e2e_comparison_diagnostic import repository_route, word_crop_diagnostic
from recognition.icdar13_annotations import load_icdar13_recognition_samples
from scoring.local_context import score_local_candidates
from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
from segmentation.data import load_segmentation_manifest
from scripts import evaluate_icdar13_e2e_faithful_vs_balanced as old


MODE = "icdar13-routing-suite-v1"
BRANCHES = (
    "faithful", "v1_repository", "balanced",
    "v1_local_repository_caption", "balanced_local_repository_caption",
    "v1_local_antirepetition", "balanced_local_antirepetition", "deepsolo_only",
)


def local_route(proposals, caption, scorer):
    """Exactly the TRAIN-tested T3-only, exact-length paired context gate."""
    scores = score_local_candidates(
        [row["text"] for row in proposals], caption, scorer, variant="exact_length",
    )
    index = max(range(len(scores)), key=lambda i: (scores[i]["confidence"], -i), default=None)
    score = scores[index] if index is not None else {"confidence": 0.0}
    selected = proposals[index] if index is not None else None
    return {
        "C": score["confidence"], "needs_fallback": score["confidence"] < .8,
        "pre_fallback_output": selected["text"] if selected else "",
        "selected_crop": selected, "context_evidence": score,
        "strategy": "exact_length T3-only; first maximum paired .6*S+.4*L; tau=.8",
        "T1_used": False,
    }


def summarize(branch, rows, gt):
    summary = old._aggregate_final(branch=branch, rows=rows, gt_by_id=gt)
    accepted_correct = accepted_both = accepted_only_parseq = replaced_correct_deep = 0
    improved = worsened = pre_correct = 0
    for row in rows:
        route = row[branch]["route"]
        references = {word["text"].casefold() for word in gt[row["sample_id"]]}
        pre = bool(route["pre_fallback_output"]) and route["pre_fallback_output"].casefold() in references
        ds = str(row["deepsolo"]["selected_output"])
        deep_correct = bool(ds) and ds.casefold() in references
        final_correct = deep_correct if route["needs_fallback"] else pre
        pre_correct += pre
        improved += final_correct and not deep_correct
        worsened += deep_correct and not final_correct
        if not route["needs_fallback"]:
            accepted_correct += pre
            raw = {str(text).casefold() for text in row["deepsolo"]["recognized_strings"]}
            both = pre and route["pre_fallback_output"].casefold() in raw
            accepted_both += both
            accepted_only_parseq += pre and not both
            replaced_correct_deep += deep_correct and not pre
    accepted = summary["no_fallback_images"]
    summary.update(
        pre_fallback_gt_membership_images=pre_correct,
        accepted_correct_images=accepted_correct,
        accepted_incorrect_images=accepted-accepted_correct,
        accepted_precision=accepted_correct/accepted if accepted else None,
        accepted_correct_also_in_deepsolo_raw=accepted_both,
        accepted_correct_absent_from_deepsolo_raw=accepted_only_parseq,
        accepted_wrong_replacing_correct_deepsolo=replaced_correct_deep,
        improved_images_vs_deepsolo_only=improved,
        worsened_images_vs_deepsolo_only=worsened,
        confidence_median=float(np.median([r[branch]["route"]["C"] for r in rows])),
        confidence_max=max(r[branch]["route"]["C"] for r in rows),
    )
    return summary


def seal_stage(root, stage, samples, contract):
    """Record/verify every finished stage byte-for-byte before downstream reuse."""
    hashes = {}
    for sample in samples:
        path = root / stage / f"{int(sample.sample_id):04d}.json"
        value = old._load_shard(path, contract=contract, sample_id=sample.sample_id)
        if value is None or value.get("image_sha256") != sample.image_sha256:
            raise RuntimeError(f"incomplete/wrong-image stage: {path}")
        hashes[path.name] = old._sha(path)
    record = {"contract_sha256": contract, "sha256": hashes}
    seal = root / "seals" / f"{stage}.json"
    if seal.exists():
        if json.loads(seal.read_text()) != record:
            raise RuntimeError(f"cached stage content changed: {stage}")
    else:
        old._write_atomic(seal, record)


def stage_exists(root, stage):
    return (root / "seals" / f"{stage}.json").is_file()


def practical_captions(args, samples, root, contract):
    from PIL import Image
    captioner = None
    try:
        for number, sample in enumerate(samples, 1):
            path = root / "06-practical-captions" / f"{int(sample.sample_id):04d}.json"
            if old._load_shard(path, contract=contract, sample_id=sample.sample_id) is None:
                if captioner is None:
                    captioner = BLIP2Captioner(BLIP2ApproxConfig(
                        generation=dict(PRACTICAL_ANTI_REPETITION_GENERATION),
                        device=args.device, dtype="float16",
                    ))
                with Image.open(old._contained(args.data_root, sample.image_path)) as image:
                    rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
                result = captioner.caption(rgb).to_dict()
                old._write_atomic(path, {
                    "contract_sha256": contract, "sample_id": sample.sample_id,
                    "image_sha256": sample.image_sha256, "caption": result,
                })
            print(f"[anti-repetition] {number}/233 sample={sample.sample_id}", flush=True)
    finally:
        if captioner is not None:
            old._release_cuda(captioner)


def extra_routes(samples, root, contract):
    scorer = None
    try:
        for number, sample in enumerate(samples, 1):
            sid = f"{int(sample.sample_id):04d}.json"
            path = root / "07-suite-routing" / sid
            existing = old._load_shard(path, contract=contract, sample_id=sample.sample_id)
            if existing is None:
                if scorer is None:
                    scorer = MPNetSemanticSimilarity(MPNetApproxConfig(device="cpu"))
                original = json.loads((root / "04-routing" / sid).read_text())
                proposal = json.loads((root / "01-proposals" / sid).read_text())
                ocr = json.loads((root / "02-parseq" / sid).read_text())
                practical = json.loads((root / "06-practical-captions" / sid).read_text())
                caption = original["T2"]
                v1 = [{"candidate_id": row["candidate_id"], "bbox": row["bbox"],
                       "text": ocr["crops"][row["ocr_key"]]["raw_text"]}
                      for row in proposal["v1_anchors"]]
                texts = [original["T1"]] + [row["text"] for row in v1] + [caption]
                emb = scorer.embeddings(texts)
                similarities = np.clip(emb[:-1] @ emb[-1], -1, 1)
                route = repository_route(
                    t1=original["T1"], caption=caption, s1=float(similarities[0]),
                    candidates=[{**row, "semantic_similarity": float(score)}
                                for row, score in zip(v1, similarities[1:])],
                )
                branches = {name: original[name] for name in ("faithful", "balanced")}
                branches["v1_repository"] = {"proposals": v1, "route": route}
                for family, crops in (("v1", v1), ("balanced", original["balanced"]["proposals"])):
                    for profile, text in (("repository_caption", caption),
                                          ("antirepetition", practical["caption"]["raw_caption"])):
                        branches[f"{family}_local_{profile}"] = {
                            "proposals": crops, "route": local_route(crops, text, scorer),
                        }
                branches["deepsolo_only"] = {"proposals": [], "route": {
                    "needs_fallback": True, "pre_fallback_output": "", "C": 0.0,
                }}
                existing = {"contract_sha256": contract, "sample_id": sample.sample_id,
                            "image_sha256": sample.image_sha256, **branches}
                old._write_atomic(path, existing)
            flags = {name: existing[name]["route"]["needs_fallback"] for name in BRANCHES}
            print(f"[routing] {number}/233 sample={sample.sample_id} fallback={flags}", flush=True)
    finally:
        if scorer is not None:
            scorer.release()


def deepsolo_control(args, samples, root, contract):
    # Run once per image for the standalone control and reuse in every branch.
    # This count must never be confused with a branch's conditional fallback count.
    adapter = None
    for number, sample in enumerate(samples, 1):
        path = root / "05-deepsolo" / f"{int(sample.sample_id):04d}.json"
        if old._load_shard(path, contract=contract, sample_id=sample.sample_id) is None:
            if adapter is None:
                adapter = DeepSoloSubprocessAdapter(
                    args.deepsolo_python, args.deepsolo_runtime_manifest,
                    host_source_root=args.deepsolo_source_root,
                )
            response = adapter.infer(
                image_path=old._contained(args.data_root, sample.image_path),
                candidate_id=QUALIFIED_CANDIDATE_ID,
                inference_options={"device": args.device, "score_threshold": None, "lexicon_type": None},
                reduction_policy="highest_score",
            ).to_dict()
            if response.get("success") is not True:
                raise RuntimeError(f"DeepSolo failed: {sample.sample_id}")
            old._write_atomic(path, {
                "contract_sha256": contract, "sample_id": sample.sample_id,
                "image_sha256": sample.image_sha256, "response": response,
                "purpose": "shared standalone control; per-branch fallback counted separately",
            })
        print(f"[DeepSolo shared control] {number}/233 sample={sample.sample_id}", flush=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "segmentation-manifest", "recognition-manifest", "data-root", "evaluation-config",
        "checkpoint", "v1-calibration", "parseq-source", "parseq-source-archive",
        "parseq-checkpoint", "deepsolo-python", "deepsolo-runtime-manifest",
        "deepsolo-source-root", "artifact-dir", "output",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda:0"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    args.data_root = args.data_root.resolve()
    if args.batch_size < 1:
        raise ValueError("positive batch size required")
    expected = {
        "segmentation_manifest": old.EXPECTED_SEGMENTATION_MANIFEST_SHA256,
        "recognition_manifest": old.EXPECTED_RECOGNITION_MANIFEST_SHA256,
        "checkpoint": old.EXPECTED_CHECKPOINT_SHA256,
        "v1_calibration": old.EXPECTED_V1_CALIBRATION_SHA256,
        "parseq_source_archive": old.PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "parseq_checkpoint": old.PARSEQ_CHECKPOINT_SHA256,
        "deepsolo_runtime_manifest": old.EXPECTED_DEEPSOLO_RUNTIME_SHA256,
    }
    inputs = {}
    for name, wanted in expected.items():
        path = getattr(args, name)
        inputs[name] = old._sha(path)
        if inputs[name] != wanted:
            raise RuntimeError(f"frozen input SHA mismatch: {name}")
    inputs["evaluation_config"] = old._sha(args.evaluation_config)
    manifest = load_segmentation_manifest(args.segmentation_manifest)
    if manifest.split_role != "test" or len(manifest.samples) != 233:
        raise RuntimeError("requires official TEST-233")
    for sample in manifest.samples:
        if old._sha(old._contained(args.data_root, sample.image_path)) != sample.image_sha256:
            raise RuntimeError(f"image checksum mismatch: {sample.sample_id}")
    calibration = json.loads(args.v1_calibration.read_text())
    params = GroupedWordBlockParams(**calibration["calibration"]["selected"]["parameters"])
    repo = Path(__file__).resolve().parents[1]
    # Includes all reused implementations, not only this orchestrator.
    sources = {str(path.relative_to(repo)): old._sha(path)
               for directory in ("scripts", "localize", "pipeline", "caption", "scoring", "recognition", "segmentation", "models", "deepsolo")
               for path in sorted((repo / directory).glob("*.py"))}
    protocol = {
        "mode": MODE, "branches": BRANCHES, "inputs": inputs, "sources": sources,
        "device": args.device, "batch_size": args.batch_size,
        "alpha": .6, "beta": .4, "tau": .8,
        "practical_generation": PRACTICAL_ANTI_REPETITION_GENERATION,
        "balanced_train_report_historical_sha256": old.EXPECTED_BALANCED_TRAIN_SHA256,
        "balanced_status": "TRAIN_REJECTED; user-requested descriptive comparison only",
        "local_context": "exact_length T3-only, as evaluated on TRAIN",
        "balanced_proposals_frozen_before_local_context": True,
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "claim_label": "approximate reproduction", "table_5_accuracy": None,
    }
    contract = old._json_hash(protocol)
    root = args.artifact_dir.resolve()
    # Validate all sealed data before either reporting cache success or resuming.
    stages = ("01-proposals", "02-parseq", "03-blip2", "04-routing",
              "06-practical-captions", "07-suite-routing", "05-deepsolo")
    for stage in stages:
        if stage_exists(root, stage):
            seal_stage(root, stage, manifest.samples, contract)
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result.get("contract_sha256") != contract or not all(stage_exists(root, s) for s in stages):
            raise RuntimeError("existing report/cache mismatch; choose a separate output directory")
        receipt = root / "report-receipt.json"
        identity = {"contract_sha256": contract, "report_sha256": old._sha(args.output)}
        if receipt.exists() and json.loads(receipt.read_text()) != identity:
            raise RuntimeError("final report checksum changed")
        if not receipt.exists() and not args.preflight_only:
            raise RuntimeError("final report receipt missing; inspect interrupted finalization before reuse")
        print(f"Verified complete report: {args.output}\nSHA256 {old._sha(args.output)}", flush=True)
        return 0
    print("[preflight] frozen inputs and TEST-233 image hashes PASS", flush=True)
    if args.preflight_only:
        return 0
    if not torch.cuda.is_available() or "RTX 3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("execute on PAPER RTX 3090, cuda:0")
    torch.cuda.init()
    root.mkdir(parents=True, exist_ok=True)
    old._write_atomic(root / "protocol.json", {"contract_sha256": contract, **protocol})
    operations = (
        ("01-proposals", lambda: old._build_proposal_shards(args=args, samples=manifest.samples, params=params, contract=contract, root=root)),
        ("02-parseq", lambda: old._run_ocr_shards(args=args, samples=manifest.samples, contract=contract, root=root)),
        ("03-blip2", lambda: old._run_caption_shards(args=args, samples=manifest.samples, contract=contract, root=root)),
        ("04-routing", lambda: old._score_and_select_shards(samples=manifest.samples, contract=contract, root=root)),
        ("06-practical-captions", lambda: practical_captions(args, manifest.samples, root, contract)),
        ("07-suite-routing", lambda: extra_routes(manifest.samples, root, contract)),
        ("05-deepsolo", lambda: deepsolo_control(args, manifest.samples, root, contract)),
    )
    for stage, execute in operations:
        if not stage_exists(root, stage):
            execute()
        seal_stage(root, stage, manifest.samples, contract)
        print(f"[stage verified] {stage}", flush=True)
    samples, _ = load_icdar13_recognition_samples(args.recognition_manifest, args.data_root)
    if len(samples) != 233 or sum(len(s.words) for s in samples) != 1095:
        raise RuntimeError("requires TEST-233 / 1095 GT words")
    gt = {s.sample_id: [{"bbox": list(w.bbox_xyxy), "text": w.transcription} for w in s.words] for s in samples}
    rows = []
    for s in samples:
        sid = f"{int(s.sample_id):04d}.json"
        row = json.loads((root / "07-suite-routing" / sid).read_text())
        deep = json.loads((root / "05-deepsolo" / sid).read_text())["response"]
        rows.append({**row, "deepsolo": deep})
    metrics = {branch: summarize(branch, rows, gt) for branch in BRANCHES}
    crop_metrics = {branch: word_crop_diagnostic(
        [r[branch]["proposals"] for r in rows], [gt[r["sample_id"]] for r in rows],
    ) for branch in ("faithful", "v1_repository", "balanced")}
    result = {
        "mode": MODE, "contract_sha256": contract, "protocol": protocol,
        "scope": {"split": "TEST", "images": 233, "gt_words": 1095},
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "deepsolo_shared_control_images": len(rows),
        "conditional_fallback_counts_are_in_each_branch": True,
        "metrics": metrics, "crop_diagnostics": crop_metrics,
        "metric_definitions": {
            "image_membership": "one final string casefold-exact to any care GT word in image; no spatial match",
            "multiset": "per-image one-to-one casefold text matches; all raw DeepSolo strings on fallback, one string otherwise",
            "crop_diagnostics": "GT selects best geometric crop post-hoc; not inference-selected accuracy",
            "fallback": "conditional routing per branch; standalone DeepSolo control computed once per image separately",
        },
        "paper_comparison": {"reported_icdar13_r50": 88.8, "reported_no_fallback": "168/229",
                             "comparable_accuracy": None, "percentage_point_gap": None,
                             "unresolved": ["E-001", "DS-002", "D-002"]},
        "history": {"reported_final_membership": "207/233=88.841%", "reported_fallback": "233/233",
                    "source": "docs/PAPER_AMBIGUITIES.md R-E2E-COMP-001; original JSON currently missing",
                    "historical_json_sha256": "e6db8027e1d0f6ec3bf5e3b574c431ad1779857330825cb14bf766f505747eee"},
    }
    # Receipt precedes publication: interruption cannot publish an unsealed report.
    report_bytes = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    old._write_atomic(root / "report-receipt.json", {
        "contract_sha256": contract, "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
    })
    old._write_atomic(args.output, result)
    print(json.dumps({name: {k: v for k, v in value.items() if k != "per_image"}
                      for name, value in metrics.items()}, indent=2), flush=True)
    print(f"JSON {args.output}\nSHA256 {old._sha(args.output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
