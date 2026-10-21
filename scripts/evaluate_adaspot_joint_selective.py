"""Replay the actual calibrated selective network, independently of OCR caches."""

import argparse
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from models.adaspot_joint import AdaSpotJoint, qualified_parseq_parts
from models.adaspot_mask_align import MaskAlignOptions
from scripts.diagnose_adaspot_phase0 import REPO, check_hash, summarize, write_json
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_adaspot_gap_context import memory
from scripts.run_adaspot_joint import predicted_rois
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.training import seed_everything


def load_completed(root, device):
    p = json.loads((root / "protocol_final.json").read_text())
    completed = json.loads((root / "training_complete.json").read_text())
    audit = json.loads((root / "independent_audit.json").read_text())
    if p["mode"] != "approximate" or audit["status"] != "PASS" or completed["status"] != "SUCCESS":
        raise ValueError("Audited approximate joint run required")
    check_hash(root / "results.json", audit["result_sha256"])
    check_hash(Path(completed["checkpoint"]), completed["checkpoint_sha256"])
    for path, expected in p["source_hashes"].items():
        check_hash(Path(path), expected)
    parts, tokenizer, _, state, _ = qualified_parseq_parts(p["parseq_source"], p["parseq_archive"], p["parseq_checkpoint"])
    del state
    model = AdaSpotJoint(parts, tokenizer, p["resolved_config"])
    payload = torch.load(completed["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    del payload
    return model.to(device).eval(), p


@torch.no_grad()
def run(args):
    if args.mode != "approximate":
        raise ValueError("Explicit approximate mode required")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    root, out = args.run_root.resolve(), args.output_root.resolve()
    if out == root or out.exists() or REPO == out or REPO in out.parents:
        raise ValueError("New external output root required")
    model, p = load_completed(root, device)
    c = p["resolved_config"]
    torch.set_num_threads(c["cpu_threads"])
    cv2.setNumThreads(1)
    seed_everything(c["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True)
    args._output_created = True
    previous = json.loads((root / "results.json").read_text())
    cache = json.loads((root / "test_cache.json").read_text())
    manifest = load_segmentation_manifest(Path(p["test_manifest"]), require_checksums=True)
    dataset = SegmentationDataset(manifest, Path(p["test_root"]), transform=_transform(p["resolved_preprocessing"], "letterbox", train=False))
    by_id = {s.sample_id: i for i, s in enumerate(manifest.samples)}
    options = MaskAlignOptions(**c["geometry"])
    rows, records, deep_calls, backbone_calls = [], [], [], []
    deep_hook = model.deep_head.projection.register_forward_pre_hook(lambda module, values: deep_calls.append(len(values[0])))
    backbone_hook = model.backbone.encoder.conv1.register_forward_pre_hook(lambda module, values: backbone_calls.append(len(values[0])))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        for index, frozen in enumerate(cache):
            sid = frozen["sample_id"]
            batch = dataset[by_id[sid]]
            with Image.open(batch["image_path"]) as image:
                wh = image.size
            evidence = model.encode(batch["image"][None].to(device))
            labels, proposals, _, _, _ = predicted_rois(evidence, batch, c, wh)
            with np.load(frozen["map_path"], allow_pickle=False) as values:
                if not np.array_equal(labels, values["labels"]):
                    raise RuntimeError("Fresh selective decoding changed cached predicted labels")
            fields = ("instance_id", "foreground_area", "bbox_native", "bbox_original", "bbox")
            expected_proposals = [{key: v[key] for key in fields} for v in frozen["predictions"]]
            if proposals != expected_proposals:
                raise RuntimeError("Selective care proposal geometry differs")
            decisions = []
            size = c["evaluation_roi_batch_size"]
            for start in range(0, len(proposals), size):
                masks = [labels == v["instance_id"] for v in proposals[start:start+size]]
                decisions.extend(model.infer_instances(evidence, masks, options=options,
                    view=c["alignment_view"], **previous["calibration_choice"]))
            emitted = []
            expected_routes = previous["routes"][index]
            for proposal, actual, expected in zip(proposals, decisions, expected_routes):
                for key in ("route", "emitted", "text"):
                    if actual[key] != expected[key]:
                        raise RuntimeError("Actual selective inference differs from cached routing: " + key)
                if abs(actual["gate_probability"] - expected["gate_probability"]) > 2e-5:
                    raise RuntimeError("Actual gate probability differs beyond declared FP32 tolerance")
                if actual["emitted"]:
                    # Preserve identical metadata for an exact per-image score replay.
                    original = next(v for v in frozen["predictions"] if v["instance_id"] == proposal["instance_id"])
                    emitted.append({**original, "text": actual["text"]})
            if len(decisions) != len(proposals) or len(decisions) != len(expected_routes):
                raise RuntimeError("Selective inference omitted instances")
            rows.append({**frozen, "predictions": emitted})
            records.append({"sample_id": sid, "decisions": decisions})
            if (index + 1) % 25 == 0:
                print(f"[actual selective] {index+1}/{len(cache)} images; deep calls={sum(deep_calls)}", flush=True)
    finally:
        deep_hook.remove()
        backbone_hook.remove()
    score = summarize(rows)
    if score != previous["arms"]["adaptive"]:
        raise RuntimeError("Actual selective aggregate/per-image scores differ")
    expected_deep = sum(v["route"] == "escalate" for row in records for v in row["decisions"])
    if sum(deep_calls) != expected_deep or expected_deep != previous["deep_calls"] or sum(backbone_calls) != len(cache):
        raise RuntimeError("Unexpected shared-backbone or deep execution count")
    write_json(out / "routes.json", records)
    receipt = {"status": "PASS", "result_class": p["result_class"], "scope": "fresh actual calibrated selective inference on every source TEST image",
        "source_result_sha256": sha256_file(root / "results.json"), "source_checkpoint_sha256": sha256_file(Path(p["source_checkpoint"])),
        "joint_checkpoint_sha256": sha256_file(root / "training/last.pt"), "implementation_sha256": sha256_file(Path(__file__)),
        "images": len(cache), "backbone_calls": sum(backbone_calls), "deep_calls": expected_deep,
        "deep_projection_batches": deep_calls, "all_care_instances": sum(len(s["predictions"]) for s in cache),
        "same_labels_boxes_routes_words_and_scores": True, "aggregate": score["aggregate"],
        "wall_seconds_including_io": time.monotonic() - started,
        "memory": memory(device, "actual shared AdaSpot calibrated selective inference; no diagnostic always-deep pass", [1, 3, 512, 512]),
        "gate_probability_absolute_tolerance": 2e-5, "no_new_training_or_threshold_selection": True}
    write_json(out / "independent_execution_audit.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=["approximate"], required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if getattr(args, "_output_created", False):
            write_json(args.output_root / "failure.json", {"status": "FAILED", "error": repr(error),
                "oom": isinstance(error, torch.cuda.OutOfMemoryError), "traceback": traceback.format_exc()})
        raise
