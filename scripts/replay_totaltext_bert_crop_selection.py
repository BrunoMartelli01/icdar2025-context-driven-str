"""Reuse genuine frozen BERT scores only for exactly identical observables.

Never opens source or destination annotation/evaluation files. This does not
approximate or fabricate missing likelihoods: any incompatibility is fatal.
"""
from __future__ import annotations

import ast
from collections import Counter
from hashlib import sha256
from pathlib import Path

from recognition.bert_crop_selection import CropPolicy, selections
from scripts.evaluate_totaltext_bert_correction import checked_file, digest, payload_digest, seal_json


def function_hash(source: str, name: str) -> str:
    matches = [node for node in ast.parse(source).body
               if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError("expected one authenticated scoring function")
    return sha256(ast.dump(matches[0], include_attributes=False).encode()).hexdigest()


def identical_observations(current, original):
    if current["samples"] != original["samples"]:
        raise ValueError("cache observations differ: image/crop/text/context/physical evidence must match exactly")


def replay(args, protocol, record, inputs):
    from scripts.evaluate_totaltext_bert_crop_selection import REPO, check_payload, path_for
    import json
    source = args.reuse_inference_dir.resolve()
    if source == args.output_dir.resolve():
        raise ValueError("cache source and destination must differ")
    old_record = json.loads((source / "prepared.json").read_text())
    old_protocol = checked_file(source / "protocol.json", old_record["protocol_sha256"])
    old_inputs = checked_file(source / "inputs.json", old_record["inputs_sha256"])
    if (old_inputs["protocol_sha256"] != old_record["protocol_sha256"]
            or old_protocol["reproduction_mode"] != "approximate"
            or old_protocol["table_5_claim"] is not False
            or old_protocol["config"]["model"] != protocol["config"]["model"]
            or old_protocol["code_sha256"]["recognition/bert_correction.py"] !=
               digest(REPO / "recognition/bert_correction.py")):
        raise ValueError("cache BERT model/masking/protocol identity differs")
    identical_observations(inputs, old_inputs)
    name = "recognition/bert_crop_selection.py"
    snapshot = source / "software_snapshot" / name
    if not snapshot.is_file() or digest(snapshot) != old_protocol["code_sha256"][name]:
        raise ValueError("cache requires a checksum-bound historical scoring source snapshot")
    engine = function_hash(snapshot.read_text(), "score_requests")
    if engine != function_hash((REPO / name).read_text(), "score_requests"):
        raise ValueError("BERT batch scoring algorithm changed; cache reuse forbidden")
    old_runtime_path = source / "runtime.json"
    old_runtime = json.loads(old_runtime_path.read_text())
    receipt_path = source / "inference_complete.json"
    receipt = json.loads(receipt_path.read_text())
    runtime_sha = digest(old_runtime_path)
    cfg = protocol["config"]["model"]
    loading = old_runtime["loading_audit"]
    if (old_runtime["inputs_sha256"] != old_record["inputs_sha256"]
            or receipt["inputs_sha256"] != old_record["inputs_sha256"]
            or receipt["runtime_sha256"] != runtime_sha
            or receipt["images"] != len(inputs["samples"])
            or set(receipt["inference_sha256"]) != set(protocol["sample_ids"])
            or old_runtime["device"] != args.device
            or old_runtime["precision"] != cfg["precision"]
            or any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs"))
            or set(loading.get("unexpected_keys", [])) != set(cfg["allowed_unused_checkpoint_keys"])
            or old_runtime["model_files_sha256"] != cfg["files_sha256"]):
        raise ValueError("cache population/runtime identity differs")
    if args.model_dir.resolve().name != cfg["revision"]:
        raise ValueError("model directory revision differs")
    for name, expected in cfg["files_sha256"].items():
        if digest(args.model_dir / name) != expected:
            raise ValueError("cached BERT model file changed")
    runtime = {**old_runtime, "inputs_sha256": record["inputs_sha256"],
               "inference_origin": "verified genuine BERT cache; no new neural inference",
               "cache_source": str(source), "cache_receipt_sha256": digest(receipt_path),
               "cache_runtime_sha256": runtime_sha, "scoring_function_ast_sha256": engine}
    seal_json(args.output_dir / "runtime.json", runtime)
    new_runtime_sha = digest(args.output_dir / "runtime.json")
    statuses = Counter()
    for sample in inputs["samples"]:
        sid = sample["sample_id"]
        path = path_for(source, "inference", sid)
        if digest(path) != receipt["inference_sha256"][sid]:
            raise ValueError("cache inference shard changed")
        old = check_payload(path, old_record, runtime_sha, sid)
        if old["image_sha256"] != sample["image_sha256"]:
            raise ValueError("cache image identity differs")
        selected = selections(sample, old["scores"], CropPolicy(**protocol["config"]["policy"]))
        value = {"inputs_sha256": record["inputs_sha256"], "runtime_sha256": new_runtime_sha,
                 "sample_id": sid, "image_sha256": sample["image_sha256"],
                 "scores": old["scores"], "unique_requests": old["unique_requests"], **selected,
                 "max_input_tokens_seen": old["max_input_tokens_seen"],
                 "gpu_peak_allocated_bytes": old["gpu_peak_allocated_bytes"],
                 "gpu_peak_reserved_bytes": old["gpu_peak_reserved_bytes"],
                 "source_inference_sha256": digest(path)}
        value["payload_sha256"] = payload_digest(value)
        seal_json(path_for(args.output_dir, "inference", sid), value)
        for group in old["scores"].values():
            statuses.update(row["status"] for rows in group.values() for row in rows)
    seal_json(args.output_dir / "inference_complete.json", {
        "inputs_sha256": record["inputs_sha256"], "runtime_sha256": new_runtime_sha,
        "images": len(inputs["samples"]), "statuses": dict(statuses),
        "inference_sha256": {s["sample_id"]: digest(path_for(args.output_dir, "inference", s["sample_id"]))
                             for s in inputs["samples"]},
        "ground_truth_use": "none; exact-observation authenticated BERT score cache only",
        "new_neural_inference": False, "cache_receipt_sha256": digest(receipt_path)})
    print(f"Replayed {len(inputs['samples'])} authenticated BERT score shards; no annotation files opened", flush=True)
