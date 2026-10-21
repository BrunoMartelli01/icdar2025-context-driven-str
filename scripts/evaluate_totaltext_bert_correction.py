"""Prepare, score and evaluate frozen Total-Text v4 PARSeq with genuine BERT.

The infer stage reads only observable inputs. Ground truth is stored separately
and consumed only by prepare's baseline audit and the evaluate stage.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
import re
import tempfile
import time

from recognition.bert_correction import BertWordScorer, CorrectionPolicy, correct_blocks
from recognition.scene_text_v1_evaluation import box_iou, evaluate_images, maximum_matching


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        value = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def payload_digest(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seal_json(path: Path, value: dict) -> None:
    """Idempotent writes; never overwrite artifacts from a different run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Conflicting sealed artifact: {path}; choose a new output directory")
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def checked_file(path: Path, expected: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    if digest(path) != expected:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return read_json(path)


def load_config(path: Path) -> dict:
    path = path.resolve()
    if path.parent.name != "approximate" or path.parent.parent.name != "configs":
        raise ValueError("BERT diagnostic requires a configuration in configs/approximate/")
    cfg = read_json(path)
    if (cfg.get("reproduction_mode") != "approximate" or cfg.get("dataset") != "total_text"
            or cfg.get("claim_label") != "approximate reproduction"
            or cfg.get("experiment_kind") != "totaltext_frozen_v4_bert_post_ocr"):
        raise ValueError("Expected explicit approximate Total-Text BERT configuration")
    if (cfg["source"]["layout"] != "letterbox_word_f1"
            or cfg["source"]["split"] != "frozen_TRAIN_validation_125"
            or cfg["source"]["variant"] not in {"word_f1_control", "word_f1_separator_head"}):
        raise ValueError("This adapter requires the frozen Total-Text letterbox TRAIN-holdout source")
    bert = cfg["bert"]
    if (not re.fullmatch(r"[0-9a-f]{40}", bert["revision"])
            or bert["precision"] != "float32"
            or bert["scoring"] != "target_only_mean_PLL_word_l2r"):
        raise ValueError("BERT requires immutable revision and explicit FP32/scoring configuration")
    for key in ("batch_size", "cpu_threads", "max_input_tokens"):
        if type(bert[key]) is not int or bert[key] < 1:
            raise ValueError(f"BERT {key} must be a positive integer")
    if type(bert["seed"]) is not int or not 0 <= bert["seed"] < 2**32:
        raise ValueError("BERT seed must be a uint32")
    if cfg["conditions"] != ["word_only", "local_ocr"]:
        raise ValueError("Expected predeclared word-only/local-OCR controls")
    if cfg["caption_condition_if_supplied"] != "local_ocr_caption":
        raise ValueError("Unsupported caption condition")
    CorrectionPolicy(**cfg["policy"])
    return cfg


def load_captions(path: Path, samples: list[dict]) -> dict[str, str]:
    obj = read_json(path)
    if (obj.get("schema_version") != 1 or not obj.get("model_id")
            or not re.fullmatch(r"[0-9a-f]{40}", obj.get("revision", ""))
            or not obj.get("generation") or not obj.get("model_files_sha256")):
        raise ValueError("Caption file requires model ID, pinned revision, file hashes and decoding parameters")
    if any(not re.fullmatch(r"[0-9a-f]{64}", v) for v in obj["model_files_sha256"].values()):
        raise ValueError("Invalid caption model file checksum")
    expected = {s["sample_id"]: s["image_sha256"] for s in samples}
    captions = {}
    for row in obj["samples"]:
        sid = row["sample_id"]
        if sid in captions or sid not in expected or row["image_sha256"] != expected[sid]:
            raise ValueError("Caption identity/checksum mismatch or duplicate image")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Caption must contain nonempty text for every image")
        captions[sid] = row["text"]
    if set(captions) != set(expected):
        raise ValueError("Caption coverage must match all Total-Text source images")
    return captions


def source_inputs(root: Path, cfg: dict, ocr_reuse_dirs=()) -> tuple[list[dict], list[dict], dict]:
    expected = cfg["source"]
    report = checked_file(root / "results.json", expected["results_sha256"])
    protocol = checked_file(root / "protocol.json", expected["protocol_sha256"])
    branch = report["variants"][expected["variant"]]
    if (branch["checkpoint_sha256"] != expected["segmentation_checkpoint_sha256"]
            or protocol["parseq_checkpoint_sha256"] != expected["parseq_checkpoint_sha256"]
            or protocol["parseq_runtime_archive_sha256"] != expected["parseq_source_archive_sha256"]
            or report["protocol"] != protocol or protocol["dataset"] != "total_text"
            or protocol["split"] != expected["split"]
            or report["images"] != expected["expected_images"]
            or branch["aggregate"]["gt_words"] != expected["expected_care_words"]):
        raise ValueError("Frozen source dataset/checkpoint/protocol mismatch")
    observations, labels, files, seen = [], [], {}, set()
    # Upstream pins the sorted JSON object, not the pretty-printed file bytes.
    # Match the exact contract in evaluate_totaltext_word_f1_separator_parseq_val.py.
    shard_protocol_sha = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    for report_sample in branch["per_image"]:
        sid = report_sample["sample_id"]
        if not re.fullmatch(r"total_text:[0-9]+", sid):
            raise ValueError("Expected a Total-Text image ID")
        path = root / expected["variant"] / "samples" / (sid.replace(":", "__") + ".json")
        sample = read_json(path)
        if (sample["sample_id"] != sid or sid in seen
                or not re.fullmatch(r"[0-9a-f]{64}", sample["image_sha256"])
                or sample["protocol_sha256"] != shard_protocol_sha
                or sample["checkpoint_sha256"] != expected["segmentation_checkpoint_sha256"]
                or sample["ground_truth"] != report_sample["ground_truth"]):
            raise ValueError(f"Frozen sample identity mismatch: {path}")
        seen.add(sid)
        proposals = sample["proposals"]
        predictions = report_sample["predictions"]
        if len(predictions) != len(proposals):
            raise ValueError(f"PARSeq/proposal count mismatch: {sid}")
        blocks = []
        for proposal, prediction in zip(proposals, predictions):
            if (any(proposal[k] != prediction[k] for k in ("bbox", "crop_rgb_sha256"))
                    or not isinstance(prediction["text"], str)):
                raise ValueError(f"Frozen PARSeq crop identity mismatch: {sid}")
            ocr_name = proposal["crop_rgb_sha256"] + ".json"
            ocr_paths = [directory / ocr_name for directory in [root / "ocr", *ocr_reuse_dirs]
                         if (directory / ocr_name).is_file()]
            if not ocr_paths:
                raise FileNotFoundError(f"Missing PARSeq record for {sid}, crop {proposal['crop_rgb_sha256']}. Supply the upstream --reuse-ocr directory used by the frozen letterbox run.")
            records = [read_json(p) for p in ocr_paths]
            if any(r != records[0] for r in records[1:]):
                raise ValueError(f"Conflicting PARSeq cache records for crop {proposal['crop_rgb_sha256']}")
            ocr = records[0]
            if (ocr["crop_rgb_sha256"] != proposal["crop_rgb_sha256"]
                    or ocr["raw_text"] != prediction["text"]
                    or ocr["parseq_checkpoint_sha256"] != expected["parseq_checkpoint_sha256"]
                    or ocr["parseq_source_archive_sha256"] != expected["parseq_source_archive_sha256"]):
                raise ValueError(f"Frozen PARSeq checkpoint/text mismatch: {sid}")
            blocks.append({"rank": proposal["rank"], "bbox": proposal["bbox"],
                           "crop_rgb_sha256": proposal["crop_rgb_sha256"], "text": prediction["text"],
                           "minimum_token_probability": ocr["minimum_token_probability"]})
            for ocr_path in ocr_paths:
                files[str(ocr_path.resolve())] = digest(ocr_path)
        if len({p["rank"] for p in blocks}) != len(blocks):
            raise ValueError(f"Duplicate crop rank: {sid}")
        observations.append({"sample_id": sid, "image_sha256": sample["image_sha256"],
                             "blocks": blocks, "caption": None})
        labels.append({"sample_id": sid, "ground_truth": sample["ground_truth"]})
        files[str(path.relative_to(root))] = digest(path)
        mask_path = root / expected["variant"] / "masks" / path.with_suffix(".png").name
        if digest(mask_path) != sample["native_mask_file_sha256"]:
            raise ValueError(f"Frozen segmentation mask checksum mismatch: {sid}")
        files[str(mask_path.relative_to(root))] = digest(mask_path)
    if (len(observations) != expected["expected_images"]
            or sum(len(s["blocks"]) for s in observations) != expected["expected_proposals"]
            or sum(len(s["ground_truth"]) for s in labels) != expected["expected_care_words"]):
        raise ValueError("Frozen source population differs; no samples may be omitted")
    metric = evaluate_images([{"ground_truth": g["ground_truth"], "predictions": s["blocks"]}
                             for s, g in zip(observations, labels)], text_available=True)
    recorded = branch["aggregate"]
    actual = {"gt_words": metric["geometry"]["gt_instances"],
              "proposals": metric["geometry"]["retained_proposals"],
              "geometry_iou_ge_25": metric["geometry"]["iou_ge_0_25_one_to_one"],
              "geometry_iou_ge_50": metric["geometry"]["iou_ge_0_50_one_to_one"],
              "geometry_iou_ge_75": metric["geometry"]["iou_ge_0_75_one_to_one"],
              "exact_casefold_iou_ge_50": metric["recognition"]["exact_casefold_spatial_one_to_one"]}
    if any(actual[k] != recorded[k] for k in actual):
        raise ValueError("Frozen PARSeq baseline fails exact cached-metric reproduction")
    return observations, labels, {"files_sha256": files, "upstream_protocol": protocol, "baseline": metric}


def code_hashes() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = (Path(__file__).resolve(), root / "recognition/bert_correction.py",
             root / "recognition/scene_text_v1_evaluation.py")
    return {str(p.relative_to(root)): digest(p) for p in files}


def prepare(args):
    cfg = load_config(args.config)
    reuse_dirs = [p.resolve() for p in args.reuse_ocr]
    if any(not p.is_dir() for p in reuse_dirs):
        raise FileNotFoundError("A declared PARSeq --reuse-ocr directory is missing")
    samples, labels, audit = source_inputs(args.source_dir.resolve(), cfg, reuse_dirs)
    caption_record = None
    conditions = list(cfg["conditions"])
    if args.captions:
        captions = load_captions(args.captions, samples)
        for sample in samples:
            sample["caption"] = captions[sample["sample_id"]]
        conditions.append(cfg["caption_condition_if_supplied"])
        caption_record = {"path": str(args.captions.resolve()), "sha256": digest(args.captions),
                          "provenance": {k: v for k, v in read_json(args.captions).items() if k != "samples"}}
    protocol = {"schema_version": 1, "reproduction_mode": "approximate",
                "claim_label": "approximate reproduction", "result_class": cfg["result_class"],
                "resolved_config": cfg, "config_sha256": digest(args.config),
                "source_dir": str(args.source_dir.resolve()), "conditions": conditions,
                "reuse_ocr_directories": [str(p) for p in reuse_dirs],
                "source_files_sha256": audit["files_sha256"],
                "upstream_protocol": audit["upstream_protocol"], "captions": caption_record,
                "implementation_sha256": code_hashes()}
    out = args.output_dir
    seal_json(out / "protocol.json", protocol)
    bind = {"protocol_sha256": digest(out / "protocol.json"),
            "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "result_class": cfg["result_class"]}
    seal_json(out / "inputs.json", {**bind, "samples": samples})
    seal_json(out / "evaluation_labels.json", {**bind, "samples": labels})
    seal_json(out / "baseline.json", {**bind, "claim_label": "approximate reproduction",
                                      "metrics": audit["baseline"]})
    seal_json(out / "prepared.json", {**bind, "inputs_sha256": digest(out / "inputs.json"),
                "evaluation_labels_sha256": digest(out / "evaluation_labels.json"),
                "baseline_sha256": digest(out / "baseline.json")})
    print(json.dumps({"claim_label": "approximate reproduction", "result_class": cfg["result_class"],
                      "images": len(samples), "proposals": sum(len(s["blocks"]) for s in samples),
                      "conditions": conditions, "baseline": audit["baseline"]["recognition"]}, indent=2))


def prepared_inputs(out: Path) -> tuple[dict, dict, dict]:
    protocol = read_json(out / "protocol.json")
    record = read_json(out / "prepared.json")
    if record["protocol_sha256"] != digest(out / "protocol.json"):
        raise ValueError("Prepared protocol checksum mismatch")
    if protocol["implementation_sha256"] != code_hashes():
        raise ValueError("Implementation changed; prepare a new versioned experiment")
    inputs = checked_file(out / "inputs.json", record["inputs_sha256"])
    if inputs["protocol_sha256"] != record["protocol_sha256"]:
        raise ValueError("Observable inputs/protocol mismatch")
    return protocol, record, inputs


def snapshot_path(cfg: dict, cache: Path) -> Path:
    return cache / ("models--" + cfg["bert"]["model_id"].replace("/", "--")) / "snapshots" / cfg["bert"]["revision"]


def fetch_model(args):
    from huggingface_hub import snapshot_download
    cfg = load_config(args.config)
    path = snapshot_download(repo_id=cfg["bert"]["model_id"], revision=cfg["bert"]["revision"],
             cache_dir=str(args.hf_cache.resolve()),
             allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"])
    print("BERT snapshot:", path)


def shard_path(out: Path, condition: str, sid: str) -> Path:
    return out / "inference" / condition / (sha256(sid.encode()).hexdigest() + ".json")


def verify_shard(path: Path, sample: dict, condition: str, record: dict, runtime_sha: str) -> dict:
    row = read_json(path)
    if row.get("payload_sha256") != payload_digest({k: v for k, v in row.items() if k != "payload_sha256"}):
        raise ValueError(f"BERT inference shard checksum differs: {path}")
    if (row["inputs_sha256"] != record["inputs_sha256"] or row["runtime_sha256"] != runtime_sha
            or row["condition"] != condition or row["sample_id"] != sample["sample_id"]
            or row["image_sha256"] != sample["image_sha256"]):
        raise ValueError(f"BERT inference shard provenance differs: {path}")
    trace = row["trace"]
    if len(trace) != len(sample["blocks"]):
        raise ValueError(f"BERT changed the crop population: {path}")
    for before, after in zip(sample["blocks"], trace):
        if any(before[k] != after[k] for k in ("rank", "bbox", "crop_rgb_sha256")) or before["text"] != after["original_text"]:
            raise ValueError(f"BERT changed a crop identity: {path}")
        selected = after["decision"]["selected_index"]
        if (type(selected) is not int or not 0 <= selected < len(after["candidates"])
                or after["parseq_minimum_token_probability"] != before["minimum_token_probability"]
                or after["candidates"][0]["text"] != before["text"]
                or after["text"] != after["candidates"][selected]["text"]):
            raise ValueError(f"BERT chosen text differs from candidate trace: {path}")
        if after["scores"]:
            from recognition.bert_correction import choose_candidate
            if after["decision"] != choose_candidate(after["candidates"], after["scores"], CorrectionPolicy(**record["policy"])):
                raise ValueError(f"BERT gate trace mismatch: {path}")
    return row


def infer(args):
    # Deliberately never open evaluation_labels.json or baseline.json here.
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    protocol, prepared, inputs = prepared_inputs(args.output_dir)
    cfg, bert = protocol["resolved_config"], protocol["resolved_config"]["bert"]
    snapshot = snapshot_path(cfg, args.hf_cache.resolve())
    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt")
    if any(not (snapshot / name).is_file() for name in required):
        raise FileNotFoundError(f"Pinned BERT weights/tokenizer missing at {snapshot}. Run fetch-model on a network-enabled host with this --hf-cache.")
    if args.device not in {"cpu", "cuda:0"}:
        raise ValueError("Select cpu or single-GPU cuda:0 explicitly")
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA:0 is unavailable; no automatic device substitution")
    random.seed(bert["seed"])
    np.random.seed(bert["seed"])
    torch.manual_seed(bert["seed"])
    torch.set_num_threads(bert["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda:0":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(bert["seed"])
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=True)
    model, loading = AutoModelForMaskedLM.from_pretrained(
        snapshot, local_files_only=True, use_safetensors=True, torch_dtype=torch.float32,
        output_loading_info=True)
    if (model.config.model_type != "bert" or loading["missing_keys"] or loading["mismatched_keys"]
            or loading["error_msgs"]
            or set(loading["unexpected_keys"]) - set(bert["allowed_unused_checkpoint_keys"])):
        raise RuntimeError(f"BERT scientifically material checkpoint mismatch: {loading}")
    print("BERT checkpoint loading audit:", json.dumps(loading), flush=True)
    model.to(args.device)
    runtime = {"schema_version": 1, "reproduction_mode": "approximate",
               "claim_label": "approximate reproduction", "inputs_sha256": prepared["inputs_sha256"],
               "model_id": bert["model_id"], "revision": bert["revision"],
               "model_files_sha256": {n: digest(snapshot / n) for n in required},
               "checkpoint_loading": loading, "device": args.device, "precision": "float32",
               "torch": torch.__version__, "torch_cuda": torch.version.cuda,
               "transformers": transformers.__version__, "numpy": np.__version__,
               "python": platform.python_version(), "platform": platform.platform(),
               "gpu": torch.cuda.get_device_name(0) if args.device == "cuda:0" else None,
               "seed": bert["seed"], "cpu_threads": bert["cpu_threads"],
               "deterministic_algorithms": True, "cudnn_benchmark": False, "tf32": False,
               "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
               "mask_task_batch_size": bert["batch_size"], "max_input_tokens": bert["max_input_tokens"],
               "model_set": [bert["model_id"]], "policy": cfg["policy"]}
    seal_json(args.output_dir / "runtime.json", runtime)
    runtime_sha = digest(args.output_dir / "runtime.json")
    scorer = BertWordScorer(tokenizer, model, device=args.device,
                            batch_size=bert["batch_size"], max_tokens=bert["max_input_tokens"])
    policy = CorrectionPolicy(**cfg["policy"])
    model_load_seconds = time.perf_counter() - started
    loading_memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda:0" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda:0" else None,
    }
    check_record = {**prepared, "policy": asdict(policy)}
    for condition in protocol["conditions"]:
        for index, sample in enumerate(inputs["samples"], 1):
            path = shard_path(args.output_dir, condition, sample["sample_id"])
            if path.exists():
                verify_shard(path, sample, condition, check_record, runtime_sha)
                continue
            tick = time.perf_counter()
            if args.device == "cuda:0":
                torch.cuda.reset_peak_memory_stats()
            trace = correct_blocks(sample["blocks"], condition=condition, caption=sample["caption"],
                                   scorer=scorer, policy=policy)
            if args.device == "cuda:0":
                torch.cuda.synchronize()
            row = {"schema_version": 1, "reproduction_mode": "approximate",
                   "claim_label": "approximate reproduction", "condition": condition,
                   "sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"],
                   "inputs_sha256": prepared["inputs_sha256"], "runtime_sha256": runtime_sha,
                   "trace": trace, "correction_seconds": time.perf_counter() - tick,
                   "model_load_seconds": model_load_seconds, "model_loading_memory": loading_memory,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda:0" else None,
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda:0" else None,
                   "max_input_tokens_seen": scorer.max_input_tokens_seen}
            row["payload_sha256"] = payload_digest(row)
            seal_json(path, row)
            if index % 10 == 0 or index == len(inputs["samples"]):
                print(f"[BERT {condition}] {index}/{len(inputs['samples'])}", flush=True)
    # Run timing varies on resume and is not part of the immutable scientific runtime.
    print(f"BERT model load {model_load_seconds:.2f}s; inference shards complete", flush=True)


def matched_gt(predictions: list[dict], ground_truth: list[dict]) -> set[int]:
    edges = [(i, j) for i, p in enumerate(predictions) for j, g in enumerate(ground_truth)
             if box_iou(p["bbox"], g["bbox"]) >= 0.5 and p["text"].casefold() == g["text"].casefold()]
    return maximum_matching(len(predictions), len(ground_truth), edges)


def evaluate(args):
    protocol, prepared, inputs = prepared_inputs(args.output_dir)
    labels = checked_file(args.output_dir / "evaluation_labels.json", prepared["evaluation_labels_sha256"])
    baseline = checked_file(args.output_dir / "baseline.json", prepared["baseline_sha256"])
    if ([s["sample_id"] for s in labels["samples"]] != [s["sample_id"] for s in inputs["samples"]]
            or labels["protocol_sha256"] != prepared["protocol_sha256"]):
        raise ValueError("Evaluation labels are not bound to the frozen image order")
    runtime = read_json(args.output_dir / "runtime.json")
    if runtime["inputs_sha256"] != prepared["inputs_sha256"]:
        raise ValueError("BERT runtime does not match observable inputs")
    runtime_sha = digest(args.output_dir / "runtime.json")
    results = {}
    check_record = {**prepared, "policy": protocol["resolved_config"]["policy"]}
    for condition in protocol["conditions"]:
        rows, details, hashes = [], [], {}
        recovered = damaged = changed = abstentions = 0
        for sample, label in zip(inputs["samples"], labels["samples"]):
            path = shard_path(args.output_dir, condition, sample["sample_id"])
            row = verify_shard(path, sample, condition, check_record, runtime_sha)
            predictions = [{k: t[k] for k in ("rank", "bbox", "text")} for t in row["trace"]]
            before = matched_gt(sample["blocks"], label["ground_truth"])
            after = matched_gt(predictions, label["ground_truth"])
            got, lost = sorted(after - before), sorted(before - after)
            recovered += len(got)
            damaged += len(lost)
            changed += sum(t["text"] != t["original_text"] for t in row["trace"])
            abstentions += sum(t["decision"]["reason"] in {"input_token_limit", "unknown_or_special_target"}
                               for t in row["trace"])
            rows.append({"ground_truth": label["ground_truth"], "predictions": predictions})
            details.append({"sample_id": sample["sample_id"], "ground_truth": label["ground_truth"],
                            "predictions": predictions, "recovered_gt_indices": got, "damaged_gt_indices": lost,
                            "correction_seconds": row["correction_seconds"],
                            "peak_allocated_bytes": row["peak_allocated_bytes"],
                            "peak_reserved_bytes": row["peak_reserved_bytes"]})
            hashes[str(path.relative_to(args.output_dir))] = digest(path)
        metrics = evaluate_images(rows, text_available=True)
        if metrics["geometry"] != baseline["metrics"]["geometry"]:
            raise ValueError("BERT changed localization metrics")
        net = metrics["recognition"]["exact_casefold_spatial_one_to_one"] - baseline["metrics"]["recognition"]["exact_casefold_spatial_one_to_one"]
        if net != recovered - damaged:
            raise ValueError("Recovered/damaged counts do not reconcile with exact-word totals")
        results[condition] = {"metrics": metrics, "recovered": recovered, "damaged": damaged,
                              "net": net, "changed_crops": changed, "explicit_abstentions": abstentions,
                              "correction_seconds_total": sum(d["correction_seconds"] for d in details),
                              "peak_allocated_bytes": max((d["peak_allocated_bytes"] for d in details
                                                            if d["peak_allocated_bytes"] is not None), default=None),
                              "peak_reserved_bytes": max((d["peak_reserved_bytes"] for d in details
                                                           if d["peak_reserved_bytes"] is not None), default=None),
                              "per_image": details, "inference_files_sha256": hashes}
    result = {"schema_version": 1, "reproduction_mode": "approximate",
              "claim_label": "approximate reproduction", "result_class": protocol["result_class"],
              "protocol_sha256": prepared["protocol_sha256"], "runtime_sha256": runtime_sha,
              "inputs_sha256": prepared["inputs_sha256"], "baseline": baseline["metrics"],
              "conditions": results, "official_polygon_protocol": False,
              "end_to_end_runtime_measured": False, "parameters_tuned_on_test": False,
              "test_loaded": False, "training_performed": False}
    seal_json(args.output_dir / "results.json", result)
    print(json.dumps({name: {k: v for k, v in r.items() if k not in {"per_image", "inference_files_sha256"}}
                      for name, r in results.items()}, indent=2))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--captions", type=Path)
    p.add_argument("--reuse-ocr", type=Path, action="append", default=[])
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("fetch-model")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--hf-cache", type=Path, required=True)
    p = sub.add_parser("infer")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--hf-cache", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = arguments()
    try:
        {"prepare": prepare, "fetch-model": fetch_model, "infer": infer, "evaluate": evaluate}[args.stage](args)
    except Exception as error:
        # Retain a failure record, then propagate the original failure. No substitutions.
        if getattr(args, "output_dir", None):
            path = args.output_dir / "errors" / (str(time.time_ns()) + ".json")
            seal_json(path, {"stage": args.stage, "error_type": type(error).__name__, "error": str(error),
                            "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
                            "result_class": "failed approximate reproduction diagnostic"})
        raise


if __name__ == "__main__":
    main()
