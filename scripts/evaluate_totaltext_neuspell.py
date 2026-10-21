"""Approximate frozen Total-Text letterbox -> PARSeq -> spelling-tuned BERT.

Stages are prepare, fetch-model, infer, evaluate. Ground truth is isolated from
the inference stage. Models/crops/results from the earlier MLM runs are retained.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import random
import re
import tempfile
import time

from recognition.neuspell_correction import (
    NeuSpellWordHead, SpellPolicy, convert_vocabulary, correct_blocks,
    git_blob_digest, load_word_head, select_spelling,
)
from recognition.scene_text_v1_evaluation import evaluate_images
from scripts.evaluate_totaltext_bert_correction import (
    checked_file, digest, load_captions, matched_gt, payload_digest, read_json,
    seal_json, shard_path, source_inputs,
)


ARMS = ("native_head", "conservative")
MODEL_FILES = ("config.json", "pytorch_model.bin", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json")
AUXILIARY_FILES = ("vocab.pkl", "vocabulary.json", "author_snippet.py")


def config(path: Path) -> dict:
    path = path.resolve()
    if path.parent.name != "approximate" or path.parent.parent.name != "configs":
        raise ValueError("NeuSpell experiment requires configs/approximate/")
    cfg = read_json(path)
    if (cfg.get("reproduction_mode") != "approximate"
            or cfg.get("claim_label") != "approximate reproduction"
            or cfg.get("dataset") != "total_text"
            or cfg.get("experiment_kind") != "totaltext_frozen_letterbox_neuspell"):
        raise ValueError("Expected explicit approximate Total-Text NeuSpell configuration")
    model, source = cfg["model"], cfg["upstream_neuspell"]
    if (not re.fullmatch(r"[0-9a-f]{40}", model["revision"])
            or not re.fullmatch(r"[0-9a-f]{40}", source["revision"])
            or any(not re.fullmatch(r"[0-9a-f]{40}", source[k])
                   for k in ("vocabulary_git_blob", "snippet_git_blob"))
            or model["precision"] != "float32"
            or model["decoder"] != "mean_subword_trained_word_head"):
        raise ValueError("NeuSpell requires pinned model/source and genuine FP32 word-head decoding")
    for key in ("label_count", "batch_size", "max_input_tokens", "saved_top_k", "cpu_threads"):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"Invalid NeuSpell {key}")
    if type(model["seed"]) is not int or not 0 <= model["seed"] < 2**32:
        raise ValueError("NeuSpell seed must be uint32")
    if (type(model["tokenizer_do_lower_case"]) is not bool
            or set(model["allowed_unused_checkpoint_keys"]) - {
                "bert.pooler.dense.bias", "bert.pooler.dense.weight", "bert.embeddings.position_ids"}
            or model["saved_top_k"] > model["label_count"]
            or not 3 <= model["max_input_tokens"] <= 512
            or type(cfg["fetch"]["timeout_seconds"]) is not int or cfg["fetch"]["timeout_seconds"] < 1):
        raise ValueError("Invalid NeuSpell tokenizer, unused-key or input/download bounds")
    if cfg["conditions"] != ["word_only", "local_ocr"] or cfg["arms"] != list(ARMS):
        raise ValueError("Unsupported NeuSpell control/selection protocol")
    if (cfg["source"]["layout"] != "letterbox_word_f1"
            or cfg["source"]["split"] != "frozen_TRAIN_validation_125"):
        raise ValueError("NeuSpell requires the frozen Total-Text TRAIN-holdout letterbox source")
    SpellPolicy(**cfg["policy"])
    return cfg


def implementation_hashes() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), root / "recognition/neuspell_correction.py",
             root / "recognition/bert_correction.py", root / "recognition/scene_text_v1_evaluation.py",
             root / "scripts/evaluate_totaltext_bert_correction.py"]
    return {str(p.relative_to(root)): digest(p) for p in paths}


def prepare(args):
    cfg = config(args.config)
    reuse = [p.resolve() for p in args.reuse_ocr]
    if any(not p.is_dir() for p in reuse):
        raise FileNotFoundError("Declared historical PARSeq cache is missing")
    samples, labels, source = source_inputs(args.source_dir.resolve(), cfg, reuse)
    conditions = list(cfg["conditions"])
    caption_source = None
    if args.captions:
        captions = load_captions(args.captions, samples)
        for sample in samples:
            sample["caption"] = captions[sample["sample_id"]]
        conditions.append("local_ocr_caption")
        caption_source = {"file": str(args.captions.resolve()), "sha256": digest(args.captions),
                          "model_provenance": {k: v for k, v in read_json(args.captions).items() if k != "samples"}}
    common = {"schema_version": 1, "reproduction_mode": "approximate",
              "claim_label": "approximate reproduction", "result_class": cfg["result_class"]}
    out = args.output_dir
    protocol = {**common, "resolved_config": cfg, "config_sha256": digest(args.config),
                "source_dir": str(args.source_dir.resolve()), "source_files_sha256": source["files_sha256"],
                "upstream_protocol": source["upstream_protocol"],
                "reuse_ocr_directories": [str(p) for p in reuse], "captions": caption_source,
                "conditions": conditions, "arms": cfg["arms"],
                "implementation_sha256": implementation_hashes()}
    seal_json(out / "protocol.json", protocol)
    bound = {**common, "protocol_sha256": digest(out / "protocol.json")}
    seal_json(out / "inputs.json", {**bound, "samples": samples})
    seal_json(out / "evaluation_labels.json", {**bound, "samples": labels})
    seal_json(out / "baseline.json", {**bound, "metrics": source["baseline"]})
    seal_json(out / "prepared.json", {**bound,
              **{name + "_sha256": digest(out / (name + ".json"))
                 for name in ("inputs", "evaluation_labels", "baseline")}})
    print(json.dumps({**common, "images": len(samples),
          "proposals": sum(len(s["blocks"]) for s in samples), "conditions": conditions,
          "baseline_exact": source["baseline"]["recognition"]["exact_casefold_spatial_one_to_one"]}, indent=2))


def prepared(out: Path):
    record = read_json(out / "prepared.json")
    protocol = checked_file(out / "protocol.json", record["protocol_sha256"])
    if protocol["implementation_sha256"] != implementation_hashes():
        raise ValueError("NeuSpell implementation changed; prepare a new experiment version")
    inputs = checked_file(out / "inputs.json", record["inputs_sha256"])
    if inputs["protocol_sha256"] != record["protocol_sha256"]:
        raise ValueError("NeuSpell observable inputs/protocol mismatch")
    return protocol, record, inputs


def snapshot_path(cfg: dict, cache: Path) -> Path:
    return cache / ("models--" + cfg["model"]["model_id"].replace("/", "--")) / "snapshots" / cfg["model"]["revision"]


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Conflicting NeuSpell asset: {path}")
        return
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_assets(cfg: dict, cache: Path, asset_dir: Path):
    manifest = read_json(asset_dir / "assets.json")
    if (manifest["model_id"] != cfg["model"]["model_id"]
            or manifest["model_revision"] != cfg["model"]["revision"]
            or manifest["upstream_neuspell"] != cfg["upstream_neuspell"]):
        raise ValueError("NeuSpell asset manifest does not match the pinned model/source")
    if (set(manifest["model_files_sha256"]) != set(MODEL_FILES)
            or set(manifest["auxiliary_files_sha256"]) != set(AUXILIARY_FILES)
            or any(not re.fullmatch(r"[0-9a-f]{64}", v) for v in
                   [manifest["authoritative_weight_sha256"], *manifest["model_files_sha256"].values(),
                    *manifest["auxiliary_files_sha256"].values()])):
        raise ValueError("NeuSpell manifest lacks complete model/vocabulary checksums")
    snapshot = snapshot_path(cfg, cache)
    for name, value in manifest["model_files_sha256"].items():
        if digest(snapshot / name) != value:
            raise ValueError(f"NeuSpell model asset checksum mismatch: {name}")
    if digest(snapshot / "pytorch_model.bin") != manifest["authoritative_weight_sha256"]:
        raise ValueError("NeuSpell weights differ from authoritative Hugging Face LFS metadata")
    for name, value in manifest["auxiliary_files_sha256"].items():
        if digest(asset_dir / name) != value:
            raise ValueError(f"NeuSpell vocabulary/source checksum mismatch: {name}")
    vocabulary = read_json(asset_dir / "vocabulary.json")
    if (vocabulary["source_git_blob_sha1"] != cfg["upstream_neuspell"]["vocabulary_git_blob"]
            or len(vocabulary["labels"]) != cfg["model"]["label_count"]):
        raise ValueError("NeuSpell decoded vocabulary/source differs")
    original = convert_vocabulary(asset_dir / "vocab.pkl",
        expected_blob=cfg["upstream_neuspell"]["vocabulary_git_blob"], expected_labels=cfg["model"]["label_count"])
    if vocabulary != original:
        raise ValueError("NeuSpell decoded label order differs from the authenticated raw vocabulary")
    if git_blob_digest((asset_dir / "author_snippet.py").read_bytes()) != cfg["upstream_neuspell"]["snippet_git_blob"]:
        raise ValueError("NeuSpell author snippet Git blob mismatch")
    evidence = cfg.get("assets_evidence")
    if evidence and (
            evidence["checkpoint_sha256"] != manifest["model_files_sha256"]["pytorch_model.bin"]
            or evidence["tokenizer_config_sha256"] != manifest["model_files_sha256"]["tokenizer_config.json"]
            or evidence["vocabulary_raw_sha256"] != manifest["auxiliary_files_sha256"]["vocab.pkl"]):
        raise ValueError("NeuSpell downloaded assets differ from the configuration provenance evidence")
    return snapshot, vocabulary, manifest


def fetch_model(args):
    import requests
    from huggingface_hub import HfApi, snapshot_download

    cfg = config(args.config)
    cache, asset_dir = args.hf_cache.resolve(), args.asset_dir.resolve()
    if (asset_dir / "assets.json").exists():
        verify_assets(cfg, cache, asset_dir)
        print("Reused verified pinned NeuSpell assets", asset_dir)
        return
    model = cfg["model"]
    info = HfApi().model_info(model["model_id"], revision=model["revision"], files_metadata=True)
    if info.sha != model["revision"]:
        raise ValueError("Hugging Face resolved a different NeuSpell model revision")
    weight = next((s for s in info.siblings if s.rfilename == "pytorch_model.bin"), None)
    if weight is None or weight.lfs is None or not re.fullmatch(r"[0-9a-f]{64}", weight.lfs.sha256):
        raise ValueError("Authoritative NeuSpell weight LFS checksum is unavailable")
    snapshot = Path(snapshot_download(model["model_id"], revision=model["revision"], cache_dir=str(cache),
        allow_patterns=list(MODEL_FILES)))
    if digest(snapshot / "pytorch_model.bin") != weight.lfs.sha256:
        raise ValueError("Downloaded NeuSpell checkpoint fails authoritative LFS checksum")
    source = cfg["upstream_neuspell"]
    for file_name, source_path, blob in (
            ("vocab.pkl", source["vocabulary_path"], source["vocabulary_git_blob"]),
            ("author_snippet.py", source["snippet_path"], source["snippet_git_blob"])):
        url = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{source_path}"
        response = requests.get(url, timeout=cfg["fetch"]["timeout_seconds"])
        response.raise_for_status()
        data = response.content
        if git_blob_digest(data) != blob:
            raise ValueError(f"NeuSpell pinned upstream blob differs: {file_name}")
        atomic_bytes(asset_dir / file_name, data)
    vocabulary = convert_vocabulary(asset_dir / "vocab.pkl", expected_blob=source["vocabulary_git_blob"],
                                     expected_labels=model["label_count"])
    seal_json(asset_dir / "vocabulary.json", vocabulary)
    manifest = {"schema_version": 1, "reproduction_mode": "approximate",
                "claim_label": "approximate reproduction", "model_id": model["model_id"],
                "model_revision": model["revision"], "upstream_neuspell": source,
                "authoritative_weight_sha256": weight.lfs.sha256,
                "model_files_sha256": {n: digest(snapshot / n) for n in MODEL_FILES},
                "auxiliary_files_sha256": {n: digest(asset_dir / n) for n in AUXILIARY_FILES}}
    seal_json(asset_dir / "assets.json", manifest)
    print("Verified NeuSpell assets", asset_dir, "checkpoint SHA256", weight.lfs.sha256)


def verify_shard(path: Path, sample: dict, condition: str, record: dict, runtime_sha: str, policy: SpellPolicy):
    row = read_json(path)
    if row.get("payload_sha256") != payload_digest({k: v for k, v in row.items() if k != "payload_sha256"}):
        raise ValueError(f"NeuSpell shard payload checksum differs: {path}")
    if (row["inputs_sha256"] != record["inputs_sha256"] or row["runtime_sha256"] != runtime_sha
            or row["sample_id"] != sample["sample_id"] or row["image_sha256"] != sample["image_sha256"]
            or row["condition"] != condition or len(row["trace"]) != len(sample["blocks"])):
        raise ValueError(f"NeuSpell shard identity/population differs: {path}")
    for before, after in zip(sample["blocks"], row["trace"]):
        if (any(before[k] != after[k] for k in ("rank", "bbox", "crop_rgb_sha256"))
                or before["text"] != after["original_text"]
                or after["selections"] != select_spelling(before["text"], after["neural"], policy)):
            raise ValueError(f"NeuSpell crop/selection trace differs: {path}")
    return row


def infer(args):
    # No GT/baseline file is opened in this stage.
    import numpy as np
    import huggingface_hub
    import rapidfuzz
    import tokenizers
    import torch
    import transformers
    from transformers import AutoTokenizer

    protocol, record, inputs = prepared(args.output_dir)
    cfg = protocol["resolved_config"]
    if not (args.asset_dir / "assets.json").is_file():
        raise FileNotFoundError("Pinned spelling-tuned NeuSpell assets missing; run fetch-model on a network-enabled host")
    snapshot, vocabulary, assets = verify_assets(cfg, args.hf_cache.resolve(), args.asset_dir.resolve())
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("NeuSpell CUDA:0 unavailable; no automatic CPU substitution")
    if args.device not in {"cpu", "cuda:0"}:
        raise ValueError("Select CPU or single-GPU CUDA:0 explicitly")
    model_cfg = cfg["model"]
    random.seed(model_cfg["seed"])
    np.random.seed(model_cfg["seed"])
    torch.manual_seed(model_cfg["seed"])
    torch.set_num_threads(model_cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == "cuda:0":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(model_cfg["seed"])
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    if tokenizer.do_lower_case != model_cfg["tokenizer_do_lower_case"]:
        raise ValueError(
            f"NeuSpell tokenizer normalization differs: expected do_lower_case={model_cfg['tokenizer_do_lower_case']}, "
            f"loaded {tokenizer.do_lower_case}. Inspect the pinned tokenizer_config.json and select an explicit "
            "APPROXIMATE configuration with a new output directory; no normalization override was applied.")
    model, loading = load_word_head(snapshot, device=args.device, expected_labels=model_cfg["label_count"],
                                    allowed_unused_keys=model_cfg["allowed_unused_checkpoint_keys"])
    decoder = NeuSpellWordHead(tokenizer, model, vocabulary, device=args.device,
              batch_size=model_cfg["batch_size"], max_input_tokens=model_cfg["max_input_tokens"],
              saved_top_k=model_cfg["saved_top_k"])
    policy = SpellPolicy(**cfg["policy"])
    runtime = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
               "inputs_sha256": record["inputs_sha256"], "assets_sha256": digest(args.asset_dir / "assets.json"),
               "model_id": model_cfg["model_id"], "revision": model_cfg["revision"],
               "model_files_sha256": assets["model_files_sha256"],
               "auxiliary_files_sha256": assets["auxiliary_files_sha256"], "loading_audit": loading,
               "decoder": model_cfg["decoder"], "device": args.device, "precision": "float32",
               "tokenizer_class": type(tokenizer).__name__, "tokenizer_do_lower_case": tokenizer.do_lower_case,
               "model_set": [model_cfg["model_id"]], "parameters": sum(p.numel() for p in model.parameters()),
               "torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
               "tokenizers": tokenizers.__version__, "huggingface_hub": huggingface_hub.__version__,
               "rapidfuzz": rapidfuzz.__version__,
               "numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform(),
               "gpu": torch.cuda.get_device_name(0) if args.device == "cuda:0" else None,
               "seed": model_cfg["seed"], "cpu_threads": model_cfg["cpu_threads"],
               "batch_size": model_cfg["batch_size"], "max_input_tokens": model_cfg["max_input_tokens"],
               "deterministic_algorithms": True, "cudnn_benchmark": False, "tf32": False,
               "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "policy": asdict(policy)}
    seal_json(args.output_dir / "runtime.json", runtime)
    runtime_sha = digest(args.output_dir / "runtime.json")
    loading_seconds = time.perf_counter() - started
    loading_memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda:0" else None,
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda:0" else None}
    for condition in protocol["conditions"]:
        for index, sample in enumerate(inputs["samples"], 1):
            path = shard_path(args.output_dir, condition, sample["sample_id"])
            if path.exists():
                verify_shard(path, sample, condition, record, runtime_sha, policy)
                continue
            tick = time.perf_counter()
            if args.device == "cuda:0":
                torch.cuda.reset_peak_memory_stats()
            trace = correct_blocks(sample["blocks"], condition=condition, caption=sample["caption"],
                                   decoder=decoder, policy=policy)
            if args.device == "cuda:0":
                torch.cuda.synchronize()
            row = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
                   "inputs_sha256": record["inputs_sha256"], "runtime_sha256": runtime_sha,
                   "sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"], "condition": condition,
                   "trace": trace, "correction_seconds": time.perf_counter() - tick,
                   "model_loading_seconds": loading_seconds, "model_loading_memory": loading_memory,
                   "model_input_shapes": [list(s) for s in sorted(decoder.input_shapes)],
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda:0" else None,
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda:0" else None}
            row["payload_sha256"] = payload_digest(row)
            seal_json(path, row)
            if index % 10 == 0 or index == len(inputs["samples"]):
                print(f"[NeuSpell {condition}] {index}/{len(inputs['samples'])}", flush=True)


def evaluate(args):
    protocol, record, inputs = prepared(args.output_dir)
    labels = checked_file(args.output_dir / "evaluation_labels.json", record["evaluation_labels_sha256"])
    baseline = checked_file(args.output_dir / "baseline.json", record["baseline_sha256"])
    if [s["sample_id"] for s in labels["samples"]] != [s["sample_id"] for s in inputs["samples"]]:
        raise ValueError("NeuSpell GT/image order differs")
    runtime = read_json(args.output_dir / "runtime.json")
    if runtime["inputs_sha256"] != record["inputs_sha256"]:
        raise ValueError("NeuSpell runtime/input contract differs")
    runtime_sha = digest(args.output_dir / "runtime.json")
    policy = SpellPolicy(**protocol["resolved_config"]["policy"])
    summaries = {}
    for condition in protocol["conditions"]:
        verified = [verify_shard(shard_path(args.output_dir, condition, sample["sample_id"]), sample,
                                 condition, record, runtime_sha, policy) for sample in inputs["samples"]]
        for arm in protocol["arms"]:
            rows, details, hashes = [], [], {}
            recovered = damaged = changed = abstained = 0
            for sample, label, shard in zip(inputs["samples"], labels["samples"], verified):
                predictions = [{"rank": t["rank"], "bbox": t["bbox"], "text": t["selections"][arm]["text"]}
                               for t in shard["trace"]]
                before, after = matched_gt(sample["blocks"], label["ground_truth"]), matched_gt(predictions, label["ground_truth"])
                got, lost = sorted(after - before), sorted(before - after)
                recovered += len(got)
                damaged += len(lost)
                changed += sum(p["text"] != b["text"] for p, b in zip(predictions, sample["blocks"]))
                abstained += sum(t["neural"]["status"] in {"input_token_limit", "unknown_or_special_subword"} for t in shard["trace"])
                rows.append({"ground_truth": label["ground_truth"], "predictions": predictions})
                details.append({"sample_id": sample["sample_id"], "ground_truth": label["ground_truth"],
                     "predictions": predictions, "recovered_gt_indices": got, "damaged_gt_indices": lost,
                     "correction_seconds": shard["correction_seconds"]})
                path = shard_path(args.output_dir, condition, sample["sample_id"])
                hashes[str(path.relative_to(args.output_dir))] = digest(path)
            metric = evaluate_images(rows, text_available=True)
            if metric["geometry"] != baseline["metrics"]["geometry"]:
                raise ValueError("NeuSpell changed the frozen localization geometry")
            net = metric["recognition"]["exact_casefold_spatial_one_to_one"] - baseline["metrics"]["recognition"]["exact_casefold_spatial_one_to_one"]
            if net != recovered - damaged:
                raise ValueError("NeuSpell exact/recovery/damage totals do not reconcile")
            summaries[condition + "/" + arm] = {"metrics": metric, "recovered": recovered, "damaged": damaged,
                "net": net, "changed_crops": changed, "explicit_abstentions": abstained, "per_image": details,
                "correction_seconds_total": sum(s["correction_seconds"] for s in verified),
                "peak_allocated_bytes": max((s["peak_allocated_bytes"] for s in verified if s["peak_allocated_bytes"] is not None), default=None),
                "peak_reserved_bytes": max((s["peak_reserved_bytes"] for s in verified if s["peak_reserved_bytes"] is not None), default=None),
                "inference_files_sha256": hashes}
    result = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
              "result_class": protocol["resolved_config"]["result_class"], "protocol_sha256": record["protocol_sha256"],
              "runtime_sha256": runtime_sha, "inputs_sha256": record["inputs_sha256"], "baseline": baseline["metrics"],
              "conditions": summaries, "test_loaded": False, "training_performed": False,
              "official_polygon_protocol": False, "parameters_tuned_on_test": False, "end_to_end_runtime_measured": False}
    seal_json(args.output_dir / "results.json", result)
    print(json.dumps({k: {n: v for n, v in s.items() if n not in {"per_image", "inference_files_sha256"}}
                      for k, s in summaries.items()}, indent=2))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--reuse-ocr", type=Path, action="append", default=[])
    p.add_argument("--captions", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("fetch-model")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--hf-cache", type=Path, required=True)
    p.add_argument("--asset-dir", type=Path, required=True)
    p = sub.add_parser("infer")
    p.add_argument("--hf-cache", type=Path, required=True)
    p.add_argument("--asset-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = arguments()
    try:
        {"prepare": prepare, "fetch-model": fetch_model, "infer": infer, "evaluate": evaluate}[args.stage](args)
    except Exception as error:
        folder = getattr(args, "output_dir", None) or getattr(args, "asset_dir", None)
        if folder:
            seal_json(folder / "errors" / (str(time.time_ns()) + ".json"),
                      {"stage": args.stage, "error_type": type(error).__name__, "error": str(error),
                       "claim_label": "approximate reproduction", "result_class": "failed NeuSpell diagnostic"})
        raise


if __name__ == "__main__":
    main()
