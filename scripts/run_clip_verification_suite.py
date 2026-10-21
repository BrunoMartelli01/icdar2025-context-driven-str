"""Download pinned CLIP snapshots, run v1/v2 sequentially, and summarize.

No dependency installations. --offline --models vit_b32 --device cpu --limit 1
is a local smoke using the existing snapshot and real frozen PARSeq inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from scripts import run_clip_word_verification as base
from scripts import run_clip_innovative_verification as innovative
from scripts.crop_clip_runtime import sha, write_json


REPO = base.REPO


def ensure_snapshot(spec: dict, path: Path, *, offline: bool) -> dict:
    required = ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "vocab.json", "merges.txt", spec["weight_file"]]
    if offline:
        identity = base.model_identity(path, spec)
        return {"source": "local_only", "model_id": spec["model_id"], "revision": spec["revision"],
                "weight_sha256": identity["files_sha256"][spec["weight_file"]],
                "upstream_verified": spec["weight_sha256"] is not None}
    if spec["weight_sha256"] and all((path/name).is_file() for name in required):
        identity = base.model_identity(path, spec)
        return {"source": "known_pinned_local_weight", "model_id": spec["model_id"], "revision": spec["revision"],
                "weight_sha256": identity["files_sha256"][spec["weight_file"]], "upstream_verified": True}
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    url = hf_hub_url(spec["model_id"], spec["weight_file"], revision=spec["revision"])
    metadata = get_hf_file_metadata(url)
    expected = metadata.etag
    if metadata.commit_hash != spec["revision"] or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("pinned HF weight metadata must identify the full commit and SHA256 LFS object")
    if spec["weight_sha256"] and spec["weight_sha256"] != expected:
        raise ValueError("remote pinned CLIP checkpoint differs from protocol checksum")
    for name in required:
        print(f"[download] {spec['model_id']} {spec['revision']} {name}", flush=True)
        hf_hub_download(spec["model_id"], name, revision=spec["revision"], local_dir=str(path))
    actual = sha(path/spec["weight_file"])
    if actual != expected:
        raise ValueError("downloaded CLIP checkpoint fails upstream SHA256 verification")
    base.model_identity(path, spec)
    return {"source": "pinned_huggingface_LFS", "model_id": spec["model_id"], "revision": spec["revision"],
            "weight_sha256": actual, "upstream_verified": True, "metadata_url": url,
            "commit_hash": metadata.commit_hash, "etag": metadata.etag, "size": metadata.size}


def child(module: str, arguments: list[str]) -> None:
    print(f"[stage] {module}", flush=True)
    subprocess.run([sys.executable, "-u", "-m", module, *arguments], cwd=REPO, check=True)


def run(args: argparse.Namespace) -> Path:
    config = base.load_config(args.config)
    paths = base.paths_profile(args.paths)
    innovation, innovation_base = innovative.load_config(args.innovative_config)
    innovation_paths = innovative.local_paths(args.innovative_paths)
    if config != innovation_base:
        raise ValueError("v1 and innovative extension must reference the same base scientific configuration")
    if len(set(args.models)) != len(args.models) or not set(args.models) <= config["models"].keys():
        raise ValueError("model names must be unique declared checkpoint variants")
    if args.limit is not None and not 1 <= args.limit < config["expected_images"]:
        raise ValueError("smoke limit must be positive and below full population size; omit --limit for a full run")
    # Every protocol/code update gets a new output tree. Device is part of its identity.
    content = {"config_sha256": sha(args.config), "innovative_config_sha256": sha(args.innovative_config),
        "paths_sha256": sha(args.paths), "innovative_paths_sha256": sha(args.innovative_paths),
        "models": args.models, "device": args.device, "limit": args.limit, "offline": args.offline,
        "software": {p: innovative.importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "Pillow")},
        "code_sha256": {p: sha(REPO/p) for p in (*innovative.CODE, "scripts/run_clip_verification_suite.py",
                                                "scripts/run_clip_verification_suite_host.sh")}}
    token = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
    root = args.output_root.resolve()/token
    write_json(root/"launch.json", {"schema_version": 1, "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction", "status": "STARTING", **content})
    print(f"[suite] output={root} device={args.device}", flush=True)
    try:
        if args.device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in this process; run the command in your host terminal or explicitly choose --device cpu")
        if args.device not in ("cpu", "cuda:0"):
            raise ValueError("choose cpu or cuda:0")
        # Verify dataset/caption/crop evidence before spending time on downloads.
        base.prepare(config, paths, args.limit)
        innovative.font_identity(innovation, innovation_paths["fonts"])
        hardware = {"device": args.device, "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda:0" else None,
                    "free_total_cuda_bytes": list(torch.cuda.mem_get_info(0)) if args.device == "cuda:0" else None}
        write_json(root/"hardware.json", hardware)
        print(f"[hardware] gpu={hardware['gpu_name']}", flush=True)
        local = json.loads(args.paths.read_text())
        local["output_root"] = str(root/"classic")
        local_innovative = json.loads(args.innovative_paths.read_text())
        local_innovative["output_root"] = str(root/"innovative")
        write_json(root/"classic_paths.json", local)
        write_json(root/"innovative_paths.json", local_innovative)
        comparison = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
                      "status": "RUNNING", "models": {}, "table_5_claim": False, "test_loaded": False}
        for name in args.models:
            receipt = ensure_snapshot(config["models"][name], paths["models"][name], offline=args.offline)
            write_json(root/"downloads"/(name+".json"), receipt)
            shared = ["--model", name, "--device", args.device]
            limit = ["--limit", str(args.limit)] if args.limit is not None else []
            child("scripts.run_clip_word_verification", ["--config", str(args.config), "--paths",
                  str(root/"classic_paths.json"), *shared, *limit])
            original = root/"classic"/name/(f"smoke-{args.limit}" if args.limit else "train125")
            child("scripts.run_clip_innovative_verification", ["--config", str(args.innovative_config),
                  "--paths", str(root/"innovative_paths.json"), "--base-run", str(original), *shared])
            advanced = root/"innovative"/name/"all-methods"
            report = json.loads((advanced/"results.json").read_text())
            comparison["models"][name] = {"revision": config["models"][name]["revision"],
                "classic_results": str(original/"results.json"), "innovative_results": str(advanced/"results.json"),
                "innovative_results_sha256": sha(advanced/"results.json"), "images": report["images"],
                "result_class": report["result_class"], "summary": report["summary"],
                "baseline_correctness": report["baseline_correctness"], "cuda_peak": report["cuda_peak"],
                "download_provenance": receipt}
            write_json(root/"comparison.json", comparison)
        comparison["status"] = "SUCCESS"
        write_json(root/"comparison.json", comparison)
        lines = ["# CLIP checkpoint and innovative-method comparison", "",
                 "approximate reproduction; TRAIN development; not independent TEST/Table5.", "",
                 "| Model | Rule | Correct | Recovered | Damaged |", "|---|---|---:|---:|---:|"]
        for name, report in comparison["models"].items():
            for arm, metric in sorted(report["summary"].items(), key=lambda v: (-v[1]["exact"], v[0])):
                lines.append(f"| {name} | {arm} | {metric['exact']} / {metric['care_words']} | {metric['recovered']} | {metric['damaged']} |")
        (root/"COMPARISON.md").write_text("\n".join(lines)+"\n")
        print(f"\nCOMPLETED: {root/'comparison.json'}\nReadable table: {root/'COMPARISON.md'}", flush=True)
        return root
    except Exception as error:
        write_json(root/"suite_failure.json", {"schema_version": 1, "status": "FAILED",
            "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["vit_b32", "vit_b16", "vit_l14"])
    parser.add_argument("--device", default="cuda:0", choices=["cpu", "cuda:0"])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/clip-verification-host-suite"))
    parser.add_argument("--config", type=Path, default=REPO/"configs/approximate/clip_word_verification_train_v1.json")
    parser.add_argument("--paths", type=Path, default=REPO/"configs/approximate/clip_word_verification_local_paths_v1.json")
    parser.add_argument("--innovative-config", type=Path, default=REPO/"configs/approximate/clip_innovative_verification_train_v2.json")
    parser.add_argument("--innovative-paths", type=Path, default=REPO/"configs/approximate/clip_innovative_verification_local_paths_v2.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
