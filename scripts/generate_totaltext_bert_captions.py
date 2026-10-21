"""Approximate SHA-bound Florence scene context for frozen Total-Text OCR inputs.

Uses an existing isolated pure-Python dependency target; never installs packages.
No annotation, baseline or evaluation-label file is opened by this script.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

from scripts.evaluate_totaltext_bert_correction import digest, payload_digest, read_json, seal_json


def load_config(path):
    if path.resolve().parent.name != "approximate" or path.resolve().parent.parent.name != "configs":
        raise ValueError("Caption methods require configs/approximate/")
    cfg = read_json(path)
    if (cfg["reproduction_mode"] != "approximate" or cfg["claim_label"] != "approximate reproduction"
            or cfg["experiment_kind"] != "totaltext_bert_method_battery" or cfg["dataset"] != "total_text"):
        raise ValueError("Expected explicit approximate Total-Text BERT methods config")
    return cfg


def bind_rgb(inputs, manifest, root, expected_manifest_sha):
    import yaml
    if digest(manifest) != expected_manifest_sha:
        raise ValueError("Florence RGB lookup manifest checksum differs")
    obj = yaml.safe_load(manifest.read_text())
    if obj["dataset"] != "total_text" or obj["split_role"] != "train":
        raise ValueError("Florence RGB manifest must describe Total-Text TRAIN")
    by_id = {s["id"]: s for s in obj["samples"]}
    if len(by_id) != len(obj["samples"]):
        raise ValueError("Duplicate image IDs in RGB lookup manifest")
    result = []
    for sample in inputs["samples"]:
        item = by_id[sample["sample_id"]]
        relative = Path(item["image"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root.resolve()):
            raise ValueError("RGB image escapes configured data root")
        if item["image_sha256"] != sample["image_sha256"] or digest(path) != sample["image_sha256"]:
            raise ValueError("Florence image does not match the frozen OCR input")
        result.append({"sample_id": sample["sample_id"], "image_sha256": sample["image_sha256"],
                       "image_path": str(path)})
    if len({s["sample_id"] for s in result}) != len(result):
        raise ValueError("Duplicate frozen caption input IDs")
    return result


def caption_path(out, sid):
    from hashlib import sha256
    return out / "samples" / (sha256(sid.encode()).hexdigest() + ".json")


def checked_caption(path, sample, runtime_sha):
    obj = read_json(path)
    if (obj["payload_sha256"] != payload_digest({k:v for k,v in obj.items() if k != "payload_sha256"})
            or any(obj[k] != sample[k] for k in ("sample_id", "image_sha256"))
            or obj["runtime_sha256"] != runtime_sha or not isinstance(obj["text"],str) or not obj["text"].strip()):
        raise ValueError("Florence caption shard checksum/identity/coverage differs")
    return obj


def run(args):
    cfg = load_config(args.config)
    inputs = read_json(args.inputs)
    if len(inputs["samples"]) != cfg["source"]["expected_images"]:
        raise ValueError("Caption input population differs from frozen source")
    samples = bind_rgb(inputs, args.manifest, args.data_root, cfg["caption"]["manifest_sha256"])
    cap = cfg["caption"]
    snapshot = args.hf_cache / ("models--" + cap["model_id"].replace("/","--")) / "snapshots" / cap["revision"]
    for name, sha in cap["files_sha256"].items():
        if digest(snapshot / name) != sha:
            raise ValueError(f"Florence pinned model file checksum differs: {name}")
    if digest(snapshot / "model.safetensors") != cap["weight_sha256"]:
        raise ValueError("Florence pinned weights differ")
    if digest(args.runtime_deps / "einops/einops.py") != cap["einops_source_sha256"]:
        raise ValueError("Existing isolated einops dependency differs")
    protocol = {"schema_version":1, "reproduction_mode":"approximate", "claim_label":"approximate reproduction",
        "result_class":"Total-Text TRAIN-holdout scene caption experiment", "resolved_config":cfg,
        "config_sha256":digest(args.config), "implementation_sha256":digest(Path(__file__)),
        "inputs_sha256":digest(args.inputs), "manifest_sha256":digest(args.manifest), "samples":samples,
        "runtime_dependency_path":str(args.runtime_deps.resolve()), "device":args.device,
        "module_cache":str(args.module_cache.resolve())}
    seal_json(args.output_dir / "protocol.json", protocol)
    os.environ["HF_MODULES_CACHE"] = str(args.module_cache.resolve())
    sys.path.insert(0, str(args.runtime_deps.resolve()))
    import numpy as np
    import torch
    import transformers
    import einops
    import timm
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    if einops.__version__ != cap["einops_version"]:
        raise ValueError("Existing einops version differs")
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("Florence CUDA unavailable; no automatic CPU substitution")
    random.seed(cfg["model"]["seed"])
    np.random.seed(cfg["model"]["seed"])
    torch.manual_seed(cfg["model"]["seed"])
    torch.set_num_threads(cap["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == "cuda:0":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(cfg["model"]["seed"])
        torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    model, loading = AutoModelForCausalLM.from_pretrained(snapshot, trust_remote_code=True,
        local_files_only=True, use_safetensors=True, torch_dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
        raise RuntimeError(f"Florence checkpoint compatibility failure: {loading}")
    model.to(args.device).eval()
    runtime = {"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "protocol_sha256":digest(args.output_dir/"protocol.json"), "model_id":cap["model_id"],
        "revision":cap["revision"],"model_files_sha256":cap["files_sha256"],"loading_audit":loading,
        "model_set":[cap["model_id"]],"device":args.device,"precision":"float32","batch_size":1,
        "task":cap["task"],"generation":cap["generation"],"seed":cfg["model"]["seed"],
        "cpu_threads":cap["cpu_threads"],"deterministic_algorithms":True,"tf32":False,
        "python":platform.python_version(),"torch":torch.__version__,"transformers":transformers.__version__,
        "numpy":np.__version__,"einops":einops.__version__,"timm":timm.__version__,
        "parameters":sum(p.numel() for p in model.parameters())}
    seal_json(args.output_dir/"runtime.json",runtime)
    runtime_sha = digest(args.output_dir/"runtime.json")
    load_seconds = time.perf_counter()-tick
    for index,sample in enumerate(samples,1):
        path=caption_path(args.output_dir,sample["sample_id"])
        if path.exists():
            checked_caption(path,sample,runtime_sha)
            continue
        if args.max_images is not None and index > args.max_images:
            break
        if args.device == "cuda:0":
            torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        with Image.open(sample["image_path"]) as opened:
            image=opened.convert("RGB")
        if digest(Path(sample["image_path"])) != sample["image_sha256"]:
            raise ValueError("RGB image changed after caption preparation")
        features=processor(text=cap["task"],images=image,return_tensors="pt")
        shapes={k:list(v.shape) for k,v in features.items()}
        features={k:v.to(args.device) for k,v in features.items()}
        with torch.inference_mode():
            generated=model.generate(**features,**cap["generation"])
        if args.device == "cuda:0":
            torch.cuda.synchronize()
        raw=processor.batch_decode(generated,skip_special_tokens=False)[0]
        text=processor.post_process_generation(raw,task=cap["task"],image_size=image.size)[cap["task"]]
        if not isinstance(text,str) or not text.strip():
            raise ValueError("Florence emitted an empty or non-text caption")
        row={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
            "sample_id":sample["sample_id"],"image_sha256":sample["image_sha256"],"runtime_sha256":runtime_sha,
            "text":text.strip(),"raw_generation":raw,"generated_tokens":int(generated.shape[-1]),
            "latency_seconds":time.perf_counter()-start,"model_loading_seconds":load_seconds,"input_shapes":shapes,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated() if args.device=="cuda:0" else None,
            "peak_reserved_bytes":torch.cuda.max_memory_reserved() if args.device=="cuda:0" else None}
        row["payload_sha256"]=payload_digest(row)
        seal_json(path,row)
        print(f"[Florence scene] {index}/{len(samples)} {row['latency_seconds']:.2f}s: {text[:110]}",flush=True)
    complete=all(caption_path(args.output_dir,s["sample_id"]).is_file() for s in samples)
    if not complete:
        print("Partial caption smoke; no complete captions.json emitted",flush=True)
        return
    rows=[checked_caption(caption_path(args.output_dir,s["sample_id"]),s,runtime_sha) for s in samples]
    output={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":"complete frozen TRAIN-holdout captions", "model_id":cap["model_id"],"revision":cap["revision"],
        "generation":cap["generation"],"model_files_sha256":cap["files_sha256"],"runtime_sha256":runtime_sha,
        "protocol_sha256":runtime["protocol_sha256"], "samples":[{k:r[k] for k in ("sample_id","image_sha256","text")} for r in rows],
        "caption_seconds_total":sum(r["latency_seconds"] for r in rows),"test_loaded":False,"training_performed":False}
    seal_json(args.output_dir/"captions.json",output)
    print("Complete Total-Text scene captions",len(rows),digest(args.output_dir/"captions.json"),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("config","inputs","manifest","data-root","hf-cache","runtime-deps","module-cache","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda:0"),required=True)
    p.add_argument("--max-images",type=int)
    args=p.parse_args()
    if args.max_images is not None and args.max_images < 1:
        p.error("--max-images must be positive")
    try:
        run(args)
    except Exception as e:
        seal_json(args.output_dir/"errors"/(str(time.time_ns())+".json"),
            {"error_type":type(e).__name__,"error":str(e),"reproduction_mode":"approximate",
             "claim_label":"approximate reproduction","result_class":"failed scene caption experiment"})
        raise


if __name__=="__main__":
    main()
