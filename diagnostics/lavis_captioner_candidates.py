"""Five-image caption diagnostics; isolated from all scientific pipeline defaults."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import statistics
import sys
import time
import traceback
import urllib.request
from pathlib import Path

SAMPLES = (1, 25, 26, 27, 28)
ROOT = Path("artifacts/diagnostics/lavis-captioner-candidates-20260914")
LAVIS_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/77/1b/"
    "112981873d44de113ecaab8084da82606871ad1e914e00c6ceb68f96b394/"
    "salesforce_lavis-1.0.2-py3-none-any.whl"
)
LAVIS_WHEEL_SHA = "489782efc2d98d8e878112b2758f82ac2e6c10dfcfa087db6656465c7784d49f"
CANDIDATES = {
    "blip-base-coco": {
        "model": "blip_caption", "model_type": "base_coco", "backend": "lavis",
        "checkpoint_url": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP/blip_coco_caption_base.pth",
        "backbone": "ViT-Base", "is_blip2": False,
        "generation": {"use_nucleus_sampling": False, "num_beams": 3,
                       "min_length": 10, "max_length": 80},
    },
    "blip2-vitl": {
        "model": "blip2", "model_type": "pretrain_vitL", "backend": "lavis",
        "checkpoint_url": "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_vitL.pth",
        "backbone": "CLIP ViT-L + Q-Former; no OPT/T5", "is_blip2": True,
        "generation": {"use_nucleus_sampling": False, "num_beams": 3,
                       "min_length": 10, "max_length": 80},
    },
    "florence2-base-ft": {
        "model": "microsoft/Florence-2-base-ft", "model_type": "base-ft",
        "backend": "transformers", "backbone": "DaViT + sequence decoder",
        "is_blip2": False,
        "revision": "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",
        "weight_sha256": "58757d657ff44051314c8030b68e04cb1bb618ca9a4885418f111f6fb708185a",
        "generation": {"max_new_tokens": 512, "num_beams": 3, "do_sample": False},
        "tasks": ["<MORE_DETAILED_CAPTION>", "<OCR>"],
    },
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with part.open("w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    part.replace(path)


def write_json(path, obj):
    atomic(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def text_metrics(text, prompt=""):
    words = text.split()
    tokens = re.findall(r"\w+", text.casefold())
    tri = [tuple(tokens[i:i + 3]) for i in range(max(0, len(tokens) - 2))]
    return {"word_count": len(words),
            "repeated_trigram_fraction": 1 - len(set(tri)) / len(tri) if tri else 0.0,
            "prompt_echo": bool(prompt.strip() and prompt.casefold().strip() in text.casefold())}


def archive_empty_preflight(root, old, new):
    scientific = ("schema_version", "mode", "classification", "scope", "sample_ids",
                  "images", "candidates", "reference", "selection")
    if any(old.get(k) != new.get(k) for k in scientific):
        raise RuntimeError("Frozen protocol changed; use a new diagnostic directory")
    names = ["protocol.json", "runner.py", "run_host.sh", "comparison.json", "comparison.md", "SHA256SUMS"]
    for candidate in CANDIDATES:
        path = root / (candidate + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("samples") != {} or record.get("status") not in ("FAILED", "PARTIAL"):
                raise RuntimeError("Saved model outputs exist; preserve the frozen protocol and use a new directory")
        names.extend([candidate + ".json", candidate + ".md"])
    history = root / "history" / digest(old)
    if history.exists() or (root / "history").is_symlink():
        raise RuntimeError("Unexpected protocol history target")
    history.mkdir(parents=True)
    for name in names:
        path = root / name
        if path.exists():
            path.rename(history / name)
    review = root / "qualitative-review.md"
    if review.is_file():
        shutil.copyfile(review, history / review.name)
    print(f"[protocol] runtime fix; archived empty failed preflight at {history}", flush=True)


def prepare(args):
    images = {}
    for sid in SAMPLES:
        paths = [args.data_root / f"img_{sid}{ext}" for ext in (".jpg", ".JPG", ".png", ".jpeg")]
        paths = [p for p in paths if p.is_file()]
        if len(paths) != 1:
            raise ValueError(f"Expected one image for sample {sid}: {paths}")
        images[str(sid)] = {"path": str(paths[0].resolve()), "sha256": sha(paths[0])}
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    reference = json.loads(args.reference.read_text())
    if reference["model_id"] != "microsoft/Florence-2-large-ft":
        raise ValueError("Wrong reference model")
    if {int(x) for x in reference["samples"]} != set(SAMPLES):
        raise ValueError("Wrong reference samples")
    obj = {"schema_version": 1, "mode": "APPROXIMATE",
           "classification": "PRACTICAL_APPROXIMATION", "scope": "FIVE_IMAGE_VISUAL_DIAGNOSTIC",
           "sample_ids": list(SAMPLES), "images": images, "candidates": CANDIDATES,
           "reference": {"path": str(args.reference.resolve()), "sha256": sha(args.reference)},
           "runner_sha256": sha(__file__),
           "host_helper_sha256": sha(Path(__file__).with_name("run_lavis_captioner_candidates_host.sh")),
           "packaging_repair_sha256": sha(Path(__file__).with_name("repair_decord_wheel.py")),
           "selection": "visual review only; no accuracy tuning",
           "evidence": [
               {"classification": "PAPER_SPECIFIED", "value": "T2 is a BLIP-2 scene caption (Fig.6)."},
               {"classification": "UNRESOLVED", "ambiguity": "C-001", "value": "BLIP-2 ViT-B does not identify a checkpoint."},
               {"classification": "ASSUMPTION", "value": "Compare pretrained alternatives without fitting any weights; native beam generation; no instruction prompt for BLIP/Q-Former."},
               {"classification": "ASSUMPTION", "value": "LAVIS 1.0.2 official wheel, pinned by SHA256, in a separate environment. Florence reuses qualified main; einops is isolated in a pure-Python target."},
               {"classification": "ASSUMPTION", "value": "Decord Linux wheel metadata retagged to its official filename py3-none; only WHEEL/RECORD change, all runtime payload bytes verified identical."},
               {"classification": "UNRESOLVED", "value": "Legacy Florence-large report lacks image hashes, explicit revision, dtype and measured parameter count; timings are historical generation-only."},
           ]}
    target = args.output_dir / "protocol.json"
    if target.exists() and json.loads(target.read_text()) != obj:
        old = json.loads(target.read_text())
        if getattr(args, "resume_runtime_repair", False):
            scientific = ("schema_version", "mode", "classification", "scope", "sample_ids",
                          "images", "candidates", "reference", "selection")
            if any(old.get(k) != obj.get(k) for k in scientific):
                raise RuntimeError("Frozen scientific protocol changed; runtime repair refused")
            revision = args.output_dir / "runtime-revisions" / digest(obj)
            revision.mkdir(parents=True, exist_ok=True)
            write_json(revision / "runtime.json", {"frozen_protocol_sha256": digest(old),
                       "runtime_protocol": obj, "note": "Explicit loader repair; existing results unchanged."})
            for source, name in ((Path(__file__), "runner.py"),
                                 (Path(__file__).with_name("run_lavis_captioner_candidates_host.sh"), "run_host.sh")):
                shutil.copyfile(source, revision / name)
            return old
        archive_empty_preflight(args.output_dir, old, obj)
    if not target.exists():
        write_json(target, obj)
        shutil.copyfile(__file__, args.output_dir / "runner.py")
        shutil.copyfile(Path(__file__).with_name("run_lavis_captioner_candidates_host.sh"),
                        args.output_dir / "run_host.sh")
        shutil.copyfile(Path(__file__).with_name("repair_decord_wheel.py"),
                        args.output_dir / "repair_decord_wheel.py")
    return obj


def verify_record(record, protocol, candidate, device):
    if record["protocol_sha256"] != digest(protocol) or record["device"] != device:
        raise RuntimeError("Incompatible saved result; refusing reuse or overwrite")
    if record["candidate"] != candidate:
        raise RuntimeError("Wrong candidate")
    for sid, row in record["samples"].items():
        if row["image_sha256"] != protocol["images"][sid]["sha256"]:
            raise RuntimeError(f"Changed image {sid}")
        if not isinstance(row["caption"], str) or not row["caption"].strip():
            raise RuntimeError(f"Invalid caption {sid}")
        if row["result_sha256"] != digest({k: v for k, v in row.items() if k != "result_sha256"}):
            raise RuntimeError(f"Corrupt saved sample {sid}")
        if record.get("status") == "COMPLETE" and CANDIDATES[candidate]["backend"] == "transformers" and "ocr" not in row:
            raise RuntimeError(f"Missing OCR in COMPLETE sample {sid}")


def download(url, target, expected=None):
    target = Path(target)
    receipt = target.with_suffix(target.suffix + ".sha256")
    if target.is_file():
        actual = sha(target)
        if expected and actual != expected:
            raise RuntimeError(f"Checksum mismatch {target}")
        if not expected and (not receipt.is_file() or receipt.read_text().strip() != actual):
            raise RuntimeError(f"Unverified existing checkpoint {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    print(f"[download] {url}", flush=True)
    with urllib.request.urlopen(url, timeout=600) as response, part.open("wb") as f:
        size = int(response.headers.get("Content-Length", 0))
        count = 0
        last = time.monotonic()
        while chunk := response.read(4 * 1024 * 1024):
            f.write(chunk)
            count += len(chunk)
            if time.monotonic() - last > 5:
                print(f"[download] {count // 1048576} / {size // 1048576} MiB", flush=True)
                last = time.monotonic()
        f.flush()
        os.fsync(f.fileno())
    if size and count != size:
        raise RuntimeError(f"Incomplete checkpoint {target}")
    actual = sha(part)
    if expected and actual != expected:
        raise RuntimeError(f"Checksum mismatch {target}")
    part.replace(target)
    atomic(receipt, actual + "\n")
    return target


def audit_visual_backbone(model, cfg, torch):
    """Verify the separately initialized frozen CLIP checkpoint, not random weights."""
    from lavis.models import clip_vit
    from timm.models import hub as timm_hub
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/clip_vit_L.pth"
    if cfg.model.get("vit_model") != "clip_L" or not cfg.model.get("freeze_vit", True):
        raise RuntimeError("Only the official frozen CLIP-L composite checkpoint is supported")
    path = Path(timm_hub.get_cache_dir()) / "clip_vit_L.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Official separately loaded visual checkpoint missing: {path}")
    checksum = sha(path)
    receipt = path.with_suffix(path.suffix + ".sha256")
    if receipt.exists() and receipt.read_text().strip() != checksum:
        raise RuntimeError("Separate visual checkpoint checksum mismatch")
    reference = torch.load(path, map_location="cpu", weights_only=True)
    clip_vit.interpolate_pos_embed(model.visual_encoder, reference)
    state = model.visual_encoder.state_dict()
    unexpected = sorted(set(reference) - set(state))
    # LAVIS 1.0.2 uses layers=22 and Transformer(layers-1): blocks 0..20.
    # Its official CLIP file additionally contains blocks 21 and 22, unused upstream.
    allowed_extra = {key.replace("resblocks.20.", f"resblocks.{index}.")
                     for key in state if key.startswith("transformer.resblocks.20.")
                     for index in (21, 22)}
    if set(state) - set(reference) or (unexpected and set(unexpected) != allowed_extra):
        raise RuntimeError("Separate visual checkpoint has unresolved missing/unexpected keys")
    rounded = 0
    for key, value in state.items():
        actual = value.detach().cpu()
        expected = reference[key].to(dtype=actual.dtype)
        if actual.shape != expected.shape:
            raise RuntimeError(f"Separate visual checkpoint shape mismatch: {key}")
        if torch.equal(actual, expected):
            continue
        # Upstream create_clip_vit_L rounds selected layers to FP16 before our .float().
        if cfg.model.get("vit_precision", "fp16") == "fp16" and torch.equal(actual, reference[key].half().to(actual.dtype)):
            rounded += 1
            continue
        raise RuntimeError(f"Separate visual checkpoint tensor mismatch: {key}")
    if any(p.requires_grad for p in model.visual_encoder.parameters()):
        raise RuntimeError("Separate visual encoder is not frozen")
    atomic(receipt, checksum + "\n")
    return {"checkpoint_url": url, "checkpoint_path": str(path), "checkpoint_sha256": checksum,
            "checkpoint_bytes": path.stat().st_size, "verified_tensor_count": len(state),
            "upstream_fp16_rounded_tensor_count": rounded, "frozen": True,
            "upstream_unused_keys": unexpected,
            "loader_source_sha256": sha(clip_vit.__file__)}


def load_lavis(spec, args, torch):
    if metadata.version("salesforce-lavis") != "1.0.2":
        raise RuntimeError("Use the isolated pinned LAVIS 1.0.2 environment")
    from lavis.common.registry import registry
    from lavis.models import load_preprocess
    from omegaconf import OmegaConf
    cls = registry.get_model_class(spec["model"])
    cfg_path = Path(cls.default_config_path(spec["model_type"]))
    cfg = OmegaConf.load(cfg_path)
    key = "finetuned" if cfg.model.get("load_finetuned", False) else "pretrained"
    if cfg.model[key] != spec["checkpoint_url"]:
        raise RuntimeError("Official installed config differs from declared checkpoint")
    weight = download(spec["checkpoint_url"], args.cache_root / (args.candidate + ".pth"))
    cfg.model[key] = str(weight.resolve())
    if spec["model"] == "blip2" and args.device == "cpu":
        cfg.model.vit_precision = "fp32"
    start = time.perf_counter()
    model = cls.from_config(cfg.model).float().to(args.device).eval()
    processors, _ = load_preprocess(cfg.preprocess)
    load_seconds = time.perf_counter() - start
    raw = torch.load(weight, map_location="cpu")["model"]
    state = model.state_dict()
    missing = sorted(set(state) - set(raw))
    unexpected = sorted(set(raw) - set(state))
    mismatched = [k for k in state.keys() & raw.keys() if state[k].shape != raw[k].shape]
    critical = ("visual_encoder.", "text_decoder.", "Qformer.", "query_tokens", "ln_vision.")
    separate_visual = None
    if spec["model"] == "blip2" and any(k.startswith("visual_encoder.") for k in missing):
        visual_keys = {k for k in state if k.startswith("visual_encoder.")}
        if not visual_keys.issubset(missing):
            raise RuntimeError("Partially missing visual encoder is not an official composite checkpoint")
        separate_visual = audit_visual_backbone(model, cfg, torch)
        critical = tuple(k for k in critical if k != "visual_encoder.")
    unresolved = [k for k in missing if k.startswith(critical)]
    if mismatched or unresolved:
        raise RuntimeError(f"Generation checkpoint incomplete: missing_count={len(unresolved)}, first={unresolved[:5]}; shape={mismatched[:5]}")
    del raw
    evidence = {"checkpoint_path": str(weight), "checkpoint_sha256": sha(weight),
                "checkpoint_bytes": weight.stat().st_size, "config_sha256": sha(cfg_path),
                "model_source_sha256": sha(sys.modules[cls.__module__].__file__),
                "checkpoint_loader": "official LAVIS; upstream non-strict loading audited",
                "missing_keys": missing, "unexpected_keys": unexpected,
                "shape_mismatch_keys": mismatched, "visual_preprocessor": repr(processors["eval"])}
    if separate_visual is not None:
        evidence["separate_frozen_visual_checkpoint"] = separate_visual

    def infer(image, task):
        tensor = processors["eval"](image).unsqueeze(0).to(args.device)
        return model.generate({"image": tensor}, **spec["generation"])[0].strip()
    return model, infer, evidence, load_seconds


def load_florence(spec, args, torch):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoProcessor
    snapshot = Path(snapshot_download(spec["model"], revision=spec["revision"],
                    cache_dir=str(args.cache_root / "huggingface"),
                    ignore_patterns=["pytorch_model.bin", "*.onnx", "*.msgpack", "*.h5"]))
    weight = snapshot / "model.safetensors"
    if sha(weight) != spec["weight_sha256"]:
        raise RuntimeError("Florence weight checksum mismatch")
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(snapshot), trust_remote_code=True)
    model, info = AutoModelForCausalLM.from_pretrained(
        str(snapshot), trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Florence checkpoint compatibility failure: {info}")
    model = model.to(args.device).eval()
    load_seconds = time.perf_counter() - start
    evidence = {"revision": spec["revision"], "snapshot": str(snapshot),
                "checkpoint_sha256": sha(weight), "checkpoint_bytes": weight.stat().st_size,
                "source_hashes": {p.name: sha(p) for p in snapshot.glob("*.py")},
                "loading_info": info, "visual_preprocessor": repr(processor.image_processor)}

    def infer(image, task):
        inputs = processor(text=task, images=image, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        output = model.generate(**inputs, **spec["generation"])
        raw = processor.batch_decode(output, skip_special_tokens=False)[0]
        result = processor.post_process_generation(raw, task=task, image_size=image.size)[task]
        if not isinstance(result, str):
            raise TypeError(f"Unexpected task output: {result}")
        return result.strip()
    return model, infer, evidence, load_seconds


def markdown(record):
    lines = [f"# {record['candidate']}", "", f"Status: {record['status']}",
             "Classification: PRACTICAL_APPROXIMATION; five-image visual diagnostic", "",
             f"Model: {record['spec']['model']} / {record['spec']['model_type']}",
             f"Generation: `{json.dumps(record['spec']['generation'])}`", ""]
    for sid, row in record["samples"].items():
        lines += [f"## Sample {sid}", "", row["caption"], "",
                  f"Words {row['word_count']}; repetition {row['repeated_trigram_fraction']:.3f}; latency {row['latency_seconds']:.3f}s", ""]
        if "ocr" in row:
            lines += [f"OCR (separate task, never appended to T2): {row['ocr']}", ""]
    lines += ["Qualitative review: inspect caption against frozen RGB; repetition metrics do not detect hallucinations.", ""]
    if "error" in record:
        lines += [record["error"]["type"] + ": " + record["error"]["message"], ""]
    return "\n".join(lines)


def persist(args, record):
    jp = args.output_dir / (args.candidate + ".json")
    write_json(jp, record)
    atomic(jp.with_suffix(".md"), markdown(record))
    print(f"JSON {jp} SHA256 {sha(jp)}", flush=True)


def run(args, protocol):
    spec = CANDIDATES[args.candidate]
    jp = args.output_dir / (args.candidate + ".json")
    record = json.loads(jp.read_text()) if jp.exists() else {
        "candidate": args.candidate, "spec": spec, "samples": {},
        "protocol_sha256": digest(protocol), "device": args.device,
        "mode": "APPROXIMATE", "classification": "PRACTICAL_APPROXIMATION"}
    # An empty failed preflight contains no model outputs; a host device change
    # is allowed only in this case. Partial successful results remain frozen.
    if record.get("status") == "FAILED" and not record["samples"] and "checkpoint" not in record:
        record["device"] = args.device
    verify_record(record, protocol, args.candidate, args.device)
    if set(record["samples"]) == {str(x) for x in SAMPLES} and record["status"] == "COMPLETE":
        print(f"[reuse] {args.candidate} 5/5 SHA256 {sha(jp)}", flush=True)
        return 0
    try:
        import torch
        from PIL import Image
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        if args.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("Requested PAPER GPU unavailable; no automatic device substitution")
        record["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
            "executable": sys.executable, "torch": torch.__version__, "threads": args.threads,
            "seed": 0, "dtype": "float32", "batch_size": 1,
            "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()}}
        record["execution_runner_sha256"] = sha(__file__)
        # FP32 keeps the two legacy LAVIS encoders/decoders on one compatible dtype.
        record["runtime_choice"] = {"classification": "HARDWARE_DEVIATION", "value": "FP32; sequential models; CPU or explicitly requested PAPER GPU"}
        loader = load_lavis if spec["backend"] == "lavis" else load_florence
        model, infer, evidence, seconds = loader(spec, args, torch)
        record.update(checkpoint=evidence, load_seconds=seconds,
                      parameters=sum(p.numel() for p in model.parameters()), status="PARTIAL")
        record.pop("error", None)
        persist(args, record)
        for index, sid in enumerate(SAMPLES, 1):
            key = str(sid)
            needs_ocr = spec["backend"] == "transformers"
            if key in record["samples"] and (not needs_ocr or "ocr" in record["samples"][key]):
                print(f"[reuse] {args.candidate} sample {sid}", flush=True)
                continue
            image = Image.open(protocol["images"][key]["path"]).convert("RGB")
            if args.device != "cpu":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            if key in record["samples"]:
                row = dict(record["samples"][key])
                row.pop("result_sha256")
                caption = row["caption"]
            else:
                start = time.perf_counter()
                with torch.inference_mode():
                    caption = infer(image, "<MORE_DETAILED_CAPTION>")
                if args.device != "cpu":
                    torch.cuda.synchronize()
                latency = time.perf_counter() - start
                if not caption:
                    raise ValueError(f"Empty caption sample {sid}")
                row = {"caption": caption, "image_sha256": protocol["images"][key]["sha256"],
                       "latency_seconds": latency, "latency_scope": "preprocess + generation + decode; excludes load",
                       "qualitative_status": "PENDING_VISUAL_REVIEW", **text_metrics(caption)}
                row["result_sha256"] = digest(row)
                record["samples"][key] = dict(row)
                persist(args, record)
                row.pop("result_sha256")
            if needs_ocr and "ocr" not in row:
                start = time.perf_counter()
                with torch.inference_mode():
                    row["ocr"] = infer(image, "<OCR>")
                if args.device != "cpu":
                    torch.cuda.synchronize()
                row["ocr_latency_seconds"] = time.perf_counter() - start
            if args.device != "cpu":
                row["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                row["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            row["result_sha256"] = digest(row)
            record["samples"][key] = row
            persist(args, record)
            print(f"[{args.candidate}] {index}/5 sample {sid}: {caption}", flush=True)
        record["status"] = "COMPLETE"
        persist(args, record)
        return 0
    except Exception as exc:
        record["status"] = "FAILED"
        record["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
        persist(args, record)
        traceback.print_exc()
        return 1


def compare(args, protocol):
    rows = {}
    for name in CANDIDATES:
        path = args.output_dir / (name + ".json")
        if not path.is_file():
            rows[name] = {"status": "NOT_RUN", "reason": "No model results available"}
            continue
        record = json.loads(path.read_text())
        verify_record(record, protocol, name, record["device"])
        if record["status"] == "COMPLETE" and len(record["samples"]) != 5:
            raise RuntimeError("False COMPLETE result")
        rows[name] = {"status": record["status"], "device": record["device"],
                      "parameters": record.get("parameters"), "checkpoint": record.get("checkpoint"),
                      "samples": record["samples"], "json_sha256": sha(path)}
        if record["samples"]:
            rows[name]["mean_latency_seconds"] = statistics.mean(r["latency_seconds"] for r in record["samples"].values())
        if "error" in record:
            rows[name]["error"] = record["error"]["message"]
    ref = json.loads(Path(protocol["reference"]["path"]).read_text())
    rows["florence2-large-ft"] = {"status": "HISTORICAL_REUSED", "json_sha256": protocol["reference"]["sha256"],
        "parameters_measured": None, "parameters_model_card": 770_000_000,
        "mean_latency_seconds": statistics.mean(r["caption_latency_seconds"] for r in ref["samples"].values()),
        "latency_comparability": "GPU generation-only; new runs include CPU/GPU preprocessing; do not compare speed directly",
        "samples": ref["samples"], "provenance_limitations": "Legacy file lacks pinned revision and image hashes"}
    complete = all(rows[n]["status"] == "COMPLETE" for n in CANDIDATES)
    obj = {"protocol_sha256": digest(protocol), "classification": "PRACTICAL_APPROXIMATION",
           "status": "AWAITING_VISUAL_REVIEW" if complete else "INCOMPLETE_COMPARISON",
           "candidates": rows, "full_train": "HOLD", "recommendation": None,
           "note": "No winner until all three new candidates have results and RGB visual review; OCR task is separate from T2."}
    path = args.output_dir / "comparison.json"
    write_json(path, obj)
    lines = ["# Captioner comparison", "", "| Candidate | Status | Parameters | Mean seconds |",
             "|---|---|---:|---:|"]
    for name, row in rows.items():
        lines.append(f"| {name} | {row['status']} | {row.get('parameters', 'unmeasured')} | {row.get('mean_latency_seconds', '—')} |")
    lines += ["", obj["note"], "", "Full TRAIN: HOLD.", ""]
    atomic(path.with_suffix(".md"), "\n".join(lines))
    print(f"JSON {path} SHA256 {sha(path)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", choices=tuple(CANDIDATES))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--resume-runtime-repair", action="store_true",
                    help="Preserve frozen protocol/results; archive an explicit runtime-only repair")
    ap.add_argument("--data-root", required=True, type=Path, help="Frozen five-sample image directory")
    ap.add_argument("--reference", type=Path, default=Path("artifacts/diagnostics/captioner-candidates-20260914/florence2-large-ft.json"))
    ap.add_argument("--output-dir", type=Path, default=ROOT)
    ap.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    ap.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    protocol = prepare(args)
    if args.compare:
        compare(args, protocol)
        return 0
    if args.prepare:
        print(f"PROTOCOL {args.output_dir / 'protocol.json'} SHA256 {sha(args.output_dir / 'protocol.json')}")
        return 0
    if not args.candidate:
        ap.error("Choose --candidate, --prepare or --compare")
    return run(args, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
