"""PAPER-only CUDA export; resumable persistent probabilities, exact frozen checks."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from diagnostics.group_coherence_probability_train import load_context, cached_native, read_probability
from diagnostics.gt_mask_localization_oracle import save_json, sha, verify_file
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter, FrozenSegmentationConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "oracle", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", choices=["cuda:0"], required=True)
    a = p.parse_args()
    config, oracle, source, manifest, source_path, data_root = load_context(a.config, a.oracle)
    checkpoint_sha = oracle["protocol"]["checkpoint_sha256"]
    _, first = cached_native(manifest.samples[0], source, source_path, checkpoint_sha)
    checkpoint = Path(first["segmentation_metadata"]["checkpoint"]["path"])
    verify_file(checkpoint, checkpoint_sha)
    contract = {"scope": config["scope"], "checkpoint_sha256": checkpoint_sha,
        "oracle_sha256": sha(a.oracle), "evaluation_config_sha256": config["evaluation_config_sha256"],
        "source_report_sha256": sha(source_path), "device": a.device,
        "batch_size": 1, "precision": "float32", "seed": 0,
        "cudnn_benchmark": False, "deterministic_algorithms": False,
        "allow_tf32_matmul": False, "allow_tf32_cudnn": True,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "export_code_sha256": sha(__file__), "adapter_sha256": sha("pipeline/frozen_segmentation.py")}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = a.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != contract: raise RuntimeError("existing probability export contract differs")
    else: save_json(protocol_path, contract)
    pending = []
    for sample in manifest.samples:
        binary, meta = cached_native(sample, source, source_path, checkpoint_sha)
        if (a.output_dir / f"{sample.sample_id}.json").exists():
            read_probability(a.output_dir, sample.sample_id, binary, meta)
        else: pending.append(sample)
    print(f"[probability export] reusable={229-len(pending)}/229 missing={len(pending)}", flush=True)
    if not pending:
        save_json(a.output_dir / "manifest.json", {"protocol": contract,
            "samples": [{"sample_id": s.sample_id,
                         "metadata_sha256": sha(a.output_dir / f"{s.sample_id}.json")}
                        for s in manifest.samples]})
        print("Verified complete probability cache; no GPU inference required", flush=True)
        return
    if not torch.cuda.is_available(): raise RuntimeError("Run this export on PAPER cuda:0; no CPU inference fallback")
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(False)
    torch.cuda.init(); torch.cuda.reset_peak_memory_stats(0)
    model = None
    try:
        model = FrozenAuthorProtocolSegmenter(FrozenSegmentationConfig(Path(config["evaluation_config"]), checkpoint, a.device))
        for index, sample in enumerate(pending, 1):
            binary, meta = cached_native(sample, source, source_path, checkpoint_sha)
            path = data_root / sample.image_path
            verify_file(path, meta["image_sha256"])
            with Image.open(path) as opened: rgb = np.asarray(opened.convert("RGB"))
            start = time.perf_counter()
            result = model(rgb)
            probability = result["probability_native"]
            expected = meta["segmentation_metadata"]["native_probability_sha256"]
            if hashlib.sha256(probability.tobytes()).hexdigest() != expected or not np.array_equal(result["mask_native"], binary):
                raise RuntimeError(f"Historical native probability/mask mismatch sample={sample.sample_id}; refusing to mix inference artifacts")
            target = a.output_dir / f"{sample.sample_id}.npy"
            temporary = a.output_dir / f"{sample.sample_id}.partial.npy"
            np.save(temporary, probability, allow_pickle=False); temporary.replace(target)
            entry = {"sample_id": sample.sample_id, "checkpoint_sha256": checkpoint_sha,
                "image_sha256": meta["image_sha256"], "file_sha256": sha(target),
                "array_sha256": expected, "seconds": time.perf_counter()-start,
                "native_shape": list(probability.shape), "input_shape": result["metadata"]["input_shape"],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
            save_json(a.output_dir / f"{sample.sample_id}.json", entry)
            print(f"[probability export CUDA] {index}/{len(pending)} sample={sample.sample_id} verified", flush=True)
    except Exception as error:
        save_json(a.output_dir / "last-error.json", {"type": type(error).__name__, "message": str(error),
            "oom": isinstance(error, torch.cuda.OutOfMemoryError),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)})
        raise
    finally:
        if model is not None: model.release()
        gc.collect(); torch.cuda.empty_cache()
    entries = []
    for sample in manifest.samples:
        binary, meta = cached_native(sample, source, source_path, checkpoint_sha)
        read_probability(a.output_dir, sample.sample_id, binary, meta)
        entries.append({"sample_id": sample.sample_id, "metadata_sha256": sha(a.output_dir / f"{sample.sample_id}.json")})
    save_json(a.output_dir / "manifest.json", {"protocol": contract, "gpu": torch.cuda.get_device_name(0), "samples": entries})
    print("MANIFEST", a.output_dir / "manifest.json", "SHA256", sha(a.output_dir / "manifest.json"), flush=True)


if __name__ == "__main__": main()
