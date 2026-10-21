"""Qualified fixed-PARSeq budget ablation for the TRAIN-selected Guarded V3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import torch

from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer)
from scripts.analyze_native128_failures import augment, box_matrices
from scripts.evaluate_cached_totaltext_padding import crop_hash
from scripts.evaluate_native128_crop_limit import reconstruct, select
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.training import seed_everything

REPO = Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(path)


def utc(): return datetime.now(timezone.utc).isoformat()


def config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO/"configs/approximate" not in path.parents:
        raise ValueError("Explicit isolated APPROXIMATE configuration required")
    c = read(path)
    if c["reproduction_mode"] != mode or c["experiment_kind"] != "mask_crop_three_tests_v1":
        raise ValueError("Unsupported experiment configuration")
    for key in ("checkpoint", "parseq_checkpoint", "parseq_archive", "train_manifest", "test_manifest", "training_contract"):
        if file_sha256(Path(c["paths"][key])) != c["sha256"][key]:
            raise ValueError("Input SHA256 mismatch: "+key)
    if c["sha256"]["parseq_checkpoint"] != PARSEQ_CHECKPOINT_SHA256 or c["sha256"]["parseq_archive"] != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise ValueError("Qualified PARSeq artifacts required")
    selected = read(Path(c["paths"]["training_contract"]).parent/"checkpoint_selection.json")["best_bbox_word_f1"]
    if selected["epoch"] != 447 or selected["checkpoint_sha256"] != c["sha256"]["checkpoint"]:
        raise ValueError("Frozen TRAIN-selected V3/447 required")
    return c


def setup(c):
    rt = c["runtime"]
    if rt["device"] != "cuda:0" or rt["precision"] != "float32" or not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA FP32 runtime required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != rt["cublas_workspace_config"]:
        raise RuntimeError("Set the declared CUBLAS_WORKSPACE_CONFIG before launch")
    torch.set_num_threads(rt["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    seed_everything(c["seed"], deterministic_algorithms=True)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < rt["minimum_free_gpu_bytes"]:
        raise RuntimeError("Insufficient free VRAM; existing jobs are preserved")
    torch.cuda.reset_peak_memory_stats()
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "numpy": np.__version__, "python": platform.python_version(),
            "other_gpu_jobs": subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                check=True, capture_output=True, text=True).stdout,
            **rt, "deterministic_algorithms": True, "tf32": False}


def measure(proposals, words, ocr):
    iou, _, _ = box_matrices(proposals, words)
    texts = np.asarray([[ocr[p["crop_array_sha256"]]["raw_text"].casefold() == w["text"].casefold()
                        for w in words] for p in proposals], bool).reshape(len(proposals), len(words))
    exact = augment((iou >= .5) & texts)
    geometry = augment(iou >= .5, exact)
    return {"gt_words": len(words), "proposals": len(proposals), "correct": len(exact),
            "localized": len(geometry), "correct_gt_indices": sorted(exact.values()),
            "localized_gt_indices": sorted(geometry.values())}


def aggregate(rows):
    out = {k: sum(r[k] for r in rows) for k in ("gt_words", "proposals", "correct", "localized")}
    n, g, tp = out["proposals"], out["gt_words"], out["correct"]
    out.update(precision=tp/n if n else 0., recall=tp/g if g else 0., f1=2*tp/(n+g) if n+g else 0.)
    return out


class CropOCR:
    """Hash-bound exact RGB cache; misses use the unchanged qualified runtime."""
    def __init__(self, c, output, source_cache=None):
        self.c, self.output, self.model = c, Path(output), None
        self.rows, self.reused, self.inferred = {}, 0, 0
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.touch(exist_ok=False)
        if source_cache is not None:
            source = Path(source_cache)
            protocol = read(source/"protocol.json")
            adapters = [h for k,h in protocol["code_sha256"].items() if k.endswith("/recognition/parseq.py")]
            if adapters != [file_sha256(REPO/"recognition/parseq.py")]:
                raise ValueError("Cached recognizer implementation differs")
            if protocol["parseq_checkpoint_sha256"] != PARSEQ_CHECKPOINT_SHA256 or not protocol["parseq_decode_ar"] or protocol["parseq_refine_iters"] != 1:
                raise ValueError("Cached recognition model/decoding differs")
            protocol_sha = file_sha256(source/"protocol.json")
            for line in (source/"ocr.jsonl").open():
                row = json.loads(line)
                if row["protocol_sha256"] != protocol_sha: raise ValueError("OCR cache protocol differs")
                key = row["crop_array_sha256"]
                if key in self.rows and self.rows[key] != row: raise ValueError("Conflicting source OCR")
                self.rows[key] = row

    def recognizer(self):
        if self.model is None:
            p = self.c["paths"]
            self.model = ParseqApproxRecognizer(ParseqApproxConfig(
                runtime_source=Path(p["parseq_source"]), checkpoint=Path(p["parseq_checkpoint"]),
                device=self.c["runtime"]["device"], runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                runtime_source_archive=Path(p["parseq_archive"])))
        return self.model

    def image(self, rgb, proposals):
        pending = {}
        for p in proposals:
            x0,y0,x1,y1 = p["padded_bbox_original"]
            crop = np.ascontiguousarray(rgb[y0:y1,x0:x1])
            key = crop_hash(crop)
            if "crop_array_sha256" in p and p["crop_array_sha256"] != key:
                raise ValueError("Crop pixels differ from the frozen source")
            p["crop_array_sha256"] = key
            if key in self.rows: self.reused += 1
            else: pending.setdefault(key, crop)
        keys = list(pending)
        size = self.c["runtime"]["ocr_batch_size"]
        for start in range(0, len(keys), size):
            batch = keys[start:start+size]
            values = self.recognizer().recognize_candidate_batch([pending[k] for k in batch])
            if len(values) != len(batch): raise RuntimeError("OCR output cardinality differs")
            with self.output.open("a") as stream:
                for key, value in zip(batch, values):
                    row = {**value.to_dict(), "crop_array_sha256": key, "source": "new qualified PARSeq inference"}
                    self.rows[key] = row; stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                    self.inferred += 1

    def release(self):
        if self.model is not None: self.model.release(); self.model = None
        gc.collect(); torch.cuda.empty_cache()


def run(args):
    c = config(args.config, args.mode)
    if args.limit_images is not None and not 1 <= args.limit_images <= 300:
        raise ValueError("Smoke image limit must be between1 and300")
    output = args.output.resolve()
    if output.exists(): raise FileExistsError("Choose a fresh output directory")
    output.mkdir(parents=True)
    hardware = setup(c); started = time.monotonic()
    source = Path(c["paths"]["source_ocr"])
    protocol = read(source/"protocol.json"); source_sha = file_sha256(source/"protocol.json")
    source_paths = [source/n for n in ("protocol.json", "results.json", "audit.json", "ground_truth.json", "ocr.jsonl")]
    if read(source/"audit.json")["status"] != "PASS": raise ValueError("Source audit did not pass")
    resolved = {"config": c, "config_sha256": file_sha256(args.config), "runtime": hardware,
                "mode": "approximate", "result_class": "smoke-test result" if args.limit_images else "approximate reproduction; exploratory TEST300",
                "source_sha256": {str(p): file_sha256(p) for p in source_paths},
                "code_sha256": {str(p):file_sha256(p) for p in (Path(__file__), REPO/"recognition/parseq.py")},
                "metric": "maximum one-to-one unpadded bbox IoU>=.5 + exact Unicode casefold; punctuation retained; all crops/care GT count"}
    write(output/"resolved_config.json", resolved)
    ids = protocol["sample_ids"][:args.limit_images] if args.limit_images else protocol["sample_ids"]
    manifest = load_segmentation_manifest(Path(c["paths"]["test_manifest"]), require_checksums=True)
    samples = {s.sample_id:s for s in manifest.samples}
    ground_truth = read(source/"ground_truth.json")["ground_truth"]
    ocr = CropOCR(c, output/"ocr_new.jsonl", source)
    rows = {str(cap):[] for cap in c["budgets"]}; shards = {}; source_shas = {}
    try:
        for index, sid in enumerate(ids, 1):
            path = source/"epoch-0447/proposals"/(sid.replace(":", "__")+".json")
            cached = read(path); source_shas[str(path)] = file_sha256(path)
            if cached["sample_id"] != sid or cached["protocol_sha256"] != source_sha:
                raise ValueError("Source shard identity differs")
            image_path = Path(c["paths"]["test_data_root"])/samples[sid].image_path
            if file_sha256(image_path) != cached["image_sha256"] or cached["image_sha256"] != samples[sid].image_sha256:
                raise ValueError("RGB checksum differs")
            with Image.open(image_path) as im: rgb = np.ascontiguousarray(im.convert("RGB"))
            proposals = select(reconstruct(cached), cached["image_hw"], area=0, padding=1, limit=None)
            fields = ("contour_index", "bbox_native", "bbox", "padded_bbox_original")
            if [[p[k] for k in fields] for p in proposals[:10]] != [[p[k] for k in fields] for p in cached["variants"]["1"]["proposals"]]:
                raise ValueError("Top10 baseline geometry does not replay")
            ocr.image(rgb, proposals)
            for cap in c["budgets"]:
                selected = proposals[:cap]
                rows[str(cap)].append({"sample_id": sid, **measure(selected, ground_truth[sid], ocr.rows)})
            shards[sid] = proposals
            if index % 25 == 0 or index == len(ids):
                print(f"[budget] {index}/{len(ids)} new_OCR={ocr.inferred}", flush=True)
        results = {key: aggregate(values) for key,values in rows.items()}
        if not args.limit_images and (len(ids) != 300 or results["10"]["gt_words"] != 2204 or results["10"]["correct"] != 795 or results["10"]["proposals"] != 2086):
            raise AssertionError("Full V3 baseline failed to replay")
        for cap, values in rows.items():
            gains = losses = 0
            for base,row in zip(rows["10"], values):
                a,b = set(base["correct_gt_indices"]),set(row["correct_gt_indices"])
                gains += len(b-a); losses += len(a-b)
            results[cap].update(gained=gains, lost=losses)
        used = {p["crop_array_sha256"] for pp in shards.values() for p in pp}
        write(output/"ocr_used.json", {k:ocr.rows[k] for k in sorted(used)})
        write(output/"proposals.json", shards)
        write(output/"per_image.json", rows)
        write(output/"source_shards.json", source_shas)
        report = {"status":"SUCCESS", "claim_label":"approximate reproduction", "results":results,
                  "new_ocr":ocr.inferred,"elapsed_seconds":time.monotonic()-started,
                  "memory":{"models":["qualified PARSeq AR+refine1"],"precision":"float32", "batch_size":c["runtime"]["ocr_batch_size"],
                            "input_shape":[c["runtime"]["ocr_batch_size"],3,32,128],"device":c["runtime"]["device"],
                            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"oom":False}}
        write(output/"results.json", report); print(json.dumps(report),flush=True)
    finally: ocr.release()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",required=True,choices=["approximate"]);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--limit-images",type=int)
    args=p.parse_args()
    output_existed=args.output.exists()
    try: run(args)
    except Exception as error:
        if not output_existed:
            write(args.output/"failure.json",{"status":"FAILED","error":str(error),"traceback":traceback.format_exc(),
                "oom":isinstance(error,torch.cuda.OutOfMemoryError),"utc":utc()})
        raise
