"""Matched detached/connected auxiliary CTC supervision of the Text+Kernel U-Net."""

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset

from localize.word_multitask import word_instances, parseq_word_proposals
from models.unet_auxiliary_ocr import AuxiliaryCTCHead
from models.unet_word_research import WordResearchAGUNet
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer)
from scripts.diagnose_adaspot_phase0 import summarize, paired_changes
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_adaspot_gap_decoder import state_digest
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.totaltext_word_kernel import WordKernelDataset, instance_geometry
from segmentation.training import seed_everything, runtime_environment
from segmentation.unet_auxiliary_ocr import (vocabulary, word_supervision,
    auxiliary_loss, segmentation_objective, selection_key)

REPO = Path(__file__).resolve().parents[1]
ARMS = ("detached", "connected")


def json_write(path, value, *, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Immutable artifact changed: " + str(path))
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    tmp.replace(path)


def atomic_save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    torch.save(value, tmp); tmp.replace(path)


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO/"configs/approximate" not in path.parents:
        raise ValueError("Explicit approximate configuration namespace required")
    c = json.loads(path.read_text())
    if (c.get("schema_version") != 1 or c.get("reproduction_mode") != mode
            or c.get("claim_label") != "approximate reproduction"
            or c.get("experiment_kind") != "shared_unet_auxiliary_ctc_v1"):
        raise ValueError("Invalid scientific configuration")
    positive = ("epochs", "warmup_epochs", "batch_size", "cpu_threads", "reader_hidden",
        "maximum_label_length", "maximum_words_per_image", "fit_images", "calibration_images",
        "evaluation_images", "evaluation_words", "ocr_batch_size")
    if any(type(c[k]) is not int or c[k] < 1 for k in positive):
        raise ValueError("Positive integer experiment sizes required")
    if (type(c["num_workers"]) is not int or c["num_workers"] < 0
            or c["candidate_epochs"] != sorted(set([0, *c["candidate_epochs"], c["epochs"]]))
            or any(type(v) is not int or not 0 <= v <= c["epochs"] for v in c["candidate_epochs"])
            or c["augmentation"] != "disabled_in_both_arms"):
        raise ValueError("Invalid candidate epochs or unsupported augmentation")
    if (len(c["feature_crop_hw"]) != 2 or any(type(v) is not int or v < 1 for v in c["feature_crop_hw"])
            or any(not np.isfinite(c[k]) or c[k] <= 0 for k in
                   ("decoder_learning_rate", "reader_learning_rate", "weight_decay", "ocr_weight", "gradient_clip_norm"))):
        raise ValueError("Invalid crop or optimizer/loss settings")
    if not c["scientific_choices"] or any(x.get("classification") not in
            ("PAPER_SPECIFIED", "REPOSITORY_SPECIFIED", "ASSUMPTION", "HARDWARE_DEVIATION", "UNRESOLVED")
            or not x.get("rationale") or not x.get("expected_effect") for x in c["scientific_choices"]):
        raise ValueError("Classified scientific choices required")
    loss = c["loss"]
    if (not np.isfinite(list(loss.values())).all()
            or min(loss["lambda_text"], loss["lambda_kernel"], loss["smooth"]) <= 0
            or min(loss["bce_weight"], loss["dice_weight"]) < 0
            or loss["bce_weight"] + loss["dice_weight"] <= 0
            or not 1 <= loss["separation_weight"] <= 20):
        raise ValueError("Invalid segmentation loss weights")
    return c


def split_population(manifest, source_config, config):
    train, evaluation, sha = _split_indices(manifest, source_config["seed"], source_config["validation_count"])
    if sha != source_config["validation_ids_sha256"]:
        raise ValueError("Original source evaluation split changed")
    order = np.random.default_rng(config["split_seed"]).permutation(len(train))
    calib_set = {train[i] for i in order[:config["calibration_images"]]}
    result = {"fit": [i for i in train if i not in calib_set],
              "calibration": [i for i in train if i in calib_set], "evaluation": evaluation}
    for key in result:
        if len(result[key]) != config[key+"_images"]:
            raise ValueError("Unexpected population: " + key)
    if any(set(result[a]) & set(result[b]) for a, b in
           (("fit", "calibration"), ("fit", "evaluation"), ("calibration", "evaluation"))):
        raise ValueError("Fit/calibration/evaluation overlap")
    return result


def memory(device, model_set, shape):
    return {"device": str(device), "gpu": torch.cuda.get_device_name(device),
        "dtype": "float32", "model_set": model_set, "input_shape": list(shape),
        "allocated_bytes": torch.cuda.max_memory_allocated(device),
        "reserved_bytes": torch.cuda.max_memory_reserved(device), "oom": False}


class AuxiliaryDataset(WordKernelDataset):
    def __init__(self, base, *, distance_fraction, transcripts):
        super().__init__(base, distance_fraction=distance_fraction)
        self.transcripts = transcripts

    def __getitem__(self, index):
        result = super().__getitem__(index)
        result["ocr_texts"] = self.transcripts.get(result["sample_id"], [])
        return result


def collate(rows):
    from torch.utils.data import default_collate
    texts = [row["ocr_texts"] for row in rows]
    result = default_collate([{k: v for k, v in row.items() if k != "ocr_texts"} for row in rows])
    result["ocr_texts"] = texts
    return result


def loader(dataset, indices, c, *, epoch=None):
    indices = list(indices)
    if epoch is not None:
        generator = torch.Generator().manual_seed(c["seed"] + epoch * 1000003)
        indices = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
    return DataLoader(Subset(dataset, indices), batch_size=c["batch_size"], shuffle=False,
        num_workers=c["num_workers"], collate_fn=collate,
        timeout=60 if c["num_workers"] else 0,
        generator=torch.Generator().manual_seed(c["seed"]))


def build_model(source_state, source_config, device):
    model = WordResearchAGUNet("control", skip_floor=source_config["skip_floor"],
                              embedding_dimensions=4, encoder_pretrained=False)
    model.load_state_dict(source_state, strict=True)
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("Qualified encoder must remain frozen")
    return model.to(device)


def make_reader(c, characters, device):
    return AuxiliaryCTCHead(len(characters)+1, hidden=c["reader_hidden"]).to(device)


def epoch_train(model, reader, optimizer, dataset, indices, characters, c, device,
                *, epoch, arm, warmup=False):
    model.eval() if warmup else model.train()
    if model.encoder.training:
        raise RuntimeError("Frozen encoder BatchNorm entered training mode")
    reader.train(); totals = Counter(); seen = []; steps = 0; first = None
    batches = loader(dataset, indices, c, epoch=epoch)
    started = time.monotonic()
    for bi, batch in enumerate(batches):
        seen.extend(batch["sample_id"])
        supervision = []
        for i, texts in enumerate(batch["ocr_texts"]):
            rows, counts = word_supervision(batch["owners_native"][i].numpy(),
                batch["validity"][i, 0].numpy(), texts, characters,
                max_length=c["maximum_label_length"], time_steps=c["feature_crop_hw"][1],
                max_words=c["maximum_words_per_image"], epoch=epoch)
            supervision.append(rows); totals.update(counts)
        if not batch["validity"].any() or warmup and not any(supervision):
            totals["skipped_unsupervised_batches"] += 1
            continue
        local = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        if warmup:
            with torch.no_grad(): evidence = model.forward_evidence(local["image"])
            segmentation = evidence["words"].sum()*0
        else:
            evidence = model.forward_evidence(local["image"])
            segmentation = segmentation_objective(evidence["words"], local, c["loss"])
        ocr = auxiliary_loss(evidence["features"], reader, supervision,
            output_hw=c["feature_crop_hw"], detach=warmup or arm == "detached", stats=totals)
        loss = ocr if warmup else segmentation + c["ocr_weight"]*ocr
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training objective")
        probe = None
        if first is None and not warmup and any(supervision):
            feature_gradient = torch.autograd.grad(ocr, evidence["features"], retain_graph=True,
                                                    allow_unused=True)[0]
            norm = float(feature_gradient.abs().sum()) if feature_gradient is not None else 0.
            if (arm == "detached" and norm != 0) or (arm == "connected" and norm <= 0):
                raise RuntimeError("Auxiliary feature gradient violates paired intervention")
            probe = {"ocr_to_features_gradient_l1": norm}
        loss.backward()
        if any(p.grad is not None for p in model.encoder.parameters()):
            raise RuntimeError("Frozen encoder received a gradient")
        for module in (model, reader):
            params = [p for p in module.parameters() if p.requires_grad and p.grad is not None]
            if params:
                torch.nn.utils.clip_grad_norm_(params, c["gradient_clip_norm"], error_if_nonfinite=True)
        reader_grad = sum(float(p.grad.abs().sum()) for p in reader.parameters() if p.grad is not None)
        optimizer.step(); steps += 1
        if probe is not None:
            first = {**probe, "reader_gradient_l1": reader_grad, "status": "PASS",
                "device": str(device), "batch_sample_ids": batch["sample_id"],
                "supervised_words": sum(map(len, supervision))}
        totals.update(loss=float(loss.detach()), segmentation_loss=float(segmentation.detach()), ocr_loss=float(ocr.detach()))
        if bi == 0 or (bi+1) % 50 == 0:
            print(f"[{arm}{' warmup' if warmup else ''}] epoch={epoch+1} batch={bi+1}/{len(batches)} loss={float(loss.detach()):.5f}", flush=True)
    expected = {dataset.base.manifest.samples[i].sample_id for i in indices}
    if len(seen) != len(indices) or len(set(seen)) != len(seen) or set(seen) != expected or not steps:
        raise RuntimeError("Incomplete/unsupervised epoch population")
    record = {"epoch": epoch+1, "samples": len(seen), "sample_ids": seen,
        "sample_order_sha256": hashlib.sha256("\n".join(seen).encode()).hexdigest(),
        "steps": steps, "seconds": time.monotonic()-started,
        **{k: v/steps if k in ("loss", "segmentation_loss", "ocr_loss") else v for k, v in totals.items()},
        "first_step": first, "memory": memory(device, "Text+Kernel U-Net + training-only CTC head/AdamW", [c["batch_size"],3,512,512])}
    return record


def freeze_predictions(model, dataset, indices, sc, args, c, device, directory):
    receipt = directory/"frozen.json"
    if receipt.exists():
        result = json.loads(receipt.read_text())
        for row in result["rows"]:
            if sha256_file(Path(row["maps_path"])) != row["maps_sha256"]:
                raise ValueError("Frozen prediction cache changed")
        return result
    model.eval(); rows = []; native = Counter(); pixels = Counter()
    # One image per forward is identical for all candidates/reference.
    with torch.inference_mode():
        for index in indices:
            batch = dataset[index]; sample = dataset.base.manifest.samples[index]
            if sha256_file(Path(batch["image_path"])) != sample.image_sha256:
                raise ValueError("Prediction RGB checksum changed")
            p = model(batch["image"][None].to(device)).sigmoid()[0].cpu().numpy()
            with Image.open(batch["image_path"]) as im: rgb = np.asarray(im.convert("RGB"))
            canvas = letterbox_canvas_mask((rgb.shape[1], rgb.shape[0]), native_size=p.shape[-1])
            labels, audit, _ = word_instances(p, canvas, sc["postprocessing"])
            valid = batch["validity"][0].numpy().astype(bool)
            ignored = [int(i) for i in np.unique(labels[labels > 0]) if not np.any((labels == i) & valid)]
            care = labels.copy(); care[np.isin(care, ignored)] = 0
            # Expected original IDs are preserved even when a word vanishes natively.
            with Image.open(args.word_root/sample.word_owner_annotation_path) as im:
                original_ids = np.unique(np.asarray(im)); expected = original_ids[original_ids > 0].tolist()
            counts = instance_geometry(care, batch["owners_native"].numpy(), expected)
            native.update(counts)
            truth = batch["mask"][0].numpy().astype(bool); foreground = p[0] >= sc["postprocessing"]["text_threshold"]
            pixel = {"tp": int((foreground & truth & valid).sum()), "fp": int((foreground & ~truth & valid).sum()),
                     "fn": int((~foreground & truth & valid).sum())}; pixels.update(pixel)
            proposals = parseq_word_proposals(labels, (rgb.shape[1], rgb.shape[0]), padding=0, min_area=0, max_instances=None)
            for prop in proposals:
                prop["ignored"] = prop["instance_id"] in ignored
                x1,y1,x2,y2 = prop["bbox"]; crop = np.ascontiguousarray(rgb[y1:y2,x1:x2])
                digest = crop_array_hash(crop); path = args.output_root/"crops"/(digest+".png")
                if not path.exists(): path.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(crop).save(path)
                prop.update(crop_rgb_sha256=digest, crop_path=str(path))
            path = directory/"maps"/(sample.sample_id.replace(":","__")+".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, probabilities=p, labels=labels, owners=batch["owners_native"].numpy(),
                validity=valid, text_target=truth, expected_owner_ids=np.array(expected), canvas=canvas)
            rows.append({"sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                "proposals": proposals, "native": counts, "pixel": pixel, "audit": audit,
                "maps_path": str(path), "maps_sha256": sha256_file(path)})
    result = {"rows": rows, "native_counts": dict(native), "pixel_counts": dict(pixels),
              "memory": memory(device, "Text+Kernel U-Net only; CTC head absent", [1,3,512,512])}
    json_write(receipt, result, immutable=True)
    return result


def recognize(frozen, args, c, device):
    jobs = {p["crop_rgb_sha256"]: Path(p["crop_path"]) for item in frozen for row in item["rows"]
            for p in row["proposals"] if not p["ignored"]}
    cache = args.output_root/"parseq_cache"; cache.mkdir(exist_ok=True)
    identity = {"checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256, "archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                "decode_ar": True, "refine_iters": 1, "device": str(device), "batch_size": c["ocr_batch_size"]}
    readings = {}; pending = []
    for digest, path in sorted(jobs.items()):
        with Image.open(path) as im: crop = np.asarray(im.convert("RGB"))
        if crop_array_hash(crop) != digest: raise ValueError("Frozen OCR RGB changed")
        entry = cache/(digest+".json")
        if entry.exists():
            value = json.loads(entry.read_text())
            if value["identity"] != identity or value["crop_rgb_sha256"] != digest:
                raise ValueError("PARSeq cache provenance changed")
            readings[digest] = value["reading"]
        else: pending.append((digest, path))
    measured = None
    if pending:
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
            checkpoint=args.parseq_checkpoint, runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_archive, device=str(device)))
        try:
            torch.cuda.reset_peak_memory_stats(device)
            for start in range(0, len(pending), c["ocr_batch_size"]):
                group = pending[start:start+c["ocr_batch_size"]]; crops = []
                for _, path in group:
                    with Image.open(path) as im: crops.append(np.asarray(im.convert("RGB")))
                predictions = recognizer.recognize_candidate_batch(crops)
                if len(predictions) != len(group): raise RuntimeError("PARSeq omitted crops")
                for (digest, _), prediction in zip(group, predictions):
                    readings[digest] = prediction.to_dict()
                    json_write(cache/(digest+".json"), {"identity": identity,
                        "crop_rgb_sha256": digest, "reading": readings[digest]}, immutable=True)
                if start % (c["ocr_batch_size"]*50) == 0:
                    print(f"[PARSeq] {min(start+c['ocr_batch_size'],len(pending))}/{len(pending)} new crops", flush=True)
            measured = memory(device, "qualified genuine PARSeq AR1 only", [c["ocr_batch_size"],3,32,128])
        finally: recognizer.release()
    return readings, measured


def score_frozen(frozen, readings, manifest, args):
    by_id = {s.sample_id: s for s in manifest.samples}; rows = []
    for row in frozen["rows"]:
        sample = by_id[row["sample_id"]]
        with Image.open(args.word_root/sample.image_path) as im: size = im.size
        gt, sha = _train_gt(sample, args.word_root, size)
        predictions = [{**p, "text": readings[p["crop_rgb_sha256"]]["raw_text"]}
                       for p in row["proposals"] if not p["ignored"]]
        rows.append({"sample_id": row["sample_id"], "annotation_sha256": sha,
                     "ground_truth": gt, "predictions": predictions})
    return summarize(rows)


def preflight(args, c, manifest, dataset, populations):
    transcripts = {}; records = []; total = Counter()
    for i, sample in enumerate(manifest.samples):
        item = dataset[i]
        with Image.open(args.word_root/sample.word_owner_annotation_path) as im:
            original = np.asarray(im); owners = sorted(int(v) for v in np.unique(original[original > 0]))
        if owners != list(range(1,len(owners)+1)):
            raise ValueError("Owner IDs must retain care annotation order")
        native = sorted(int(v) for v in np.unique(item["owners_native"].numpy()) if v > 0)
        if not set(native) <= set(owners): raise ValueError("Native owners invented word IDs")
        row = {"sample_id": sample.sample_id, "source_words":len(owners), "native_words":len(native)}
        if i in populations["fit"]:
            with Image.open(item["image_path"]) as im: size = im.size
            gt, sha = _train_gt(sample, args.word_root, size)
            if len(gt) != len(owners): raise ValueError("GT transcript/owner order count differs")
            transcripts[sample.sample_id] = [g["text"] for g in gt]; row["annotation_sha256"] = sha
        records.append(row); total.update(source_words=len(owners), native_words=len(native))
        if (i+1) % 250 == 0: print(f"[preflight] {i+1}/{len(dataset)}",flush=True)
    validation_words = sum(records[i]["source_words"] for i in populations["evaluation"])
    if args.action != "smoke" and validation_words != c["evaluation_words"]:
        raise ValueError("Evaluation word denominator changed")
    characters = vocabulary(text for values in transcripts.values() for text in values)
    return transcripts, characters, {"status":"PASS", "images":len(dataset), **total,
        "evaluation_words":validation_words, "per_image":records, "characters":characters,
        "fit_transcripts":transcripts, "evaluation_transcripts_read":False}


def rng_state():
    return {"python": random.getstate(), "numpy":np.random.get_state(),
            "torch":torch.get_rng_state(), "cuda":torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"])


def run(args):
    c = read_config(args.config,args.mode); device = torch.device(args.device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise RuntimeError("Explicit available cuda:N device required; no CPU fallback")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    torch.cuda.init(); torch.set_num_threads(c["cpu_threads"]); cv2.setNumThreads(1)
    seed_everything(c["seed"],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    receipt = json.loads((args.source_run/"run_manifest.json").read_text())
    source_contract = json.loads((args.source_run/"run_contract.json").read_text())
    source_checkpoint = args.source_run/"best.pt"
    if (receipt["status"] != "SUCCESS" or not receipt["training_complete"]
            or source_contract["branch"] != "control"
            or sha256_file(source_checkpoint) != c["source_checkpoint_sha256"]
            or receipt["checkpoint"]["sha256"] != c["source_checkpoint_sha256"]):
        raise ValueError("Qualified Text+Kernel source required")
    sc = source_contract["resolved_config"]
    manifest_path = args.word_root/sc["train_manifest"]
    if sha256_file(manifest_path) != sc["word_manifest_sha256"]:
        raise ValueError("Dataset manifest SHA changed")
    manifest = load_segmentation_manifest(manifest_path,require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("Official TRAIN manifest required; TEST forbidden")
    populations = split_population(manifest,sc,c)
    result_class = "smoke-test result" if args.action == "smoke" else "approximate reproduction"
    if args.action == "smoke":
        populations = {"fit":populations["fit"][:8], "calibration":populations["calibration"][:2],
                       "evaluation":populations["evaluation"][:2]}
        c = {**c,"warmup_epochs":1,"epochs":1,"candidate_epochs":[0,1],"num_workers":0}
    if (sha256_file(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256
            or sha256_file(args.parseq_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("Qualified PARSeq artifacts changed")
    out = args.output_root.resolve(); args.output_root = out
    if any(out == p.resolve() or p.resolve() in out.parents for p in (REPO,args.source_run,args.word_root)):
        raise ValueError("New external output root required")
    sources = {str(p.resolve()):sha256_file(p) for package in
        ("models","segmentation","recognition","localize","scripts","diagnostics","scoring","pipeline","caption")
        for p in (REPO/package).rglob("*.py") if "__pycache__" not in p.parts}
    sources.update({str(p.resolve()):sha256_file(p) for p in
        (args.config,source_checkpoint,args.source_run/"run_manifest.json",args.source_run/"run_contract.json",
         manifest_path,args.parseq_checkpoint,args.parseq_archive)})
    protocol = {"mode":"approximate","result_class":result_class,"resolved_config":c,
        "source_contract":source_contract,"source_hashes":sources,"device":str(device),
        "gpu":torch.cuda.get_device_name(device),"dtype":"float32","tf32":False,
        "deterministic_algorithms":True,"input_shape":[c["batch_size"],3,512,512],
        "split_indices":populations,"split_ids":{k:[manifest.samples[i].sample_id for i in v] for k,v in populations.items()},
        "paths":{k:str(getattr(args,k).resolve()) for k in
            ("word_root","source_run","parseq_source","parseq_checkpoint","parseq_archive")},
        "metric":"original RGB half-open bboxIoU>=.5 maximum one-to-one matching AND exact Unicode casefold; punctuation retained; allcare words/misses/empty+duplicate predictions; wholly invalid predicted instances ignored only after geometry",
        "selection":"calibration-only exact count/Hmean/earliest epoch, baseline eligible; freeze before evaluation access",
        "test_access":False,"independent_validation_claim":False}
    out.mkdir(parents=True,exist_ok=True)
    # Prevent concurrent duplicate processes; completed/epoch-boundary resume is explicit.
    import fcntl
    lock=(out/"runner.lock").open("a+")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    json_write(out/"protocol.json",protocol,immutable=True)
    if (out/"results.json").exists():
        result=json.loads((out/"results.json").read_text())
        if result["status"] != "SUCCESS":raise ValueError("Invalid completed receipt")
        return result
    json_write(out/"runtime_environment.json",{**runtime_environment(),"gpu":protocol["gpu"],
        "cuda":torch.version.cuda,"python_executable":os.sys.executable},immutable=True)
    transform=_transform(source_contract["resolved_preprocessing"],"letterbox",train=False)
    base=SegmentationDataset(manifest,args.word_root,transform=transform)
    dataset=AuxiliaryDataset(base,distance_fraction=sc["targets"]["distance_fraction"],transcripts={})
    if (out/"preflight.json").exists():
        pre=json.loads((out/"preflight.json").read_text());transcripts=pre["fit_transcripts"];characters=pre["characters"]
        for row in pre["per_image"]:
            if "annotation_sha256" in row:
                number=row["sample_id"].split(":",1)[1]
                path=args.word_root/"datasets/total_text/train/annotations"/(number+".txt")
                if sha256_file(path) != row["annotation_sha256"]:
                    raise ValueError("Fit transcript annotation changed on resume")
    else:
        transcripts,characters,pre=preflight(args,c,manifest,dataset,populations)
        json_write(out/"preflight.json",pre,immutable=True)
    dataset.transcripts=transcripts
    if args.action == "preflight":return pre
    payload=torch.load(source_checkpoint,map_location="cpu",weights_only=False)
    if payload["run_contract"] != source_contract:raise ValueError("Checkpoint scientific contract differs")
    source_state=payload["model_state_dict"];del payload
    model=build_model(source_state,sc,device);encoder_sha=state_digest(model.encoder)
    reader=make_reader(c,characters,device);inference_parameters=sum(p.numel() for p in model.parameters())
    aux_parameters=sum(p.numel() for p in reader.parameters())
    reader_initial_sha = state_digest(reader)
    torch.cuda.reset_peak_memory_stats(device)
    warm=out/"warmup.pt";history=[]
    optimizer=torch.optim.AdamW(reader.parameters(),lr=c["reader_learning_rate"],weight_decay=c["weight_decay"])
    initial_digest=state_digest(model)
    if warm.exists():
        saved=torch.load(warm,map_location="cpu",weights_only=False)
        reader.load_state_dict(saved["reader"],strict=True);optimizer.load_state_dict(saved["optimizer"])
        history=saved["history"];restore_rng(saved["rng"])
    for epoch in range(len(history),c["warmup_epochs"]):
        record=epoch_train(model,reader,optimizer,dataset,populations["fit"],characters,c,device,
                           epoch=epoch,arm="common",warmup=True)
        history.append(record)
        if state_digest(model) != initial_digest:raise RuntimeError("Warmup mutated source weights or BN")
        atomic_save(warm,{"reader":reader.state_dict(),"optimizer":optimizer.state_dict(),"history":history,"rng":rng_state()})
        json_write(out/"warmup.json",{"status":"SUCCESS" if len(history)==c["warmup_epochs"] else "RUNNING",
            "history":history,"source_state_unchanged":True,"auxiliary_parameters":aux_parameters,
            "inference_parameters":inference_parameters,"warmup_checkpoint_sha256":sha256_file(warm),
            "reader_initial_sha256": reader_initial_sha, "reader_trained_sha256": state_digest(reader)})
    warm_state={k:v.detach().cpu().clone() for k,v in reader.state_dict().items()}
    del reader,optimizer,model;gc.collect();torch.cuda.empty_cache()
    candidates={};training={}
    for arm in ARMS:
        seed_everything(c["seed"],deterministic_algorithms=True)
        model=build_model(source_state,sc,device);reader=make_reader(c,characters,device)
        reader.load_state_dict(warm_state,strict=True)
        optimizer=torch.optim.AdamW([
            {"params":[p for p in model.parameters() if p.requires_grad],"lr":c["decoder_learning_rate"]},
            {"params":reader.parameters(),"lr":c["reader_learning_rate"]}],weight_decay=c["weight_decay"])
        directory=out/arm;directory.mkdir(exist_ok=True);last=directory/"last.pt";history=[]
        if last.exists():
            saved=torch.load(last,map_location="cpu",weights_only=False)
            model.load_state_dict(saved["model"],strict=True);reader.load_state_dict(saved["reader"],strict=True)
            optimizer.load_state_dict(saved["optimizer"]);history=saved["history"];restore_rng(saved["rng"])
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(len(history),c["epochs"]):
            json_write(out/"progress.json",{"status":"RUNNING","stage":"training","arm":arm,"epoch":epoch+1,"total_epochs":c["epochs"]})
            record=epoch_train(model,reader,optimizer,dataset,populations["fit"],characters,c,device,
                               epoch=epoch,arm=arm)
            if state_digest(model.encoder) != encoder_sha:raise RuntimeError("Frozen encoder/BN changed")
            history.append(record)
            if epoch+1 in c["candidate_epochs"]:
                atomic_save(directory/"checkpoints"/f"{epoch+1:03d}.pt",{"model":model.state_dict(),"epoch":epoch+1})
            atomic_save(last,{"model":model.state_dict(),"reader":reader.state_dict(),"optimizer":optimizer.state_dict(),
                "history":history,"rng":rng_state()})
            json_write(directory/"training.json",{"status":"SUCCESS" if len(history)==c["epochs"] else "RUNNING",
                "history":history,"completed_epochs":len(history),"encoder_sha256":encoder_sha,
                "last_checkpoint_sha256":sha256_file(last),"inference_parameters":inference_parameters,
                "auxiliary_parameters":aux_parameters})
            print(f"[{arm}] completed epoch {epoch+1}/{c['epochs']} loss={record['loss']:.5f} OCR={record['ocr_loss']:.5f} seconds={record['seconds']:.1f}",flush=True)
        training[arm]=json.loads((directory/"training.json").read_text())
        del optimizer,reader;gc.collect();torch.cuda.empty_cache()
        candidates[arm]=[]
        for epoch in c["candidate_epochs"]:
            state=source_state if epoch==0 else torch.load(directory/"checkpoints"/f"{epoch:03d}.pt",map_location="cpu",weights_only=True)["model"]
            model.load_state_dict(state,strict=True)
            folder=out/"calibration"/("baseline" if epoch==0 else arm+f"_{epoch:03d}")
            frozen=freeze_predictions(model,dataset,populations["calibration"],sc,args,c,device,folder)
            candidates[arm].append({"epoch":epoch,"frozen":frozen})
        del model;gc.collect();torch.cuda.empty_cache()
    json_write(out/"progress.json",{"status":"RUNNING","stage":"calibration_PARSeq"})
    readings,calibration_memory=recognize([p["frozen"] for values in candidates.values() for p in values],args,c,device)
    selections={};calibration={}
    for arm,values in candidates.items():
        scored=[]
        for candidate in values:
            report=score_frozen(candidate["frozen"],readings,manifest,args)
            scored.append({"epoch":candidate["epoch"],**report})
        calibration[arm]=scored;choice=max(scored,key=selection_key)
        epoch=choice["epoch"];checkpoint=source_checkpoint if epoch==0 else out/arm/"checkpoints"/f"{epoch:03d}.pt"
        selections[arm]={"epoch":epoch,"aggregate":choice["aggregate"],"checkpoint_path":str(checkpoint),"checkpoint_sha256":sha256_file(checkpoint)}
    json_write(out/"calibration.json",calibration,immutable=True)
    json_write(out/"selections.json",selections,immutable=True)
    del readings,candidates;gc.collect();torch.cuda.empty_cache()
    json_write(out/"progress.json",{"status":"RUNNING","stage":"evaluation_predictions","selections":selections})
    model=build_model(source_state,sc,device);frozen={}
    for arm in ("baseline",*ARMS):
        epoch=0 if arm=="baseline" else selections[arm]["epoch"]
        state=source_state if epoch==0 else torch.load(selections[arm]["checkpoint_path"],map_location="cpu",weights_only=True)["model"]
        model.load_state_dict(state,strict=True)
        frozen[arm]=freeze_predictions(model,dataset,populations["evaluation"],sc,args,c,device,out/"evaluation"/arm)
    del model,state,source_state;gc.collect();torch.cuda.empty_cache()
    readings,evaluation_memory=recognize(list(frozen.values()),args,c,device)
    arms={arm:{"ocr":score_frozen(item,readings,manifest,args),"native_counts":item["native_counts"],
               "pixel_counts":item["pixel_counts"]} for arm,item in frozen.items()}
    for arm,value in arms.items():
        if value["ocr"]["aggregate"]["gt_words"] != pre["evaluation_words"]:
            raise ValueError("Evaluation omitted care words")
        value["changes_vs_baseline"]=paired_changes(arms["baseline"]["ocr"],value["ocr"])
    paired=paired_changes(arms["detached"]["ocr"],arms["connected"]["ocr"])
    result={"status":"SUCCESS","result_class":result_class,"protocol":protocol,"selections":selections,
        "training":training,"arms":arms,"connected_vs_detached":paired,
        "inference_parameters":inference_parameters,"training_only_parameters":aux_parameters,
        "calibration_ocr_memory":calibration_memory,"evaluation_ocr_memory":evaluation_memory}
    lines=["# Auxiliary shared-feature OCR — "+result_class,"",
        "TRAIN fit/calibration/evaluation; original evaluation reused. Not official TEST, polygon scoring or Table5.","",
        "| Arm | Selected epoch | Localized | Exact / all GT | Precision | Recall | Hmean | Gained / lost vs baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm,value in arms.items():
        a=value["ocr"]["aggregate"];changes=value["changes_vs_baseline"]
        lines.append(f"| {arm} | {selections[arm]['epoch'] if arm!='baseline' else 0} | {a['geometry_iou_ge_50']} | {a['exact_casefold_iou_ge_50']} / {a['gt_words']} | {a['precision']:.2%} | {a['recall']:.2%} | {a['hmean']:.2%} | {changes['recovered']} / {changes['damaged']} |")
    lines.extend(["",f"Connected versus detached: {paired['net']:+d} exact words ({paired['recovered']} gained / {paired['damaged']} lost).",
        f"Inference parameters: {inference_parameters}; training-only reader: {aux_parameters}. Reader discarded at inference.",
        "Single seed/reused evaluation; no independent generalization or guaranteed80percent claim."])
    (out/"comparison.md").write_text("\n".join(lines)+"\n")
    # Fixed-first examples, never chosen by OCR outcome.
    for arm in ("baseline",*ARMS):
        for row in arms[arm]["ocr"]["per_image"][:8]:
            sample=next(s for s in manifest.samples if s.sample_id==row["sample_id"])
            with Image.open(args.word_root/sample.image_path) as im: image=im.convert("RGB")
            draw=ImageDraw.Draw(image)
            for p in row["predictions"]:draw.rectangle(p["bbox"],outline="orange",width=2)
            path=out/"visualizations"/arm/(row["sample_id"].replace(":","__")+".jpg");path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
    json_write(out/"results.json",result,immutable=True)
    json_write(out/"progress.json",{"status":"SUCCESS","stage":"complete","selections":selections,"connected_vs_detached":paired})
    print(json.dumps({k:v["ocr"]["aggregate"] for k,v in arms.items()},indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["approximate"],required=True)
    for name in ("config","source-run","word-root","parseq-source","parseq-checkpoint","parseq-archive","output-root"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",required=True)
    parser.add_argument("--action",choices=["preflight","smoke","run"],required=True)
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        if (args.output_root/"protocol.json").exists():
            value={"status":"FAILED","error":str(error),"traceback":traceback.format_exc(),
                   "oom":isinstance(error,torch.cuda.OutOfMemoryError)}
            if torch.cuda.is_initialized():
                value.update(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            json_write(args.output_root/"failure.json",value)
        raise


if __name__ == "__main__":main()
