"""Frozen TRAIN125/TEST300 CLIP image router and full-scene DeepSolo evaluation.

APPROXIMATE only. Prepare is GPU-independent when cached segmentation is used.
Complete resumes preparation and executes missing DeepSolo requests on the host.
Evaluate uses existing outputs only. Splits always have separate aggregates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
import torch

from deepsolo.adapter import DeepSoloSubprocessAdapter
from deepsolo.candidates import verify_host_source_identity
from deepsolo.contracts import DeepSoloResponse
from localize.word_guided_contours import word_contour_crops
from recognition.clip_image_router import accepted
from recognition.clip_router_evaluation import (
    aggregate_split, deepsolo_predictions, evaluate_predictions, fixed_image_score, native_mask,
)
from recognition.group_word_selection import reading_confidence
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer)
from scripts import run_clip_image_router as battery
from scripts.crop_clip_runtime import _cuda_peak, sha, write_json
from scripts.evaluate_totaltext_highres_contour_padding import _ground_truth, crop_array_hash
from scripts.run_clip_high_recall_admission import payload_digest, seal
from scripts.run_clip_image_router_word_advantage import WordAdvantageScorer
from scripts.run_clip_word_verification import model_identity, paths_profile
from segmentation.data import load_segmentation_manifest
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.training import seed_everything

REPO = Path(__file__).resolve().parents[1]
CODE = (*battery.CODE, "scripts/run_clip_router_e2e.py", "recognition/clip_router_evaluation.py",
        "scripts/run_clip_image_router_word_advantage.py", "recognition/parseq.py",
        "localize/word_guided_contours.py", "localize/word_multitask.py",
        "localize/independent_word_slots.py", "segmentation/totaltext_image_frequency_augment.py",
        "deepsolo/adapter.py", "deepsolo/contracts.py", "deepsolo/candidates.py",
        "deepsolo/worker.py", "deepsolo/reduction.py")


def read_approximate(path):
    if path.resolve().parent != REPO / "configs/approximate":
        raise ValueError("Isolated APPROXIMATE configuration required")
    return json.loads(path.read_text())


def resolve(value):
    return (Path(value) if Path(value).is_absolute() else REPO / value).resolve()


def checked(path, digest):
    return battery.checked(path, digest)


def key(sid):
    return sid.replace(":", "__")


def frame(path):
    with Image.open(path) as image:
        wh, orientation = list(image.size), image.getexif().get(274, 1)
    if type(orientation) is not int or orientation not in range(9):
        raise ValueError("Unsupported EXIF orientation: " + str(path))
    return wh, orientation


def validity(sample, data_root, image_wh):
    path = data_root / sample.validity_annotation_path
    if sha(path) != sample.validity_annotation_sha256:
        raise ValueError("Changed evaluation validity mask")
    with Image.open(path) as image:
        image = image.convert("L")
        if list(image.size) != image_wh or not set(np.unique(image)).issubset({0, 255}):
            raise ValueError("Invalid original validity geometry")
        return native_mask(image, 256)


def qualify(args):
    cfg, raw_paths = read_approximate(args.config), read_approximate(args.paths)
    if (args.mode != "approximate" or cfg["reproduction_mode"] != "approximate"
            or cfg["claim_label"] != "approximate reproduction" or cfg["table_5_claim"] is not False
            or cfg["schema_version"] != 1 or cfg["model"] != "vit_b16"
            or not 0 <= cfg["threshold"] <= 1 or cfg["tn"] != 0 or cfg["matching_iou"] != .5
            or cfg["precision"] != "float32"
            or any(type(cfg[k]) is not int or cfg[k] < 1 for k in ("cpu_threads", "parseq_batch_size"))):
        raise ValueError("Unsupported whole-pipeline scientific protocol")
    if raw_paths["path_base"] != "repository_root":
        raise ValueError("Invalid path profile")
    paths = {k: resolve(v) for k, v in raw_paths.items() if k not in ("schema_version", "path_base")}
    if args.output_root:
        paths["output_root"] = args.output_root.resolve()
    if paths["output_root"].is_relative_to(REPO):
        raise ValueError("Large outputs require an isolated external directory")
    router_cfg, router_paths, observed = battery.load_inputs(SimpleNamespace(
        mode=args.mode, config=REPO/cfg["router_config"], paths=REPO/cfg["router_paths"], limit=None))
    fallback = read_approximate(REPO/cfg["fallback_config"])
    fallback_paths = read_approximate(REPO/cfg["fallback_paths"])
    if (fallback["reproduction_mode"] != "approximate" or fallback["device"] != "cuda:0"
            or fallback["claim_label"] != "approximate reproduction" or fallback["table_5_claim"] is not False):
        raise ValueError("Unsupported fallback protocol")
    runtime_path = resolve(fallback_paths["deepsolo_runtime_manifest"])
    if sha(runtime_path) != fallback["runtime_sha256"]:
        raise ValueError("Qualified DeepSolo runtime changed")
    runtime = verify_host_source_identity(runtime_path, resolve(fallback_paths["deepsolo_source_root"]))
    if sha(resolve(fallback_paths["deepsolo_source_root"])/"adet/modeling/text_spotter.py") != cfg["deepsolo_threshold_source_sha256"]:
        raise ValueError("Qualified inclusive post-inference threshold implementation changed")
    candidate = runtime["candidates"][fallback["candidate"]]
    if (candidate["checkpoint_sha256"] != fallback["checkpoint_sha256"]
            or sha(Path(candidate["checkpoint_path"])) != fallback["checkpoint_sha256"]):
        raise ValueError("DeepSolo checkpoint changed")
    reference_root = paths["reference_run"]
    reference_plan = checked(reference_root/"resolved_configuration.json", cfg["reference_plan_sha256"])
    reference_scores = checked(reference_root/"scores_by_image.json", cfg["reference_scores_sha256"])
    checked(reference_root/"results.json", cfg["reference_results_sha256"])
    if (reference_plan["config"] != router_cfg
            or reference_scores["fit_sample_ids"] != reference_plan["split"]["calibration"]
            or set(reference_scores["scores"]) != {r["sample_id"] for r in observed}):
        raise ValueError("Frozen TRAIN calibration provenance differs")
    for name, digest in reference_plan["code_sha256"].items():
        if sha(REPO/name) != digest:
            raise ValueError("Historical router source changed: " + name)
    spec = {**json.loads((REPO/router_cfg["model_config"]).read_text())["models"][cfg["model"]],
            "weight_sha256": router_cfg["model_weight_sha256"][cfg["model"]]}
    identity = model_identity(paths_profile(router_paths["model_paths_profile"])["models"][cfg["model"]], spec)
    if identity != reference_plan["model"]:
        raise ValueError("CLIP calibration/model identity differs")
    for name, values in reference_scores["rank_reference"].items():
        if not values or values != sorted(values) or not np.isfinite(values).all():
            raise ValueError("Invalid frozen TRAIN reference: " + name)
    geometry = json.loads((router_paths["source_run"]/"geometry_results.json").read_text())
    recognition = json.loads((router_paths["source_run"]/"recognition_results.json").read_text())
    seg_manifest = json.loads((router_paths["segmentation_run"]/"run_manifest.json").read_text())
    contract = seg_manifest["run_contract"]
    if sha(router_paths["segmentation_run"]/seg_manifest["checkpoint"]["path"]) != router_cfg["source_checkpoint_sha256"]:
        raise ValueError("Selected kernel checkpoint changed")
    for name, digest in contract["source_sha256"].items():
        if sha(REPO/name) != digest:
            raise ValueError("Historical kernel source changed: " + name)
    train_root = Path(contract["runtime_inputs"]["target_root"])
    train_manifest = train_root/contract["config"]["train_manifest"]
    if sha(train_manifest) != contract["config"]["train_manifest_sha256"]:
        raise ValueError("TRAIN source manifest changed")
    train_samples = {s.sample_id:s for s in load_segmentation_manifest(train_manifest, require_checksums=True).samples}
    geos = {r["sample_id"]:r for r in geometry["branches"]["text_kernel"]["seeded"]["per_image"]}
    train, asset_hashes = [], {}
    for r in observed:
        sid, gr = r["sample_id"], geos[r["sample_id"]]
        wh, orientation = frame(Path(r["image_path"]))
        valid = validity(train_samples[sid], train_root, wh)
        with np.load(router_paths["segmentation_run"]/"validation_exports"/key(sid)/"maps.npz", allow_pickle=False) as z:
            if not np.array_equal(valid, z["validity"]):
                raise ValueError("TRAIN native validity replay differs")
        if sha(Path(gr["annotation_path"])) != gr["annotation_sha256"]:
            raise ValueError("TRAIN annotation changed")
        score_path = reference_root/"scores"/(key(sid)+".json")
        old = json.loads(score_path.read_text())
        if (old["plan_sha256"] != cfg["reference_plan_sha256"] or old["payload_sha256"] != payload_digest(old)
                or [{k:v for k,v in w.items() if k != "clip_word_advantage"} for w in old["rows"]] != r["rows"]):
            raise ValueError("Frozen TRAIN score rows differ")
        if fixed_image_score(old["rows"], reference_scores["rank_reference"]) != reference_scores["scores"][sid]["native_word_advantage"]:
            raise ValueError("Frozen TRAIN score replay differs")
        asset_hashes[str(score_path)] = sha(score_path)
        train.append({"observed": {**r,"rows":old["rows"]}, "sample": train_samples[sid],
                      "data_root": train_root, "validity": valid, "ground_truth": gr["ground_truth"],
                      "image_wh":wh,"orientation":orientation,"ignored_ids":gr["ignored_prediction_ids"]})
    test_plan = checked(paths["kernel_test_run"]/"resolved_configuration.json", cfg["kernel_test_plan_sha256"])
    test_result = checked(paths["kernel_test_run"]/"results.json", cfg["kernel_test_results_sha256"])
    if (test_result["status"] != "SUCCESS" or test_plan["partial"]
            or test_plan["source_runs"]["kernel"]["report"] != seg_manifest
            or test_plan["preprocessing"] != contract["resolved_preprocessing"]
            or test_plan["test_manifest_sha256"] != cfg["test_manifest_sha256"]
            or sha(paths["test_manifest"]) != cfg["test_manifest_sha256"]):
        raise ValueError("Complete TEST kernel map provenance differs")
    for name, digest in test_plan["source_sha256"].items():
        if sha(REPO/name) != digest:
            raise ValueError("Historical TEST source changed: " + name)
    test_manifest = load_segmentation_manifest(paths["test_manifest"], require_checksums=True)
    if (test_manifest.dataset != "total_text" or test_manifest.split_role != "test"
            or len(test_manifest.samples) != cfg["test_images"]
            or test_plan["sample_ids"] != [s.sample_id for s in test_manifest.samples]):
        raise ValueError("Official TEST population differs")
    test = []
    for s in test_manifest.samples:
        image_path = paths["test_data_root"]/s.image_path
        if sha(image_path) != s.image_sha256:
            raise ValueError("TEST original image changed")
        wh, orientation = frame(image_path)
        gt, annotation_sha = _ground_truth(s, paths["test_data_root"], tuple(wh))
        map_path = paths["kernel_test_run"]/"maps/kernel"/(key(s.sample_id)+".npz")
        asset_hashes[str(map_path)] = sha(map_path)
        asset_hashes[str(paths["test_data_root"]/"datasets/total_text/test/annotations"/(s.sample_id.split(":")[1]+".txt"))] = annotation_sha
        test.append({"observed":{"sample_id":s.sample_id,"image_path":str(image_path),
                     "image_sha256":s.image_sha256},"sample":s,"data_root":paths["test_data_root"],
                     "validity":validity(s, paths["test_data_root"], wh), "ground_truth":gt,
                     "image_wh":wh,"orientation":orientation,"map_path":map_path})
    if ({r["observed"]["sample_id"] for r in train} & {r["observed"]["sample_id"] for r in test}
            or {r["observed"]["image_sha256"] for r in train} & {r["observed"]["image_sha256"] for r in test}):
        raise ValueError("TRAIN/TEST image leakage")
    for name, rows in (("train",train),("test",test)):
        if len(rows) != cfg[name+"_images"] or sum(len(r["ground_truth"]) for r in rows) != cfg[name+"_care_words"]:
            raise ValueError("Frozen split denominator differs: " + name)
    ds_plan = checked(paths["deepsolo_test_run"]/"resolved_configuration.json", cfg["deepsolo_test_plan_sha256"])
    if (ds_plan["manifest_sha256"] != cfg["test_manifest_sha256"]
            or ds_plan["runtime_manifest_sha256"] != fallback["runtime_sha256"]
            or ds_plan["checkpoint_sha256"] != fallback["checkpoint_sha256"]
            or ds_plan["detector_inference_threshold"] > fallback["score_threshold"]):
        raise ValueError("DeepSolo TEST cache contract differs")
    for name, digest in ds_plan["code_sha256"].items():
        # Historical crop-slot evaluation changed; raw inference modules did not.
        if name.startswith("deepsolo/") and sha(REPO/name) != digest:
            raise ValueError("Cached DeepSolo inference source changed")
    for r in test:
        path = paths["deepsolo_test_run"]/"detector"/(key(r["observed"]["sample_id"])+".json")
        asset_hashes[str(path)] = sha(path)
        old = json.loads(path.read_text())
        if (old["sample_id"] != r["observed"]["sample_id"] or old["image_sha256"] != r["observed"]["image_sha256"]
                or old["image_wh"] != r["image_wh"] or old["plan_sha256"] != cfg["deepsolo_test_plan_sha256"]):
            raise ValueError("DeepSolo TEST image provenance differs")
        validate_deepsolo(old["response"], fallback, ds_plan["detector_inference_threshold"])
        r["ds_cache_path"] = path
    return cfg,paths,router_cfg,router_paths,recognition,spec,identity,fallback,fallback_paths,reference_scores,asset_hashes,{"train":train,"test":test}


def validate_deepsolo(response, cfg, actual_threshold):
    r = DeepSoloResponse.from_dict(response).to_dict()
    a = r["artifact"]
    load = a["load_validation"]
    if (not r["success"] or a["checkpoint_sha256"] != cfg["checkpoint_sha256"]
            or a["candidate"]["artifact_id"] != cfg["candidate"]
            or a["config_sha256"] != a["candidate"]["qualified_config_sha256"]
            or a["inference_options"] != {"device":cfg["device"],"lexicon_type":None,"score_threshold":actual_threshold}
            or a["resolved_inference_threshold"] != actual_threshold
            or r["placement"]["device"] != cfg["device"] or not load["strict_compatibility_check"]
            or any(load[n] for n in ("missing_keys","unexpected_keys","incorrect_shapes"))):
        raise ValueError("Failed/incompatible raw DeepSolo output")
    return r


def cached(path, contract):
    r = json.loads(path.read_text())
    if r["contract"] != contract or r["payload_sha256"] != payload_digest(r):
        raise ValueError("Cached shard contract/payload differs: " + str(path))
    return r


def save_shard(path, contract, value):
    r = {"contract":contract, **value}
    r["payload_sha256"] = payload_digest(r)
    seal(path,r)
    return r


def original_crops(observed, rows):
    if sha(Path(observed["image_path"])) != observed["image_sha256"]:
        raise ValueError("Original RGB changed during execution")
    with Image.open(observed["image_path"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    crops = []
    for w in rows:
        x1,y1,x2,y2 = w["bbox"]
        if not (0 <= x1 < x2 <= rgb.shape[1] and 0 <= y1 < y2 <= rgb.shape[0]):
            raise ValueError("Invalid original crop")
        crop = np.ascontiguousarray(rgb[y1:y2,x1:x2])
        if "crop_rgb_sha256" in w and crop_array_hash(crop) != w["crop_rgb_sha256"]:
            raise ValueError("Changed original RGB crop")
        crops.append(crop)
    return crops


def prepare(split, rows, root, plan_sha, cfg, router_cfg, recognition, identity, spec, reference, device):
    ocr_model = None
    prepared = []
    try:
        for i,r in enumerate(rows,1):
            observed = r["observed"]
            contract = {"plan_sha256":plan_sha,"sample_id":observed["sample_id"],"image_sha256":observed["image_sha256"]}
            path = root/split/"parseq"/(key(observed["sample_id"])+".json")
            if path.exists():
                shard = cached(path,contract)
            elif split == "train":
                shard = save_shard(path,contract,{"observed":observed,"ignored_ids":r["ignored_ids"],
                    "origin":"frozen TRAIN PARSeq and CLIP","ground_truth_in_routing":False})
            else:
                start=time.monotonic()
                with np.load(r["map_path"], allow_pickle=False) as z:
                    probabilities = z["probabilities"]
                canvas = letterbox_canvas_mask(tuple(r["image_wh"]), native_size=256)
                # Seeded replay retains every unseeded text component, matching TRAIN.
                source_contract = json.loads(Path(cfg["_kernel_manifest"]).read_text())["run_contract"]
                proposals, labels, audit = word_contour_crops(probabilities,canvas,tuple(r["image_wh"]),
                    source_contract["config"]["postprocessing"],{},method="seeded")
                crops = original_crops(observed,proposals)
                if crops and ocr_model is None:
                    p=recognition["protocol"]["paths"]
                    ocr_model=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(p["parseq_source"]),
                        checkpoint=Path(p["parseq_checkpoint"]),runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                        runtime_source_archive=Path(p["parseq_source_archive"]),device=device,decode_ar=True,refine_iters=1))
                words, ocr_records = [], []
                for first in range(0,len(crops),cfg["parseq_batch_size"]):
                    batch=crops[first:first+cfg["parseq_batch_size"]]
                    values=ocr_model.recognize_candidate_batch(batch)
                    if len(values)!=len(batch):raise ValueError("PARSeq batch population changed")
                    for proposal,crop,value in zip(proposals[first:first+len(batch)],batch,values):
                        ocr={**value.to_dict(),"crop_rgb_sha256":crop_array_hash(crop),
                            "parseq_checkpoint_sha256":PARSEQ_CHECKPOINT_SHA256,
                            "parseq_source_archive_sha256":PARSEQ_RUNTIME_ARCHIVE_SHA256,
                            "decode_ar":True,"refine_iters":1}
                        native=reading_confidence(ocr)
                        words.append({"instance_id":proposal["instance_id"],"bbox":proposal["bbox"],
                            "raw_text":ocr["raw_text"],"crop_rgb_sha256":ocr["crop_rgb_sha256"],
                            "native":native,"confidence":native["confidence_balanced"]})
                        ocr_records.append(ocr)
                ignored=[int(v) for v in np.unique(labels[labels>0]) if not np.any((labels==v)&r["validity"])]
                shard=save_shard(path,contract,{"observed":{**observed,"rows":words},"ocr_records":ocr_records,
                    "ignored_ids":ignored,"proposals":proposals,"audit":audit,"ground_truth_in_routing":False,
                    "runtime":{"seconds":time.monotonic()-start,"cuda_peak":_cuda_peak(device),
                        "model_set":["PARSeq"],"precision":"float32","batch_size":cfg["parseq_batch_size"],
                        "input_shape":[cfg["parseq_batch_size"],3,32,128],"device":device,"oom":False}})
            prepared.append((r,shard))
            if i%25==0 or i==len(rows):print(f"[{split} PARSeq] {i}/{len(rows)}",flush=True)
    finally:
        if ocr_model is not None:ocr_model.release()
        del ocr_model
        gc.collect()
        if device.startswith("cuda"):torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    model, scorer, routes = None,None,[]
    try:
        for i,(r,ocr) in enumerate(prepared,1):
            observed=ocr["observed"]; path=root/split/"routes"/(key(observed["sample_id"])+".json")
            if path.exists():
                route=cached(path,ocr["contract"])
            else:
                start=time.monotonic();words=observed["rows"]
                if split == "test":
                    if words and scorer is None:
                        from transformers import CLIPModel,CLIPProcessor
                        model,info=CLIPModel.from_pretrained(identity["local_path"],local_files_only=True,
                            use_safetensors=spec["weight_file"].endswith("safetensors"),output_loading_info=True)
                        if any(info.values()):raise ValueError("CLIP checkpoint key mismatch: "+str(info))
                        model=model.eval().float().to(device)
                        scorer=WordAdvantageScorer(model,CLIPProcessor.from_pretrained(identity["local_path"],local_files_only=True),router_cfg,device)
                    advantages=scorer.score(words,original_crops(observed,words)) if words else []
                    words=[{**w,"clip_word_advantage":v} for w,v in zip(words,advantages)]
                score=fixed_image_score(words,reference)
                route=save_shard(path,ocr["contract"],{"observed":{**observed,"rows":words},"score":score,
                    "threshold":cfg["threshold"],"action":"accept_parseq" if accepted(score,cfg["threshold"]) else "deepsolo",
                    "ground_truth_in_routing":False,"ignored_ids":ocr["ignored_ids"],
                    "runtime":{"seconds":time.monotonic()-start,"cuda_peak":_cuda_peak(device),"device":device,
                        "model_set":[spec["model_id"]] if split=="test" else [],"precision":"float32",
                        "batch_size":router_cfg["execution"]["batch_size"],"input_shape":[3,224,224],"oom":False}})
            routes.append(route)
            if i%25==0 or i==len(rows):print(f"[{split} CLIP routes] {i}/{len(rows)}",flush=True)
    finally:
        del model,scorer
        gc.collect()
        if device.startswith("cuda"):torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    seal(root/split/"routing_requests.json",{"plan_sha256":plan_sha,"threshold":cfg["threshold"],
        "images":[{"sample_id":r["observed"]["sample_id"],"score":r["score"],"action":r["action"]} for r in routes],
        "ground_truth_in_routing":False,"reference_fit_split":"TRAIN80","test_fitting":False})
    return routes


def evaluate(split, rows, routes, root, plan_sha, cfg, fallback, fallback_paths, execute):
    adapter=None; evaluated=[]
    for i,(sample,route) in enumerate(zip(rows,routes),1):
        observed=route["observed"];sid=observed["sample_id"]
        contract={"plan_sha256":plan_sha,"sample_id":sid,"image_sha256":observed["image_sha256"]}
        raw_origin=None
        if route["action"] == "accept_parseq":
            preds=[{"instance_id":w["instance_id"],"bbox":w["bbox"],"text":w["raw_text"],
                    "ignored":w["instance_id"] in route["ignored_ids"]} for w in observed["rows"]]
        else:
            path=root/split/"deepsolo"/(key(sid)+".json")
            if path.exists():
                output=cached(path,contract)
            elif split=="test":
                source=sample["ds_cache_path"];old=json.loads(source.read_text());response=old["response"]
                output=save_shard(path,contract,{"response":response,"origin":{"path":str(source),"sha256":sha(source),
                    "conversion":"inclusive post-inference score filter","original_threshold":response["artifact"]["resolved_inference_threshold"],
                    "effective_score_threshold":fallback["score_threshold"]}})
            elif execute:
                if adapter is None:
                    adapter=DeepSoloSubprocessAdapter(Path(fallback_paths["deepsolo_python"]).absolute(),
                        resolve(fallback_paths["deepsolo_runtime_manifest"]),
                        host_source_root=resolve(fallback_paths["deepsolo_source_root"]),repository_root=REPO)
                response=adapter.infer(image_path=Path(observed["image_path"]),candidate_id=fallback["candidate"],
                    inference_options={"device":fallback["device"],"score_threshold":fallback["score_threshold"],"lexicon_type":None},
                    reduction_policy="reading_order_space_join").to_dict()
                validate_deepsolo(response,fallback,fallback["score_threshold"])
                output=save_shard(path,contract,{"response":response,"origin":{"kind":"fresh qualified full-image inference"}})
            else:
                raise RuntimeError("Missing TRAIN DeepSolo output; run --stage complete on GPU host: "+sid)
            actual=output["response"]["artifact"]["resolved_inference_threshold"]
            response=validate_deepsolo(output["response"],fallback,actual)
            if actual>fallback["score_threshold"]:raise ValueError("Cached DeepSolo threshold loses required detections")
            preds=deepsolo_predictions(response,sample["image_wh"],sample["orientation"],sample["validity"],fallback["score_threshold"])
            raw_origin={"path":str(path),"sha256":sha(path),"receipt":output["origin"]}
        metrics=evaluate_predictions(preds,sample["ground_truth"],iou_threshold=cfg["matching_iou"])
        result={"sample_id":sid,"status":"SUCCESS","action":route["action"],"router_score":route["score"],
            "predictions_all":preds,"ground_truth":sample["ground_truth"],"metrics":metrics,
            "raw_deepsolo":raw_origin,"image_wh":sample["image_wh"],"exif_orientation":sample["orientation"]}
        save_shard(root/split/"evaluation"/(key(sid)+".json"),contract,result)
        evaluated.append(result)
        if i%10==0 or i==len(rows):print(f"[{split} final evaluation] {i}/{len(rows)}",flush=True)
    ids=[r["observed"]["sample_id"] for r in rows]
    expected_words=sum(len(r["ground_truth"]) for r in rows)
    combined=aggregate_split(evaluated,ids,expected_words)
    parseq_rows=[]
    for sample,route in zip(rows,routes):
        predictions=[{"bbox":w["bbox"],"text":w["raw_text"],"ignored":w["instance_id"] in route["ignored_ids"]} for w in route["observed"]["rows"]]
        parseq_rows.append({"sample_id":sample["observed"]["sample_id"],"status":"SUCCESS","action":"accept_parseq",
                           "metrics":evaluate_predictions(predictions,sample["ground_truth"],iou_threshold=cfg["matching_iou"])})
    baseline=aggregate_split(parseq_rows,ids,expected_words)
    if split=="train" and not cfg.get("_smoke_limit") and (baseline["tp"],baseline["detected_words"])!=(cfg["train_baseline_tp"],cfg["train_baseline_care_outputs"]):
        raise ValueError("Frozen TRAIN baseline metric replay differs")
    report={"status":"SUCCESS","reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":"smoke-test result" if cfg.get("_smoke_limit") else "approximate reproduction",
        "split":"TRAIN_heldout125" if split=="train" else "TEST_official300","table_5_claim":False,
        "plan_sha256":plan_sha,"threshold":cfg["threshold"],"combined_model":combined,"parseq_baseline":baseline,
        "per_image":evaluated,"protocol":{"aggregation":"micro word counts over every image of this split",
        "matching":"one-to-one maximum cardinality; exclusive bbox IoU>=0.5 and Unicode casefold exact; punctuation retained",
        "accuracy_on_detected_words":"TP/(TP+FP)","accuracy_tn0":"TP/(TP+FP+FN)",
        "ignore":"evaluation-only native support entirely invalid; DeepSolo polygon, U-Net exact instance",
        "empty_envelope":"retain as unmatchable FP","empty_predictions":"all care GT become FN","failed_images":"fail complete aggregate"},
        "test_fitting":False}
    seal(root/split/"results.json",report)
    return report


def run(args):
    values=qualify(args)
    cfg,paths,router_cfg,router_paths,recognition,spec,identity,fallback,fallback_paths,reference,assets,splits=values
    if args.limit is not None:
        if not 0<args.limit<=min(len(r) for r in splits.values()):raise ValueError("Invalid smoke limit")
        splits={name:rows[:args.limit] for name,rows in splits.items()}
    if args.check:
        print(json.dumps({"status":"PREFLIGHT_PASS","train_images":len(splits['train']),"test_images":len(splits['test']),
            "train_accepted":sum(accepted(fixed_image_score(r['observed']['rows'],reference['rank_reference']),cfg['threshold']) for r in splits['train']),
            "threshold":cfg['threshold'],"main_cuda_available":torch.cuda.is_available(),"test_deepsolo_cache":"qualified"}),flush=True)
        return None
    if args.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; explicitly use --device cpu or run on GPU host")
    torch.set_num_threads(cfg["cpu_threads"]);seed_everything(cfg["seed"],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.device.startswith("cuda"):torch.cuda.reset_peak_memory_stats(args.device)
    cfg={**cfg,"_kernel_manifest":str(router_paths["segmentation_run"]/"run_manifest.json"),"_smoke_limit":args.limit}
    plan={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction","table_5_claim":False,
        "result_class":"smoke-test result" if args.limit else "approximate reproduction","config":cfg,
        "config_sha256":sha(args.config),"paths":{k:str(v) for k,v in paths.items()},"device":args.device,
        "router_config":router_cfg,"router_paths":{k:str(v) for k,v in router_paths.items() if k!='fonts'},
        "fallback_config":fallback,"fallback_paths":fallback_paths,"clip_model":identity,
        "split_ids":{name:[r['observed']['sample_id'] for r in rows] for name,rows in splits.items()},
        "asset_sha256":assets,"code_sha256":{n:sha(REPO/n) for n in CODE},
        "reference_fit_split":"TRAIN80","threshold_fit_split":"TRAIN125","test_fitting":False,
        "software":{"torch":torch.__version__,"numpy":np.__version__,"pillow":Image.__version__,
            "transformers":__import__('transformers').__version__,"opencv":__import__('cv2').__version__,
            "python":__import__('sys').version},"deterministic_algorithms":True,"tf32":False,
        "gpu":torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None}
    root=paths["output_root"]/hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    seal(root/"resolved_configuration.json",plan);plan_sha=sha(root/"resolved_configuration.json")
    selected=list(splits) if args.split=='both' else [args.split]
    try:
        reports={}
        for split in selected:
            if args.stage=='evaluate':
                routes=[cached(root/split/'routes'/(key(r['observed']['sample_id'])+'.json'),
                    {"plan_sha256":plan_sha,"sample_id":r['observed']['sample_id'],"image_sha256":r['observed']['image_sha256']}) for r in splits[split]]
            else:
                routes=prepare(split,splits[split],root,plan_sha,cfg,router_cfg,recognition,identity,spec,reference['rank_reference'],args.device)
            if args.stage!='prepare':
                reports[split]=evaluate(split,splits[split],routes,root,plan_sha,cfg,fallback,fallback_paths,args.stage=='complete')
                print(json.dumps({"split":split,"results":str(root/split/'results.json'),"combined_model":reports[split]['combined_model']}),flush=True)
        if all((root/name/'results.json').exists() for name in splits):
            both={name:json.loads((root/name/'results.json').read_text()) for name in splits}
            seal(root/'results.json',{"status":"SUCCESS","claim_label":"approximate reproduction","table_5_claim":False,
                "result_class":plan['result_class'],"plan_sha256":plan_sha,
                "splits":{name:{"results_path":str(root/name/'results.json'),"results_sha256":sha(root/name/'results.json'),
                    "combined_model":r['combined_model'],"parseq_baseline":r['parseq_baseline']} for name,r in both.items()},
                "splits_aggregated_together":False})
        print(json.dumps({"status":"PREPARED" if args.stage=='prepare' else 'SUCCESS',"output_root":str(root),"splits":selected}),flush=True)
        return root
    except Exception as error:
        write_json(root/'failure.json',{"status":"FAILED","stage":args.stage,"split":args.split,"error":str(error),
            "traceback":traceback.format_exc(),"out_of_memory":isinstance(error,(MemoryError,torch.cuda.OutOfMemoryError)),
            "cuda_peak":_cuda_peak(args.device),"worker_response":error.response.to_dict() if getattr(error,'response',None) else None})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['approximate'])
    p.add_argument('--config',type=Path,default=REPO/'configs/approximate/clip_router_e2e_v1.json')
    p.add_argument('--paths',type=Path,default=REPO/'configs/approximate/clip_router_e2e_local_paths_v1.json')
    p.add_argument('--stage',choices=['prepare','complete','evaluate'],default='complete')
    p.add_argument('--split',choices=['train','test','both'],default='both')
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu')
    p.add_argument('--output-root',type=Path);p.add_argument('--limit',type=int);p.add_argument('--check',action='store_true')
    run(p.parse_args())


if __name__=='__main__':main()
