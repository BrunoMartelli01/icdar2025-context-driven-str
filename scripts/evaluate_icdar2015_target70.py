"""Frozen approximate ICDAR TEST500 evaluation; all care and emitted outputs count."""

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from recognition.icdar2015_word_f1 import aggregate_counts,sample_counts
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE,PARSEQ_CHECKPOINT_SHA256,PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig,ParseqApproxRecognizer,verify_parseq_runtime_source)
from scripts.evaluate_icdar2015_improvement_test import original_rgb,ground_truth,crop_hash
from scripts.train_icdar2015_totaltext_transfer import write_json
from segmentation.data import file_sha256,load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.icdar2015_improvement_runtime import REPO
from segmentation.icdar2015_improvements import binary_proposals,make_transform,valid_canvas
from segmentation.icdar2015_operating_policy import component_crop
from segmentation.icdar2015_target70 import emitted_predictions,load_profile
from segmentation.icdar2015_target70_selection import verify_frozen_selection,load_selected_model
from segmentation.training import seed_everything,runtime_environment


def load_test_config(path):
    config=json.loads(path.read_text())
    if path.resolve().parent.name!="approximate":raise ValueError("Explicit approximate evaluation profile required")
    expected={"reproduction_mode":"approximate","claim_label":"approximate reproduction","dataset":"icdar2015",
        "split_role":"test","expected_images":500,"expected_care_words":2077,"expected_ignored_gt":3153,
        "segmentation_batch_size":1,"text_normalization":"unicode_casefold_then_isalnum",
        "matching_iou_threshold":.5,"prediction_ignore_policy":"all_outputs_count_including_ignored_regions",
        "ground_truth_policy":"exclude_official_ignored_instances_keep_all_care_words",
        "connectivity":8,"min_component_area":0,"maximum_proposals":None,"padding_native_pixels":0}
    if any(config.get(key)!=value for key,value in expected.items()):raise ValueError("Unchanged complete TEST500 word protocol required")
    if config["parseq_batch_size"]<1:raise ValueError("Positive OCR batch size required")
    return config


def require_ocr_qualification(qualification,chosen,device):
    if (qualification["status"]!="PASS" or qualification["images"]!=100 or qualification["care_words"]!=430
            or qualification["failed_samples"]!=0 or qualification["oom"] is not False
            or any(qualification[key]!=0 for key in ("text_differences","token_id_differences","output_policy_differences"))
            or qualification["protocol"]["operating_results_sha256"]!=chosen["policy_report_sha256"]
            or qualification["protocol"]["selected"]!=chosen["policy"] or qualification["protocol"]["device"]!=device
            or qualification["raw_aggregate"]!=chosen["policy"]["raw_aggregate"]
            or qualification["emitted_aggregate"]!=chosen["policy"]["emitted_aggregate"]):
        raise ValueError("Exact full selected-validation OCR device qualification required")


def reconstruct(shard,rgb,output,shape,policy):
    path=output/shard["mask_path"]
    if file_sha256(path)!=shard["mask_sha256"]:raise ValueError("Cached TEST mask changed")
    with np.load(path) as saved:probability,prediction=saved["probability"],saved["prediction"]
    if (probability.shape!=tuple(x//2 for x in shape) or not np.isfinite(probability).all()
            or not np.array_equal(prediction,(probability>=policy["probability_threshold"])&valid_canvas(shard["image_wh"],shape))):
        raise ValueError("Cached threshold/canvas replay differs")
    proposals,labels=binary_proposals(prediction,shard["image_wh"],shape)
    if len(proposals)!=len(shard["proposals"]):raise ValueError("All raw components must remain present")
    crops=[]
    for proposal,saved in zip(proposals,shard["proposals"]):
        if any(proposal[k]!=saved[k] for k in proposal):raise ValueError("Cached raw component differs")
        crop=component_crop(rgb,labels,proposal,shape,policy["crop_method"])
        if crop_hash(crop)!=saved["crop_rgb_sha256"]:raise ValueError("Frozen TEST crop changed")
        crops.append(crop)
    return crops


def run(args):
    started=time.monotonic();config=load_test_config(args.config)
    if file_sha256(args.selection_profile)!=config["selection_profile_sha256"]:raise ValueError("Selection profile changed")
    selection=verify_frozen_selection(args.selection,args.selection_profile);chosen=selection["selected"];policy=chosen["policy"]
    training_path=REPO/selection["config"]["training_profile"]
    if file_sha256(training_path)!=config["training_config_sha256"]:raise ValueError("Training profile changed")
    training=load_profile(training_path);shape=chosen["image_hw"]
    qualification=json.loads(args.ocr_qualification.read_text());require_ocr_qualification(qualification,chosen,args.device)
    if config["parseq_batch_size"]!=qualification["protocol"]["cpu_protocol"]["batch_size"]:
        raise ValueError("Qualified OCR batch size changed")
    for name,digest in qualification["protocol"]["source_sha256"].items():
        if file_sha256(REPO/name)!=digest:raise ValueError("OCR qualification source changed: "+name)
    if file_sha256(args.test_manifest)!=config["test_manifest_sha256"]:raise ValueError("Official TEST manifest changed")
    manifest=load_segmentation_manifest(args.test_manifest,require_checksums=True)
    if manifest.dataset!="icdar2015" or manifest.split_role!="test" or len(manifest.samples)!=500:
        raise ValueError("Complete official TEST500 required")
    if (file_sha256(args.parseq_checkpoint)!=PARSEQ_CHECKPOINT_SHA256
            or file_sha256(args.parseq_source_archive)!=PARSEQ_RUNTIME_ARCHIVE_SHA256):raise ValueError("Qualified PARSeq artifacts changed")
    runtime=verify_parseq_runtime_source(args.parseq_source,source_type=PARSEQ_ARCHIVE_SOURCE,source_archive=args.parseq_source_archive)
    if qualification["protocol"]["cpu_protocol"]["runtime"]!=runtime:raise ValueError("Qualified OCR runtime changed")
    resolved={**config,"image_hw":shape,"probability_threshold":policy["probability_threshold"],
        "crop_method":policy["crop_method"],"emission_policy":policy["emission_policy"]}
    sources=("scripts/evaluate_icdar2015_target70.py","segmentation/icdar2015_target70_selection.py",
        "segmentation/icdar2015_improvements.py","segmentation/icdar2015_operating_policy.py","segmentation/icdar2015_target70.py",
        "scripts/evaluate_icdar2015_improvement_test.py","recognition/icdar2015_word_f1.py","recognition/deepsolo_word_f1.py","recognition/parseq.py")
    protocol={"reproduction_mode":"approximate","claim_label":"approximate reproduction","result_class":"approximate reproduction",
        "evaluation_config":resolved,"config_sha256":file_sha256(args.config),"selection_sha256":file_sha256(args.selection),
        "selected":chosen,"ocr_device_qualification_sha256":file_sha256(args.ocr_qualification),"parseq_runtime":runtime,
        "parseq_checkpoint_sha256":PARSEQ_CHECKPOINT_SHA256,"device":args.device,"precision":"float32","tf32":False,
        "source_sha256":{p:file_sha256(REPO/p) for p in sources},
        "paths":{name:str(getattr(args,name).resolve()) for name in ("selection","selection_profile","ocr_qualification",
            "test_manifest","data_root","parseq_source","parseq_source_archive","parseq_checkpoint")},
        "primary_population":"all outputs emitted by frozen TRAIN-selected policy, including ignored regions",
        "raw_population":"all connected components, including blank OCR and ignored regions"}
    if args.preflight_only:return {"status":"PREFLIGHT_PASS","images":len(manifest.samples),"selected":chosen["name"]}
    output=args.output_dir.resolve();root=args.data_root.resolve();device=torch.device(args.device)
    if output==REPO or REPO in output.parents:raise ValueError("Scientific outputs must remain outside Git")
    if device.type!="cuda" or not torch.cuda.is_available():raise ValueError("Explicit qualified CUDA device required; no substitution")
    protocol_sha=sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest();protocol_path=output/"protocol.json"
    if output.exists():
        if not args.resume or not protocol_path.exists() or json.loads(protocol_path.read_text())!=protocol:
            raise ValueError("Existing TEST output requires exact-protocol --resume")
    else:write_json(protocol_path,protocol)
    seed_everything(training["training"]["seed"],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    _prepare_cuda_memory_tracking(device)
    shards={};pending=[]
    for sample in manifest.samples:
        path=output/"samples"/(sample.sample_id.replace(":","_")+".json")
        if path.exists():
            shard=json.loads(path.read_text())
            if shard["protocol_sha256"]!=protocol_sha or shard["image_sha256"]!=sample.image_sha256:
                raise ValueError("Cached sample protocol changed")
            shards[sample.sample_id]=shard
        else:pending.append(sample)
    if pending:
        model=load_selected_model(selection).to(device).eval();transform=make_transform(training,shape,augment=False)
        with torch.inference_mode():
            for index,sample in enumerate(pending,1):
                rgb=original_rgb(sample,root);source=Image.fromarray(rgb)
                image,_,_=transform.transform_with_validity(source,Image.new("L",source.size),Image.new("L",source.size,255),0)
                probability=model(image[None].to(device)).sigmoid()[0,0].cpu().numpy()
                if probability.shape!=tuple(x//2 for x in shape) or not np.isfinite(probability).all():raise ValueError("Invalid native probability")
                prediction=(probability>=policy["probability_threshold"])&valid_canvas(source.size,shape)
                proposals,labels=binary_proposals(prediction,source.size,shape)
                for rank,p in enumerate(proposals):
                    p.update(rank=rank,crop_rgb_sha256=crop_hash(component_crop(rgb,labels,p,shape,policy["crop_method"])))
                key=sample.sample_id.replace(":","_");path=output/"masks"/(key+".npz");path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():
                    with np.load(path) as old:
                        if not np.array_equal(old["probability"],probability) or not np.array_equal(old["prediction"],prediction):
                            raise ValueError("Orphaned mask differs from deterministic replay")
                else:
                    temporary=path.with_suffix(".tmp")
                    with temporary.open("wb") as stream:np.savez_compressed(stream,probability=probability,prediction=prediction)
                    temporary.replace(path)
                shard={"sample_id":sample.sample_id,"image_sha256":sample.image_sha256,"image_wh":list(source.size),
                    "protocol_sha256":protocol_sha,"mask_path":str(path.relative_to(output)),"mask_sha256":file_sha256(path),"proposals":proposals}
                write_json(output/"samples"/(key+".json"),shard);shards[sample.sample_id]=shard
                if index%50==0 or index==len(pending):print(f"TEST masks {len(shards)}/500",flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    segmentation_peak={"allocated":torch.cuda.max_memory_allocated(device),"reserved":torch.cuda.max_memory_reserved(device)}
    _prepare_cuda_memory_tracking(device);recognizer=None;pending_crops={}
    # Rebuild every raw crop, even on cache reopening. Neither this stage nor
    # the preceding neural inference reads TEST annotations or ignore regions.
    for sample in manifest.samples:
        shard=shards[sample.sample_id];rgb=original_rgb(sample,root)
        crops=reconstruct(shard,rgb,output,shape,policy)
        for proposal,crop in zip(shard["proposals"],crops):
            digest=proposal["crop_rgb_sha256"];path=output/"ocr"/(digest+".json")
            if path.exists():
                cached=json.loads(path.read_text())
                if cached["protocol_sha256"]!=protocol_sha or cached["crop_rgb_sha256"]!=digest:raise ValueError("OCR cache changed")
            else:pending_crops.setdefault(digest,crop)
    items=list(pending_crops.items());batch=config["parseq_batch_size"]
    if items:
        recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,checkpoint=args.parseq_checkpoint,
            device=args.device,runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=args.parseq_source_archive))
        for start in range(0,len(items),batch):
            chunk=items[start:start+batch];outputs=recognizer.recognize_candidate_batch([rgb for _,rgb in chunk])
            if len(outputs)!=len(chunk):raise RuntimeError("Incomplete TEST OCR batch")
            for (digest,_),predicted in zip(chunk,outputs):
                write_json(output/"ocr"/(digest+".json"),{"protocol_sha256":protocol_sha,"crop_rgb_sha256":digest,**predicted.to_dict()})
        recognizer.release()
    ocr_peak={"allocated":torch.cuda.max_memory_allocated(device),"reserved":torch.cuda.max_memory_reserved(device)}
    rows=[];ignored_total=0
    # Only after predictions are frozen do annotations enter the metric.
    for sample in manifest.samples:
        shard=shards[sample.sample_id];words,ignored,annotation_sha=ground_truth(sample,root,tuple(shard["image_wh"]))
        ignored_total+=ignored;raw=[]
        for p in shard["proposals"]:
            path=output/"ocr"/(p["crop_rgb_sha256"]+".json");o=json.loads(path.read_text())
            raw.append({**p,"text":o["raw_text"],"sequence_confidence":o["sequence_confidence_native"],
                "mean_token_log_probability":o["mean_token_log_probability"],"ocr_file_sha256":file_sha256(path)})
        emitted=emitted_predictions(raw,policy["emission_policy"])
        rows.append({"sample_id":sample.sample_id,"ground_truth":words,"ignored_gt_count":ignored,"annotation_sha256":annotation_sha,
            "raw_predictions":raw,"predictions":emitted,"raw_counts":sample_counts(raw,words,iou_threshold=config["matching_iou_threshold"]),
            "counts":sample_counts(emitted,words,iou_threshold=config["matching_iou_threshold"])})
    raw=aggregate_counts([r["raw_counts"] for r in rows]);emitted=aggregate_counts([r["counts"] for r in rows])
    if (raw["images"]!=500 or emitted["images"]!=500 or raw["gt_words"]!=2077 or emitted["gt_words"]!=2077
            or ignored_total!=config["expected_ignored_gt"]):raise RuntimeError("Complete TEST care/ignore populations changed")
    result={"status":"SUCCESS","reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":"approximate reproduction","protocol":protocol,"raw_aggregate":raw,"aggregate":emitted,
        "rejected_outputs":raw["predictions"]-emitted["predictions"],"ignored_gt":ignored_total,"failed_samples":0,
        "per_image":rows,"device":args.device,"gpu":torch.cuda.get_device_name(device),"precision":"float32",
        "segmentation_peak_bytes":segmentation_peak,"ocr_peak_bytes":ocr_peak,"oom":False,
        "segmentation_input_shape":[1,3,*shape],"parseq_batch_size":batch,
        "runtime":runtime_environment(),"elapsed_seconds_this_invocation":time.monotonic()-started}
    path=output/"results.json"
    if path.exists():
        previous=json.loads(path.read_text())
        if any(previous[k]!=result[k] for k in ("protocol","aggregate","raw_aggregate","per_image")):
            raise ValueError("Completed TEST result changed")
        return previous
    write_json(path,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","selection","selection-profile","ocr-qualification","test-manifest","data-root",
                 "parseq-source","parseq-source-archive","parseq-checkpoint","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",required=True);parser.add_argument("--resume",action="store_true")
    parser.add_argument("--preflight-only",action="store_true");args=parser.parse_args()
    try:result=run(args)
    except Exception as error:
        initialized=torch.cuda.is_initialized()
        write_json(args.output_dir/"failure.json",{"status":"FAILED","reproduction_mode":"approximate","claim_label":"approximate reproduction",
            "error_type":type(error).__name__,"error":str(error),"device":args.device,"precision":"float32",
            "oom":isinstance(error,torch.cuda.OutOfMemoryError),"peak_allocated_bytes":torch.cuda.max_memory_allocated() if initialized else None,
            "peak_reserved_bytes":torch.cuda.max_memory_reserved() if initialized else None})
        raise
    print(json.dumps({k:result[k] for k in ("status","aggregate","raw_aggregate") if k in result}),flush=True)


if __name__=="__main__":main()
