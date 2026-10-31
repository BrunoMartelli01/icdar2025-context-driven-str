"""Full fixed-TRAIN-validation CPU/GPU OCR and output-policy equivalence."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from recognition.icdar2015_word_f1 import aggregate_counts,sample_counts
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE,PARSEQ_CHECKPOINT_SHA256,
    ParseqApproxConfig,ParseqApproxRecognizer,verify_parseq_runtime_source)
from scripts.evaluate_icdar2015_totaltext_transfer_word_f1 import crop_hash
from scripts.train_icdar2015_totaltext_transfer import write_json
from segmentation.data import file_sha256
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.icdar2015_improvement_runtime import REPO,load_training_population,references
from segmentation.icdar2015_improvements import binary_proposals,valid_canvas
from segmentation.icdar2015_operating_policy import component_crop
from segmentation.icdar2015_target70 import emitted_predictions,load_profile
from segmentation.training import seed_everything,runtime_environment


def prediction_fields(output):
    return {"text":output["raw_text"],"sequence_confidence":output["sequence_confidence_native"],
        "mean_token_log_probability":output["mean_token_log_probability"]}


def compare_outputs(cpu,gpu,policy):
    if cpu["raw_text"]!=gpu["raw_text"]:raise ValueError("CPU/GPU OCR text differs")
    if list(cpu["token_ids"])!=list(gpu["token_ids"]):raise ValueError("CPU/GPU token IDs differ")
    a,b=prediction_fields(cpu),prediction_fields(gpu)
    if bool(emitted_predictions([a],policy))!=bool(emitted_predictions([b],policy)):
        raise ValueError("CPU/GPU output policy decision differs")
    differences={}
    for field in ("sequence_confidence_native","mean_token_log_probability"):
        left,right=cpu[field],gpu[field]
        if (left is None)!=(right is None):raise ValueError("CPU/GPU empty confidence differs")
        if left is not None:
            if not math.isfinite(left) or not math.isfinite(right):raise ValueError("Nonfinite OCR confidence")
            differences[field]=abs(left-right)
    return differences


def run(args):
    report=json.loads(args.operating_results.read_text());contract=report["run_contract"];config=contract["config"]
    if report["status"]!="SUCCESS" or report["reproduction_mode"]!="approximate" or report["claim_label"]!="approximate reproduction":
        raise ValueError("Completed approximate validation operating results required")
    if (args.config.resolve().parent.name!="approximate" or file_sha256(args.config)!=contract["config_sha256"]
            or json.loads(args.config.read_text())!=config):raise ValueError("Operating profile changed")
    profile=REPO/config["training_profile"]
    if file_sha256(profile)!=config["training_profile_sha256"]:raise ValueError("Training profile changed")
    training=load_profile(profile);root=args.data_root.resolve();shape=contract["image_hw"]
    for name,digest in contract["source_sha256"].items():
        if file_sha256(REPO/name)!=digest:raise ValueError("Operating source changed: "+name)
    if file_sha256(args.geometry)!=contract["geometry_sha256"]:raise ValueError("Geometry report changed")
    selected=report["selected"];path=args.operating_results.parent/selected["per_image_file"]
    if file_sha256(path)!=selected["per_image_sha256"]:raise ValueError("Selected validation outputs changed")
    rows=json.loads(path.read_text());manifest,_,val=load_training_population(training,root);refs=references(manifest,root,val)
    if len(rows)!=100 or {r["sample_id"] for r in rows}!=set(refs):raise ValueError("Complete fixed VAL100 required")
    runtime=verify_parseq_runtime_source(args.parseq_source,source_type=PARSEQ_ARCHIVE_SOURCE,source_archive=args.parseq_source_archive)
    cpu_protocol=report["ocr_protocol"]
    if (cpu_protocol["device"]!="cpu" or cpu_protocol["precision"]!="float32" or cpu_protocol["runtime"]!=runtime
            or cpu_protocol["checkpoint_sha256"]!=PARSEQ_CHECKPOINT_SHA256
            or file_sha256(args.parseq_checkpoint)!=PARSEQ_CHECKPOINT_SHA256
            or cpu_protocol["adapter_sha256"]!=file_sha256(REPO/"recognition/parseq.py")
            or cpu_protocol["batch_size"]!=training["parseq_batch_size"]):raise ValueError("Qualified CPU OCR protocol changed")
    cpu_sha=sha256(json.dumps(cpu_protocol,sort_keys=True).encode()).hexdigest()
    device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available():raise ValueError("Explicit available CUDA device required")
    if args.output_dir.exists():raise ValueError("Preserve previous qualification")
    protocol={"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "operating_results_sha256":file_sha256(args.operating_results),"geometry_sha256":file_sha256(args.geometry),
        "selected":selected,"device":args.device,"precision":"float32","tf32":False,
        "batch_size":training["parseq_batch_size"],"cpu_protocol":cpu_protocol,
        "paths":{key:str(getattr(args,key).resolve()) for key in ("data_root","mask_dir","cpu_cache","geometry","operating_results",
            "parseq_source","parseq_source_archive","parseq_checkpoint")},
        "source_sha256":{p:file_sha256(REPO/p) for p in ("scripts/qualify_icdar2015_target70_ocr_device.py",
            "segmentation/icdar2015_operating_policy.py","segmentation/icdar2015_target70.py","recognition/parseq.py")}}
    write_json(args.output_dir/"protocol.json",protocol)
    samples={s.sample_id:s for s in manifest.samples};details={r["sample_id"]:r for r in json.loads(args.geometry.read_text())["per_image"]}
    crops={};cpu_outputs={}
    for row in rows:
        key=row["sample_id"];sample=samples[key];path=root/sample.image_path
        if file_sha256(path)!=sample.image_sha256:raise ValueError("Validation RGB changed")
        with Image.open(path) as image:rgb=np.asarray(image.convert("RGB"))
        if row["ground_truth"]!=refs[key]["ground_truth"]:raise ValueError("Validation GT changed")
        path=args.mask_dir/details[key]["mask_file"]
        if file_sha256(path)!=details[key]["mask_sha256"]:raise ValueError("Probability map changed")
        with np.load(path) as maps:probability=maps["probability"].copy()
        mask=(probability>=selected["probability_threshold"])&valid_canvas(refs[key]["image_wh"],shape)
        proposals,labels=binary_proposals(mask,refs[key]["image_wh"],shape)
        if len(proposals)!=len(row["predictions"]):raise ValueError("Raw component population differs")
        for p,saved in zip(proposals,row["predictions"]):
            if any(p[k]!=saved[k] for k in p):raise ValueError("Raw component geometry differs")
            rgb_crop=component_crop(rgb,labels,p,shape,selected["crop_method"]);digest=crop_hash(rgb_crop)
            if digest!=saved["crop_rgb_sha256"]:raise ValueError("Selected RGB crop differs")
            path=args.cpu_cache/(digest+".json")
            if file_sha256(path)!=saved["ocr_sha256"]:raise ValueError("CPU OCR cache changed")
            cached=json.loads(path.read_text())
            if cached["ocr_protocol_sha256"]!=cpu_sha or cached["crop_rgb_sha256"]!=digest:
                raise ValueError("CPU crop protocol changed")
            if any(saved[k]!=v for k,v in prediction_fields(cached).items()):raise ValueError("Saved CPU prediction differs")
            crops.setdefault(digest,np.ascontiguousarray(rgb_crop));cpu_outputs[digest]=cached
    seed_everything(training["training"]["seed"],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    _prepare_cuda_memory_tracking(device)
    free,total=torch.cuda.mem_get_info(device)
    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,checkpoint=args.parseq_checkpoint,
        device=args.device,runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=args.parseq_source_archive))
    items=list(crops.items());batch=training["parseq_batch_size"];gpu_outputs={};differences=[]
    for start in range(0,len(items),batch):
        chunk=items[start:start+batch];outputs=recognizer.recognize_candidate_batch([rgb for _,rgb in chunk])
        if len(outputs)!=len(chunk):raise RuntimeError("Incomplete GPU OCR batch")
        for (digest,_),output in zip(chunk,outputs):
            gpu=output.to_dict();write_json(args.output_dir/"gpu_ocr"/(digest+".json"),gpu)
            differences.append(compare_outputs(cpu_outputs[digest],gpu,selected["emission_policy"]))
            gpu_outputs[digest]=gpu
    raw_counts=[];emitted_counts=[]
    for row in rows:
        predicted=[{**p,**prediction_fields(gpu_outputs[p["crop_rgb_sha256"]])} for p in row["predictions"]]
        raw_counts.append(sample_counts(predicted,row["ground_truth"],iou_threshold=config["matching_iou_threshold"]))
        emitted_counts.append(sample_counts(emitted_predictions(predicted,selected["emission_policy"]),row["ground_truth"],
            iou_threshold=config["matching_iou_threshold"]))
    raw,emitted=aggregate_counts(raw_counts),aggregate_counts(emitted_counts)
    if raw!=selected["raw_aggregate"] or emitted!=selected["emitted_aggregate"]:
        raise ValueError("CPU/GPU aggregate metrics differ")
    result={"status":"PASS","reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":"full TRAIN validation OCR device qualification","protocol":protocol,
        "images":len(rows),"care_words":raw["gt_words"],"raw_proposals":raw["predictions"],"unique_crops":len(items),
        "text_differences":0,"token_id_differences":0,"output_policy_differences":0,
        "max_absolute_confidence_differences":{k:max(d.get(k,0.) for d in differences) for k in
            ("sequence_confidence_native","mean_token_log_probability")},
        "raw_aggregate":raw,"emitted_aggregate":emitted,"failed_samples":0,"oom":False,
        "gpu":torch.cuda.get_device_name(device),"gpu_free_bytes_before_ocr":free,"gpu_total_bytes":total,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"peak_reserved_bytes":torch.cuda.max_memory_reserved(device),
        "model_set":["qualified frozen PARSeq"],"input_shape":[batch,3,*list(recognizer.model.hparams.img_size)],
        "runtime":runtime_environment()}
    write_json(args.output_dir/"results.json",result)
    recognizer.release();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("config","operating-results","geometry","mask-dir","data-root","cpu-cache","parseq-source",
                 "parseq-source-archive","parseq-checkpoint","output-dir"):
        parser.add_argument("--"+flag,type=Path,required=True)
    parser.add_argument("--device",required=True);args=parser.parse_args()
    try:result=run(args)
    except Exception as error:
        initialized=torch.cuda.is_initialized()
        write_json(args.output_dir/"failure.json",{"status":"FAILED","reproduction_mode":"approximate",
            "claim_label":"approximate reproduction","error_type":type(error).__name__,"error":str(error),
            "oom":isinstance(error,torch.cuda.OutOfMemoryError),"device":args.device,"precision":"float32",
            "peak_allocated_bytes":torch.cuda.max_memory_allocated() if initialized else None,
            "peak_reserved_bytes":torch.cuda.max_memory_reserved() if initialized else None})
        raise
    print(json.dumps({k:result[k] for k in ("status","images","care_words","raw_proposals","unique_crops",
        "text_differences","output_policy_differences","peak_allocated_bytes","peak_reserved_bytes")}),flush=True)


if __name__=="__main__":main()
