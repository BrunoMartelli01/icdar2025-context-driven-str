"""Symbol-free spatial word F1 on frozen validation100 segmentation artifacts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from recognition.icdar2015_word_f1 import aggregate_counts, sample_counts
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer, verify_parseq_runtime_source,
)
from segmentation.data import file_sha256
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.icdar2015_improvements import binary_proposals, original_geometry, valid_canvas
from segmentation.icdar2015_improvement_runtime import load_config, load_training_population, references, REPO
from segmentation.training import runtime_environment, seed_everything
from scripts.evaluate_icdar2015_totaltext_transfer_word_f1 import crop_hash
from scripts.train_icdar2015_totaltext_transfer import write_json


def verify_geometry_counts(row, boxes, words, config):
    # Frozen v1 geometry rows contain counts. Complete checksum-pinned native
    # maps retain the boxes; regenerate them without changing training sources.
    measured=original_geometry(boxes,words,iou_threshold=config["matching_iou_threshold"])
    if any(measured[key]!=row[key] for key in ("proposals","matched_gt","gt_words")):
        raise ValueError("Frozen validation geometry replay differs")


def run(args):
    start=time.monotonic();config=load_config(args.config)
    root=args.data_root.resolve();manifest,fit,val=load_training_population(config,root)
    refs=references(manifest,root,val)
    if args.training_run is not None:
        origin=args.training_run/"run_manifest.json";trained=json.loads(origin.read_text())
        if (trained["status"]!="SUCCESS" or trained["training_complete"] is not True
                or trained["completed_epochs"]!=config["training"]["epochs"]
                or trained["run_contract"]["config"]!=config):raise ValueError("Completed matched training required")
        if file_sha256(args.training_run/"best.pt")!=trained["checkpoint"]["sha256"]:
            raise ValueError("Selected checkpoint checksum mismatch")
        for name,digest in trained["run_contract"]["source_sha256"].items():
            if file_sha256(REPO/name)!=digest:raise ValueError("Training source changed: "+name)
        shape=trained["run_contract"]["image_hw"];branch=trained["run_contract"]["branch"]
        details=args.training_run/"validation_geometry.json";mask_dir=args.training_run/"validation_masks"
    else:
        if args.screen_resolution is None:raise ValueError("Screen baseline resolution required")
        origin=args.screening_dir/"results.json";screen=json.loads(origin.read_text())
        if screen["status"]!="SUCCESS" or screen["run_contract"]["config"]!=config:
            raise ValueError("Qualified frozen-weight screening required")
        shape=config["resolutions"][args.screen_resolution];branch="source_"+args.screen_resolution
        details=args.screening_dir/args.screen_resolution/"results.json";mask_dir=details.parent/"masks"
    geometry=json.loads(details.read_text());rows=geometry["per_image"]
    if len(rows)!=len(val) or {row["sample_id"] for row in rows}!=set(refs):
        raise ValueError("Full fixed validation100 required")
    if (file_sha256(args.parseq_checkpoint)!=PARSEQ_CHECKPOINT_SHA256
            or file_sha256(args.parseq_source_archive)!=PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("Pinned PARSeq artifacts changed")
    runtime=verify_parseq_runtime_source(args.parseq_source,source_type=PARSEQ_ARCHIVE_SOURCE,
                                        source_archive=args.parseq_source_archive)
    protocol={"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":"TRAIN validation100 correct-word F1", "config_sha256":file_sha256(args.config),
        "origin_report_sha256":file_sha256(origin),"geometry_report_sha256":file_sha256(details),
        "branch":branch,"image_hw":shape,"runtime_source":runtime,
        "probability_threshold":config["probability_threshold"],
        "matching_iou_threshold":config["matching_iou_threshold"],
        "validation_ids":[manifest.samples[i].sample_id for i in val],
        "paths":{"data_root":str(root),"manifest":str(root/config["datasets"]["icdar2015"]["manifest"]),
            "geometry_report":str(details.resolve()),"mask_dir":str(mask_dir.resolve())},
        "parseq_checkpoint_sha256":PARSEQ_CHECKPOINT_SHA256,
        "parseq_batch_size":config["parseq_batch_size"],"device":args.device,"precision":"float32","tf32":False,
        "source_sha256":{name:file_sha256(REPO/name) for name in (
            "scripts/evaluate_icdar2015_improvement_validation.py","recognition/icdar2015_word_f1.py",
            "scripts/evaluate_icdar2015_totaltext_transfer_word_f1.py",
            "recognition/deepsolo_word_f1.py","recognition/parseq.py","segmentation/icdar2015_improvements.py",
            "segmentation/icdar2015_improvement_runtime.py","recognition/scene_text_datasets.py")},
        "metric":"one-to-one original bbox IoU>=0.5 and exact Unicode casefold+isalnum text",
        "ignore_policy":"all_predictions_count; official_ignored_GT_excluded; all_care_GT_retained"}
    digest=sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    p=args.output_dir/"protocol.json"
    if p.exists():
        if json.loads(p.read_text())!=protocol:raise ValueError("Validation OCR protocol changed")
    else:write_json(p,protocol)
    by_id={sample.sample_id:sample for sample in manifest.samples}
    proposals={};rgb_paths={};unique={}
    for row in rows:
        sample_id=row["sample_id"];sample=by_id[sample_id]
        rgb_paths[sample_id]=root/sample.image_path
        if file_sha256(rgb_paths[sample_id])!=sample.image_sha256:raise ValueError("Validation RGB changed")
        path=mask_dir/row["mask_file"]
        if file_sha256(path)!=row["mask_sha256"]:raise ValueError("Frozen validation mask changed")
        with np.load(path) as maps:
            prediction=maps["prediction"];probability=maps["probability"]
            expected=(probability>=config["probability_threshold"]) & valid_canvas(refs[sample_id]["image_wh"],shape)
            if not np.array_equal(prediction,expected):raise ValueError("Validation mask threshold/canvas mismatch")
        boxes,_=binary_proposals(prediction,refs[sample_id]["image_wh"],shape)
        verify_geometry_counts(row,boxes,refs[sample_id]["ground_truth"],config)
        with Image.open(rgb_paths[sample_id]) as im:rgb=np.array(im.convert("RGB"))
        for box in boxes:
            x1,y1,x2,y2=box["bbox"];key=crop_hash(rgb[y1:y2,x1:x2]);box["crop_rgb_sha256"]=key
            unique.setdefault(key,(sample_id,box))
        proposals[sample_id]=boxes
    device=torch.device(args.device)
    seed_everything(config["training"]["seed"],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device.type=="cuda":_prepare_cuda_memory_tracking(device)
    pending=[]
    for key,item in unique.items():
        p=args.output_dir/"ocr"/(key+".json")
        if p.exists():
            cache=json.loads(p.read_text())
            if cache["protocol_sha256"]!=digest or cache["crop_rgb_sha256"]!=key:
                raise ValueError("OCR cache protocol mismatch")
        else:pending.append((key,item))
    if pending:
        recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
            checkpoint=args.parseq_checkpoint,device=args.device,runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_source_archive))
        for offset in range(0,len(pending),config["parseq_batch_size"]):
            batch=pending[offset:offset+config["parseq_batch_size"]];crops=[]
            for key,(sample_id,box) in batch:
                with Image.open(rgb_paths[sample_id]) as im:rgb=np.array(im.convert("RGB"))
                x1,y1,x2,y2=box["bbox"];crop=np.ascontiguousarray(rgb[y1:y2,x1:x2])
                if crop_hash(crop)!=key:raise ValueError("Crop changed")
                crops.append(crop)
            predictions=recognizer.recognize_candidate_batch(crops)
            if len(predictions)!=len(batch):raise RuntimeError("Incomplete PARSeq batch")
            for (key,_),prediction in zip(batch,predictions):
                write_json(args.output_dir/"ocr"/(key+".json"),{"protocol_sha256":digest,
                    "crop_rgb_sha256":key,**prediction.to_dict()})
            if offset%320==0 or offset+len(batch)==len(pending):
                print(f"[validation OCR {branch}] {offset+len(batch)}/{len(pending)}",flush=True)
        recognizer.release()
    output=[]
    for row in rows:
        key=row["sample_id"];predictions=[]
        for box in proposals[key]:
            p=args.output_dir/"ocr"/(box["crop_rgb_sha256"]+".json")
            predictions.append({**box,"text":json.loads(p.read_text())["raw_text"],"ocr_sha256":file_sha256(p)})
        words=refs[key]["ground_truth"]
        output.append({"sample_id":key,"image_sha256":by_id[key].image_sha256,
            "annotation_sha256":refs[key]["annotation_sha256"],"predictions":predictions,
            "ground_truth":words,"counts":sample_counts(predictions,words,iou_threshold=config["matching_iou_threshold"])})
    aggregate=aggregate_counts([row["counts"] for row in output])
    if (aggregate["images"]!=100 or aggregate["gt_words"]!=430
            or aggregate["geometry"]["f1"]!=geometry["aggregate"]["proposal_f1_iou_50"]):
        raise ValueError("Validation population/geometry changed")
    report={"status":"SUCCESS","reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "protocol":protocol,"aggregate":aggregate,"per_image":output,"failed_samples":0,
        "runtime":runtime_environment(),"oom":False,"elapsed_seconds_this_invocation":time.monotonic()-start,
        "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
        "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None}
    path=args.output_dir/"results.json"
    if path.exists():
        previous=json.loads(path.read_text())
        if any(previous[k]!=report[k] for k in ("protocol","aggregate","per_image")):
            raise ValueError("Completed validation OCR changed")
        return previous
    write_json(path,report);print(json.dumps(aggregate,sort_keys=True),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","data-root","parseq-source","parseq-source-archive","parseq-checkpoint","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    origin=parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--training-run",type=Path)
    origin.add_argument("--screening-dir",type=Path)
    parser.add_argument("--screen-resolution",choices=("square512","square768","square1024","wide1024"))
    parser.add_argument("--device",required=True)
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        write_json(args.output_dir/"failure.json",{"status":"FAILED","error_type":type(error).__name__,
            "error":str(error),"oom":isinstance(error,torch.cuda.OutOfMemoryError)})
        raise


if __name__=="__main__":main()
