"""Post-hoc fixed-final-epoch OCR diagnostic, preserving primary selection/results."""

import argparse
import gc
import json
from pathlib import Path
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from models.adaspot_gap_context import GapContextHead
from localize.word_multitask import parseq_word_proposals
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE, ParseqApproxConfig, ParseqApproxRecognizer
from scripts.diagnose_adaspot_phase0 import REPO, check_hash, paired_changes, summarize, write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_adaspot_gap_context import load_validation, memory, native_result


def run(args):
    path=args.config.resolve()
    if args.mode!="approximate" or REPO/"configs/approximate" not in path.parents:
        raise ValueError("Explicit APPROXIMATE namespace required")
    config=json.loads(path.read_text())
    if config["analysis_kind"]!="post_hoc_fixed_final_epoch_diagnostic" or config["reproduction_mode"]!=args.mode:
        raise ValueError("Explicit post-hoc final diagnostic required")
    root,out=args.run_root.resolve(),args.output_root.resolve()
    device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("GPU required; no CPU fallback")
    torch.cuda.init()
    if out==REPO or REPO in out.parents or out==root or root in out.parents:raise ValueError("Separate external diagnostic output required")
    primary=json.loads((root/"results.json").read_text());protocol=primary["protocol"]
    c=protocol["resolved_config"]
    if primary["status"]!="SUCCESS" or c["epochs"]!=config["expected_final_epoch"]:raise ValueError("Complete fixed-budget primary result required")
    hashes=dict(protocol["source_hashes"])
    for p,h in hashes.items():check_hash(Path(p),h)
    for p in (root/"results.json",path,Path(__file__).resolve()):hashes[str(p)]=sha256_file(p)
    out.mkdir(parents=True,exist_ok=False);args._output_created=True
    torch.set_num_threads(c["cpu_threads"]);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    previous=Path(protocol["previous_probe"]);source=Path(protocol["source_run"])
    cached=json.loads((previous/"feature_manifest.json").read_text())
    validation=load_validation(cached["validation"],source)
    baseline=json.loads((Path(protocol["phase0_run"])/"results.json").read_text())["arms"]["pred/axis/ar1"]
    ground_truth={s["sample_id"]:s["ground_truth"] for s in baseline["per_image"]}
    frozen,unique,native={},{},{}
    torch.cuda.reset_peak_memory_stats(device)
    for name,arm in primary["arms"].items():
        checkpoint=root/name/"last.pt";hashes[str(checkpoint)]=sha256_file(checkpoint)
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if state["epoch"]!=config["expected_final_epoch"]:raise ValueError("Last checkpoint epoch differs")
        head=GapContextHead(arm["kernel_size"],initial_bias=c["initial_bias"]).to(device)
        head.load_state_dict(state["head"],strict=True)
        frozen[name]=[];native[name]=[]
        for item in validation:
            sid=item["sample_id"]
            row,gap,labels=native_result(head,item,protocol["source_contract"]["resolved_config"]["postprocessing"],device)
            native[name].append(row)
            with Image.open(item["proposal"]["image_path"]) as image:rgb=np.asarray(image.convert("RGB"))
            predictions=[]
            for p in parseq_word_proposals(labels,(rgb.shape[1],rgb.shape[0]),padding=0,min_area=0,max_instances=None):
                if p["instance_id"] in row["ignored_ids"]:continue
                x0,y0,x1,y1=p["bbox"];crop=np.ascontiguousarray(rgb[y0:y1,x0:x1]);digest=crop_array_hash(crop)
                dest=out/"crops"/(digest+".png");dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():Image.fromarray(crop).save(dest)
                unique[digest]=dest;predictions.append({**p,"crop_rgb_sha256":digest})
            frozen[name].append({"sample_id":sid,"ground_truth":ground_truth[sid],"predictions":predictions})
            dest=out/name/"predicted_maps"/(sid.replace(":","__")+".npz");dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,gap_probability=gap,labels=labels)
    heads_memory=memory(device,"one fixed final Gap head at a time",[64,256,256])
    write_json(out/"frozen_inputs.json",frozen)
    del head,state,validation;gc.collect();torch.cuda.empty_cache()
    r=None;readings={}
    try:
        r=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(protocol["parseq_runtime_source"]),
            checkpoint=Path(protocol["parseq_checkpoint"]),runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=Path(protocol["parseq_runtime_archive"]),device=str(device)))
        torch.cuda.reset_peak_memory_stats(device);jobs=sorted(unique.items())
        for start in range(0,len(jobs),c["image_batch_size"]):
            batch=jobs[start:start+c["image_batch_size"]];crops=[]
            for digest,path in batch:
                with Image.open(path) as image:crop=np.asarray(image.convert("RGB"))
                if crop_array_hash(crop)!=digest:raise ValueError("Frozen crop changed")
                crops.append(crop)
            for (digest,_),p in zip(batch,r.recognize_candidate_batch(crops)):readings[digest]=p.to_dict()
        ocr_memory=memory(device,"qualified PARSeq AR1 only",[c["image_batch_size"],3,32,128])
    finally:
        if r is not None:r.release()
    write_json(out/"ocr_readings.json",readings)
    results={};order={s["sample_id"]:i for i,s in enumerate(baseline["per_image"])}
    for name,samples in frozen.items():
        for sample in samples:sample["predictions"]=[{**p,"text":readings[p["crop_rgb_sha256"]]["raw_text"]} for p in sample["predictions"]]
        samples.sort(key=lambda s:order[s["sample_id"]]);scored=summarize(samples)
        if scored["aggregate"]["gt_words"]!=c["expected_words"]:raise ValueError("Care denominator changed")
        results[name]={"epoch":config["expected_final_epoch"],"ocr":scored,"native_per_image":native[name],
            "changes_vs_baseline":paired_changes(baseline,scored)}
    for p,h in hashes.items():check_hash(Path(p),h)
    write_json(out/"results.json",{"status":"SUCCESS","result_class":"approximate reproduction",
        "analysis_kind":config["analysis_kind"],"resolved_config":config,"source_hashes":hashes,"arms":results,
        "primary_results_unchanged":True,"memory":{"heads":heads_memory,"ocr":ocr_memory}})
    lines=["# Fixed epoch20 — post-hoc diagnostic, approximate reproduction","",
        "No checkpoint selection/promotion. Primary experiment results remain unchanged. Reused TRAIN holdout125/all983words.","",
        "| Trained head at epoch20 | Exact /983 | Localized RGB | Proposals | Hmean | Recovered / damaged vs baseline524 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name,result in results.items():
        a,ch=result["ocr"]["aggregate"],result["changes_vs_baseline"]
        lines.append(f"| {name} | {a['exact_casefold_iou_ge_50']} | {a['geometry_iou_ge_50']} | {a['proposals']} | {a['hmean']:.4%} | {ch['recovered']} / {ch['damaged']} |")
    (out/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v["ocr"]["aggregate"] for k,v in results.items()},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=["approximate"],required=True)
    for name in ("config","run-root","output-root"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",required=True);args=p.parse_args()
    try:run(args)
    except Exception as error:
        if getattr(args,"_output_created",False):
            failure={"status":"FAILED","error":str(error),"traceback":traceback.format_exc(),"oom":isinstance(error,torch.cuda.OutOfMemoryError)}
            if torch.cuda.is_available():failure.update(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(args.output_root/"failure.json",failure)
        raise
