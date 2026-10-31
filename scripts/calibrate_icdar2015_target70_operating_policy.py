"""Calibrate a predeclared U-Net operating grid on complete fixed TRAIN VAL100."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from recognition.icdar2015_word_f1 import sample_counts,aggregate_counts
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE,ParseqApproxConfig,ParseqApproxRecognizer
from scripts.evaluate_icdar2015_totaltext_transfer_word_f1 import crop_hash
from scripts.train_icdar2015_totaltext_transfer import write_json
from segmentation.data import file_sha256
from segmentation.icdar2015_improvement_runtime import REPO,load_training_population,references
from segmentation.icdar2015_improvements import binary_proposals,valid_canvas
from segmentation.icdar2015_operating_policy import component_crop,operating_selection_key
from segmentation.icdar2015_target70 import load_profile,ValidationOCR,calibrate_emission


def recognize(crops,ocr):
    pending={}
    for crop in crops:
        digest=crop_hash(crop);path=ocr.cache/(digest+".json")
        if path.exists():
            old=json.loads(path.read_text())
            if old["crop_rgb_sha256"]!=digest or old["ocr_protocol_sha256"]!=ocr.protocol_sha:
                raise ValueError("Cached crop protocol changed")
        else:pending.setdefault(digest,crop)
    items=list(pending.items());size=ocr.config["parseq_batch_size"]
    if items and ocr.recognizer is None:
        args=ocr.args
        ocr.recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
            checkpoint=args.parseq_checkpoint,device="cpu",runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_source_archive))
    for start in range(0,len(items),size):
        batch=items[start:start+size];outputs=ocr.recognizer.recognize_candidate_batch([rgb for _,rgb in batch])
        if len(outputs)!=len(batch):raise RuntimeError("Incomplete operating-policy OCR")
        for (digest,_),output in zip(batch,outputs):
            write_json(ocr.cache/(digest+".json"),{"crop_rgb_sha256":digest,"ocr_protocol_sha256":ocr.protocol_sha,**output.to_dict()})


def run(args):
    config=json.loads(args.config.read_text())
    if (args.config.resolve().parent.name!="approximate" or config["reproduction_mode"]!="approximate"
            or config["claim_label"]!="approximate reproduction"):
        raise ValueError("Explicit approximate operating profile required")
    path=REPO/config["training_profile"]
    if file_sha256(path)!=config["training_profile_sha256"]:raise ValueError("Training profile changed")
    training=load_profile(path);root=args.data_root.resolve();shape=args.image_hw
    manifest,_,val=load_training_population(training,root);refs=references(manifest,root,val)
    details=json.loads(args.geometry.read_text());expected=json.loads(args.expected_raw_report.read_text())["aggregate"]
    if len(details["per_image"])!=100 or {r["sample_id"] for r in details["per_image"]}!=set(refs):
        raise ValueError("Complete fixed TRAIN VAL100 required")
    if args.output_dir.exists():raise ValueError("Preserve existing operating-policy output")
    contract={"config":config,"config_sha256":file_sha256(args.config),"geometry_sha256":file_sha256(args.geometry),
        "expected_raw_sha256":file_sha256(args.expected_raw_report),"image_hw":shape,
        "source_sha256":{p:file_sha256(REPO/p) for p in ("segmentation/icdar2015_operating_policy.py",
            "scripts/calibrate_icdar2015_target70_operating_policy.py","segmentation/icdar2015_target70.py")}}
    write_json(args.output_dir/"run_contract.json",contract)
    ocr=ValidationOCR(args,training);samples={s.sample_id:s for s in manifest.samples};maps={};images={}
    for row in details["per_image"]:
        key=row["sample_id"];path=args.mask_dir/row["mask_file"]
        if file_sha256(path)!=row["mask_sha256"]:raise ValueError("Source probability map changed")
        with np.load(path) as saved:
            probability=saved["probability"].copy();prediction=saved["prediction"].copy()
        if probability.shape!=tuple(d//2 for d in shape):raise ValueError("Native shape mismatch")
        if not np.array_equal(prediction,(probability>=.5)&valid_canvas(refs[key]["image_wh"],shape)):
            raise ValueError("Source threshold/canvas replay changed")
        maps[key]=probability;path=root/samples[key].image_path
        if file_sha256(path)!=samples[key].image_sha256:raise ValueError("Validation RGB changed")
        with Image.open(path) as image:images[key]=np.asarray(image.convert("RGB"))
    candidates=[]
    for threshold in config["probability_thresholds"]:
        components={key:binary_proposals((maps[key]>=threshold)&valid_canvas(refs[key]["image_wh"],shape),
            refs[key]["image_wh"],shape) for key in refs}
        for method in config["crop_methods"]:
            rows=[];crops=[]
            for key,(proposals,labels) in components.items():
                predictions=[]
                for proposal in proposals:
                    crop=component_crop(images[key],labels,proposal,shape,method)
                    if not crop.size:raise ValueError("Empty crop; no output omissions")
                    digest=crop_hash(crop);crops.append(crop)
                    predictions.append({**proposal,"crop_rgb_sha256":digest})
                rows.append({"sample_id":key,"ground_truth":refs[key]["ground_truth"],"predictions":predictions})
            recognize(crops,ocr)
            for row in rows:
                for p in row["predictions"]:
                    path=ocr.cache/(p["crop_rgb_sha256"]+".json");out=json.loads(path.read_text())
                    p.update(text=out["raw_text"],sequence_confidence=out["sequence_confidence_native"],
                        mean_token_log_probability=out["mean_token_log_probability"],ocr_sha256=file_sha256(path))
                row["counts"]=sample_counts(row["predictions"],row["ground_truth"],iou_threshold=config["matching_iou_threshold"])
            raw=aggregate_counts([r["counts"] for r in rows]);calibration=calibrate_emission(rows,training)
            if threshold==.5 and method=="axis0" and raw!=expected:raise ValueError("Qualified raw source replay differs")
            candidate={"probability_threshold":threshold,"crop_method":method,"raw_aggregate":raw,
                "emitted_aggregate":calibration["aggregate"],"emission_policy":calibration["policy"]}
            directory=args.output_dir/f"threshold_{threshold}_{method}";write_json(directory/"per_image.json",rows)
            write_json(directory/"calibration.json",calibration)
            candidate.update(per_image_file=str((directory/"per_image.json").relative_to(args.output_dir)),
                per_image_sha256=file_sha256(directory/"per_image.json"))
            candidates.append(candidate);write_json(args.output_dir/"candidates.json",candidates)
            print(f"threshold={threshold} crop={method} raw_F1={raw['alphanumeric_casefold']['f1']:.6f} emitted_F1={calibration['aggregate']['alphanumeric_casefold']['f1']:.6f}",flush=True)
    selected=max(candidates,key=operating_selection_key)
    write_json(args.output_dir/"results.json",{"status":"SUCCESS","reproduction_mode":"approximate",
        "claim_label":"approximate reproduction","result_class":"TRAIN validation operating-policy calibration",
        "run_contract":contract,"candidates":candidates,"selected":selected,"ocr_protocol":ocr.protocol,"failed_samples":0})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","data-root","geometry","mask-dir","expected-raw-report","parseq-source",
                 "parseq-source-archive","parseq-checkpoint","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--image-hw",nargs=2,type=int,required=True)
    run(parser.parse_args())


if __name__=="__main__":main()
