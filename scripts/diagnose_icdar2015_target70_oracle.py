"""Perfect-localization OCR ceiling on fixed TRAIN validation, never TEST."""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from recognition.deepsolo_word_f1 import normalize_word
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE,ParseqApproxConfig,ParseqApproxRecognizer
from recognition.scene_text_datasets import parse_icdar2015_annotation
from scripts.evaluate_icdar2015_totaltext_transfer_word_f1 import crop_hash
from scripts.train_icdar2015_totaltext_transfer import write_json
from segmentation.data import file_sha256
from segmentation.icdar2015_improvement_runtime import REPO,load_training_population,references
from segmentation.icdar2015_target70 import load_profile


def quad_crop(rgb,points):
    """Official ICDAR ordered TL,TR,BR,BL corners; bilinear, replicated border."""
    quad=np.asarray(points,dtype=np.float32)
    if quad.shape!=(4,2) or not cv2.isContourConvex(quad):
        raise ValueError("Expected a convex ordered official quadrilateral")
    width=int(np.ceil(max(np.linalg.norm(quad[1]-quad[0]),np.linalg.norm(quad[2]-quad[3]))))+1
    height=int(np.ceil(max(np.linalg.norm(quad[3]-quad[0]),np.linalg.norm(quad[2]-quad[1]))))+1
    if min(width,height)<2:raise ValueError("Degenerate quadrilateral crop")
    target=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype=np.float32)
    mapping=cv2.getPerspectiveTransform(quad,target)
    return cv2.warpPerspective(rgb,mapping,(width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def run(args):
    config=json.loads(args.config.read_text())
    if (args.config.resolve().parent.name!="approximate" or config["reproduction_mode"]!="approximate"
            or config["claim_label"]!="approximate reproduction" or config["device"]!="cpu"):
        raise ValueError("Explicit approximate CPU diagnostic required")
    profile=REPO/config["training_profile"]
    if file_sha256(profile)!=config["training_profile_sha256"]:raise ValueError("Training profile changed")
    training=load_profile(profile);root=args.data_root.resolve()
    manifest,_,val=load_training_population(training,root);refs=references(manifest,root,val)
    if len(val)!=config["expected_images"]:raise ValueError("Validation image count changed")
    if args.output_dir.exists():raise ValueError("Preserve existing oracle results")
    crops=[];rows=[]
    for index in val:
        sample=manifest.samples[index];key=sample.sample_id;path=root/sample.image_path
        if file_sha256(path)!=sample.image_sha256:raise ValueError("Validation RGB changed")
        with Image.open(path) as source:rgb=np.asarray(source.convert("RGB"))
        h,w=rgb.shape[:2]
        ann=root/Path(sample.image_path).parent.parent/"annotations"/(key.split(":",1)[1]+".txt")
        if file_sha256(ann)!=refs[key]["annotation_sha256"]:raise ValueError("Annotation changed")
        instances=[i for i in parse_icdar2015_annotation(ann,image_size_wh=(w,h)) if not i.ignore]
        if len(instances)!=len(refs[key]["ground_truth"]):raise ValueError("Care count changed")
        for number,(instance,gt) in enumerate(zip(instances,refs[key]["ground_truth"])):
            for method in config["methods"]:
                if method.startswith("axis"):
                    pad=int(method[4:]);x1,y1,x2,y2=gt["bbox"]
                    crop=rgb[max(0,y1-pad):min(h,y2+pad),max(0,x1-pad):min(w,x2+pad)]
                elif method=="quad0":crop=quad_crop(rgb,instance.polygon_xy)
                else:raise ValueError("Unknown predeclared method")
                if crop.size==0:raise ValueError("Empty crop; no omissions permitted")
                crops.append(np.ascontiguousarray(crop))
                rows.append({"sample_id":key,"word_index":number,"method":method,"ground_truth":gt,
                    "quad_xy":instance.polygon_xy,"image_sha256":sample.image_sha256,
                    "annotation_sha256":refs[key]["annotation_sha256"],"crop_rgb_sha256":crop_hash(crop)})
    if len(rows)!=config["expected_care_words"]*len(config["methods"]):raise ValueError("Oracle denominator changed")
    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
        checkpoint=args.parseq_checkpoint,device="cpu",runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
        runtime_source_archive=args.parseq_source_archive))
    start=time.monotonic();size=config["ocr_batch_size"]
    for i in range(0,len(crops),size):
        outputs=recognizer.recognize_candidate_batch(crops[i:i+size])
        if len(outputs)!=len(crops[i:i+size]):raise RuntimeError("Incomplete oracle recognition")
        for row,output in zip(rows[i:i+size],outputs):
            row["ocr"]=output.to_dict()
            row["correct"]=bool(normalize_word(output.raw_text)) and normalize_word(output.raw_text)==normalize_word(row["ground_truth"]["text"])
    aggregate={}
    for method in config["methods"]:
        subset=[r for r in rows if r["method"]==method];correct=sum(r["correct"] for r in subset)
        aggregate[method]={"words":len(subset),"correct":correct,"incorrect":len(subset)-correct,"accuracy":correct/len(subset)}
    report={"status":"SUCCESS","reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":config["result_class"],"deployable":False,"config":config,
        "config_sha256":file_sha256(args.config),"source_sha256":{p:file_sha256(REPO/p) for p in (
            "scripts/diagnose_icdar2015_target70_oracle.py","recognition/parseq.py","recognition/scene_text_datasets.py")},
        "checkpoint_sha256":file_sha256(args.parseq_checkpoint),"device":"cpu","precision":"float32",
        "images":len(val),"care_words":config["expected_care_words"],"failed_samples":0,
        "aggregate":aggregate,"elapsed_seconds":time.monotonic()-start,"per_word":rows}
    write_json(args.output_dir/"results.json",report)
    print(json.dumps({"status":"SUCCESS","deployable":False,"aggregate":aggregate}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","data-root","parseq-source","parseq-source-archive","parseq-checkpoint","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":main()
