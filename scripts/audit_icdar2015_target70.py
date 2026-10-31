"""Independent CSV/SciPy word metrics and complete raw/emitted TEST populations."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import label,find_objects
import yaml

from scripts.audit_icdar2015_word_f1 import independent_tp
from scripts.train_icdar2015_totaltext_transfer import write_json
from segmentation.icdar2015_operating_policy import component_crop


def digest(path):return sha256(path.read_bytes()).hexdigest()


def independently_emitted(prediction,policy):
    if policy["confidence"]=="sequence_confidence":confidence=prediction["sequence_confidence"]
    elif policy["confidence"]=="mean_token_confidence":
        logp=prediction["mean_token_log_probability"];confidence=math.exp(logp) if logp is not None else 0.
    else:raise ValueError("Unknown frozen confidence field")
    normalized="".join(character for character in prediction["text"].casefold() if character.isalnum())
    return confidence>=policy["threshold"] and len(normalized)>=policy["minimum_alphanumeric_length"]


def aggregate_metric(tp,predictions,words):
    return {"tp":tp,"fp":predictions-tp,"fn":words-tp,"precision":tp/predictions if predictions else None,
        "recall":tp/words if words else None,"f1":2*tp/(predictions+words) if predictions+words else None}


def audit(path,root):
    result=json.loads(path.read_text());protocol=result["protocol"];config=protocol["evaluation_config"]
    if (result["status"]!="SUCCESS" or result["reproduction_mode"]!="approximate" or result["failed_samples"]!=0
            or config["expected_images"]!=500 or config["expected_care_words"]!=2077
            or config["matching_iou_threshold"]!=.5 or config["connectivity"]!=8
            or config["prediction_ignore_policy"]!="all_outputs_count_including_ignored_regions"):
        raise ValueError("Complete successful unchanged TEST500 evaluation required")
    manifest=Path(protocol["paths"]["test_manifest"])
    if digest(manifest)!=config["test_manifest_sha256"]:raise ValueError("Official TEST manifest changed")
    selection=Path(protocol["paths"]["selection"])
    if digest(selection)!=protocol["selection_sha256"]:raise ValueError("Frozen selection changed")
    frozen=json.loads(selection.read_text())
    if frozen["selected"]!=protocol["selected"] or frozen["test_labels_used_for_selection"] is not False:
        raise ValueError("TEST outputs do not match the frozen TRAIN selection")
    chosen=protocol["selected"]
    if digest(Path(chosen["checkpoint"]))!=chosen["checkpoint_sha256"]:raise ValueError("Frozen checkpoint changed")
    protocol_sha=sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    population=yaml.safe_load(manifest.read_text());samples={s["id"]:s for s in population["samples"]}
    rows=result["per_image"]
    if len(samples)!=500 or len(rows)!=500 or {r["sample_id"] for r in rows}!=set(samples):
        raise ValueError("Every official TEST image must remain in the population")
    modes=("geometry","casefold_exact","alphanumeric_casefold")
    totals={population:{mode:0 for mode in modes} for population in ("raw","emitted")}
    counts={"raw":0,"emitted":0};gt_total=ignored_total=crops_verified=0
    shape=config["image_hw"];nh,nw=shape[0]//2,shape[1]//2
    for row in rows:
        key=row["sample_id"];sample=samples[key];rgb_path=root/sample["image"]
        if digest(rgb_path)!=sample["image_sha256"]:raise ValueError("Official RGB changed")
        with Image.open(rgb_path) as image:rgb=np.asarray(image.convert("RGB"))
        h,w=rgb.shape[:2];divisor=math.gcd(*shape);rw,rh=shape[1]//divisor,shape[0]//divisor
        scale=max((w+rw-1)//rw,(h+rh-1)//rh);cw,ch=rw*scale,rh*scale;left,top=(cw-w)//2,(ch-h)//2
        canvas=Image.new("L",(cw,ch),0);canvas.paste(255,(left,top,left+w,top+h))
        canvas=np.asarray(canvas.resize((nw,nh),Image.Resampling.NEAREST))>0
        shard=json.loads((path.parent/"samples"/(key.replace(":","_")+".json")).read_text())
        if (shard["protocol_sha256"]!=protocol_sha or shard["image_sha256"]!=sample["image_sha256"]
                or shard["image_wh"]!=[w,h]):raise ValueError("Frozen segmentation identity differs")
        mask_path=path.parent/shard["mask_path"]
        if digest(mask_path)!=shard["mask_sha256"]:raise ValueError("Frozen mask bytes changed")
        with np.load(mask_path) as saved:probability,prediction=saved["probability"],saved["prediction"]
        if prediction.shape!=(nh,nw) or not np.array_equal(prediction,(probability>=config["probability_threshold"])&canvas):
            raise ValueError("Independent threshold/canvas reconstruction differs")
        labels,number=label(prediction,structure=np.ones((3,3)))
        measured=[]
        for component,slices in enumerate(find_objects(labels),1):
            yy,xx=slices;measured.append((int((labels[slices]==component).sum()),(xx.start,yy.start,xx.stop,yy.stop)))
        raw=row["raw_predictions"]
        reported=[(p["component_area"],tuple(p["bbox_native"])) for p in raw]
        if len(raw)!=number or Counter(measured)!=Counter(reported):raise ValueError("Independent raw component populations differ")
        if [p["component_area"] for p in raw]!=sorted([p["component_area"] for p in raw],reverse=True):
            raise ValueError("Raw components are not sorted by pixel area")
        # The renderer is explicitly shared. Population, boxes, CSV parsing,
        # output rejection and maximum matching above/below are independent.
        _,render_labels,_,_=cv2.connectedComponentsWithStats(prediction.astype(np.uint8),connectivity=8)
        for p in raw:
            a,b,c,d=p["bbox_native"]
            box=[max(0,a*cw//nw-left),max(0,b*ch//nh-top),
                min(w,(c*cw+nw-1)//nw-left),min(h,(d*ch+nh-1)//nh-top)]
            if p["bbox"]!=box:raise ValueError("Independent inverse coordinate mapping differs")
            crop=component_crop(rgb,render_labels,p,shape,config["crop_method"])
            expected=sha256(np.asarray(crop.shape,dtype="<i8").tobytes()+np.ascontiguousarray(crop).tobytes()).hexdigest()
            if expected!=p["crop_rgb_sha256"]:raise ValueError("Frozen crop byte replay differs")
            ocr=path.parent/"ocr"/(expected+".json")
            if digest(ocr)!=p["ocr_file_sha256"]:raise ValueError("Frozen OCR bytes changed")
            decoded=json.loads(ocr.read_text())
            if (decoded["protocol_sha256"]!=protocol_sha or decoded["crop_rgb_sha256"]!=expected
                    or decoded["raw_text"]!=p["text"] or decoded["sequence_confidence_native"]!=p["sequence_confidence"]
                    or decoded["mean_token_log_probability"]!=p["mean_token_log_probability"]):
                raise ValueError("Raw OCR transcript/confidence differs")
            crops_verified+=1
        expected_emitted=[p for p in raw if independently_emitted(p,config["emission_policy"])]
        if row["predictions"]!=expected_emitted:raise ValueError("Independent output-policy replay differs")
        annotation=rgb_path.parent.parent/"annotations"/(key.split(":",1)[1]+".txt")
        if digest(annotation)!=row["annotation_sha256"]:raise ValueError("Official annotation changed")
        words=[];ignored=0
        for fields in csv.reader(annotation.read_text(encoding="utf-8-sig").splitlines(),skipinitialspace=True):
            if not fields or all(not f.strip() for f in fields):continue
            points=[int(x) for x in fields[:8]];text=",".join(fields[8:]).strip()
            if text=="###":ignored+=1;continue
            xs,ys=points[::2],points[1::2]
            words.append({"text":text,"bbox":[min(xs),min(ys),min(w,max(xs)+1),min(h,max(ys)+1)]})
        if [{"text":x["text"],"bbox":x["bbox"]} for x in row["ground_truth"]]!=words or ignored!=row["ignored_gt_count"]:
            raise ValueError("Independent complete care/ignore parsing differs")
        for population,predictions,stored in (("raw",raw,row["raw_counts"]),("emitted",expected_emitted,row["counts"])):
            if stored["predictions"]!=len(predictions) or stored["gt_words"]!=len(words):raise ValueError("Per-image population differs")
            for mode in modes:
                tp=independent_tp(predictions,words,config["matching_iou_threshold"],mode)
                if tp!=stored[mode+"_tp"]:raise ValueError("Independent per-image maximum matching differs")
                totals[population][mode]+=tp
            counts[population]+=len(predictions)
        gt_total+=len(words);ignored_total+=ignored
    metrics={}
    if gt_total!=2077 or ignored_total!=3153 or ignored_total!=result["ignored_gt"]:raise ValueError("Official complete denominators changed")
    for population,summary in (("raw",result["raw_aggregate"]),("emitted",result["aggregate"])):
        if summary["images"]!=500 or summary["gt_words"]!=gt_total or summary["predictions"]!=counts[population]:
            raise ValueError("Aggregate populations differ")
        metrics[population]={mode:aggregate_metric(totals[population][mode],counts[population],gt_total) for mode in modes}
        if any(summary[mode]!=value for mode,value in metrics[population].items()):raise ValueError("Independent aggregate F1 differs")
    if result["rejected_outputs"]!=counts["raw"]-counts["emitted"]:raise ValueError("Rejected-output count differs")
    return {"status":"PASS","reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":"complete TEST500 independent metric/population audit","source_results_sha256":digest(path),
        "auditor_sha256":digest(Path(__file__)),"independent_matcher_sha256":digest(Path(__file__).with_name("audit_icdar2015_word_f1.py")),
        "images":500,"care_words":gt_total,"ignored_gt":ignored_total,"raw_predictions":counts["raw"],
        "emitted_predictions":counts["emitted"],"raw_crop_hashes_verified":crops_verified,"metrics":metrics,
        "independent_checks":["CSV annotations","SciPy components","inverse coordinates","output-policy membership","SciPy matching","micro F1"],
        "shared_crop_renderer":"segmentation.icdar2015_operating_policy.component_crop; byte replay, not independent rendering",
        "neural_inference_recomputed":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("results","data-root","output"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError("Preserve existing audit")
    result=audit(args.results,args.data_root.resolve());write_json(args.output,result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
