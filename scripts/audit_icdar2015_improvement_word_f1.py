"""Independent CSV/SciPy audit of dynamic-resolution validation/TEST word F1."""

import argparse
import csv
from hashlib import sha256
import json
from math import gcd
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects, label
import yaml

from scripts.audit_icdar2015_word_f1 import independent_tp
from scripts.train_icdar2015_totaltext_transfer import write_json


def digest(path):return sha256(path.read_bytes()).hexdigest()


def audit(results,root):
    report=json.loads(results.read_text());protocol=report["protocol"]
    is_test="evaluation_config" in protocol
    if is_test:
        config=protocol["evaluation_config"];shape=config["image_hw"]
        manifest=Path(protocol["paths"]["test_manifest"])
        expected_images,expected_words=config["expected_images"],config["expected_care_words"]
        selection=Path(protocol["paths"]["selection_report"])
        if digest(selection)!=protocol["selection_report_sha256"]:raise ValueError("Frozen selection changed")
        selection_report=json.loads(selection.read_text())
        if selection_report["selected_branch"]!=protocol["selected_branch"]:raise ValueError("Unselected TEST branch")
        details=None
    else:
        config=protocol;shape=protocol["image_hw"];manifest=Path(protocol["paths"]["manifest"])
        expected_images,expected_words=100,430
        geometry_path=Path(protocol["paths"]["geometry_report"])
        if digest(geometry_path)!=protocol["geometry_report_sha256"]:raise ValueError("Validation geometry changed")
        details={r["sample_id"]:r for r in json.loads(geometry_path.read_text())["per_image"]}
    population=yaml.safe_load(manifest.read_text())
    by_id={row["id"]:row for row in population["samples"]}
    expected_ids=set(by_id) if is_test else set(protocol["validation_ids"])
    if {r["sample_id"] for r in report["per_image"]}!=expected_ids or len(report["per_image"])!=expected_images:
        raise ValueError("Full evaluation population required")
    totals={name:0 for name in ("geometry","casefold_exact","alphanumeric_casefold")}
    n_pred=n_gt=ignored=crops=0
    nh,nw=shape[0]//2,shape[1]//2
    for row in report["per_image"]:
        sample_id=row["sample_id"];sample=by_id[sample_id];rgb_path=root/sample["image"]
        if digest(rgb_path)!=sample["image_sha256"]:raise ValueError("RGB checksum changed")
        with Image.open(rgb_path) as im:rgb=np.array(im.convert("RGB"))
        h,w=rgb.shape[:2]
        divisor=gcd(*shape);rw,rh=shape[1]//divisor,shape[0]//divisor
        scale=max((w+rw-1)//rw,(h+rh-1)//rh);cw,ch=rw*scale,rh*scale
        left,top=(cw-w)//2,(ch-h)//2
        canvas=Image.new("L",(cw,ch),0);canvas.paste(255,(left,top,left+w,top+h))
        canvas=np.asarray(canvas.resize((nw,nh),Image.Resampling.NEAREST))>0
        if is_test:
            shard=json.loads((results.parent/"samples"/(sample_id.replace(":","_")+".json")).read_text())
            mask=results.parent/shard["mask_path"];expected_hash=shard["mask_sha256"]
        else:
            saved=details[sample_id];mask=Path(protocol["paths"]["mask_dir"])/saved["mask_file"]
            expected_hash=saved["mask_sha256"]
        if digest(mask)!=expected_hash:raise ValueError("Mask file changed")
        with np.load(mask) as maps:probability,prediction=maps["probability"],maps["prediction"]
        if prediction.shape!=(nh,nw) or not np.array_equal(prediction,(probability>=config["probability_threshold"])&canvas):
            raise ValueError("Independent threshold/canvas reconstruction differs")
        labels,number=label(prediction,structure=np.ones((3,3)))
        measured=[]
        for identity,slices in enumerate(find_objects(labels),1):
            yy,xx=slices
            measured.append((int((labels[slices]==identity).sum()),(xx.start,yy.start,xx.stop,yy.stop)))
        saved=[(p["component_area"],tuple(p["bbox_native"])) for p in row["predictions"]]
        if number!=len(saved) or sorted(measured)!=sorted(saved):raise ValueError("Independent components differ")
        annotation=rgb_path.parent.parent/"annotations"/(sample_id.split(":",1)[1]+".txt")
        if digest(annotation)!=row["annotation_sha256"]:raise ValueError("Annotation changed")
        words=[]
        for source in csv.reader(annotation.read_text(encoding="utf-8-sig").splitlines(),skipinitialspace=True):
            if not source or all(not v.strip() for v in source):continue
            points=[int(v) for v in source[:8]];text=",".join(source[8:]).strip()
            if text=="###":ignored+=1;continue
            xs,ys=points[::2],points[1::2]
            words.append({"text":text,"bbox":[min(xs),min(ys),min(w,max(xs)+1),min(h,max(ys)+1)]})
        if [{"text":r["text"],"bbox":r["bbox"]} for r in row["ground_truth"]]!=words:
            raise ValueError("Independent care GT differs")
        for p in row["predictions"]:
            a,b,c,d=p["bbox_native"]
            inverted=[max(0,a*cw//nw-left),max(0,b*ch//nh-top),
                min(w,(c*cw+nw-1)//nw-left),min(h,(d*ch+nh-1)//nh-top)]
            if p["bbox"]!=inverted:raise ValueError("Independent inverse coordinates differ")
            x1,y1,x2,y2=p["bbox"];crop=np.ascontiguousarray(rgb[y1:y2,x1:x2])
            key=sha256(np.asarray(crop.shape,dtype="<i8").tobytes()+crop.tobytes()).hexdigest()
            if key!=p["crop_rgb_sha256"]:raise ValueError("Crop RGB differs")
            ocr=results.parent/"ocr"/(key+".json")
            if digest(ocr)!=p["ocr_file_sha256" if is_test else "ocr_sha256"] or json.loads(ocr.read_text())["raw_text"]!=p["text"]:
                raise ValueError("OCR output differs")
            crops+=1
        for mode in totals:
            tp=independent_tp(row["predictions"],words,config["matching_iou_threshold"],mode)
            if tp!=row["counts"][mode+"_tp"]:raise ValueError("Independent maximum matching differs")
            totals[mode]+=tp
        n_pred+=len(row["predictions"]);n_gt+=len(words)
    if n_pred!=report["aggregate"]["predictions"] or n_gt!=expected_words:raise ValueError("Full denominators differ")
    if is_test and ignored!=report["ignored_gt"]:raise ValueError("Ignored GT denominator differs")
    metrics={}
    for mode,tp in totals.items():
        measured={"tp":tp,"fp":n_pred-tp,"fn":n_gt-tp,"precision":tp/n_pred if n_pred else None,
            "recall":tp/n_gt if n_gt else None,"f1":2*tp/(n_pred+n_gt) if n_pred+n_gt else None}
        if measured!=report["aggregate"][mode]:raise ValueError("Independent aggregate metric differs")
        metrics[mode]=measured
    return {"status":"PASS","reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_sha256":digest(results),"auditor_sha256":digest(Path(__file__)),
        "independent_matcher_sha256":digest(Path(__file__).with_name("audit_icdar2015_word_f1.py")),
        "images":expected_images,"care_words":n_gt,"predictions":n_pred,"crop_hashes_verified":crops,
        "ignored_gt":ignored,"image_hw":shape,"metrics":metrics}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("results","data-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();report=audit(args.results,args.data_root)
    write_json(args.output,report);print(json.dumps(report,sort_keys=True))


if __name__=="__main__":main()
