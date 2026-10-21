#!/usr/bin/env python3
"""Compare original-coordinate padding with padding at native 128, first10 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image
from rapidfuzz.fuzz import ratio

from pipeline.frozen_segmentation import backproject_binary_mask
from recognition.icdar13_annotations import load_icdar13_recognition_samples
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE, ParseqApproxConfig, ParseqApproxRecognizer


EXPECTED_CHECKPOINT_SHA256 = "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2"
EXAMPLES = ("Tiredness", "GREEK", "intelligent", "Harwich", "Ipswich")
MIN_AREA_ORIGINAL = 150
PAD = 6
CONNECTIVITY = 8
NMAX = 10


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    intersection=max(0,ix2-ix1)*max(0,iy2-iy1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.0


def _map_native_box(box: tuple[int,int,int,int], width: int, height: int) -> tuple[int,int,int,int]:
    x1,y1,x2,y2=box
    return (math.floor(x1*width/128),math.floor(y1*height/128),math.ceil(x2*width/128),math.ceil(y2*height/128))


def _native_candidates(mask: np.ndarray, rgb: np.ndarray) -> tuple[int,list[dict]]:
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,CONNECTIVITY)
    height,width=rgb.shape[:2]; candidates=[]
    # Mapped area exactly follows nearest-neighbor projection, preserving the
    # CURRENT Amin population while changing only the coordinate frame of pad.
    for component_id in range(1,count):
        native=(labels==component_id)
        projected=backproject_binary_mask(native.astype(np.uint8),(height,width))
        mapped_area=int(projected.sum())
        if mapped_area<MIN_AREA_ORIGINAL: continue
        x=int(stats[component_id,cv2.CC_STAT_LEFT]); y=int(stats[component_id,cv2.CC_STAT_TOP])
        w=int(stats[component_id,cv2.CC_STAT_WIDTH]); h=int(stats[component_id,cv2.CC_STAT_HEIGHT])
        native_bbox=(x,y,x+w,y+h)
        padded_native=(max(0,x-PAD),max(0,y-PAD),min(128,x+w+PAD),min(128,y+h+PAD))
        candidates.append({"component_id":component_id,"component_area_original_equivalent":mapped_area,"bbox_native":native_bbox,"padded_bbox_native":padded_native,"bbox_original":_map_native_box(native_bbox,width,height),"padded_bbox_original":_map_native_box(padded_native,width,height)})
    candidates.sort(key=lambda v:(-v["component_area_original_equivalent"],v["bbox_native"][1],v["bbox_native"][0],v["component_id"]))
    for rank,value in enumerate(candidates[:NMAX],1): value["rank"]=rank
    return count-1,candidates[:NMAX]


def _geometry(boxes: list[tuple[int,int,int,int]], gt: list[tuple[int,int,int,int]]) -> dict:
    best=[max((_iou(word,box) for box in boxes),default=0.0) for word in gt]
    return {"gt_words":len(gt),"iou_ge_0_25":sum(v>=.25 for v in best),"iou_ge_0_50":sum(v>=.5 for v in best),"iou_ge_0_75":sum(v>=.75 for v in best)}


def _ocr_metrics(crops: list[dict], gt_boxes: list[tuple[int,int,int,int]], gt_words: tuple) -> dict:
    exact=fuzzy=0
    for crop in crops:
        best=max(range(len(gt_boxes)),key=lambda idx:_iou(tuple(crop["bbox"]),gt_boxes[idx]))
        truth=gt_words[best].transcription; text=crop["text"]
        exact+=text.casefold()==truth.casefold(); fuzzy+=ratio(text.casefold(),truth.casefold())>=80
    return {"casefold_exact":exact,"rapidfuzz_casefold_ge_0_8":fuzzy}


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--native-visual-dir",type=Path,required=True);p.add_argument("--gpu-reference-report",type=Path,required=True)
    p.add_argument("--trace-index",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--parseq-source",type=Path,required=True);p.add_argument("--parseq-source-archive",type=Path,required=True);p.add_argument("--parseq-checkpoint",type=Path,required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--output",type=Path,required=True)
    return p.parse_args()


def main() -> int:
    a=parse_args(); output=a.output.resolve()
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    if _sha_file(a.checkpoint.resolve())!=EXPECTED_CHECKPOINT_SHA256: raise RuntimeError("segmentation checkpoint SHA256 mismatch")
    samples,_=load_icdar13_recognition_samples(a.manifest,a.data_root);samples=samples[:10]
    reference=json.loads(a.gpu_reference_report.read_text()); refs={str(v["sample_id"]):v for v in reference["frozen_test"]["frozen_segmentation"]["mask_hashes"]}
    trace=json.loads(a.trace_index.read_text()); traces={str(v["sample_id"]):v for v in trace["samples"]}
    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=a.parseq_source,checkpoint=a.parseq_checkpoint,device=a.device,runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=a.parseq_source_archive))
    per_sample=[]; paired_expansion=[]
    try:
        for number,sample in enumerate(samples,1):
            print(f"[native-pad] {number}/10 sample_id={sample.sample_id}",flush=True)
            rgb=np.asarray(Image.open(sample.image_path).convert("RGB")); width,height=sample.image_size_wh
            native_png=a.native_visual_dir/f"sample-{int(sample.sample_id):04d}-pred-native-128.png"
            panel=Image.open(native_png).convert("L")
            if panel.size != (512,536): raise RuntimeError(f"unexpected titled native-mask panel geometry for sample {sample.sample_id}: {panel.size}")
            # The visual diagnostic prepends a 24-pixel title. Recover only the
            # 512x512 nearest-neighbor mask panel, never title pixels.
            native=np.asarray(panel.crop((0,24,512,536)).resize((128,128),Image.Resampling.NEAREST))
            native=(native>0).astype(np.uint8)
            if _sha_array(native)!=refs[sample.sample_id]["native_mask_sha256"]: raise RuntimeError(f"native mask hash mismatch sample {sample.sample_id}")
            raw_count,b_candidates=_native_candidates(native,rgb)
            frozen=json.loads(Path(traces[sample.sample_id]["result"]).read_text()); a_blocks=frozen["segmentation"]["localized_blocks"]; a_texts=frozen["T3"]["all"]
            audit=frozen["segmentation"]["localization_audit"]
            if len(a_blocks)!=len(a_texts) or raw_count!=len(audit["retained"])+len(audit["rejected"]): raise RuntimeError(f"CURRENT provenance mismatch sample {sample.sample_id}")
            a_crops=[{"rank":b["rank"],"component_id":b["component_id"],"bbox":b["padded_bbox"],"text":t["raw_text"]} for b,t in zip(a_blocks,a_texts)]
            b_crops=[]
            for candidate in b_candidates:
                x1,y1,x2,y2=candidate["padded_bbox_original"]
                recognized=recognizer.recognize(rgb[y1:y2,x1:x2].copy(),input_role="T3_localized_crop")
                b_crops.append({"rank":candidate["rank"],"component_id":candidate["component_id"],"bbox":candidate["padded_bbox_original"],"bbox_native":candidate["bbox_native"],"padded_bbox_native":candidate["padded_bbox_native"],"text":recognized.raw_text})
            if len(a_crops)!=len(b_crops): raise RuntimeError(f"component selection count changed sample {sample.sample_id}")
            for left,right,candidate,audit_block in zip(a_crops,b_crops,b_candidates,audit["retained"]):
                if candidate["component_area_original_equivalent"]!=audit_block["component_area"] or max(abs(x-y) for x,y in zip(candidate["bbox_original"],audit_block["bbox"]))>1:
                    raise RuntimeError(f"component geometry/order changed sample {sample.sample_id} rank {left['rank']}")
                aw,ah=left["bbox"][2]-left["bbox"][0],left["bbox"][3]-left["bbox"][1];bw,bh=right["bbox"][2]-right["bbox"][0],right["bbox"][3]-right["bbox"][1]
                paired_expansion.append({"width_ratio":bw/aw,"height_ratio":bh/ah,"area_ratio":bw*bh/(aw*ah),"width_pixels":bw-aw,"height_pixels":bh-ah})
            gt_boxes=[(w.bbox_xyxy[0],w.bbox_xyxy[1],w.bbox_xyxy[2]+1,w.bbox_xyxy[3]+1) for w in sample.words]
            per_sample.append({"sample_id":sample.sample_id,"raw_component_count":raw_count,"A_CURRENT":{"crops":a_crops,"geometry":_geometry([tuple(v["bbox"]) for v in a_crops],gt_boxes),"ocr":_ocr_metrics(a_crops,gt_boxes,sample.words)},"B_NATIVE_PAD":{"crops":b_crops,"geometry":_geometry([tuple(v["bbox"]) for v in b_crops],gt_boxes),"ocr":_ocr_metrics(b_crops,gt_boxes,sample.words)}})
    finally: recognizer.release()
    aggregate={}
    for method in ("A_CURRENT","B_NATIVE_PAD"):
        aggregate[method]={"retained_crops":sum(len(v[method]["crops"]) for v in per_sample),**{key:sum(v[method]["geometry"][key] for v in per_sample) for key in ("gt_words","iou_ge_0_25","iou_ge_0_50","iou_ge_0_75")},"parseq_exact":sum(v[method]["ocr"]["casefold_exact"] for v in per_sample),"parseq_fuzzy_ge_0_8":sum(v[method]["ocr"]["rapidfuzz_casefold_ge_0_8"] for v in per_sample)}
    aggregate["paired_crop_expansion_B_over_A"]={key:float(np.mean([v[key] for v in paired_expansion])) for key in paired_expansion[0]}
    examples=[]
    for wanted in EXAMPLES:
        found=False
        for sample,record in zip(samples,per_sample):
            for gt_index,word in enumerate(sample.words):
                if word.transcription.casefold()!=wanted.casefold(): continue
                gt=(word.bbox_xyxy[0],word.bbox_xyxy[1],word.bbox_xyxy[2]+1,word.bbox_xyxy[3]+1)
                item={"sample_id":sample.sample_id,"gt_text":word.transcription,"gt_index":gt_index+1}
                for method in ("A_CURRENT","B_NATIVE_PAD"):
                    crop=max(record[method]["crops"],key=lambda v:_iou(tuple(v["bbox"]),gt));item[method]={"bbox":crop["bbox"],"text":crop["text"],"bbox_iou":_iou(tuple(crop["bbox"]),gt)}
                examples.append(item);found=True;break
            if found:break
        if not found: raise RuntimeError(f"required example absent: {wanted}")
    report={"schema_version":1,"classification":"APPROXIMATE_DIAGNOSTIC","scope":"first 10 ICDAR13 test only","new_segmentation_inference":False,"selection_or_tuning":False,"checkpoint_sha256":EXPECTED_CHECKPOINT_SHA256,"fixed_parameters":{"Amin_original_equivalent":MIN_AREA_ORIGINAL,"padding":PAD,"connectivity":CONNECTIVITY,"nmax":NMAX},"comparison":{"A_CURRENT":"component bbox mapped/original then padding=6 original pixels","B_NATIVE_PAD":"component bbox native128 then padding=6 native pixels then bbox remap to original"},"aggregate":aggregate,"required_examples":examples,"per_sample":per_sample}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=output.parent,delete=False,encoding="utf-8") as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");temporary=Path(stream.name)
    os.replace(temporary,output);print(json.dumps(aggregate,indent=2),flush=True);print(output,flush=True);print(_sha_file(output),flush=True);return 0


if __name__=="__main__":raise SystemExit(main())
