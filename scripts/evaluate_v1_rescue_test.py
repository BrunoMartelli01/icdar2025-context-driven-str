#!/usr/bin/env python3
"""Frozen TEST evaluation of practical-remap-v1.1-rescue against v1."""

from __future__ import annotations

import argparse,hashlib,json,os,tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz.fuzz import ratio

from localize.grouped_word_blocks import GroupedWordBlockParams,group_word_blocks
from localize.practical_remap_protocol import proposal_geometry
from localize.v1_local_rescue import LocalRescueParams,local_rescue
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter,FrozenSegmentationConfig
from recognition.icdar13_annotations import load_icdar13_recognition_samples
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE,ParseqApproxConfig,ParseqApproxRecognizer


RESCUE=LocalRescueParams(max_normalized_gap=.4,max_gap_to_internal_ratio=1.5,min_vertical_overlap=.5,max_center_y_ratio=.5,max_height_ratio=2.5)
PROMOTE_EXACT_GAIN=3;PROMOTE_FUZZY_GAIN=3;MAX_MULTIWORD_INCREASE=2
EXAMPLES=("Tiredness","GREEK","intelligent","Harwich","Ipswich","10","WINES","inhabited","(A120)")
SIGNIFICANT_GT_COVERAGE=.25


def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _area(box):return max(0,box[2]-box[0])*max(0,box[3]-box[1])
def _intersection(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def _iou(a,b):
    inter=_intersection(a,b);union=_area(a)+_area(b)-inter
    return inter/union if union else 0.


def _wordness(crops,words):
    records=[]
    for crop in crops:
        matches=[]
        for index,word in enumerate(words,1):
            gt=(word.bbox_xyxy[0],word.bbox_xyxy[1],word.bbox_xyxy[2]+1,word.bbox_xyxy[3]+1);inter=_intersection(crop["bbox"],gt)
            matches.append((index,word.transcription,_iou(crop["bbox"],gt),inter/max(1,_area(gt))))
        matches.sort(key=lambda v:(-v[2],-v[3],v[0]));best=matches[0];significant=sum(v[3]>=SIGNIFICANT_GT_COVERAGE for v in matches)
        category="MULTIWORD_MERGE" if significant>=2 else "WORD_GOOD" if best[2]>=.5 else "WORD_PARTIAL" if significant==1 else "UNMATCHED"
        records.append({**crop,"best_gt_index":best[0],"best_gt_text":best[1],"bbox_iou":best[2],"significant_gt_words":significant,"classification":category})
    return records


def _per_gt(crops,words):
    values=[]
    for index,word in enumerate(words,1):
        gt=(word.bbox_xyxy[0],word.bbox_xyxy[1],word.bbox_xyxy[2]+1,word.bbox_xyxy[3]+1)
        best=max(crops,key=lambda v:_iou(v["bbox"],gt),default=None);text=None if best is None else best["text"]
        values.append({"gt_index":index,"gt_text":word.transcription,"bbox_iou":0. if best is None else _iou(best["bbox"],gt),"crop_bbox":None if best is None else best["bbox"],"text":text,"exact":text is not None and text.casefold()==word.transcription.casefold(),"fuzzy_ge_0_8":text is not None and ratio(text.casefold(),word.transcription.casefold())>=80})
    return values


def _aggregate_ocr(records):return {"denominator_gt_words":len(records),"available_predictions":sum(v["text"] is not None for v in records),"casefold_exact":sum(v["exact"] for v in records),"rapidfuzz_casefold_ge_0_8":sum(v["fuzzy_ge_0_8"] for v in records)}
def _aggregate_wordness(records):
    c=Counter(v["classification"] for v in records)
    return {"total_crops":len(records),**{k:c[k] for k in ("WORD_GOOD","WORD_PARTIAL","MULTIWORD_MERGE","UNMATCHED")}}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--v1-calibration",type=Path,required=True);p.add_argument("--v1-test-report",type=Path,required=True);p.add_argument("--gpu-reference-report",type=Path,required=True);p.add_argument("--evaluation-config",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--parseq-source",type=Path,required=True);p.add_argument("--parseq-source-archive",type=Path,required=True);p.add_argument("--parseq-checkpoint",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--output",type=Path,required=True);return p.parse_args()


def main():
    a=parse_args();output=a.output.resolve()
    if output.exists():raise FileExistsError(f"refusing to overwrite {output}")
    samples,_=load_icdar13_recognition_samples(a.manifest,a.data_root);assert len(samples)==233
    cal=json.loads(a.v1_calibration.read_text());guard=cal["train_only_guard"]
    if guard["sample_count"]!=229 or guard["test_data_loaded"] is not False:raise ValueError("v1 parameters are not frozen train-only")
    v1_params=GroupedWordBlockParams(**cal["calibration"]["selected"]["parameters"])
    cached=json.loads(a.v1_test_report.read_text());cached_words={(str(v["sample_id"]),int(v["gt_index"])):v["ocr_comparison"]["PRACTICAL_REMAP"] for v in cached["first_ten_parseq"]["per_word"]}
    if len(cached_words)!=66:raise ValueError("v1 OCR cache must contain first10/66 words")
    refs=json.loads(a.gpu_reference_report.read_text());reference={str(v["sample_id"]):v for v in refs["frozen_test"]["frozen_segmentation"]["mask_hashes"]}
    segmenter=FrozenAuthorProtocolSegmenter(FrozenSegmentationConfig(a.evaluation_config,a.checkpoint,a.device));masks=[];geometry_rows=[];first10_groups=[]
    try:
        for number,sample in enumerate(samples,1):
            with Image.open(sample.image_path) as image:rgb=np.asarray(image.convert("RGB"))
            seg=segmenter(rgb);expected=reference[sample.sample_id]
            if seg["metadata"]["native_mask_sha256"]!=expected["native_mask_sha256"] or seg["metadata"]["output_mask_sha256"]!=expected["projected_mask_sha256"]:raise RuntimeError(f"frozen GPU mask drift sample {sample.sample_id}")
            v1=group_word_blocks(seg["mask"],rgb,v1_params);raw_boxes={v.component_id:v.bbox for v in v1.raw_components};rescue=local_rescue(seg["mask"],rgb,v1_params,RESCUE,raw_boxes)
            v1_boxes=[v.padded_bbox for v in v1.retained];rescue_boxes=[v.padded_bbox for v in rescue.retained];gt=[(w.bbox_xyxy[0],w.bbox_xyxy[1],w.bbox_xyxy[2]+1,w.bbox_xyxy[3]+1) for w in sample.words]
            geometry_rows.append((v1_boxes,rescue_boxes,gt,len(v1.groups_before_cap),len(rescue.groups_before_cap),seg["metadata"]["output_mask_sha256"]))
            if number<=10:first10_groups.append((sample,rgb,v1,rescue))
            if number==1 or number%10==0 or number==233:print(f"[test masks] {number}/233 sample_id={sample.sample_id}",flush=True)
    finally:seg_release=segmenter.release()
    def geometry(which):
        rows=[(r[which],r[2],r[3+which],0,0) for r in geometry_rows]
        return proposal_geometry(rows).as_dict()
    v1_geometry,rescue_geometry=geometry(0),geometry(1)
    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=a.parseq_source,checkpoint=a.parseq_checkpoint,device=a.device,runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=a.parseq_source_archive))
    rescue_words=[];rescue_wordness=[];v1_wordness=[]
    try:
        for number,(sample,rgb,v1,rescue) in enumerate(first10_groups,1):
            rescue_crops=[]
            for rank,block in enumerate(rescue.retained,1):
                x1,y1,x2,y2=block.padded_bbox;text=recognizer.recognize(rgb[y1:y2,x1:x2].copy(),input_role="T3_localized_crop").raw_text
                rescue_crops.append({"rank":rank,"bbox":block.padded_bbox,"text":text})
            rescue_words.extend({"sample_id":sample.sample_id,**v} for v in _per_gt(rescue_crops,sample.words));rescue_wordness.extend({"sample_id":sample.sample_id,**v} for v in _wordness(rescue_crops,sample.words))
            graph=cached["first_ten_parseq"]["per_sample_graphs"][sample.sample_id]
            v1_crops=[{"rank":v["rank"],"bbox":v["padded_bbox"],"text":v["parseq"]["raw_text"]} for v in graph["retained_groups_with_parseq"]]
            v1_wordness.extend({"sample_id":sample.sample_id,**v} for v in _wordness(v1_crops,sample.words));print(f"[PARSeq rescue] {number}/10 sample_id={sample.sample_id} crops={len(rescue_crops)}",flush=True)
    finally:parseq_release=recognizer.release()
    v1_words=[]
    for sample in samples[:10]:
        for index,word in enumerate(sample.words,1):
            value=cached_words[(sample.sample_id,index)];metrics=value["text_metrics"]
            v1_words.append({"sample_id":sample.sample_id,"gt_index":index,"gt_text":word.transcription,"text":value["text"],"exact":metrics["casefold_exact"],"fuzzy_ge_0_8":metrics["rapidfuzz_ratio_casefold"]>=.8})
    v1_ocr,rescue_ocr=_aggregate_ocr(v1_words),_aggregate_ocr(rescue_words);v1_w,rescue_w=_aggregate_wordness(v1_wordness),_aggregate_wordness(rescue_wordness)
    promote=rescue_ocr["casefold_exact"]>=v1_ocr["casefold_exact"]+PROMOTE_EXACT_GAIN and rescue_ocr["rapidfuzz_casefold_ge_0_8"]>=v1_ocr["rapidfuzz_casefold_ge_0_8"]+PROMOTE_FUZZY_GAIN and rescue_w["MULTIWORD_MERGE"]<=v1_w["MULTIWORD_MERGE"]+MAX_MULTIWORD_INCREASE
    examples=[]
    for wanted in EXAMPLES:
        rv=next(v for v in rescue_words if v["gt_text"].casefold()==wanted.casefold());vv=next(v for v in v1_words if v["gt_text"].casefold()==wanted.casefold());examples.append({"gt_text":wanted,"sample_id":rv["sample_id"],"v1_text":vv["text"],"rescue_text":rv["text"],"rescue_bbox":rv["crop_bbox"],"rescue_iou":rv["bbox_iou"]})
    report={"schema_version":1,"mode":"practical-remap-v1.1-rescue","classification":"PRACTICAL_APPROXIMATION","frozen_candidate":RESCUE.__dict__,"no_test_tuning":True,"same_mask_contract":{"samples":233,"single_mask_consumed_by_both":True},"test_geometry":{"v1":v1_geometry,"rescue":rescue_geometry,"undermerge_overmerge":None},"first10_parseq":{"v1":v1_ocr,"rescue":rescue_ocr},"first10_wordness":{"v1":v1_w,"rescue":rescue_w},"examples":examples,"promotion_rule":{"minimum_exact_gain":3,"minimum_fuzzy_gain":3,"maximum_multiword_increase":2},"decision":"PROMOTE" if promote else "REJECT","provenance":{"checkpoint_sha256":_sha(a.checkpoint),"v1_calibration_sha256":_sha(a.v1_calibration),"v1_test_report_sha256":_sha(a.v1_test_report),"segmentation_release":seg_release,"parseq_release":parseq_release}}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=output.parent,delete=False,encoding="utf-8") as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");temporary=Path(stream.name)
    os.replace(temporary,output);print(json.dumps({"test_geometry":report["test_geometry"],"first10_parseq":report["first10_parseq"],"first10_wordness":report["first10_wordness"],"decision":report["decision"]},indent=2),flush=True);print(output,flush=True);print(_sha(output),flush=True);return 0


if __name__=="__main__":raise SystemExit(main())
