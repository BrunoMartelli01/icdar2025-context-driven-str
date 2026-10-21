#!/usr/bin/env python3
"""Full TRAIN GT_BOX-vs-v1 PARSeq ceiling diagnostic; never a Table-5 result."""

from __future__ import annotations

import argparse,csv,hashlib,json,os,re,shutil,tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz.fuzz import ratio

from localize.grouped_word_blocks import GroupedWordBlockParams,group_word_blocks
from localize.practical_remap_protocol import proposal_geometry,train_task22_word_boxes
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter,FrozenSegmentationConfig
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE,ParseqApproxConfig,ParseqApproxRecognizer
from segmentation.data import load_segmentation_manifest


LINE=re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+).*"(.*)"\s*$')
SIGNIFICANT_GT_COVERAGE=.25


def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def _inter(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def _iou(a,b):
    i=_inter(a,b);u=_area(a)+_area(b)-i
    return i/u if u else 0.


def _texts(path:Path)->tuple[str,...]:
    groups=[];current=[]
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            if current:groups.append("".join(current));current=[]
            continue
        if raw.lstrip().startswith("#"):continue
        match=LINE.match(raw)
        if match is None:raise ValueError(f"malformed Task2.2 record: {path}")
        current.append(match.group(4))
    if current:groups.append("".join(current))
    return tuple(groups)


def _wordness(crop,words):
    overlaps=[(_iou(crop,word["bbox"]),_inter(crop,word["bbox"])/max(1,_area(word["bbox"])),index) for index,word in enumerate(words)]
    best=max(overlaps,key=lambda v:(v[0],v[1],-v[2]));significant=sum(v[1]>=SIGNIFICANT_GT_COVERAGE for v in overlaps)
    category="MULTIWORD_MERGE" if significant>=2 else "WORD_GOOD" if best[0]>=.5 else "WORD_PARTIAL" if significant==1 else "UNMATCHED"
    return category,best[2],best[0]


def _aggregate(rows,key):
    scores=[row[key]["rapidfuzz_casefold"] for row in rows if row[key]["available"]]
    total=len(rows);available=sum(row[key]["available"] for row in rows);exact=sum(row[key]["exact_casefold"] for row in rows);fuzzy=sum(row[key]["rapidfuzz_casefold_ge_0_8"] for row in rows)
    return {"gt_words_total":total,"available_predictions":available,"exact_casefold":exact,"exact_percent_of_all_gt":100*exact/total,"rapidfuzz_casefold_ge_0_8":fuzzy,"fuzzy_percent_of_all_gt":100*fuzzy/total,"median_rapidfuzz_available":float(np.median(scores)) if scores else None}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--train-manifest",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--v1-calibration",type=Path,required=True);p.add_argument("--evaluation-config",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--parseq-source",type=Path,required=True);p.add_argument("--parseq-source-archive",type=Path,required=True);p.add_argument("--parseq-checkpoint",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--output",type=Path,required=True);p.add_argument("--csv-output",type=Path,required=True);return p.parse_args()


def main():
    a=parse_args();output=a.output.resolve();csv_output=a.csv_output.resolve()
    if output.exists() or csv_output.exists():raise FileExistsError("refusing to overwrite frozen diagnostic")
    manifest=load_segmentation_manifest(a.train_manifest.resolve())
    if manifest.split!="official-training-all-229-author-protocol" or manifest.split_role!="train" or len(manifest.samples)!=229:raise ValueError("requires author-protocol TRAIN-229")
    calibration=json.loads(a.v1_calibration.read_text());guard=calibration["train_only_guard"]
    if guard["sample_count"]!=229 or guard["test_data_loaded"] is not False:raise ValueError("v1 calibration provenance is not train-only")
    params=GroupedWordBlockParams(**calibration["calibration"]["selected"]["parameters"])
    work=Path(tempfile.mkdtemp(prefix="train-gtbox-v1-parseq-"));records=[];geometry_rows=[]
    segmenter=FrozenAuthorProtocolSegmenter(FrozenSegmentationConfig(a.evaluation_config.resolve(),a.checkpoint.resolve(),a.device))
    try:
        for number,sample in enumerate(manifest.samples,1):
            image_path=(a.data_root/sample.image_path).resolve();mask_path=(a.data_root/sample.mask_path).resolve();annotation_path=(a.data_root/sample.validity_annotation_path).resolve()
            rgb=np.asarray(Image.open(image_path).convert("RGB"));seg=segmenter(rgb);v1=group_word_blocks(seg["mask"],rgb,params)
            boxes=train_task22_word_boxes(mask_path,annotation_path);texts=_texts(annotation_path)
            if len(boxes)!=len(texts):raise RuntimeError(f"GT geometry/text count mismatch sample {sample.sample_id}")
            words=[{"gt_index":index,"bbox":word.bbox,"text":text} for index,(word,text) in enumerate(zip(boxes,texts),1)]
            sample_dir=work/sample.sample_id;sample_dir.mkdir()
            for word in words:
                x1,y1,x2,y2=word["bbox"];path=sample_dir/f"gt-{word['gt_index']:03d}.png";Image.fromarray(rgb[y1:y2,x1:x2]).save(path);word["gt_crop"]=str(path)
            crops=[]
            for block in v1.retained:
                x1,y1,x2,y2=block.padded_bbox;path=sample_dir/f"v1-{block.rank:03d}.png";Image.fromarray(rgb[y1:y2,x1:x2]).save(path);category,best,best_iou=_wordness(block.padded_bbox,words);crops.append({"rank":block.rank,"bbox":block.padded_bbox,"path":str(path),"classification":category,"best_gt_index":best+1,"best_gt_iou":best_iou})
            geometry_rows.append(([v["bbox"] for v in crops],[v["bbox"] for v in words],len(v1.groups_before_cap),sum(len(v["raw_component_ids"])-1 for v in v1.groups_before_cap),sum(len(v["raw_component_ids"])>1 for v in v1.groups_before_cap)))
            records.append({"sample_id":sample.sample_id,"words":words,"v1_crops":crops,"mask_sha256":seg["metadata"]["output_mask_sha256"]})
            if number==1 or number%10==0 or number==229:print(f"[segmentation/crops] {number}/229 sample_id={sample.sample_id}",flush=True)
    finally:seg_release=segmenter.release()
    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=a.parseq_source,checkpoint=a.parseq_checkpoint,device=a.device,runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=a.parseq_source_archive));rows=[];ocr_count=0
    try:
        total_ocr=sum(len(v["words"])+len(v["v1_crops"]) for v in records)
        for record in records:
            for crop in record["v1_crops"]:
                crop["text"]=recognizer.recognize(np.asarray(Image.open(crop["path"]).convert("RGB")),input_role="T3_localized_crop").raw_text;ocr_count+=1
                if ocr_count%25==0 or ocr_count==total_ocr:print(f"[PARSeq] {ocr_count}/{total_ocr}",flush=True)
            for word in record["words"]:
                gt_text=recognizer.recognize(np.asarray(Image.open(word["gt_crop"]).convert("RGB")),input_role="T3_localized_crop").raw_text;ocr_count+=1
                if ocr_count%25==0 or ocr_count==total_ocr:print(f"[PARSeq] {ocr_count}/{total_ocr}",flush=True)
                candidates=record["v1_crops"];best=max(candidates,key=lambda v:_iou(v["bbox"],word["bbox"]),default=None);available=best is not None and _inter(best["bbox"],word["bbox"])>0;v1_text=best["text"] if available else None
                def metrics(text):
                    score=ratio(text.casefold(),word["text"].casefold())/100 if text is not None else None
                    return {"available":text is not None,"text":text,"exact_casefold":text is not None and text.casefold()==word["text"].casefold(),"rapidfuzz_casefold":score,"rapidfuzz_casefold_ge_0_8":score is not None and score>=.8}
                rows.append({"sample_id":record["sample_id"],"gt_index":word["gt_index"],"gt_text":word["text"],"gt_bbox":word["bbox"],"GT_BOX":metrics(gt_text),"PRACTICAL_REMAP_V1":{**metrics(v1_text),"matched_crop_rank":None if not available else best["rank"],"matched_crop_bbox":None if not available else best["bbox"],"bbox_iou":0. if not available else _iou(best["bbox"],word["bbox"]),"crop_classification":None if not available else best["classification"]}})
    finally:parseq_release=recognizer.release()
    if len(rows)!=849:raise RuntimeError(f"expected 849 GT words, got {len(rows)}")
    wordness=Counter(crop["classification"] for record in records for crop in record["v1_crops"]);geometry=proposal_geometry(geometry_rows).as_dict();gt_agg=_aggregate(rows,"GT_BOX");v1_agg=_aggregate(rows,"PRACTICAL_REMAP_V1")
    report={"schema_version":1,"classification":"APPROXIMATE_DIAGNOSTIC","table_5_claim":False,"scope":{"split":"TRAIN","images":229,"gt_words":849},"prohibitions":{"routing":False,"blip2":False,"mpnet":False,"deepsolo":False,"tuning":False},"GT_BOX":gt_agg,"PRACTICAL_REMAP_V1":{**v1_agg,"geometry":geometry,"wordness":{"total_crops":sum(wordness.values()),**{name:wordness[name] for name in ("WORD_GOOD","WORD_PARTIAL","MULTIWORD_MERGE","UNMATCHED")}}},"gap":{"exact_count":gt_agg["exact_casefold"]-v1_agg["exact_casefold"],"exact_percentage_points":gt_agg["exact_percent_of_all_gt"]-v1_agg["exact_percent_of_all_gt"],"fuzzy_count":gt_agg["rapidfuzz_casefold_ge_0_8"]-v1_agg["rapidfuzz_casefold_ge_0_8"],"fuzzy_percentage_points":gt_agg["fuzzy_percent_of_all_gt"]-v1_agg["fuzzy_percent_of_all_gt"]},"protocol":{"v1_gt_assignment":"maximum bbox IoU; positive intersection required for availability","wordness_significant_intersection":"intersection / GT bbox area >= 0.25","exact_and_fuzzy_percent_denominator":"all 849 GT words","median_fuzzy_denominator":"available predictions"},"provenance":{"checkpoint_sha256":_sha(a.checkpoint.resolve()),"v1_calibration_sha256":_sha(a.v1_calibration.resolve()),"segmentation_release":seg_release,"parseq_release":parseq_release},"per_word":rows}
    output.parent.mkdir(parents=True,exist_ok=True);csv_output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=output.parent,delete=False,encoding="utf-8") as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");temporary=Path(stream.name)
    os.replace(temporary,output)
    with tempfile.NamedTemporaryFile("w",dir=csv_output.parent,delete=False,encoding="utf-8",newline="") as stream:
        fields=("sample_id","gt_index","gt_text","gt_bbox","gtbox_text","gtbox_exact","gtbox_fuzzy","v1_text","v1_exact","v1_fuzzy","v1_bbox_iou","v1_crop_classification");writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({"sample_id":row["sample_id"],"gt_index":row["gt_index"],"gt_text":row["gt_text"],"gt_bbox":json.dumps(row["gt_bbox"]),"gtbox_text":row["GT_BOX"]["text"],"gtbox_exact":row["GT_BOX"]["exact_casefold"],"gtbox_fuzzy":row["GT_BOX"]["rapidfuzz_casefold"],"v1_text":row["PRACTICAL_REMAP_V1"]["text"],"v1_exact":row["PRACTICAL_REMAP_V1"]["exact_casefold"],"v1_fuzzy":row["PRACTICAL_REMAP_V1"]["rapidfuzz_casefold"],"v1_bbox_iou":row["PRACTICAL_REMAP_V1"]["bbox_iou"],"v1_crop_classification":row["PRACTICAL_REMAP_V1"]["crop_classification"]})
        csv_temp=Path(stream.name)
    os.replace(csv_temp,csv_output);shutil.rmtree(work);print(json.dumps({"GT_BOX":gt_agg,"PRACTICAL_REMAP_V1":report["PRACTICAL_REMAP_V1"],"gap":report["gap"]},indent=2),flush=True);print(output,flush=True);print(csv_output,flush=True);print(_sha(output),flush=True);return 0


if __name__=="__main__":raise SystemExit(main())
