#!/usr/bin/env python3
"""Calibrate the v1-local-rescue diagnostic on author-protocol TRAIN only."""

from __future__ import annotations

import argparse,json,hashlib,os,tempfile
from dataclasses import asdict
from pathlib import Path

from localize.grouped_word_blocks import GroupedWordBlockParams
from localize.practical_remap_v2_protocol import _compare_changes,_failure_and_geometry,_v1_localizer,declared_v2_grid,load_v2_train_records
from localize.v1_local_rescue import LocalRescueParams,local_rescue
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter,FrozenSegmentationConfig
from segmentation.data import load_segmentation_manifest


GRID=tuple(LocalRescueParams(gap,relative,overlap) for gap in (.20,.30,.40) for relative in (1.5,2.0) for overlap in (.5,.7))
MAX_NEW_OVERMERGE_FOR_GO=5


def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(value:dict)->dict:
    g=value["geometry"];c=value["failure_breakdown"]["categories"]
    return {"gt_words":g["gt_words"],"iou_ge_0_50":g["gt_with_proposal_iou_ge_0_50"],"precision_iou_ge_0_50":g["proposal_precision_iou_ge_0_50"],"recall_iou_ge_0_50":g["recall_iou_ge_0_50"],"f1_iou_ge_0_50":g["f1_proposal_level_iou_ge_0_50"],"undermerge":c["GROUPING_UNDERMERGE"]["count"],"overmerge":c["GROUPING_OVERMERGE"]["count"],"proposals_per_image":g["average_proposals_per_image"]}


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--train-manifest",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--v1-calibration",type=Path,required=True);p.add_argument("--evaluation-config",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--output",type=Path,required=True);return p.parse_args()


def main()->int:
    a=parse_args();output=a.output.resolve()
    if output.exists():raise FileExistsError(f"refusing to overwrite {output}")
    manifest=load_segmentation_manifest(a.train_manifest.resolve())
    if manifest.split!="official-training-all-229-author-protocol" or len(manifest.samples)!=229:raise ValueError("requires author-protocol train-229")
    v1_report=json.loads(a.v1_calibration.read_text());guard=v1_report["train_only_guard"]
    if guard["split_role"]!="train" or guard["sample_count"]!=229 or guard["test_data_loaded"] is not False:raise ValueError("v1 calibration is not train-only")
    v1_params=GroupedWordBlockParams(**v1_report["calibration"]["selected"]["parameters"])
    segmenter=FrozenAuthorProtocolSegmenter(FrozenSegmentationConfig(a.evaluation_config.resolve(),a.checkpoint.resolve(),a.device))
    try:
        records,source=load_v2_train_records(manifest,a.data_root.resolve(),lambda image:segmenter(image)["mask"],declared_v2_grid()[0],progress=lambda text:print(text,flush=True))
        v1=_failure_and_geometry(records,_v1_localizer(v1_params),label="practical_v1")
        variants=[]
        for index,params in enumerate(GRID,1):
            result=_failure_and_geometry(records,lambda record,p=params:local_rescue(record.prediction,record.image,v1_params,p,record.raw_boxes),label="v1_local_rescue")
            changes=_compare_changes(v1,result);variants.append({"parameters":asdict(params),"evaluation":result,"changes":changes})
            print(f"[grid] {index}/{len(GRID)} gap={params.max_normalized_gap} relative={params.max_gap_to_internal_ratio} overlap={params.min_vertical_overlap} F1={result['geometry']['f1_proposal_level_iou_ge_0_50']:.6f} new_overmerge={changes['v2_new_overmerge_count']}",flush=True)
    finally:release=segmenter.release()
    acceptable=[v for v in variants if v["changes"]["v2_new_overmerge_count"]<=MAX_NEW_OVERMERGE_FOR_GO]
    pool=acceptable or variants
    selected=max(pool,key=lambda v:(v["evaluation"]["geometry"]["f1_proposal_level_iou_ge_0_50"],v["changes"]["v1_undermerge_recovered_by_v2"],-v["changes"]["v2_new_overmerge_count"],-v["evaluation"]["geometry"]["average_proposals_per_image"]))
    changes=selected["changes"];v1s,rescues=_summary(v1),_summary(selected["evaluation"])
    go=bool(acceptable and rescues["f1_iou_ge_0_50"]>v1s["f1_iou_ge_0_50"] and changes["v1_undermerge_recovered_by_v2"]>0 and changes["v2_new_overmerge_count"]<=MAX_NEW_OVERMERGE_FOR_GO)
    report={"schema_version":1,"mode":"v1-local-rescue","classification":"PRACTICAL_APPROXIMATION_DIAGNOSTIC","train_only_guard":{"split":manifest.split,"samples":229,"test_loaded":False,"ocr_called":False,"gt_used_in_inference":False},"declared_grid":{"count":len(GRID),"max_normalized_gap":[.2,.3,.4],"max_gap_to_internal_ratio":[1.5,2.0],"min_vertical_overlap":[.5,.7],"fixed_max_center_y_ratio":.5,"fixed_max_height_ratio":2.5},"selection_rule":"prefer variants with <=5 new overmerge; then F1, recovered v1 undermerge, fewer new overmerge, fewer proposals","go_rule":"F1 improves, >=1 v1 undermerge recovered, <=5 new overmerge","frozen_segmentation":{"checkpoint_sha256":_sha(a.checkpoint.resolve()),"device":a.device,"release":release},"source":source,"practical_remap_v1":v1s,"selected_rescue":{"parameters":selected["parameters"],**rescues,"v1_undermerge_recovered":changes["v1_undermerge_recovered_by_v2"],"new_overmerge":changes["v2_new_overmerge_count"]},"decision":"GO" if go else "NO-GO: STOP localization","grid_results":[{"parameters":v["parameters"],"summary":_summary(v["evaluation"]),"changes":v["changes"]} for v in variants]}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=output.parent,delete=False,encoding="utf-8") as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");temporary=Path(stream.name)
    os.replace(temporary,output);print(json.dumps({"v1":v1s,"rescue":report["selected_rescue"],"decision":report["decision"]},indent=2),flush=True);print(output,flush=True);print(_sha(output),flush=True);return 0


if __name__=="__main__":raise SystemExit(main())
