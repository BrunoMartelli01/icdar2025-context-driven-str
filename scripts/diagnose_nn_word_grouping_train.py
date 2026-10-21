#!/usr/bin/env python3
"""Train-only calibration diagnostic for nn-word-grouping-v1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

from localize.grouped_word_blocks import GroupedWordBlockParams
from localize.nn_word_grouping import NNWordGroupingParams, group_nn_words
from localize.practical_remap_v2_protocol import _failure_and_geometry, _v1_localizer, load_v2_train_records, declared_v2_grid
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter, FrozenSegmentationConfig
from segmentation.data import load_segmentation_manifest


GRID = {
    "absolute": (0.25, 0.35, 0.45, 0.60),
    "local_median_ratio": (1.5, 2.0, 2.5, 3.0),
    "robust_mad": (1.5, 2.0, 2.5, 3.0),
    "local_windows": (2, 3),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grid() -> tuple[NNWordGroupingParams,...]:
    values=[NNWordGroupingParams("absolute",v,2) for v in GRID["absolute"]]
    for family in ("local_median_ratio","robust_mad"):
        values.extend(NNWordGroupingParams(family,v,w) for w in GRID["local_windows"] for v in GRID[family])
    assert len(values)==20
    return tuple(values)


def _v1_params(report: dict) -> GroupedWordBlockParams:
    guard=report["train_only_guard"]
    if guard["split_role"]!="train" or guard["sample_count"]!=229 or guard["test_data_loaded"] is not False:
        raise ValueError("v1 provenance is not train-229 only")
    return GroupedWordBlockParams(**report["calibration"]["selected"]["parameters"])


def _counts(value: dict) -> tuple[int,int]:
    categories=value["failure_breakdown"]["categories"]
    return categories["GROUPING_UNDERMERGE"]["count"],categories["GROUPING_OVERMERGE"]["count"]


def _summary(value: dict) -> dict:
    geometry=value["geometry"]; under,over=_counts(value)
    return {"parameters":value.get("parameters"),"gt_words":geometry["gt_words"],"iou_ge_0_25":geometry["gt_with_proposal_iou_ge_0_25"],"iou_ge_0_50":geometry["gt_with_proposal_iou_ge_0_50"],"iou_ge_0_75":geometry["gt_with_proposal_iou_ge_0_75"],"precision_iou_ge_0_50":geometry["proposal_precision_iou_ge_0_50"],"recall_iou_ge_0_50":geometry["recall_iou_ge_0_50"],"f1_iou_ge_0_50":geometry["f1_proposal_level_iou_ge_0_50"],"undermerge":under,"overmerge":over,"proposals_per_image":geometry["average_proposals_per_image"],"gt_words_reconstructed_by_single_group":value["failure_breakdown"]["success_count"]}


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--v1-calibration",type=Path,required=True)
    p.add_argument("--evaluation-config",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--output",type=Path,required=True)
    return p.parse_args()


def main() -> int:
    a=parse_args();output=a.output.resolve()
    if output.exists():raise FileExistsError(f"refusing to overwrite {output}")
    manifest=load_segmentation_manifest(a.train_manifest.resolve())
    if manifest.split!="official-training-all-229-author-protocol" or manifest.split_role!="train" or len(manifest.samples)!=229:raise ValueError("requires official author-protocol train-229 manifest")
    v1_report=json.loads(a.v1_calibration.read_text());v1_params=_v1_params(v1_report);grid=_grid()
    segmenter=FrozenAuthorProtocolSegmenter(FrozenSegmentationConfig(a.evaluation_config.resolve(),a.checkpoint.resolve(),a.device))
    try:
        records,source=load_v2_train_records(manifest,a.data_root.resolve(),lambda image:segmenter(image)["mask"],declared_v2_grid()[0],progress=lambda text:print(text,flush=True))
        v1=_failure_and_geometry(records,_v1_localizer(v1_params),label="practical_v1")
        variants=[]
        for index,params in enumerate(grid,1):
            result=_failure_and_geometry(records,lambda record,p=params:group_nn_words(record.prediction,record.image,p),label="nn_word_grouping_v1")
            variants.append({"parameters":asdict(params),**result})
            print(f"[grid] {index}/20 family={params.family} threshold={params.boundary_threshold} window={params.local_window} F1={result['geometry']['f1_proposal_level_iou_ge_0_50']:.6f}",flush=True)
    finally:release=segmenter.release()
    best_by_family={}
    for family in GRID:
        if family=="local_windows":continue
        members=[v for v in variants if v["parameters"]["family"]==family]
        best_by_family[family]=max(members,key=lambda v:(v["geometry"]["f1_proposal_level_iou_ge_0_50"],-_counts(v)[1],-_counts(v)[0],-v["geometry"]["average_proposals_per_image"]))
    overall=max(best_by_family.values(),key=lambda v:(v["geometry"]["f1_proposal_level_iou_ge_0_50"],-_counts(v)[1],-_counts(v)[0],-v["geometry"]["average_proposals_per_image"]))
    go=overall["geometry"]["f1_proposal_level_iou_ge_0_50"]>v1["geometry"]["f1_proposal_level_iou_ge_0_50"] and _counts(overall)[1]<=_counts(v1)[1]
    report={"schema_version":1,"mode":"nn-word-grouping-v1","classification":"PRACTICAL_APPROXIMATION_DIAGNOSTIC","train_only_guard":{"split":manifest.split,"sample_count":229,"test_loaded":False,"ocr_called":False,"gt_used_in_inference":False},"declared_grid":{"configuration_count":20,**{k:list(v) for k,v in GRID.items()}},"selection_rule":"within each family maximize F1@.5, then fewer overmerge, undermerge, proposals; overall same rule","frozen_segmentation":{"checkpoint_sha256":_sha(a.checkpoint.resolve()),"device":a.device,"release":release},"source":source,"practical_remap_v1":_summary(v1),"best_by_family":{k:_summary(v) for k,v in best_by_family.items()},"overall_best":_summary(overall),"decision":{"go":go,"label":"GO" if go else "NO-GO","criterion":"strict F1 improvement over v1 without increasing overmerge"},"grid_results":[{"parameters":v["parameters"],"summary":_summary(v)} for v in variants]}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=output.parent,delete=False,encoding="utf-8") as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");temporary=Path(stream.name)
    os.replace(temporary,output);print(json.dumps({"best_by_family":report["best_by_family"],"v1":report["practical_remap_v1"],"overall":report["overall_best"],"decision":report["decision"]},indent=2),flush=True);print(output,flush=True);print(_sha(output),flush=True);return 0


if __name__=="__main__":raise SystemExit(main())
