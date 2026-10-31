"""Replay perfect native GT targets against original care boxes on validation."""

import argparse
from pathlib import Path

from PIL import Image

from segmentation.data import file_sha256
from segmentation.icdar2015_improvements import binary_proposals, native_target_audit_maps, original_geometry
from segmentation.icdar2015_improvement_runtime import load_config, load_training_population, references
from scripts.train_icdar2015_totaltext_transfer import write_json


def run(args):
    config=load_config(args.config);manifest,fit,val=load_training_population(config,args.data_root)
    refs=references(manifest,args.data_root,val);rows={name:[] for name in config["resolutions"]}
    for index in val:
        sample=manifest.samples[index];maps=[]
        for path,digest,mode in ((sample.mask_path,sample.mask_sha256,"L"),
            (sample.validity_annotation_path,sample.validity_annotation_sha256,"L"),
            (sample.separation_annotation_path,sample.separation_annotation_sha256,"L"),
            (sample.word_owner_annotation_path,sample.word_owner_annotation_sha256,"I")):
            if file_sha256(args.data_root/path)!=digest:raise ValueError("Qualified target map changed")
            with Image.open(args.data_root/path) as im:maps.append(im.convert(mode))
        for name,shape in config["resolutions"].items():
            _,target,_,_=native_target_audit_maps(*maps,shape,threshold=config["probability_threshold"])
            ref=refs[sample.sample_id];boxes,_=binary_proposals(target,ref["image_wh"],shape)
            counts=original_geometry(boxes,ref["ground_truth"],iou_threshold=config["matching_iou_threshold"])
            rows[name].append({"sample_id":sample.sample_id,**counts})
    aggregate={}
    for name,items in rows.items():
        totals={k:sum(row[k] for row in items) for k in ("matched_gt","gt_words","proposals")}
        aggregate[name]={**totals,"geometry_f1":2*totals["matched_gt"]/(totals["gt_words"]+totals["proposals"])}
    report={"status":"SUCCESS","reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":"GT-assisted target geometry diagnostic; no network inference",
        "limits":"Perfect-target replay, not an upper bound on all possible network outputs or a deployable score. Resolution/branch selection is unchanged.",
        "config_sha256":file_sha256(args.config),"source_sha256":file_sha256(Path(__file__)),
        "aggregate":aggregate,"per_image":rows}
    if args.output.exists():raise ValueError("Preserve existing diagnostic")
    write_json(args.output,report);print(aggregate,flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","data-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":main()
