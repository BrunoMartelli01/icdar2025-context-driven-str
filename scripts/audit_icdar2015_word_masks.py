"""Full TRAIN word-separation audit of ICDAR2015 and qualified Total-Text masks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from recognition.scene_text_datasets import parse_icdar2015_annotation, parse_totaltext_annotation
from segmentation.data import file_sha256, load_segmentation_manifest, SegmentationDataset
from segmentation.icdar2015_improvements import native_target_audit_maps, owner_topology
from scripts.train_icdar2015_totaltext_transfer import write_json


def _one(task):
    cv2.setNumThreads(1)
    root, population, sample, sizes, threshold, protocol = task
    root = Path(root)
    with Image.open(root/sample.image_path) as image:
        size = image.size
    annotation = Path(sample.image_path).parent.parent/"annotations"/(sample.sample_id.split(":",1)[1]+".txt")
    instances = (parse_icdar2015_annotation(root/annotation,image_size_wh=size) if population == "icdar2015"
                 else parse_totaltext_annotation(root/annotation))
    count = sum(not item.ignore for item in instances)
    expected = range(1,count+1)
    with Image.open(root/sample.mask_path) as im:target = im.convert("L")
    with Image.open(root/sample.validity_annotation_path) as im:validity = im.convert("L")
    with Image.open(root/sample.separation_annotation_path) as im:separator = im.convert("L")
    with Image.open(root/sample.word_owner_annotation_path) as im:owner = im.convert("I")
    source = np.asarray(target)>0
    valid = np.asarray(validity)>0
    own = np.asarray(owner)
    if not np.array_equal(source,own>0) or np.any(source & ~valid):
        raise ValueError(f"Source target/ownership/ignore mismatch: {sample.sample_id}")
    result = {"sample_id":sample.sample_id,"protocol_sha256":protocol,"image_wh":size,
        "source_annotation_sha256":file_sha256(root/annotation),
        "source":owner_topology(source,own,expected),"resolutions":{}}
    for name,shape in sizes.items():
        raw,final,owners,valid = native_target_audit_maps(target,validity,separator,owner,shape,threshold=threshold)
        result["resolutions"][name] = {"image_hw":shape,"before_native_guard":owner_topology(raw,owners,expected),
            "after_native_guard":owner_topology(final,owners,expected),
            "foreground_pixels_before":int(raw.sum()),"foreground_pixels_after":int(final.sum())}
    return result


def run(args):
    config = json.loads(args.config.read_text())
    if config["reproduction_mode"] != "approximate" or args.config.resolve().parent.name != "approximate":
        raise ValueError("Explicit approximate mask-audit configuration required")
    populations = {"icdar2015":args.icdar_root.resolve(),"total_text":args.totaltext_root.resolve()}
    manifests = {}
    for name,root in populations.items():
        path=root/config["datasets"][name]["manifest"]
        if file_sha256(path)!=config["datasets"][name]["manifest_sha256"]:
            raise ValueError(f"Qualified manifest differs: {name}")
        m=load_segmentation_manifest(path,require_checksums=True)
        if m.dataset!=name or m.split_role!="train" or len(m.samples)!=config["datasets"][name]["images"]:
            raise ValueError("Complete official TRAIN population required")
        SegmentationDataset(m,root,verify_checksums=True)
        manifests[name]=m
    protocol={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":"TRAIN target integrity audit only","config_sha256":file_sha256(args.config),
        "source_sha256":file_sha256(Path(__file__)),
        "target_transform_source_sha256":file_sha256(Path(__file__).resolve().parents[1]/"segmentation/icdar2015_improvements.py"),
        "datasets":config["datasets"],"resolutions":config["resolutions"],"threshold":config["probability_threshold"]}
    digest=sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    protocol_path=args.output_dir/"protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise ValueError("Mask audit protocol changed")
    else:write_json(protocol_path,protocol)
    rows={name:{} for name in populations};tasks=[]
    for name,m in manifests.items():
        for sample in m.samples:
            path=args.output_dir/name/(sample.sample_id.replace(":","_")+".json")
            if path.exists():
                row=json.loads(path.read_text())
                if row["protocol_sha256"]!=digest:raise ValueError("Cached target audit differs")
                rows[name][sample.sample_id]=row
            else:tasks.append((str(populations[name]),name,sample,config["resolutions"],config["probability_threshold"],digest))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number,result in enumerate(pool.map(_one,tasks,chunksize=4),1):
            name=result["sample_id"].split(":",1)[0]
            write_json(args.output_dir/name/(result["sample_id"].replace(":","_")+".json"),result)
            rows[name][result["sample_id"]]=result
            if number%100==0 or number==len(tasks):print(f"[word-mask audit] {number}/{len(tasks)}",flush=True)
    aggregate={}
    for name,items in rows.items():
        if len(items)!=config["datasets"][name]["images"]:raise ValueError("Incomplete mask audit")
        source={k:sum(row["source"][k] for row in items.values()) for k in next(iter(items.values()))["source"]}
        scales={}
        for scale in config["resolutions"]:
            scales[scale]={}
            for stage in ["before_native_guard","after_native_guard"]:
                example=next(iter(items.values()))["resolutions"][scale][stage]
                scales[scale][stage]={k:sum(r["resolutions"][scale][stage][k] for r in items.values()) for k in example}
            for field in ["foreground_pixels_before","foreground_pixels_after"]:
                scales[scale][field]=sum(r["resolutions"][scale][field] for r in items.values())
        aggregate[name]={"images":len(items),"source":source,"resolutions":scales}
    report={"status":"SUCCESS",**protocol,"aggregate":aggregate}
    write_json(args.output_dir/"results.json",report)
    print(json.dumps(aggregate,sort_keys=True),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ["config","icdar-root","totaltext-root","output-dir"]:parser.add_argument("--"+flag,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    run(parser.parse_args())


if __name__=="__main__":main()
