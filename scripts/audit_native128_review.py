"""Independent numerical audit of matched native128 OCR review."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.audit_totaltext_compact_v2_selected_ocr import independent_metrics, check_distribution
from segmentation.data import file_sha256


def audit(directory):
    protocol = json.loads((directory/"protocol.json").read_text())
    results = json.loads((directory/"results.json").read_text())
    assert results["status"] == "SUCCESS" and results["protocol_sha256"] == file_sha256(directory/"protocol.json")
    for path,digest in protocol["code_sha256"].items():
        assert file_sha256(Path(path)) == digest,path
    config = protocol["config"]
    for model in config["models"]:
        for path,hash_field in [("source_manifest","source_manifest_sha256"),("config","config_sha256"),("checkpoint","checkpoint_sha256")]:
            assert file_sha256(Path(model[path])) == model[hash_field],model["id"]
    gt_data = json.loads((directory/"ground_truth.json").read_text())
    gt = gt_data["ground_truth"]
    assert set(gt) == set(protocol["sample_ids"])
    assert sum(map(len,gt.values())) == results["care_words"]
    for annotation in gt_data["annotation_provenance"]:
        assert file_sha256(Path(annotation["path"])) == annotation["sha256"]
    ocr={}
    for line in (directory/"ocr.jsonl").read_text().splitlines():
        row=json.loads(line)
        assert row["protocol_sha256"] == results["protocol_sha256"]
        assert row["crop_array_sha256"] not in ocr
        ocr[row["crop_array_sha256"]]=row
    used=set(); rows=[]
    for key,report in results["pipelines"].items():
        shards=[json.loads(p.read_text()) for p in sorted((directory/key/"proposals").glob("*.json"))]
        assert len(shards)==results["images_per_pipeline"]
        assert {s["sample_id"] for s in shards}==set(gt)
        for s in shards:
            assert s["protocol_sha256"]==results["protocol_sha256"]
            m=s["mask_measurements"]
            assert m["foreground_pixels"]==sum(x["foreground_area_native"] for x in m["components_8_connected"])
        check_distribution(report["mask_summary"]["foreground_pixels_per_image"],[s["mask_measurements"]["foreground_pixels"] for s in shards])
        for pad, metrics in report["summary"].items():
            total=Counter()
            for s in shards:
                proposals=s["variants"][pad]["proposals"]
                assert len(proposals)<=config["max_proposals"]
                for p in proposals:
                    used.add(p["crop_array_sha256"])
                    x0,y0,x1,y1=p["bbox_native"]
                    assert (x1-x0)*(y1-y0)==p["bbox_area_native"]>=report["min_bbox_area"]
                    if pad=="original6":
                        x0,y0,x1,y1=p["bbox"];h,w=s["image_hw"];d=config["original_padding"]
                        assert p["padded_bbox_original"]==[max(0,x0-d),max(0,y0-d),min(w,x1+d),min(h,y1+d)]
                        assert p["padded_bbox_native"] is None
                computed=independent_metrics(proposals,gt[s["sample_id"]],"bbox",ocr)
                total.update(computed)
                rows.append({"pipeline":key,"padding":pad,"sample_id":s["sample_id"],**computed})
            for field,value in total.items():
                source=metrics["geometry"] if field in metrics["geometry"] else metrics["recognition"]
                assert source[field]==value,(key,pad,field,source[field],value)
    assert used==set(ocr) and len(ocr)==results["unique_rgb_crops"]
    report={"status":"PASS","claim_label":"approximate reproduction","models":len(config["models"]),
        "pipelines":len(results["pipelines"]),"images_per_pipeline":results["images_per_pipeline"],
        "care_words":results["care_words"],"independent_per_image_evaluations":len(rows),
        "method":"NumPy IoU, SciPy maximum matching, Counter lexical; all models and variants",
        "original_coordinate_padding_checked":True,"source_checkpoint_hashes_verified":True,"per_image":rows}
    (directory/"audit.json").write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="per_image"}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-directory",required=True,type=Path)
    audit(p.parse_args().result_directory)
