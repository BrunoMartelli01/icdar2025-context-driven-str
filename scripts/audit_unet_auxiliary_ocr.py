"""Read-only numerical/source audit of shared-feature auxiliary OCR experiments."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from segmentation.totaltext_word_kernel import instance_geometry
from scripts.run_unet_auxiliary_ocr import json_write


def independent_counts(predictions, ground_truth):
    n, m = len(predictions), len(ground_truth)
    geometric = np.zeros((n,m), dtype=bool)
    text = np.zeros((n,m), dtype=bool)
    for i,p in enumerate(predictions):
        for j,g in enumerate(ground_truth):
            a,b=p["bbox"],g["bbox"]
            inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
            geometric[i,j]=union>0 and inter/union>=.5
            text[i,j]=geometric[i,j] and bool(p["text"]) and p["text"].casefold()==g["text"].casefold()
    def match(edges):
        if not n or not m:return 0
        a,b=linear_sum_assignment(-edges.astype(np.int32))
        return int(edges[a,b].sum())
    return {"gt_words":m,"proposals":n,"geometry_iou_ge_50":match(geometric),
            "exact_casefold_iou_ge_50":match(text)}


def check_ocr(report, expected_ids, root):
    rows=report["per_image"]
    assert len(rows)==len(expected_ids) and {r["sample_id"] for r in rows}==set(expected_ids)
    totals=Counter();crops=0
    for row in rows:
        counts=independent_counts(row["predictions"],row["ground_truth"])
        assert all(counts[k]==row["counts"][k] for k in counts),(row["sample_id"],counts)
        totals.update(counts)
        for p in row["predictions"]:
            digest=p["crop_rgb_sha256"]
            with Image.open(p["crop_path"]) as image:actual=crop_array_hash(np.asarray(image.convert("RGB")))
            assert actual==digest
            entry=json.loads((root/"parseq_cache"/(digest+".json")).read_text())
            assert p["text"]==entry["reading"]["raw_text"] and entry["crop_rgb_sha256"]==digest
            crops+=1
    a=report["aggregate"];n,g,tp=totals["proposals"],totals["gt_words"],totals["exact_casefold_iou_ge_50"]
    assert all(a[k]==v for k,v in totals.items())
    assert abs(a["precision"]-(tp/n if n else 0.))<1e-12
    assert abs(a["recall"]-(tp/g if g else 0.))<1e-12
    assert abs(a["hmean"]-(2*tp/(n+g) if n+g else 0.))<1e-12
    return len(rows),crops


def audit(root):
    root=Path(root);d=json.loads((root/"results.json").read_text())
    assert d["status"]=="SUCCESS"
    protocol=d["protocol"];c=protocol["resolved_config"];ids=protocol["split_ids"]
    for a,b in [("fit","calibration"),("fit","evaluation"),("calibration","evaluation")]:
        assert not set(ids[a])&set(ids[b])
    for path,digest in protocol["source_hashes"].items():assert sha256_file(Path(path))==digest,path
    warm=json.loads((root/"warmup.json").read_text())
    assert warm["status"]=="SUCCESS" and warm["source_state_unchanged"]
    assert warm["reader_initial_sha256"]!=warm["reader_trained_sha256"]
    assert sha256_file(root/"warmup.pt")==warm["warmup_checkpoint_sha256"]
    for arm in ("detached","connected"):
        t=d["training"][arm]
        assert t["status"]=="SUCCESS" and t["completed_epochs"]==c["epochs"]
        assert [h["epoch"] for h in t["history"]]==list(range(1,c["epochs"]+1))
        for h in t["history"]:
            assert len(h["sample_ids"])==len(ids["fit"]) and set(h["sample_ids"])==set(ids["fit"])
            assert h["steps"]>0 and h["selected_words"]>0 and not h["memory"]["oom"]
            p=h["first_step"]
            assert p["status"]=="PASS" and p["reader_gradient_l1"]>0
            assert (p["ocr_to_features_gradient_l1"]==0) if arm=="detached" else (p["ocr_to_features_gradient_l1"]>0)
        assert sha256_file(root/arm/"last.pt")==t["last_checkpoint_sha256"]
    for a,b in zip(d["training"]["detached"]["history"],d["training"]["connected"]["history"]):
        assert a["sample_order_sha256"]==b["sample_order_sha256"]
    cases=crops=0
    calibration=json.loads((root/"calibration.json").read_text())
    for arm,candidates in calibration.items():
        for candidate in candidates:
            n,k=check_ocr(candidate,ids["calibration"],root);cases+=n;crops+=k
        winner=max(candidates,key=lambda x:(x["aggregate"]["exact_casefold_iou_ge_50"],x["aggregate"]["hmean"],-x["epoch"]))
        assert winner["epoch"]==d["selections"][arm]["epoch"]
        selected=d["selections"][arm]
        assert sha256_file(Path(selected["checkpoint_path"]))==selected["checkpoint_sha256"]
    for arm,value in d["arms"].items():
        n,k=check_ocr(value["ocr"],ids["evaluation"],root);cases+=n;crops+=k
        frozen=json.loads((root/"evaluation"/arm/"frozen.json").read_text())
        native=Counter()
        for row in frozen["rows"]:
            p=Path(row["maps_path"]);assert sha256_file(p)==row["maps_sha256"]
            with np.load(p,allow_pickle=False) as saved:
                labels=saved["labels"].copy();valid=saved["validity"]
                for instance in np.unique(labels[labels>0]):
                    if not np.any((labels==instance)&valid):labels[labels==instance]=0
                counts=instance_geometry(labels,saved["owners"],saved["expected_owner_ids"].tolist())
            assert counts==row["native"];native.update(counts)
        assert dict(native)==value["native_counts"]
    # Counts and matched identities must agree on the paired net gain.
    delta=d["arms"]["connected"]["ocr"]["aggregate"]["exact_casefold_iou_ge_50"]-d["arms"]["detached"]["ocr"]["aggregate"]["exact_casefold_iou_ge_50"]
    assert delta==d["connected_vs_detached"]["net"]
    return {"status":"PASS","result_class":d["result_class"],"results_sha256":sha256_file(root/"results.json"),
        "image_candidate_cases":cases,"crop_cases":crops,"paired_net_words":delta,
        "epochs_per_arm":c["epochs"],"method":"Independent SciPy Hungarian bbox+exact text counts; reader/feature gradients, fit populations/order, calibration selection, checkpoint/source/RGB hashes and native instance replay"}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    result=audit(a.run);json_write(a.output,result,immutable=True);print(json.dumps(result,indent=2))
