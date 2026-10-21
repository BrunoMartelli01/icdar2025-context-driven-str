"""Independent counts and matched-training audit for Gap context experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_adaspot_phase0 import count, digest


def audit(root):
    if not __debug__:
        raise RuntimeError("Audit assertions must be enabled; do not use Python -O")
    result = json.loads((root / "results.json").read_text())
    assert result["status"] == "SUCCESS"
    protocol, arms = result["protocol"], result["arms"]
    config = protocol["resolved_config"]
    assert protocol["mode"] == "approximate"
    assert not set(protocol["train_ids"]) & set(protocol["validation_ids"])
    assert len(protocol["train_ids"]) == config["expected_train_images"]
    assert len(protocol["validation_ids"]) == config["expected_validation_images"]
    for path,h in protocol["source_hashes"].items(): assert digest(Path(path)) == h, path
    for path,h in result["frozen_map_sha256"].items(): assert digest(Path(path)) == h, path
    previous = Path(protocol["previous_probe"])
    old_manifest = json.loads((previous / "feature_manifest.json").read_text())
    old_pixels = torch.load(previous / "training_pixels.pt",map_location="cpu",weights_only=True)
    assert digest(previous / "training_pixels.pt") == old_manifest["train_cache_sha256"]
    manifest = json.loads((root / "training_manifest.json").read_text())
    assert digest(Path(manifest["cache_path"])) == manifest["cache_sha256"]
    pixels = torch.load(manifest["cache_path"],map_location="cpu",weights_only=True)
    counts = torch.bincount(pixels["strata"],minlength=4)
    assert counts.tolist() == manifest["stratum_counts"]
    assert torch.allclose(counts.float()*torch.tensor(manifest["stratum_weights"]),torch.full((4,),len(pixels["strata"])/4))
    offset, old_offset, central_cases = 0, 0, 0
    assert len(manifest["per_image"]) == len(old_manifest["train"])
    for row,old in zip(manifest["per_image"],old_manifest["train"]):
        assert row["sample_id"] == old["sample_id"]
        a,b,c,d = row["stratum_counts"]
        assert [a+b,c,d] == old["stratum_counts"]
        assert digest(Path(row["positions_path"])) == row["positions_sha256"]
        with np.load(row["positions_path"],allow_pickle=False) as saved:
            positions,classes = saved["positions"],saved["strata"]
        assert len(positions) == len(set(positions.tolist())) == a+b+c+d
        assert np.array_equal(classes,pixels["strata"][offset:offset+len(positions)].numpy())
        # Independent alignment check against the earlier fixed-feature probe:
        # sort the split positive groups back into the original raster order.
        centers = pixels["patches"][offset:offset+len(positions),4::9]
        ordered = torch.cat((centers[:a+b][torch.from_numpy(np.argsort(positions[:a+b]))],centers[a+b:]))
        expected = old_pixels["features"][old_offset:old_offset+len(positions)]
        assert torch.equal(ordered,expected), row["sample_id"]
        offset += len(positions); old_offset += len(positions); central_cases += len(positions)
    assert offset == old_offset == len(pixels["patches"]) == len(old_pixels["features"])
    del pixels,old_pixels
    epoch_orders=[]
    source=Path(protocol["source_run"])
    native_cases,ocr_cases,exact_by_arm=0,0,{}
    readings=json.loads((root/"ocr_readings.json").read_text())
    baseline_readings=json.loads((Path(protocol["phase0_run"])/"ar1_readings.json").read_text())
    differing_common=[key for key in readings if key in baseline_readings and readings[key]["raw_text"] != baseline_readings[key]["raw_text"]]
    for name,arm in arms.items():
        assert arm["completed_epochs"] == config["epochs"]
        assert [r["epoch"] for r in arm["history"]] == list(range(config["epochs"]+1))
        assert all(r["cuda_steps"] > 0 for r in arm["history"][1:])
        selected=max(arm["history"],key=lambda r:(r["validation"]["aggregate"]["word_f1"],
            -r["validation"]["aggregate"]["gt_words_touched_by_fused_components"],-r["epoch"]))
        assert selected["epoch"] == arm["best_epoch"]
        epoch_orders.append([r.get("sample_order_sha256") for r in arm["history"]])
        assert digest(root/name/"best.pt") == arm["best_checkpoint_sha256"]
        smoke=json.loads((root/name/"first_cuda_step.json").read_text())
        assert smoke["device"].startswith("cuda") and smoke["weight_change_linf"]>0 and smoke["point_native_error"]<=2e-5
        assert smoke["parameters"] == arm["parameters"] == (65 if name == "pointwise" else 577)
        geometry={r["sample_id"]:r["metrics"] for r in selected["validation"]["per_image"]}
        for sid,metrics in geometry.items():
            with np.load(root/name/"predicted_maps"/(sid.replace(":","__")+".npz")) as saved:
                labels=saved["labels"]
            with np.load(source/"control/validation_exports"/sid.replace(":","__")/"maps.npz") as maps:
                valid,owners,owner_ids=maps["validity"],maps["owners"],maps["expected_owner_ids"]
            def boxes(raster,ids):
                output=[]
                for i in ids:
                    yy,xx=np.where(raster==i)
                    if len(xx):output.append({"bbox":[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]})
                return output
            care_ids=[i for i in np.unique(labels[labels>0]) if np.any((labels==i)&valid)]
            pred=boxes(labels,care_ids);gt=boxes(owners,owner_ids)
            assert count(pred,gt,False)==metrics["matched_gt_iou_ge_50"]
            assert len(pred)==metrics["raw_proposals"]
            native_cases+=1
        exact=localized=words=proposals=0
        assert {r["sample_id"] for r in arm["ocr"]["per_image"]}==set(protocol["validation_ids"])
        for row in arm["ocr"]["per_image"]:
            pred,gt=row["predictions"],row["ground_truth"]
            e,g=count(pred,gt,True),count(pred,gt,False)
            assert e==row["counts"]["exact_casefold_iou_ge_50"]
            assert g==row["counts"]["geometry_iou_ge_50"]
            for p in pred:
                assert p["text"]==readings[p["crop_rgb_sha256"]]["raw_text"]
            exact+=e;localized+=g;words+=len(gt);proposals+=len(pred);ocr_cases+=1
        total=arm["ocr"]["aggregate"]
        assert words==total["gt_words"]==config["expected_words"]
        assert exact==total["exact_casefold_iou_ge_50"] and localized==total["geometry_iou_ge_50"]
        assert proposals==total["proposals"]
        assert total["hmean"]==2*exact/(words+proposals)
        assert exact-524==arm["changes_vs_baseline"]["net"]
        exact_by_arm[name]=exact
    assert epoch_orders[0]==epoch_orders[1]
    assert not differing_common, f"Unchanged crop OCR differed from baseline: {differing_common}"
    report={"status":"PASS","method":"independent SciPy Hungarian native/RGB counts; exact center-feature equality against previous probe; epoch/order/cache/selection audit",
        "native_image_cases":native_cases,"ocr_image_cases":ocr_cases,"central_feature_cases":central_cases,
        "completed_epochs_per_arm":config["epochs"],"exact_by_arm":exact_by_arm,
        "common_crop_ocr_differences":differing_common,"result_sha256":digest(root/"results.json"),"auditor_sha256":digest(Path(__file__))}
    text=json.dumps(report,indent=2,sort_keys=True)+"\n"
    path=root/"independent_audit.json"
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    print(text)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-root",type=Path,required=True)
    audit(p.parse_args().run_root)
