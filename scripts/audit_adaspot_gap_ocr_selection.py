"""Independent calibration-only OCR selection and evaluation count audit."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from localize.adaspot_gap_preserve import decode_gap
from scripts.audit_adaspot_phase0 import count, digest
from scripts.audit_adaspot_gap_adaptation import check_native
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash


def check_ocr(value, population, readings):
    assert len(value["per_image"]) == len(population)
    assert {s["sample_id"] for s in value["per_image"]} == population
    total = Counter()
    for sample in value["per_image"]:
        pred, gt = sample["predictions"], sample["ground_truth"]
        counts = {"gt_words": len(gt), "proposals": len(pred), "geometry_iou_ge_50": count(pred, gt, False), "exact_casefold_iou_ge_50": count(pred, gt, True)}
        assert all(sample["counts"][k] == v for k,v in counts.items())
        assert all(p["text"] == readings[p["crop_rgb_sha256"]]["raw_text"] for p in pred)
        total.update(counts)
    assert all(value["aggregate"][k] == v for k,v in total.items())
    assert value["aggregate"]["hmean"] == 2*total["exact_casefold_iou_ge_50"]/(total["gt_words"]+total["proposals"])
    return total


def audit(root):
    if not __debug__: raise RuntimeError("Assertions required")
    result = json.loads((root/"results.json").read_text()); assert result["status"] == "SUCCESS"
    protocol = result["protocol"]; assert protocol["mode"] == "approximate" and not protocol["selection_uses_evaluation"]
    assert result["result_class"] == "approximate reproduction"
    for path,h in protocol["source_hashes"].items(): assert digest(Path(path)) == h, path
    fit,calibration,evaluation = [set(protocol[k]) for k in ("fit_ids","calibration_ids","evaluation_ids")]
    assert tuple(map(len,(fit,calibration,evaluation))) == (1030,100,125)
    assert not (fit & calibration or fit & evaluation or calibration & evaluation)
    candidates = json.loads((root/"calibration_candidates.json").read_text())
    assert len(candidates) == len(protocol["candidates"]) == 21
    assert [{k:v for k,v in r.items() if k!="ocr"} for r in candidates] == protocol["candidates"]
    readings = json.loads((root/"calibration_readings.json").read_text())
    frozen = json.loads((root/"calibration_frozen_inputs.json").read_text())
    for row in candidates:
        total = check_ocr(row["ocr"],calibration,readings)
        assert total["gt_words"] == candidates[0]["ocr"]["aggregate"]["gt_words"]
        for raw,scored in zip(frozen[row["name"]],row["ocr"]["per_image"]):
            assert raw["sample_id"] == scored["sample_id"] and raw["ground_truth"] == scored["ground_truth"]
            assert [{k:v for k,v in p.items() if k!="text"} for p in scored["predictions"]] == raw["predictions"]
    selected = max(candidates,key=lambda r:(r["ocr"]["aggregate"]["exact_casefold_iou_ge_50"],r["ocr"]["aggregate"]["hmean"],-r["epoch"],r["threshold"] if r["threshold"] is not None else 1.,r["decoder"]=="cut"))
    assert selected == result["selection"] == json.loads((root/"selection.json").read_text())
    readings.update(json.loads((root/"evaluation_readings.json").read_text()))
    for h in readings:
        with Image.open(root/"crops"/(h+".png")) as image: assert crop_array_hash(np.asarray(image.convert("RGB"))) == h
    base=Path(protocol["source_run"]);pp=json.loads((base/"control/run_contract.json").read_text())["resolved_config"]["postprocessing"]
    exact_by_arm={};native_cases=0
    for name,arm in result["arms"].items():
        total=check_ocr(arm["ocr"],evaluation,readings);assert total["gt_words"]==983
        exact_by_arm[name]=total["exact_casefold_iou_ge_50"]
        choice=selected if name=="selected" else {"threshold":None,"decoder":"cut"}
        for row in arm["native"]["per_image"]:
            sid=row["sample_id"]
            with np.load(root/name/"evaluation_maps"/(sid.replace(":","__")+".npz")) as z: labels,gap=z["labels"],z["gap_probability"]
            with np.load(base/"control/validation_exports"/sid.replace(":","__")/"maps.npz") as z:maps={k:z[k] for k in z.files}
            assert np.array_equal(labels,decode_gap(maps["probabilities"],gap,maps["canvas"],pp,choice["threshold"],choice["decoder"]))
            check_native(labels,maps,row["counts"]);native_cases+=1
        assert arm["changes_vs_baseline"]["net"]==total["exact_casefold_iou_ge_50"]-524
    assert exact_by_arm["baseline"]==524
    report={"status":"PASS","method":"independent SciPy Hungarian per-image exact/localization counts; all calibration-only choices and crop/source hashes",
        "calibration_image_arm_cases":len(calibration)*len(candidates),"evaluation_image_arm_cases":len(evaluation)*len(result["arms"]),
        "native_image_cases":native_cases,"selected_epoch":selected["epoch"],"selected_threshold":selected["threshold"],"selected_decoder":selected["decoder"],
        "exact_by_arm":exact_by_arm,"result_sha256":digest(root/"results.json"),"auditor_sha256":digest(Path(__file__))}
    text=json.dumps(report,indent=2,sort_keys=True)+"\n";p=root/"independent_audit.json"
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)
    print(text)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-root",type=Path,required=True);audit(p.parse_args().run_root)
