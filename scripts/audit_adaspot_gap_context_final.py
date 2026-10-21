"""Independent audit of the explicitly post-hoc fixed final-epoch diagnostic."""

import argparse
import json
from pathlib import Path

from scripts.audit_adaspot_phase0 import count, digest


def audit(root,primary_root):
    if not __debug__:raise RuntimeError("Audit assertions must be enabled")
    data=json.loads((root/"results.json").read_text())
    primary=json.loads((primary_root/"results.json").read_text())
    primary_audit=json.loads((primary_root/"independent_audit.json").read_text())
    assert digest(primary_root/"results.json")==primary_audit["result_sha256"]
    assert data["status"]=="SUCCESS" and data["analysis_kind"]=="post_hoc_fixed_final_epoch_diagnostic"
    assert data["primary_results_unchanged"] is True
    for path,h in data["source_hashes"].items():assert digest(Path(path))==h,path
    readings=json.loads((root/"ocr_readings.json").read_text())
    counts={};cases=0
    for name,result in data["arms"].items():
        assert result["epoch"]==primary["protocol"]["resolved_config"]["epochs"]==20
        last={r["sample_id"]:r for r in primary["arms"][name]["history"][-1]["validation"]["per_image"]}
        for row in result["native_per_image"]:
            assert row["metrics"]==last[row["sample_id"]]["metrics"]
            assert row["audit"]==last[row["sample_id"]]["audit"]
        exact=localized=words=proposals=0
        for row in result["ocr"]["per_image"]:
            pred,gt=row["predictions"],row["ground_truth"]
            e,g=count(pred,gt,True),count(pred,gt,False)
            assert e==row["counts"]["exact_casefold_iou_ge_50"]
            assert g==row["counts"]["geometry_iou_ge_50"]
            for p in pred:assert p["text"]==readings[p["crop_rgb_sha256"]]["raw_text"]
            exact+=e;localized+=g;words+=len(gt);proposals+=len(pred);cases+=1
        a=result["ocr"]["aggregate"]
        assert words==a["gt_words"]==primary["protocol"]["resolved_config"]["expected_words"]
        assert exact==a["exact_casefold_iou_ge_50"] and localized==a["geometry_iou_ge_50"]
        assert proposals==a["proposals"] and a["hmean"]==2*exact/(words+proposals)
        assert exact-524==result["changes_vs_baseline"]["net"]
        counts[name]=exact
    report={"status":"PASS","analysis_kind":data["analysis_kind"],"image_cases":cases,
        "exact_by_arm":counts,"epoch20_native_metrics_and_audits_match_training":True,
        "primary_result_unchanged_sha256":digest(primary_root/"results.json"),
        "result_sha256":digest(root/"results.json"),"auditor_sha256":digest(Path(__file__))}
    text=json.dumps(report,indent=2,sort_keys=True)+"\n";path=root/"independent_audit.json"
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    print(text)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root",type=Path,required=True);p.add_argument("--primary-root",type=Path,required=True)
    args=p.parse_args();audit(args.run_root,args.primary_root)
