"""Freeze one completed branch using validation correct-word F1, before TEST."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from segmentation.data import file_sha256
from segmentation.icdar2015_improvement_runtime import load_config
from scripts.train_icdar2015_totaltext_transfer import write_json


def choose(rows):
    return sorted(rows,key=lambda name:(-rows[name]["word_f1"],-rows[name]["geometry_f1"],
        rows[name]["rgb_pixels"],name))[0]


def verify_selection(path, training_run, config):
    report=json.loads(path.read_text())
    if (report["status"]!="FROZEN_BEFORE_TEST" or report["config"]!=config
            or report["selected_branch"]!=choose(report["branches"])
            or report["selector_sha256"]!=file_sha256(Path(__file__))):
        raise ValueError("Invalid TRAIN-only branch selection")
    selected=report["branches"][report["selected_branch"]]
    if Path(selected["training_run"]).resolve()!=training_run.resolve():
        raise ValueError("TEST may evaluate only the validation-selected branch")
    for row in report["branches"].values():
        for field,hash_field in (("validation_report","validation_report_sha256"),
                                 ("training_report","training_report_sha256")):
            if file_sha256(Path(row[field]))!=row[hash_field]:raise ValueError("Selection evidence changed")
        if file_sha256(Path(row["training_run"])/"best.pt")!=row["checkpoint_sha256"]:
            raise ValueError("Selection checkpoint changed")
        validation=json.loads(Path(row["validation_report"]).read_text())
        if (row["word_f1"]!=validation["aggregate"]["alphanumeric_casefold"]["f1"]
                or row["geometry_f1"]!=validation["aggregate"]["geometry"]["f1"]):
            raise ValueError("Selection scores differ from frozen validation")
    return report


def run(args):
    config=load_config(args.config);rows={};ids=None
    for branch in config["branches"]:
        v=args.validation_root/branch/"results.json";t=args.suite_root/branch/"run_manifest.json"
        validation,trained=json.loads(v.read_text()),json.loads(t.read_text())
        if (validation["status"]!="SUCCESS" or validation["protocol"]["branch"]!=branch
                or validation["protocol"]["origin_report_sha256"]!=file_sha256(t)
                or validation["protocol"]["config_sha256"]!=file_sha256(args.config)
                or trained["run_contract"]["config"]!=config or not trained["training_complete"]
                or trained["completed_epochs"]!=config["training"]["epochs"]
                or validation["aggregate"]["images"]!=100 or validation["aggregate"]["gt_words"]!=430):
            raise ValueError("All four completed matched branches and validation100 required")
        population={row["sample_id"] for row in validation["per_image"]}
        if ids is not None and ids!=population:raise ValueError("Validation populations differ")
        ids=population;shape=trained["run_contract"]["image_hw"]
        if file_sha256(args.suite_root/branch/"best.pt")!=trained["checkpoint"]["sha256"]:
            raise ValueError("Best checkpoint changed")
        rows[branch]={"word_f1":validation["aggregate"]["alphanumeric_casefold"]["f1"],
            "geometry_f1":validation["aggregate"]["geometry"]["f1"],"rgb_pixels":shape[0]*shape[1],
            "image_hw":shape,"checkpoint_epoch":trained["checkpoint"]["epoch"],
            "checkpoint_sha256":trained["checkpoint"]["sha256"],
            "training_run":str((args.suite_root/branch).resolve()),
            "training_report":str(t.resolve()),"training_report_sha256":file_sha256(t),
            "validation_report":str(v.resolve()),"validation_report_sha256":file_sha256(v)}
    report={"status":"FROZEN_BEFORE_TEST","reproduction_mode":"approximate",
        "claim_label":"approximate reproduction","config":config,"config_sha256":file_sha256(args.config),
        "selected_branch":choose(rows),"branches":rows,"validation_ids":sorted(ids),
        "created_utc":datetime.now(timezone.utc).isoformat(),"selector_sha256":file_sha256(Path(__file__))}
    if args.output.exists():
        previous=verify_selection(args.output,Path(rows[report["selected_branch"]]["training_run"]),config)
        if previous["branches"]!=rows:raise ValueError("Frozen selection differs")
        return previous
    write_json(args.output,report);print(json.dumps(report,sort_keys=True),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","suite-root","validation-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":main()
