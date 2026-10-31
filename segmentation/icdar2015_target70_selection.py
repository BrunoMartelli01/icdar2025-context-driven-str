"""Freeze approximate ICDAR model/policy choices exclusively from TRAIN VAL100."""

import json
from pathlib import Path

from models.icdar2015_improvements import ImprovementAGUNet
from models.icdar2015_target70 import Target70AGUNet
from recognition.icdar2015_word_f1 import aggregate_counts,sample_counts
from segmentation.data import file_sha256
from segmentation.icdar2015_improvement_runtime import REPO,training_config as legacy_training_config
from segmentation.icdar2015_operating_policy import operating_selection_key
from segmentation.icdar2015_target70 import load_profile,training_config,calibrate_emission
from segmentation.training import load_checkpoint


def load_selection_profile(path):
    config=json.loads(path.read_text())
    if (path.resolve().parent.name!="approximate" or config["reproduction_mode"]!="approximate"
            or config["claim_label"]!="approximate reproduction"):
        raise ValueError("Explicit approximate selection profile required")
    for prefix in ("training","operating","source_training"):
        if file_sha256(REPO/config[prefix+"_profile"])!=config[prefix+"_profile_sha256"]:
            raise ValueError("Selection profile dependency changed: "+prefix)
    if config["required_branches"]!=["long768","adapt1280"] or config["include_qualified_source"] is not True:
        raise ValueError("Both declared training branches and qualified source are required")
    return config


def completed_run(run,branch,profile,profile_sha,fit_ids,val_ids):
    path=run/"run_manifest.json"
    if not path.exists():raise ValueError("Full training has not completed: "+branch)
    report=json.loads(path.read_text());contract=report["run_contract"]
    if (report["status"]!="SUCCESS" or report["training_complete"] is not True
            or report["result_class"]!="approximate reproduction"
            or report["completed_epochs"]!=profile["training"]["epochs"]):
        raise ValueError("Only completed full training can enter selection: "+branch)
    if (contract["branch"]!=branch or contract["config_sha256"]!=profile_sha or contract["config"]!=profile
            or contract["branch_config"]!=profile["branches"][branch]
            or contract["smoke_steps"] is not None or contract["precision"]!="float32"):
        raise ValueError("Training contract differs: "+branch)
    if contract["train_ids"]!=fit_ids or contract["validation_ids"]!=val_ids or contract["evaluated_ids"]!=val_ids:
        raise ValueError("Fixed TRAIN populations differ: "+branch)
    required={"models/icdar2015_target70.py","segmentation/icdar2015_target70.py","scripts/train_icdar2015_target70.py"}
    if not required.issubset(contract["source_sha256"]):raise ValueError("Training source provenance is incomplete")
    for name,digest in contract["source_sha256"].items():
        if file_sha256(REPO/name)!=digest:raise ValueError("Training scientific source changed: "+name)
    history=json.loads((run/"epochs.json").read_text())
    if ([r["epoch"] for r in history]!=list(range(1,profile["training"]["epochs"]+1))
            or any(r["train"]["samples"]!=len(fit_ids) for r in history)):
        raise ValueError("Incomplete full training history: "+branch)
    checkpoint=run/report["checkpoint"]["path"]
    if file_sha256(checkpoint)!=report["checkpoint"]["sha256"]:raise ValueError("Selected checkpoint changed")
    return report


def verified_policy(path,operating,operating_sha,geometry,words,shape,refs,training):
    result=json.loads(path.read_text());contract=result["run_contract"]
    if (result["status"]!="SUCCESS" or result["reproduction_mode"]!="approximate"
            or result["claim_label"]!="approximate reproduction" or result["failed_samples"]!=0
            or contract["config"]!=operating or contract["config_sha256"]!=operating_sha
            or contract["geometry_sha256"]!=file_sha256(geometry)
            or contract["expected_raw_sha256"]!=file_sha256(words) or contract["image_hw"]!=shape):
        raise ValueError("Operating policy provenance differs")
    for name,digest in contract["source_sha256"].items():
        if file_sha256(REPO/name)!=digest:raise ValueError("Operating source changed: "+name)
    candidates=result["candidates"]
    expected={(threshold,method) for threshold in operating["probability_thresholds"] for method in operating["crop_methods"]}
    if len(candidates)!=len(expected) or {(c["probability_threshold"],c["crop_method"]) for c in candidates}!=expected:
        raise ValueError("Complete predeclared operating grid required")
    if result["selected"]!=max(candidates,key=operating_selection_key):raise ValueError("Operating winner changed")
    for candidate in candidates:
        source=path.parent/candidate["per_image_file"]
        if file_sha256(source)!=candidate["per_image_sha256"]:raise ValueError("Per-image policy outputs changed")
        rows=json.loads(source.read_text())
        if len(rows)!=len(refs) or {r["sample_id"] for r in rows}!=set(refs):raise ValueError("Complete fixed validation required")
        for row in rows:
            if row["ground_truth"]!=refs[row["sample_id"]]["ground_truth"]:raise ValueError("Care GT changed")
        raw=aggregate_counts([sample_counts(r["predictions"],r["ground_truth"],
            iou_threshold=operating["matching_iou_threshold"]) for r in rows])
        if raw!=candidate["raw_aggregate"]:raise ValueError("Raw validation metric replay changed")
        # Replay every gate against all original care words; a selected policy
        # cannot conceal a removed short word or an omitted empty prediction.
        replay=calibrate_emission(rows,training)
        if replay["policy"]!=candidate["emission_policy"] or replay["aggregate"]!=candidate["emitted_aggregate"]:
            raise ValueError("Validation policy/metric replay changed")
        for metric in (candidate["raw_aggregate"],candidate["emitted_aggregate"]):
            if metric["images"]!=len(refs) or metric["gt_words"]!=sum(len(r["ground_truth"]) for r in refs.values()):
                raise ValueError("Validation denominator changed")
    return result["selected"]


def model_selection_key(candidate):
    return (*operating_selection_key(candidate["policy"]),-candidate["image_hw"][0]*candidate["image_hw"][1],
        -candidate["effective_epoch"],candidate["kind"]=="source",candidate["name"])


def verify_frozen_selection(path,profile_path):
    result=json.loads(path.read_text());config=load_selection_profile(profile_path)
    if (result["status"]!="FROZEN_FROM_TRAIN_VALIDATION" or result["reproduction_mode"]!="approximate"
            or result["claim_label"]!="approximate reproduction" or result["config"]!=config
            or result["config_sha256"]!=file_sha256(profile_path) or result["test_labels_used_for_selection"] is not False
            or result["selected"]!=max(result["candidates"],key=model_selection_key)):
        raise ValueError("Frozen TRAIN selection differs")
    expected={"source768",*config["required_branches"]}
    if len(result["candidates"])!=len(expected) or {c["name"] for c in result["candidates"]}!=expected:
        raise ValueError("Every declared final candidate must remain present")
    training=load_profile(REPO/config["training_profile"])
    for name,digest in result["source_sha256"].items():
        if file_sha256(REPO/name)!=digest:raise ValueError("Selection source changed: "+name)
    for candidate in result["candidates"]:
        if (file_sha256(Path(candidate["checkpoint"]))!=candidate["checkpoint_sha256"]
                or file_sha256(Path(candidate["training_run"])/"run_manifest.json")!=candidate["training_run_manifest_sha256"]
                or file_sha256(Path(candidate["policy_report"]))!=candidate["policy_report_sha256"]):
            raise ValueError("Frozen candidate artifacts changed")
        report=json.loads((Path(candidate["training_run"])/"run_manifest.json").read_text())
        policy=json.loads(Path(candidate["policy_report"]).read_text())
        if (policy["selected"]!=candidate["policy"] or policy["run_contract"]["image_hw"]!=candidate["image_hw"]
                or candidate["checkpoint_sha256"]!=report["checkpoint"]["sha256"]
                or candidate["epoch"]!=report["checkpoint"]["epoch"]
                or report["run_contract"]["train_ids"]!=result["train_ids"]
                or report["run_contract"]["validation_ids"]!=result["validation_ids"]):
            raise ValueError("Frozen candidate identity/policy differs")
        if candidate["kind"]=="source":
            if (candidate["name"]!="source768" or candidate["checkpoint_sha256"]!=training["source_checkpoint_sha256"]
                    or candidate["training_run_manifest_sha256"]!=training["source_run_manifest_sha256"]):
                raise ValueError("Qualified source identity differs")
        elif (candidate["kind"]!="trained" or candidate["name"] not in config["required_branches"]
                or report["status"]!="SUCCESS" or report["training_complete"] is not True
                or report["completed_epochs"]!=training["training"]["epochs"]
                or report["run_contract"]["config_sha256"]!=config["training_profile_sha256"]):
            raise ValueError("Full completed final training required")
    return result


def load_selected_model(selection):
    chosen=selection["selected"];run=Path(chosen["training_run"])
    report_path=run/"run_manifest.json"
    if file_sha256(report_path)!=chosen["training_run_manifest_sha256"]:raise ValueError("Frozen training report changed")
    report=json.loads(report_path.read_text());checkpoint=Path(chosen["checkpoint"])
    if file_sha256(checkpoint)!=chosen["checkpoint_sha256"]:raise ValueError("Frozen checkpoint changed")
    profile=load_profile(REPO/selection["config"]["training_profile"])
    if chosen["kind"]=="source":
        legacy=json.loads((REPO/selection["config"]["source_training_profile"]).read_text())
        model=ImprovementAGUNet(normalization="batchnorm",max_groups=legacy["group_norm_max_groups"],skip_floor=legacy["skip_floor"])
        training=legacy_training_config(legacy,report["run_contract"]["branch_config"]["loss"])
    else:
        branch=profile["branches"][chosen["name"]]
        model=Target70AGUNet(encoder_stages=branch["encoder_trainable_stages"],max_groups=profile["group_norm_max_groups"],skip_floor=profile["skip_floor"])
        training=training_config(profile,branch)
    load_checkpoint(checkpoint,model=model,expected_architecture=model.architecture_identity,
        expected_run_contract=report["run_contract"],expected_training_config=training)
    return model.eval()
