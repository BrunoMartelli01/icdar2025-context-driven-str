"""Freeze the completed source/long/native comparison without reading TEST."""

import argparse
import json
from pathlib import Path

from scripts.train_icdar2015_totaltext_transfer import write_json
from segmentation.data import file_sha256
from segmentation.icdar2015_improvement_runtime import REPO,load_training_population,references
from segmentation.icdar2015_target70 import load_profile
from segmentation.icdar2015_target70_selection import (load_selection_profile,completed_run,
    verified_policy,model_selection_key)


def run(args):
    config=load_selection_profile(args.config);training=load_profile(REPO/config["training_profile"])
    operating=json.loads((REPO/config["operating_profile"]).read_text())
    manifest,fit,val=load_training_population(training,args.data_root.resolve());refs=references(manifest,args.data_root.resolve(),val)
    fit_ids=[manifest.samples[i].sample_id for i in fit];val_ids=[manifest.samples[i].sample_id for i in val]
    reports={}
    for branch in config["required_branches"]:
        reports[branch]=completed_run(getattr(args,branch+"_run"),branch,training,
            config["training_profile_sha256"],fit_ids,val_ids)
    source_path=args.source_run/"run_manifest.json"
    if file_sha256(source_path)!=training["source_run_manifest_sha256"]:raise ValueError("Qualified source report changed")
    source=json.loads(source_path.read_text());checkpoint=args.source_run/source["checkpoint"]["path"]
    if file_sha256(checkpoint)!=training["source_checkpoint_sha256"]:raise ValueError("Qualified source checkpoint changed")
    if source["run_contract"]["train_ids"]!=fit_ids or source["run_contract"]["validation_ids"]!=val_ids:
        raise ValueError("Source used a different TRAIN split")
    candidates=[]
    for name in ("source768",*config["required_branches"]):
        reference=name=="source768";run=args.source_run if reference else getattr(args,name+"_run")
        report=source if reference else reports[name]
        policy_path=args.source_policy if reference else getattr(args,name+"_policy")
        directory=args.long768_run/"validation/epoch_000" if reference else run/"selected_validation"
        shape=source["run_contract"]["image_hw"] if reference else training["branches"][name]["image_hw"]
        policy=verified_policy(policy_path,operating,config["operating_profile_sha256"],directory/"geometry.json",
            directory/"words.json",shape,refs,training)
        candidates.append({"name":name,"kind":"source" if reference else "trained","training_run":str(run.resolve()),
            "training_run_manifest_sha256":file_sha256(run/"run_manifest.json"),
            "checkpoint":str((run/report["checkpoint"]["path"]).resolve()),"checkpoint_sha256":report["checkpoint"]["sha256"],
            "epoch":report["checkpoint"]["epoch"],"effective_epoch":training["source_effective_epoch"]+(0 if reference else report["checkpoint"]["epoch"]),
            "image_hw":shape,"policy":policy,"policy_report":str(policy_path.resolve()),"policy_report_sha256":file_sha256(policy_path)})
    selected=max(candidates,key=model_selection_key)
    result={"status":"FROZEN_FROM_TRAIN_VALIDATION","reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "config":config,"config_sha256":file_sha256(args.config),"train_ids":fit_ids,"validation_ids":val_ids,
        "validation_care_words":sum(len(r["ground_truth"]) for r in refs.values()),"candidates":candidates,"selected":selected,
        "source_sha256":{p:file_sha256(REPO/p) for p in ("scripts/select_icdar2015_target70.py",
            "segmentation/icdar2015_target70_selection.py","segmentation/icdar2015_operating_policy.py",
            "segmentation/icdar2015_target70.py","recognition/icdar2015_word_f1.py")},"test_labels_used_for_selection":False}
    if args.output.exists():
        if json.loads(args.output.read_text())!=result:raise ValueError("Preserve the existing frozen selection")
    else:write_json(args.output,result)
    print(json.dumps({"selected":selected["name"],"validation_emitted_f1":selected["policy"]["emitted_aggregate"]["alphanumeric_casefold"]["f1"]}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","data-root","source-run","source-policy","long768-run","long768-policy","adapt1280-run","adapt1280-policy","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":main()
