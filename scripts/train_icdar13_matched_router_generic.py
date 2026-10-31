"""Fit a NEW small native U-Net gate against the qualified90.17% Generic expert."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import torch

from scripts.icdar_budget_data import read, sha, key, annotation, predictions
from scripts.run_yolo_f1_quality_study import seal
from scripts.icdar_budget_stages import score, route
from scripts.finish_icdar13_word_refit import full_maps, audit_heads
from scripts.run_icdar13_matched_word_training import model, recognizer, read_rows
from recognition.icdar13_matched_router import router_rows, exact_accepted, validation_fallback_count
from recognition.icdar_budget_router import observation, lexical_counts, calibration_cut, accept
from recognition.lightweight_unet_router import fit_head, paired_utility
from recognition.f1_quality_study import aggregate
from recognition.icdar_budget_deployment import predict as single_predict
from recognition.trainable_image_router import oracle
from recognition.fallback_feasibility import target_minimum, json_point
from recognition.router_fallback_replay import ordered_frontier
from segmentation.icdar13_word_refit import strict_fallback_count

REPO = Path(__file__).resolve().parents[1]


def independent_counts(pp, gt):
    norm = lambda s: "".join(c for c in s.casefold() if c.isalnum())
    care = [p for p in pp if not p["ignored"]]
    a = Counter(norm(p["text"]) for p in care if norm(p["text"]))
    b = Counter(norm(g["text"]) for g in gt if norm(g["text"]))
    tp = sum((a & b).values())
    return dict(tp=tp, fp=len(care)-tp, fn=len(gt)-tp, detected_words=len(care), gt_words=len(gt),
        all_outputs=len(pp), ignored_outputs=len(pp)-len(care))


def prepare(args):
    if args.mode != "approximate" or args.config.resolve().parent != REPO / "configs/approximate":
        raise ValueError("Explicit approximate configuration namespace required")
    c = read(args.config)
    if c["mode"] != args.mode or c["claim_label"] != "approximate reproduction":
        raise ValueError("Mode mismatch")
    for file, digest in c["pinned_inputs"].items():
        if sha(file) != digest:
            raise ValueError("Changed pinned input: " + file)
    light = read(Path(c["light_root"]) / "resolved_configuration.json")
    selected = read(Path(c["light_root"]) / "selection_frozen.json")["winner"]
    if light["mode"] != "approximate" or read(Path(c["light_root"])/"results.json")["winner"] != selected:
        raise ValueError("Changed qualified light expert")
    parts = {k:list(v) for k, v in light["partitions"].items()}
    for name, expected in c["expected_populations"].items():
        if len(parts[name]) != expected:
            raise ValueError("Unexpected population: " + name)
    if args.smoke:
        parts = {k:ids[:(8 if k=="fit" else 4)] for k, ids in parts.items()}
    records = light["train_records"] + light["test_records"]
    groups = [{r["rgb_sha256"] for r in records if r["sample_id"] in ids} for ids in parts.values()]
    if any(a & b for a, b in itertools.combinations(groups, 2)):
        raise ValueError("RGB leakage across partitions")
    teacher = read(c["teacher_counts"])
    if (read(c["teacher_audit"])["counts_sha256"] != sha(c["teacher_counts"])
            or teacher["checkpoint_sha256"] != read(c["fallback_profile"])["checkpoint_sha256"]):
        raise ValueError("Teacher qualification mismatch")
    root = Path(c["output_root"]) / ("smoke" if args.smoke else "full")
    if root.exists():
        raise ValueError("Run already prepared; preserve it and use explicit --run-root")
    root.mkdir(parents=True)
    code = {str(p.relative_to(REPO)):sha(p) for folder in ("models", "recognition", "scripts", "segmentation", "localize", "caption", "scoring", "pipeline", "research", "deepsolo") for p in (REPO/folder).rglob("*.py")}
    plan = dict(mode="approximate", claim_label="approximate reproduction", config=c, light=light, selected=selected,
        records=records, partitions=parts, code_sha256=code, smoke=args.smoke,
        result_class="smoke-test result" if args.smoke else "approximate reproduction", config_sha256=sha(args.config),
        new_router_training=True, test_used_for_training_or_selection=False, actual_new_deepsolo_calls=0,
        hardware=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
            precision="float32", unet_batch=1, parseq_batch=c["parseq_batch_size"], image_hw=light["config"]["image_hw"],
            shared_gpu=True, inference_scope="Missing FIT U-Net/PARSeq only; all DeepSolo outputs reused; router fitting on CPU"))
    seal(root/"resolved_configuration.json", plan)
    for file, digest in code.items():
        dest=root/"code_snapshot"/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/file,dest)
        if sha(dest)!=digest:
            raise ValueError("Code changed while freezing")
    shutil.copytree(REPO/"configs",root/"code_snapshot/configs")
    return root


def population(plan, split):
    lookup = {r["sample_id"]:r for r in plan["records"]}
    return [lookup[s] for s in plan["partitions"][split]]


def observations(plan, root):
    c=plan["config"];source=Path(c["light_root"]);choice=plan["selected"];settings=choice["settings"]
    light_plan=plan["light"]
    if light_plan["config"]["device"] != c["device"]:
        raise ValueError("No implicit inference-device substitution")
    if torch.cuda.mem_get_info()[0] < c["minimum_free_gpu_bytes"]:
        raise RuntimeError("Insufficient shared GPU memory for the qualified FP32 inference")
    net=model(light_plan,choice["checkpoint"]).requires_grad_(False).eval();ocr=recognizer(light_plan)
    try:
        for i,r in enumerate(population(plan,"fit"),1):
            path=root/"experts"/(key(r["sample_id"])+".json")
            if path.exists():
                old=read(path)
                if old["image_sha256"]!=r["image_sha256"] or sha(old["maps_path"])!=old["maps_sha256"]:
                    raise ValueError("Changed resumed FIT observation")
                continue
            torch.cuda.reset_peak_memory_stats()
            probabilities=full_maps(net,r,light_plan["config"])
            _,raw,expansion=read_rows(probabilities,r,settings,ocr,c["parseq_batch_size"])
            maps=root/"maps"/(key(r["sample_id"])+".npz");maps.parent.mkdir(exist_ok=True)
            if maps.exists():
                with np.load(maps) as z:
                    if not np.array_equal(z["probabilities"],probabilities):
                        raise ValueError("Interrupted FIT map differs")
            else:
                np.savez_compressed(maps,probabilities=probabilities)
            seal(path,dict(rows=router_rows(raw,settings["minimum_transcript_probability"]),raw_rows=raw,
                maps_path=str(maps),maps_sha256=sha(maps),image_sha256=r["image_sha256"],ground_truth_used=False,
                gpu_allocated=torch.cuda.max_memory_allocated(),gpu_reserved=torch.cuda.max_memory_reserved(),expansion=expansion))
            if i%25==0 or i==len(plan["partitions"]["fit"]):
                print(json.dumps(dict(stage="new_light_fit_observations",images=i,total=len(plan["partitions"]["fit"]))),flush=True)
    finally:
        ocr.release();del net;torch.cuda.empty_cache()
    for split in ("selection","calibration","test"):
        for r in population(plan,split):
            stem=key(r["sample_id"])
            folder=source/"policy_selection"/choice["candidate"] if split=="selection" else source/"evaluation/selected"/split
            sample_path=(folder/("threshold_"+str(settings["text_threshold"])+"_padding_"+str(settings["padding"]))/"samples"/(stem+".json") if split=="selection" else folder/"samples"/(stem+".json"))
            data=read(sample_path);maps=folder/"maps"/(stem+".npz")
            if data["image_sha256"]!=r["image_sha256"] or sha(maps)!=data["maps_sha256"]:
                raise ValueError("Qualified cached light observation changed")
            rows=router_rows(data["raw_rows"],settings["minimum_transcript_probability"])
            if split!="selection" and [(v["raw_text"],v["bbox"]) for v in rows]!=[(v["raw_text"],v["bbox"]) for v in data["rows"]]:
                raise ValueError("Router adapter changed qualified text/crops")
            seal(root/"experts"/(stem+".json"),dict(rows=rows,maps_path=str(maps),maps_sha256=sha(maps),
                image_sha256=r["image_sha256"],ground_truth_used=False,reused_source=str(sample_path),reused_source_sha256=sha(sample_path)))


def collect(plan,root,ids):
    c=plan["config"];teacher=read(c["teacher_counts"]);lookup={r["sample_id"]:r for r in plan["records"]}
    features,branches,keys={},{},{}
    for sid in ids:
        r=lookup[sid];d=read(root/"experts"/(key(sid)+".json"));t=teacher["samples"][sid]
        if d["ground_truth_used"] or d["image_sha256"]!=r["image_sha256"] or t["image_sha256"]!=r["image_sha256"]:
            raise ValueError("GT-blind expert/RGB identity changed")
        if sha(d["maps_path"])!=d["maps_sha256"] or sha(t["evidence_path"])!=t["evidence_sha256"]:
            raise ValueError("Expert evidence changed")
        with np.load(d["maps_path"]) as z:maps=z["probabilities"]
        groups=observation(d["rows"],r["image_wh"],maps,"square_letterbox",settings=c["features"])
        features[sid]=dict(groups=groups,scale=len(d["rows"])+1,eligible=bool(d["rows"]))
        gt,valid=annotation(r);pp=predictions(d["rows"],r,valid);light=lexical_counts(pp,gt)
        evidence=read(t["evidence_path"]);deep=independent_counts(evidence["predictions"],gt)
        if (light!=independent_counts(pp,gt) or deep!=t["counts"] or deep!=evidence["counts"]
                or deep["gt_words"]!=light["gt_words"]):
            raise ValueError("Independent expert count or care-population replay failed")
        branches[sid]=dict(light=light,deepsolo=deep);keys[sid]=r["rgb_sha256"]
    return features,branches,keys


def train(plan,root):
    c=plan["config"];p=plan["partitions"];ids=p["fit"]+p["selection"]+p["calibration"]
    features,branches,keys=collect(plan,root,ids)
    reference=aggregate(branches[s]["deepsolo"] for s in p["fit"])["f1"]
    weights=np.asarray([features[s]["scale"] for s in p["fit"]],float)
    target=np.asarray([paired_utility(branches[s],reference)/features[s]["scale"] for s in p["fit"]])
    np.savez_compressed(root/"training_features.npz",ids=np.array(ids),targets=target,weights=weights,
        **{g:np.stack([features[s]["groups"][g] for s in ids]) for g in features[ids[0]]["groups"]})
    seal(root/"training_branches.json",dict(branches=branches,tie_keys=keys,fit_ids=p["fit"],selection_ids=p["selection"],calibration_ids=p["calibration"]))
    n=validation_fallback_count(len(p["selection"]),c["expected_populations"]["test"],c["primary_deepsolo_images"])
    trials=[]
    for groups in c["feature_groups"]:
        x=np.stack([np.concatenate([features[s]["groups"][g] for g in groups]) for s in p["fit"]])
        for index,recipe in enumerate(c["heads"]):
            bundle=fit_head(groups,x,target,weights,recipe,c["seed"])
            bundle["cost"]["requires_unet_maps"]="maps" in groups
            bundle.update(training_ids=p["fit"],reference_f1=reference)
            path=root/"routers"/("_".join(groups)+"_"+str(index)+".joblib");path.parent.mkdir(exist_ok=True)
            if path.exists():raise ValueError("Refuse to overwrite trained router weights")
            joblib.dump(bundle,path);scores=score(bundle,features,p["selection"])
            point=route(branches,p["selection"],exact_accepted(scores,tie_keys={s:keys[s] for s in p["selection"]},fallback_images=n))
            trials.append(dict(groups=groups,recipe=recipe,model_path=str(path.relative_to(root)),model_sha256=sha(path),
                scalar_count=bundle["stored_scalar_count"],serialized_bytes=path.stat().st_size,selection=point))
    best=max(v["selection"]["combined"]["f1"] for v in trials)
    winner=min([v for v in trials if v["selection"]["combined"]["f1"]>=best-c["selection_tolerance"]],
        key=lambda v:(v["scalar_count"],v["serialized_bytes"],-v["selection"]["combined"]["f1"],v["model_path"]))
    bundle=joblib.load(root/winner["model_path"]);values=score(bundle,features,p["calibration"]);policies=[]
    primary_cal=validation_fallback_count(len(p["calibration"]),c["expected_populations"]["test"],c["primary_deepsolo_images"])
    quotas=[("primary_61",primary_cal)]+[(str(cap),strict_fallback_count(len(p["calibration"]),cap)) for cap in c["calibration_fraction_controls"]]
    for name,count in quotas:
        q=(len(p["calibration"])-count)/len(p["calibration"]);policy=calibration_cut(values,p["calibration"],q,keys)
        policies.append(dict(policy,name=name,calibration=route(branches,p["calibration"],accept(values,policy,keys))))
    audited=audit_heads(root,plan,features,branches,trials)
    seal(root/"trials.json",trials);seal(root/"frozen_router.json",dict(winner=winner,policies=policies,
        fit_ids=p["fit"],selection_ids=p["selection"],calibration_ids=p["calibration"],test_used_for_selection=False,
        new_router_training=True,no_refit_after_calibration=True,independent_head_fits=audited))
    joblib.dump(dict(bundle=bundle,feature_settings=c["features"],policies=policies,expert=plan["selected"]),root/"deployment.joblib")
    print(json.dumps(dict(stage="NEW_ROUTER_TRAINED",head=winner["recipe"],groups=winner["groups"],independent_fits=audited)),flush=True)


def evaluate(plan,root):
    c=plan["config"];p=plan["partitions"];frozen=read(root/"frozen_router.json");winner=frozen["winner"]
    if sha(root/winner["model_path"])!=winner["model_sha256"]:raise ValueError("Trained weights changed")
    bundle=joblib.load(root/winner["model_path"]);features,branches,keys=collect(plan,root,p["test"]);scores=score(bundle,features,p["test"])
    n=validation_fallback_count(len(p["test"]),c["expected_populations"]["test"],c["primary_deepsolo_images"])
    accepted=exact_accepted(scores,keys,n);primary=route(branches,p["test"],accepted)
    if primary["deepsolo_images"]!=n:raise ValueError("User exact61-image budget violated")
    if not plan["smoke"] and (len(p["test"])!=c["expected_populations"]["test"] or primary["combined"]["gt_words"]!=c["expected_test_words"]):
        raise ValueError("Incomplete official TEST denominator")
    scalar=[dict(policy=policy,**route(branches,p["test"],accept(scores,policy,keys))) for policy in frozen["policies"]]
    quotas={str(cap):route(branches,p["test"],exact_accepted(scores,keys,strict_fallback_count(len(p["test"]),cap))) for cap in c["diagnostic_fraction_controls"]}
    upper=oracle(branches,p["test"],[s for s in p["test"] if scores[s] is not None],len(accepted))
    diagnostic=target_minimum(branches,c["target_f1_diagnostic"],eligible=[s for s in scores if scores[s] is not None])
    diagnostic["witness"]=None if diagnostic["witness"] is None else json_point(diagnostic["witness"])
    all_light=route(branches,p["test"],p["test"]);all_deep=route(branches,p["test"],[])
    oldroot=Path(c["historical_router_root"]);oldchoice=read(oldroot/"frozen_router.json")["choice"]
    if sha(oldroot/oldchoice["model_path"])!=oldchoice["model_sha256"]:
        raise ValueError("Frozen reference router changed")
    oldbundle=joblib.load(oldroot/oldchoice["model_path"]);oldvalues=score(oldbundle,features,p["test"])
    oldcontrol=route(branches,p["test"],exact_accepted(oldvalues,keys,n))
    maximum_error=0.
    for r in population(plan,"test"):
        d=read(root/"experts"/(key(r["sample_id"])+".json"))
        with np.load(d["maps_path"]) as z:maps=z["probabilities"]
        replay=single_predict(bundle,c["features"],d["rows"],r["image_wh"],tie_key=r["rgb_sha256"],policies=frozen["policies"],probabilities=maps,frame="square_letterbox")
        expected=scores[r["sample_id"]]
        if (replay["score"] is None)!=(expected is None):raise ValueError("Single-image eligibility changed")
        if expected is not None:maximum_error=max(maximum_error,abs(expected-replay["score"]))
        for policy in frozen["policies"]:
            if replay["accept_light"][str(policy["target_acceptance"])]!=(r["sample_id"] in accept(scores,policy,keys)):
                raise ValueError("Single-image calibration decision replay failed")
    if maximum_error>c["numerical_replay_tolerance"]:raise ValueError("Single-image trained gate replay failed")
    seal(root/"test_scores.json",dict(scores=scores,branches=branches,tie_keys=keys))
    seal(root/"frontier.json",dict(scope="TEST-descriptive fixed trained-score ranks; no new threshold selected",points=ordered_frontier(branches,scores,keys)))
    result=dict(status="SUCCESS",mode="approximate",claim_label="approximate reproduction",result_class=plan["result_class"],
        new_router_training=True,winner=winner,primary=primary,frozen_old_gate_control=oldcontrol,thresholds=scalar,predeclared_controls=quotas,
        conditional_gt_oracle=dict(**route(branches,p["test"],upper["accepted_ids"]),certificate=upper["certificate_max_residual"]),
        target90_gt_diagnostic=diagnostic,all_light=all_light,all_deepsolo=all_deep,actual_new_deepsolo_calls=0,
        actual_new_light_inference_images=len(p["fit"]),test_used_for_selection=False,routing_scope="Logical decisions on cached outputs; new CPU router fits and183 fresh FIT light inferences")
    seal(root/"results.json",result);seal(root/"independent_audit.json",dict(status="PASS",head_fit_replays=len(read(root/"trials.json")),
        expert_count_replays=len(p["fit"]+p["selection"]+p["calibration"]+p["test"]),single_image_gate_replays=len(p["test"]),
        maximum_score_error=maximum_error,results_sha256=sha(root/"results.json"),exact_primary_deepsolo_images=n))
    print(json.dumps(dict(stage="COMPLETE",primary=primary,all_deepsolo=all_deep)),flush=True)


def execute(root):
    plan=read(root/"resolved_configuration.json");c=plan["config"]
    if plan["mode"]!="approximate":raise ValueError("Approximate mode required")
    for file,digest in plan["code_sha256"].items():
        if sha(REPO/file)!=digest:raise ValueError("Frozen source changed: "+file)
    for file,digest in c["pinned_inputs"].items():
        if sha(file)!=digest:raise ValueError("Pinned input changed")
    torch.set_num_threads(c["cpu_threads"]);np.random.seed(c["seed"]);torch.manual_seed(c["seed"]);torch.cuda.manual_seed_all(c["seed"])
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    try:
        observations(plan,root);train(plan,root);evaluate(plan,root)
    except BaseException as error:
        seal(root/("failure_"+str(time.time_ns())+".json"),dict(error=str(error),traceback=traceback.format_exc(),out_of_memory=isinstance(error,torch.cuda.OutOfMemoryError)))
        raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--mode",required=True);p.add_argument("--config",type=Path);p.add_argument("--run-root",type=Path);p.add_argument("--smoke",action="store_true");args=p.parse_args()
    if args.mode!="approximate":raise ValueError("Explicit approximate mode required")
    root=args.run_root.resolve() if args.run_root else prepare(args)
    if REPO!=(root/"code_snapshot").resolve():
        subprocess.run([sys.executable,"-u","-m","scripts.train_icdar13_matched_router_generic","--mode",args.mode,"--run-root",str(root)],
            cwd=root/"code_snapshot",env={**os.environ,"PYTHONPATH":str(root/"code_snapshot")},check=True)
        return
    execute(root)


if __name__=="__main__":main()
