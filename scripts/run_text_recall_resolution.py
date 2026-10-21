"""Matched text coverage experiments: loss x input resolution, APPROXIMATE only."""

import argparse
from dataclasses import asdict
import fcntl
import gc
from hashlib import sha256
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from models.unet_adv import PracticalTotalTextHighResAGUNet
from localize.word_multitask import parseq_word_proposals
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json, _split_indices, _epoch_log, _append_epoch
from scripts.train_totaltext_v4_image_frequency_ablation import _config as source_config, CONFIG_PATH as SOURCE_PATH
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.train import EpochShuffleSampler
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.text_recall_provenance import ensure_experiment_contract
from segmentation.training import build_optimizer, load_checkpoint, save_checkpoint, seed_everything, runtime_environment
from segmentation.word_multitask_evaluation import render_word_maps, aggregate_word_metrics
from segmentation.word_multitask_provenance import json_contract
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.text_recall import (
    TextRecallTrainingConfig, TextRecallDataset, make_transform, train_recall_epoch,
    canonical_probability, coverage_counts, aggregate_coverage, operating_point, selection_key,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO/"configs/approximate/text_recall_resolution_v1.json"
BRANCHES = {"control512": {"input_size":512,"loss":"bce_dice"},
            "tversky512": {"input_size":512,"loss":"bce_tversky"},
            "control768": {"input_size":768,"loss":"bce_dice"},
            "tversky768": {"input_size":768,"loss":"bce_tversky"}}
SOURCES = ["scripts/run_text_recall_resolution.py", "segmentation/text_recall.py", "models/unet_adv.py",
    "segmentation/text_recall_provenance.py", "segmentation/evaluate.py",
    "segmentation/data.py", "segmentation/transforms.py", "segmentation/training.py", "segmentation/train.py",
    "segmentation/word_multitask.py", "segmentation/word_multitask_provenance.py",
    "segmentation/totaltext_word_kernel.py", "segmentation/totaltext_image_frequency_augment.py",
    "segmentation/word_multitask_evaluation.py", "localize/word_multitask.py", "localize/crop_recovery.py",
    "scripts/train_totaltext_v4_highres_ablation.py", "scripts/train_totaltext_v4_image_frequency_ablation.py",
    "recognition/scene_text_v1_evaluation.py", "diagnostics/render_train_visual_sheets.py"]


def read_config(path, mode):
    path=Path(path).resolve()
    if mode!="approximate" or REPO/"configs/approximate" not in path.parents:
        raise ValueError("Use explicit --mode approximate and configs/approximate/")
    cfg=json.loads(path.read_text()); source=source_config()
    if (cfg.get("schema_version")!=1 or cfg.get("reproduction_mode")!=mode
        or cfg.get("claim_label")!="approximate reproduction" or cfg.get("branches")!=BRANCHES
        or cfg.get("architecture_identity")!=PracticalTotalTextHighResAGUNet.PRACTICAL_IDENTITY):
        raise ValueError("Invalid recall factorial configuration")
    if ((REPO/cfg["source_config"]).resolve()!=SOURCE_PATH
        or sha256_file(SOURCE_PATH)!=cfg["source_config_sha256"]):
        raise ValueError("Qualified preprocessing configuration changed")
    data=cfg["dataset"];t=cfg["training"];loss=cfg["loss"];e=cfg["evaluation"]
    if (data["manifest"]!=source["train_manifest"] or data["manifest_sha256"]!=source["train_manifest_sha256"]
        or (data["images"],data["validation_images"],data["validation_words"])!=(1255,125,983)
        or t["seed"]!=source["training"]["seed"] or cfg["skip_floor"]!=source["training"]["skip_floor"]):
        raise ValueError("Frozen TRAIN dataset/split/seed/architecture differs")
    if (len(cfg["base_checkpoint_sha256"])!=64 or cfg["base_checkpoint_epoch"]<1
        or t["precision"]!="float32" or t["deterministic_algorithms"] is not True
        or type(t["epochs"]) is not int or type(t["num_workers"]) is not int or t["num_workers"]<0
        or type(t["loader_timeout_seconds"]) is not int or t["loader_timeout_seconds"]<1):
        raise ValueError("Invalid provenance/precision/worker configuration")
    if (not np.isfinite(list(loss.values())).all() or min(loss.values())<=0
        or not np.isclose(loss["fp_weight"]+loss["fn_weight"],1)
        or loss["fn_weight"]<=loss["fp_weight"] or loss["separation_weight"]!=3.):
        raise ValueError("Invalid or unmatched recall loss weights")
    thresholds=e["thresholds"]
    if (e["canonical_size"]!=256 or not thresholds or thresholds!=sorted(set(thresholds))
        or not np.isfinite(thresholds).all() or not all(0<x<1 for x in thresholds)
        or not np.isfinite([e["minimum_pixel_precision"],e["complete_word_coverage"]]).all()
        or not 0<e["minimum_pixel_precision"]<=1 or not 0<e["complete_word_coverage"]<=1
        or e["resampling"]!="sigmoid_then_CPU_bilinear_align_corners_false"
        or e["primary"]!="complete_word_rate_then_mean_word_coverage_at_precision_floor_then_pixel_F2"
        or e["ground_truth"]!="unchanged_canonical_v4_valid_targets_and_all_983_expected_word_owners"
        or e["ignore"]!="exclude_invalid_pixels_and_letterbox_padding"
        or e["missing_word"]!="keep_in_denominator_with_zero_coverage" or e["test_usage"]!="none"):
        raise ValueError("Unsupported canonical validation protocol")
    for branch in BRANCHES: training_config(cfg,branch)
    return cfg,source


def training_config(cfg, branch):
    t=cfg["training"]
    return TextRecallTrainingConfig(epochs=t["epochs"],seed=t["seed"],optimizer=t["optimizer"],
        learning_rate=t["learning_rate"],weight_decay=t["weight_decay"],batch_size=t["batch_size"],
        adam_betas=tuple(t["adam_betas"]),adam_epsilon=t["adam_epsilon"],
        learning_rate_schedule=t["learning_rate_schedule"],checkpoint_selection=t["checkpoint_selection"],
        deterministic_algorithms=t["deterministic_algorithms"],separation_weight=cfg["loss"]["separation_weight"],
        loss="masked_BCE_Dice" if cfg["branches"][branch]["loss"]=="bce_dice" else "masked_BCE_Tversky")


def fixed_json(path, data):
    normalized=json_contract(data)
    if path.exists():
        if json.loads(path.read_text())!=normalized:
            raise ValueError("Existing experiment contract differs; select a NEW output directory: "+str(path))
    else: _json(path,normalized)


def guard_output(output, target, base):
    if output==REPO or any(output==p or p in output.parents or output in p.parents for p in (target,base.parent)):
        raise ValueError("Output must be isolated from the dataset and old experiment")


def load_references(base, cfg, manifest, val_ids, target):
    result=json.loads((base/"run_manifest.json").read_text())
    validation=json.loads((base/"validation.json").read_text()); digest=cfg["base_checkpoint_sha256"]
    if (result.get("status")!="SUCCESS" or result["checkpoint"]["sha256"]!=digest
        or result["checkpoint"]["epoch"]!=cfg["base_checkpoint_epoch"]
        or validation["checkpoint_sha256"]!=digest or result["run_contract"]["variant"]!="text"):
        raise ValueError("Base must be the completed SHA-pinned single-text experiment")
    if [r["sample_id"] for r in validation["per_image"]]!=[manifest.samples[i].sample_id for i in val_ids]:
        raise ValueError("Canonical validation sample order differs")
    refs={};hashes={};historical={str(t):[] for t in cfg["evaluation"]["thresholds"]}
    dataset=TextRecallDataset(SegmentationDataset(manifest,target,
        transform=make_transform(source_config(),input_size=512,train=False)))
    for index,row in zip(val_ids,validation["per_image"]):
        path=base/"validation_exports"/row["sample_id"].replace(":","__")/"maps.npz"
        digest=sha256_file(path)
        if digest!=row["exports"]["maps.npz"]: raise ValueError("Canonical map export checksum differs")
        with np.load(path,allow_pickle=False) as maps:
            ref={k:maps[v].copy() for k,v in {"target":"text_target","validity":"validity","owners":"owners",
                "expected_owner_ids":"expected_owner_ids","canvas":"canvas"}.items()}
            batch=dataset[index]
            if (ref["target"].shape!=(256,256)
                or not np.array_equal(ref["target"],batch["mask"][0].numpy().astype(bool))
                or not np.array_equal(ref["validity"],batch["validity"][0].numpy().astype(bool))
                or not np.array_equal(ref["owners"],batch["owners_native"].numpy())):
                raise ValueError("Canonical targets differ from the qualified current dataset")
            for t in cfg["evaluation"]["thresholds"]:
                historical[str(t)].append({"sample_id":row["sample_id"],**coverage_counts(maps["probabilities"][0],
                    ref,t,complete_coverage=cfg["evaluation"]["complete_word_coverage"])})
        refs[row["sample_id"]]=ref;hashes[row["sample_id"]]=digest
    if sum(len(r["expected_owner_ids"]) for r in refs.values())!=cfg["dataset"]["validation_words"]:
        raise ValueError("Canonical word denominator differs")
    aggregates={t:aggregate_coverage(rows) for t,rows in historical.items()}
    return refs,hashes,{"result_class":"approximate reproduction","training":"historical reference, not a matched continuation",
        "thresholds":aggregates,"operating_point":operating_point(aggregates,
            minimum_precision=cfg["evaluation"]["minimum_pixel_precision"]),"per_threshold_images":historical}


def evaluate(model, batches, refs, device, cfg, *, export_dir=None, image_paths=None, export_threshold=None):
    model.eval();e=cfg["evaluation"];rows={str(t):[] for t in e["thresholds"]};geometry=[];exports=[];seen=set()
    with torch.inference_mode():
        for batch in batches:
            probabilities=model(batch["image"].to(device)).sigmoid().cpu().numpy()
            if probabilities.shape!=tuple(batch["mask"].shape): raise ValueError("Native probability/target shape differs")
            for i,sample_id in enumerate(batch["sample_id"]):
                if sample_id in seen: raise ValueError("Duplicate validation sample")
                seen.add(sample_id);ref=refs[sample_id];native=probabilities[i,0]
                p=canonical_probability(native,size=e["canonical_size"])
                for t in e["thresholds"]:
                    rows[str(t)].append({"sample_id":sample_id,**coverage_counts(p,ref,t,complete_coverage=e["complete_word_coverage"])})
                if export_dir is not None:
                    count,labels=cv2.connectedComponents(((p>=export_threshold)&ref["canvas"]).astype(np.uint8),connectivity=8)
                    with Image.open(image_paths[sample_id]) as opened: rgb=opened.convert("RGB")
                    proposals=parseq_word_proposals(labels,rgb.size,padding=0,min_area=0,max_instances=None)
                    care=labels.copy()
                    for label in range(1,count):
                        if not np.any((labels==label)&ref["validity"]): care[labels==label]=0
                    geometry.append({"sample_id":sample_id,**instance_geometry(care,ref["owners"],ref["expected_owner_ids"])})
                    directory=export_dir/sample_id.replace(":","__");directory.mkdir(parents=True,exist_ok=True)
                    arrays={"probabilities":p[None],"native_probabilities":native[None],"labels":labels,
                        "text_target":ref["target"],"validity":ref["validity"],"owners":ref["owners"],
                        "canvas":ref["canvas"],"expected_owner_ids":ref["expected_owner_ids"]}
                    if (directory/"maps.npz").exists():
                        with np.load(directory/"maps.npz",allow_pickle=False) as old:
                            if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                                raise ValueError("Existing validation maps differ")
                    else: np.savez_compressed(directory/"maps.npz",**arrays)
                    render_word_maps(rgb,p[None],proposals,directory/"predictions.png")
                    render_word_maps(rgb,ref["target"][None].astype(np.float32),[],directory/"targets.png")
                    _json(directory/"proposals.json",{"sample_id":sample_id,"result_class":"approximate reproduction",
                        "image_path":str(image_paths[sample_id]),"image_sha256":sha256_file(image_paths[sample_id]),
                        "threshold":export_threshold,"proposals":proposals,"role":"foreground components; offset network will separate words"})
                    exports.append({"sample_id":sample_id,"files":{name:sha256_file(directory/name) for name in
                        ("maps.npz","predictions.png","targets.png","proposals.json")}})
    aggregates={t:aggregate_coverage(part) for t,part in rows.items()}
    result={"thresholds":aggregates,"operating_point":operating_point(aggregates,
        minimum_precision=e["minimum_pixel_precision"]),"per_threshold_images":rows}
    if export_dir is not None:
        totals={k:sum(r[k] for r in geometry) for k in geometry[0] if k!="sample_id"}
        result.update(exports=exports,secondary_component_geometry=aggregate_word_metrics(totals),
                      secondary_component_geometry_per_image=geometry)
    return result


def memory(device, cfg, branch, batch_size=None):
    size=cfg["branches"][branch]["input_size"];batch=batch_size or cfg["training"]["batch_size"]
    return {"device":str(device),"precision":"float32","batch_size":batch,"input_shape":[batch,3,size,size],
        "model_set":"single text U-Net, sequential branches; offset network not loaded","oom":False,
        "gpu_name":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
        "peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None}


def initialize_model(base, cfg):
    model=PracticalTotalTextHighResAGUNet(skip_floor=cfg["skip_floor"],encoder_pretrained=False)
    saved=load_checkpoint(base/"best.pt",model=model,expected_architecture=model.architecture_identity)
    if (saved["epoch"]!=cfg["base_checkpoint_epoch"] or saved.get("run_contract",{}).get("variant")!="text"
        or saved["run_contract"]["config"]["skip_floor"]!=cfg["skip_floor"]): raise ValueError("Warm start metadata differs")
    receipt={"source_checkpoint_sha256":cfg["base_checkpoint_sha256"],"source_epoch":saved["epoch"],
        "architecture":model.architecture_identity,"parameters":sum(p.numel() for p in model.parameters()),
        "transfer":"all model tensors strict; fresh optimizer; encoder/encoder BN frozen"}
    del saved
    return model,receipt


def train(directory, model, optimizer, loaders, refs, sampler, transform, cfg, branch, contract, device, stop=None):
    tc=training_config(cfg,branch);last=directory/"last.pt";best=directory/"best.pt";log=directory/"epochs.jsonl"
    records=_epoch_log(log);start,step,best_epoch,best_point=1,0,None,None
    def save(path,epoch,trained,point):
        return save_checkpoint(path,model=model,optimizer=optimizer,epoch=epoch,global_step=step,config=tc,
            resolved_config=json.dumps(contract,sort_keys=True),architecture=model.architecture_identity,
            result_class="approximate reproduction",run_contract=contract,train_metrics=trained,validation_metrics=point)
    if best.exists():
        saved=load_checkpoint(best,model=model,expected_architecture=model.architecture_identity,
            expected_run_contract=contract,expected_training_config=tc)
        best_epoch,best_point=saved["epoch"],saved["validation_metrics"];del saved
    if last.exists():
        saved=load_checkpoint(last,model=model,optimizer=optimizer,expected_architecture=model.architecture_identity,
            expected_run_contract=contract,expected_training_config=tc,restore_rng=True)
        start,step=saved["epoch"]+1,saved["global_step"]
        if len(records)==start-2:
            _append_epoch(log,{"epoch":start-1,"global_step":step,"train":saved["train_metrics"],
                "validation":saved["validation_metrics"],"last_checkpoint_sha256":sha256_file(last),
                "recovered_from_checkpoint":True});records=_epoch_log(log)
        if len(records)!=start-1 or sha256_file(last)!=records[-1]["last_checkpoint_sha256"]:
            raise ValueError("Epoch log/latest checkpoint disagree")
        if best_point is None or selection_key(saved["validation_metrics"],start-1)>selection_key(best_point,best_epoch):
            best_epoch,best_point=start-1,saved["validation_metrics"];save(best,best_epoch,saved["train_metrics"],best_point)
        del saved
    elif records or (best.exists() and best_epoch!=0): raise ValueError("Training history without resumable latest checkpoint")
    elif best.exists():
        saved=load_checkpoint(best,model=model,optimizer=optimizer,expected_architecture=model.architecture_identity,
            expected_run_contract=contract,expected_training_config=tc,restore_rng=True);del saved
    else:
        initial=evaluate(model,loaders[1],refs,device,cfg)
        best_epoch,best_point=0,initial["operating_point"]
        _json(directory/"initial.json",initial);save(best,0,{},best_point)
    if [r["epoch"] for r in records]!=list(range(1,start)): raise ValueError("Non-contiguous epoch history")
    budget=tc.epochs if stop is None else stop
    if not 1<=budget<=tc.epochs: raise ValueError("--stop-after-epoch outside configured budget")
    for epoch in range(start,budget+1):
        sampler.set_epoch(epoch-1);transform.set_epoch(epoch-1)
        trained=train_recall_epoch(model,loaders[0],optimizer,device,loss_name=cfg["branches"][branch]["loss"],loss_config=cfg["loss"])
        measured=evaluate(model,loaders[1],refs,device,cfg);point=measured["operating_point"]
        if (point["images"],point["gt_words"])!=(cfg["dataset"]["validation_images"],cfg["dataset"]["validation_words"]):
            raise ValueError("Validation denominator changed")
        step+=trained["steps"];digest=save(last,epoch,trained,point)
        if selection_key(point,epoch)>selection_key(best_point,best_epoch):
            best_epoch,best_point=epoch,point;save(best,epoch,trained,point)
        _append_epoch(log,{"epoch":epoch,"global_step":step,"train":trained,"validation":point,
            "all_thresholds":measured["thresholds"],"last_checkpoint_sha256":digest})
        print(f"[{branch}] {epoch}/{tc.epochs} cover90={point['complete_word_rate']:.4f} "
              f"mean={point['mean_word_coverage']:.4f} P={point['pixel_precision']:.4f} "
              f"R={point['pixel_recall']:.4f} threshold={point['threshold']} best={best_epoch}",flush=True)
    completed=len(_epoch_log(log))
    if completed!=tc.epochs: return {"status":"IN_PROGRESS","completed_epochs":completed,"configured_epochs":tc.epochs}
    result={"status":"SUCCESS","result_class":"approximate reproduction","run_contract":contract,
        "completed_epochs":completed,"checkpoint":{"epoch":best_epoch,"path":"best.pt","sha256":sha256_file(best)},
        "validation_operating_point":best_point,"memory":memory(device,cfg,branch),"environment":runtime_environment()}
    _json(directory/"run_manifest.json",result);return result


def qualify(datasets):
    result={}
    for size,dataset in datasets.items():
        rows=[]
        for index in range(len(dataset)):
            batch=dataset[index];owners=batch["owners_native"].numpy();present=len(np.unique(owners[owners>0]))
            rows.append({"sample_id":batch["sample_id"],"expected_words":batch["expected_words"],"native_words":present,
                "lost_words":batch["expected_words"]-present,"target_sha256":sha256(batch["mask"].numpy().tobytes()).hexdigest()})
            if (index+1)%250==0: print(f"[targets {size}] {index+1}/{len(dataset)}",flush=True)
        result[str(size)]={"images":len(rows),"native_words":sum(r["native_words"] for r in rows),
            "lost_words":sum(r["lost_words"] for r in rows),"per_image":rows}
    return {"status":"PASS","result_class":"smoke-test result","augmentation":"disabled","resolutions":result}


def run(args):
    cfg,source=read_config(args.config,args.mode)
    for arg,key in ((args.num_workers,"num_workers"),(args.batch_size,"batch_size")):
        if arg is not None:
            if type(arg) is not int or arg<(1 if key=="batch_size" else 0): raise ValueError("Invalid explicit "+key)
            cfg["training"][key]=arg
            cfg["scientific_choices"].append({"classification":"HARDWARE_DEVIATION","component":"explicit_"+key,
                "rationale":"Explicit CLI override, applied to every branch; no automatic OOM fallback"})
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; use CPU preflight/smoke or the GPU host")
    target=args.target_root.resolve();base=args.base_run.resolve();output=args.output_root.resolve();guard_output(output,target,base)
    if sha256_file(base/"best.pt")!=cfg["base_checkpoint_sha256"]: raise ValueError("Base checkpoint SHA256 differs")
    manifest_path=target/cfg["dataset"]["manifest"]
    if sha256_file(manifest_path)!=cfg["dataset"]["manifest_sha256"]: raise ValueError("Dataset manifest checksum differs")
    manifest=load_segmentation_manifest(manifest_path,require_checksums=True)
    if manifest.dataset!="total_text" or manifest.split_role!="train" or len(manifest.samples)!=1255:
        raise ValueError("Only qualified Total-Text TRAIN1255 is supported")
    train_ids,val_ids,split_sha=_split_indices(manifest,cfg["training"]["seed"],cfg["dataset"]["validation_images"])
    if split_sha!=cfg["dataset"]["validation_ids_sha256"]: raise ValueError("Frozen validation split differs")
    refs,ref_hashes,historical=load_references(base,cfg,manifest,val_ids,target)
    common=json_contract({"schema_version":1,"reproduction_mode":"approximate","result_class":"approximate reproduction",
        "config":cfg,"config_sha256":sha256_file(args.config),"resolved_preprocessing":source,
        "target_root":str(target),"base_run":str(base),"manifest_sha256":sha256_file(manifest_path),
        "base_checkpoint_sha256":cfg["base_checkpoint_sha256"],"validation_ids_sha256":split_sha,
        "canonical_reference_sha256":ref_hashes,"source_sha256":{name:sha256_file(REPO/name) for name in SOURCES}})
    output.mkdir(parents=True,exist_ok=True)
    with (output/"runner.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stub_contract={**common,"branch":"control512","training_config":asdict(training_config(cfg,"control512")),
                       "execution_device":"cuda:0"}
        ensure_experiment_contract(output,common,historical,stub_contract);args._owns_output=True
        fixed_json(output/"historical_reference.json",historical)
        if args.action=="compare": return compare(output,cfg)
        if args.action=="preflight":
            path=output/"target_qualification.json"
            if path.exists():
                report=json.loads(path.read_text())
                if report.get("status")!="PASS" or report["run_contract"]!=common: raise ValueError("Target preflight contract differs")
                return report
            datasets={s:TextRecallDataset(SegmentationDataset(manifest,target,transform=make_transform(source,input_size=s,train=False))) for s in (512,768)}
            report=qualify(datasets);report["run_contract"]=common;_json(path,report);return report
        summaries=[];images={s.sample_id:target/s.image_path for s in manifest.samples}
        for branch in list(cfg["branches"]) if args.branch=="all" else [args.branch]:
            args._active_branch=branch;spec=cfg["branches"][branch];tc=training_config(cfg,branch)
            stage=output/"smoke"/str(device).replace(":","_")/branch if args.action=="smoke" else output/"branches"/branch
            stage.mkdir(parents=True,exist_ok=True)
            contract=json_contract({**common,"branch":branch,"training_config":asdict(tc),"execution_device":str(device)})
            fixed_json(stage/"run_contract.json",contract);completed=stage/"run_manifest.json"
            if completed.exists():
                result=json.loads(completed.read_text())
                if result["status"]!="SUCCESS" or result["run_contract"]!=contract or sha256_file(stage/"best.pt")!=result["checkpoint"]["sha256"]:
                    raise ValueError("Completed training provenance differs")
                if args.action=="train": summaries.append(result);continue
                if args.action=="evaluate" and (stage/"validation.json").exists():
                    cached=json.loads((stage/"validation.json").read_text())
                    if cached["run_contract"]!=contract or cached["checkpoint_sha256"]!=result["checkpoint"]["sha256"]:
                        raise ValueError("Cached evaluation provenance differs")
                    for sample in cached["exports"]:
                        folder=stage/"validation_exports"/sample["sample_id"].replace(":","__")
                        for name,digest in sample["files"].items():
                            if sha256_file(folder/name)!=digest: raise ValueError("Cached validation export changed")
                    summaries.append(cached);continue
            elif args.action=="evaluate": raise ValueError("Evaluation requires completed training")
            transforms=[make_transform(source,input_size=spec["input_size"],train=x) for x in (True,False)]
            datasets=[TextRecallDataset(SegmentationDataset(manifest,target,transform=t)) for t in transforms]
            seed_everything(tc.seed,deterministic_algorithms=True);model,receipt=initialize_model(base,cfg)
            fixed_json(stage/"warm_start.json",receipt)
            if device.type=="cuda": _prepare_cuda_memory_tracking(device)
            model.to(device);optimizer=build_optimizer(model,tc);sampler=EpochShuffleSampler(len(train_ids),seed=tc.seed)
            options={"num_workers":cfg["training"]["num_workers"],"pin_memory":device.type=="cuda",
                "timeout":cfg["training"]["loader_timeout_seconds"] if cfg["training"]["num_workers"] else 0}
            loaders=[DataLoader(Subset(datasets[0],train_ids),batch_size=tc.batch_size,sampler=sampler,**options),
                     DataLoader(Subset(datasets[1],val_ids),batch_size=1,shuffle=False,**options)]
            if args.action=="smoke":
                before=model.head.weight.detach().clone()
                trained=train_recall_epoch(model,[next(iter(loaders[0]))],optimizer,device,loss_name=spec["loss"],loss_config=cfg["loss"])
                if torch.equal(before,model.head.weight): raise RuntimeError("Real batch did not update text head")
                result={"status":"PASS","result_class":"smoke-test result","run_contract":contract,"train":trained,
                    "validation_probe":evaluate(model,[next(iter(loaders[1]))],refs,device,cfg),"parameters":receipt["parameters"],
                    "memory":memory(device,cfg,branch),"environment":runtime_environment()}
                _json(stage/"smoke.json",result)
            elif args.action=="train":
                result=train(stage,model,optimizer,loaders,refs,sampler,transforms[0],cfg,branch,contract,device,args.stop_after_epoch)
            else:
                saved=load_checkpoint(stage/"best.pt",model=model,expected_architecture=model.architecture_identity,
                    expected_run_contract=contract,expected_training_config=tc)
                result=evaluate(model,loaders[1],refs,device,cfg,export_dir=stage/"validation_exports",image_paths=images,
                    export_threshold=saved["validation_metrics"]["threshold"])
                if result["operating_point"]!=saved["validation_metrics"]: raise ValueError("Selected checkpoint replay differs")
                result.update(status="SUCCESS",result_class="approximate reproduction",run_contract=contract,
                    checkpoint_sha256=sha256_file(stage/"best.pt"),memory=memory(device,cfg,branch,batch_size=1))
                _json(stage/"validation.json",result);del saved
            summaries.append(result);del model,optimizer,loaders,datasets;gc.collect()
            if device.type=="cuda": torch.cuda.empty_cache()
        return {"status":"SUCCESS" if all(r["status"] in ("PASS","SUCCESS") for r in summaries) else "IN_PROGRESS",
                "action":args.action,"branches":len(summaries),"output_root":str(output)}


def compare(output,cfg):
    rows={};orders={};common=json.loads((output/"experiment_contract.json").read_text())
    for branch in cfg["branches"]:
        directory=output/"branches"/branch;result=json.loads((directory/"run_manifest.json").read_text())
        validation=json.loads((directory/"validation.json").read_text())
        if (result["status"]!="SUCCESS" or result["run_contract"]["config"]!=cfg
            or result["run_contract"]["source_sha256"]!=common["source_sha256"]
            or validation["run_contract"]!=result["run_contract"]
            or sha256_file(directory/"best.pt")!=result["checkpoint"]["sha256"]
            or validation["checkpoint_sha256"]!=result["checkpoint"]["sha256"]
            or validation["operating_point"]!=result["validation_operating_point"]): raise ValueError("Comparison artifacts differ")
        records=_epoch_log(directory/"epochs.jsonl")
        if len(records)!=cfg["training"]["epochs"]: raise ValueError("Incomplete comparison epoch history")
        orders[branch]=[r["train"]["sample_order_sha256"] for r in records]
        rows[branch]={**cfg["branches"][branch],"selected_epoch":result["checkpoint"]["epoch"],
            "checkpoint_sha256":result["checkpoint"]["sha256"],"operating_point":validation["operating_point"],
            "all_thresholds":validation["thresholds"],"secondary_component_geometry":validation["secondary_component_geometry"]}
    if any(o!=orders["control512"] for o in orders.values()): raise ValueError("Matched sample order differs")
    def delta(a,b):
        return {k:rows[a]["operating_point"][k]-rows[b]["operating_point"][k] for k in
            ("complete_word_rate","mean_word_coverage","pixel_precision","pixel_recall","pixel_f2")}
    result={"status":"SUCCESS","result_class":"approximate reproduction","reproduction_mode":"approximate",
        "branches":rows,"matched_sample_order":True,"all_precision_requirements_met":all(r["operating_point"]["precision_requirement_met"] for r in rows.values()),
        "effects":{"loss_at512":delta("tversky512","control512"),"resolution_with_dice":delta("control768","control512"),
            "resolution_with_tversky":delta("tversky768","tversky512"),"combined":delta("tversky768","control512")},
        "limitations":"One seed, reused TRAIN holdout; no TEST, PARSeq or auxiliary-offset end-to-end claim"}
    _json(output/"comparison.json",result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("approximate","strict"),required=True);p.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    p.add_argument("--target-root",type=Path,required=True);p.add_argument("--base-run",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True);p.add_argument("--device",choices=("cpu","cuda:0"),required=True)
    p.add_argument("--action",choices=("preflight","smoke","train","evaluate","compare"),required=True)
    p.add_argument("--branch",choices=("all",*BRANCHES),default="all");p.add_argument("--num-workers",type=int)
    p.add_argument("--batch-size",type=int);p.add_argument("--stop-after-epoch",type=int);args=p.parse_args()
    try:
        report=run(args)
        print(json.dumps({k:v for k,v in report.items() if k not in ("run_contract","resolutions","per_threshold_images")},indent=2))
    except Exception as error:
        if getattr(args,"_owns_output",False):
            cfg,_=read_config(args.config,args.mode);branch=getattr(args,"_active_branch",args.branch)
            details={"status":"FAILED","type":type(error).__name__,"message":str(error),"action":args.action,"branch":branch,
                "result_class":"smoke-test result" if args.action=="smoke" else "approximate reproduction","device":args.device}
            if branch in BRANCHES:
                if args.batch_size is not None: cfg["training"]["batch_size"]=args.batch_size
                details.update(memory(torch.device(args.device),cfg,branch))
            details["oom"]=isinstance(error,torch.cuda.OutOfMemoryError)
            _json(args.output_root/("failure-"+str(time.time_ns())+".json"),details)
        raise
    return 0


if __name__=="__main__": raise SystemExit(main())
