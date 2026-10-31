"""Synthetic text pretraining and fixed-split ICDAR fine-tuning with real AdamW."""

import argparse
import json
from pathlib import Path
import socket
import time

import torch
from torch.utils.data import DataLoader, Subset

from models.icdar2015_target70 import Target70AGUNet
from segmentation.icdar2015_encoder_decay import word_selection_key
from segmentation.icdar2015_synthetic_training import stage_config,training_config,optimizer as new_optimizer,load_source,CountedBatches
from segmentation.icdar2015_synthtext_cache import CachedSynthTextBCE
from segmentation.data import SegmentationDataset, file_sha256
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.icdar2015_improvements import OwnerDataset, make_transform
from segmentation.icdar2015_improvement_runtime import (
    REPO, evaluate_geometry, load_training_population, references, selection_key, source_hashes,
)
from segmentation.icdar2015_target70 import load_profile,TrainingSupervision,set_learning_rates,ValidationOCR,train_epoch
from segmentation.icdar2015_target70_ohem import TrainingPixels
from segmentation.train import EpochShuffleSampler
from segmentation.training import (load_checkpoint,save_checkpoint,seed_everything,
    restore_rng_state,runtime_environment)
from scripts.train_icdar2015_totaltext_transfer import write_json


def run(args):
    base=load_profile(args.config)
    if base['experiment_kind']!='icdar2015_synthtext_pretraining_v1':raise ValueError('Explicit synthetic research profile required')
    config=stage_config(base,args.branch);branch=config['branches'][args.branch];training=training_config(config,branch)
    if args.validation_batch_size<1 or (args.smoke_steps is not None and args.smoke_steps<1):raise ValueError('Positive execution limits required')
    synthetic_sha=file_sha256(args.synthetic_manifest);synthetic=json.loads(args.synthetic_manifest.read_text())
    preparation=json.loads((args.synthetic_manifest.parent/'results.json').read_text())
    if (preparation['status']!='SUCCESS' or preparation['manifest_sha256']!=synthetic_sha
            or synthetic['images']!=50000 or synthetic['run_contract']['config_sha256']!=config['synthetic_data_profile_sha256']):
        raise ValueError('Complete qualified50000-image cache required')
    for name,h in synthetic['run_contract']['source_sha256'].items():
        if file_sha256(REPO/name)!=h:raise ValueError('Synthetic target implementation changed: '+name)
    root=args.data_root.resolve();manifest,fit,val=load_training_population(config,root)
    SegmentationDataset(manifest,root,verify_checksums=True)
    evaluated=val[:branch["physical_batch_size"]] if args.smoke_steps else val
    refs=references(manifest,root,evaluated);shape=config["validation_image_hw"]
    training_indices=list(range(synthetic["images"])) if args.branch=="pretrain" else fit
    result_class="smoke-test result" if args.smoke_steps else "approximate reproduction"
    contract={"config":config,"config_sha256":file_sha256(args.config),"branch":args.branch,
        "branch_config":branch,"source_sha256":source_hashes("models/icdar2015_target70.py", "segmentation/icdar2015_target70.py",
            "segmentation/icdar2015_encoder_decay.py", "segmentation/icdar2015_target70_ohem.py",
            "segmentation/icdar2015_synthtext_cache.py", "segmentation/icdar2015_synthetic_training.py", "scripts/train_icdar2015_synthetic.py"),
        "train_ids":([r["sample_id"] for r in synthetic["samples"]] if args.branch=="pretrain" else [manifest.samples[i].sample_id for i in fit]),
        "icdar_fitting_ids":[manifest.samples[i].sample_id for i in fit],
        "synthetic_manifest_sha256":synthetic_sha,"synthetic_manifest":str(args.synthetic_manifest.resolve()),
        "synthetic_image_root":str(args.synthetic_image_root.resolve()),"synthetic_cache_root":str(args.synthetic_cache_root.resolve()),
        "validation_ids":[manifest.samples[i].sample_id for i in val],
        "evaluated_ids":[manifest.samples[i].sample_id for i in evaluated],
        "data_root":str(root),"source_checkpoint":str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256":file_sha256(args.source_checkpoint),
        "source_report":str(args.source_report.resolve()),"source_report_sha256":file_sha256(args.source_report),
        "source_validation":str(args.source_validation.resolve()),"source_validation_sha256":file_sha256(args.source_validation),
        "validation_batch_size":args.validation_batch_size,"device":args.device,
        "precision":"float32","tf32":False,"smoke_steps":args.smoke_steps,"result_class":result_class}
    out=args.output_dir.resolve();contract_path=out/"run_contract.json"
    if out.exists():
        if not args.resume or not contract_path.exists() or json.loads(contract_path.read_text())!=contract:
            raise ValueError("Existing output requires exact-contract --resume")
        if (out/"run_manifest.json").exists():
            report=json.loads((out/"run_manifest.json").read_text())
            if report["status"]!="SUCCESS" or report["run_contract"]!=contract or file_sha256(out/"best.pt")!=report["checkpoint"]["sha256"]:
                raise ValueError("Completed run changed")
            return report
    else:
        if args.resume:raise ValueError("Cannot resume missing run")
        write_json(contract_path,contract);write_json(out/"resolved_config.json",{"profile":config,"execution":contract})
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; no substitution")
    seed_everything(training.seed,deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device.type=="cuda":_prepare_cuda_memory_tracking(device)
    network=Target70AGUNet(encoder_stages=branch['encoder_trainable_stages'],skip_floor=config['skip_floor'],max_groups=config['group_norm_max_groups']).to(device)
    payload=load_source(network,args,config,synthetic_sha);del payload
    optimizer=new_optimizer(network,config,branch);source_step=0
    tx=make_transform(config,branch['image_hw'],augment=True);vx=make_transform(config,shape,augment=False)
    if args.branch=='pretrain':
        data=CachedSynthTextBCE(synthetic,image_root=args.synthetic_image_root,cache_root=args.synthetic_cache_root,transform=tx)
    else:
        data=TrainingSupervision(TrainingPixels(SegmentationDataset(manifest,root,transform=tx,verify_checksums=False)),negative_ignore=True)
    validation=OwnerDataset(SegmentationDataset(manifest,root,transform=vx,verify_checksums=False))
    sampler=EpochShuffleSampler(len(training_indices),seed=training.seed)
    options={"num_workers":config["training"]["num_workers"],"pin_memory":device.type=="cuda"}
    loader=DataLoader(Subset(data,training_indices),batch_size=branch["physical_batch_size"],sampler=sampler,**options)
    val_loader=DataLoader(Subset(validation,evaluated),batch_size=args.validation_batch_size,shuffle=False,**options)
    ocr=ValidationOCR(args,config)
    records=[];steps=source_step;best_epoch=0;geometry_best_epoch=0;start=time.monotonic()
    if args.resume:
        payload=load_checkpoint(out/"last.pt",model=network,optimizer=optimizer,
            expected_architecture=network.architecture_identity,expected_run_contract=contract,
            expected_training_config=training,restore_rng=True)
        state=payload["training_selection_metrics"];records=json.loads(state["records_json"])
        best_epoch=state["best_epoch"];best_word=state["best_word"];best_geometry=state["best_geometry"]
        best_emitted=state["best_emitted"];best_policy=state["best_policy"]
        geometry_best=state["geometry_best"];geometry_best_epoch=state["geometry_best_epoch"]
        steps=payload["global_step"]
        if len(records)!=payload["epoch"]:raise ValueError("Resume history/checkpoint mismatch")
        del payload
    else:
        directory=out/"validation"/"epoch_000"
        details=evaluate_geometry(network,val_loader,refs,config,shape,device=device,save_dir=directory/"masks")
        word=ocr.evaluate(details,refs,manifest,root,shape,directory/"masks")
        write_json(directory/"geometry.json",details);write_json(directory/"words.json",word)
        best_geometry=geometry_best=details["aggregate"];best_word=word["aggregate"]
        best_emitted=word["emitted_aggregate"];best_policy=word["emission_policy"]
        if not args.smoke_steps:
            expected=json.loads(args.source_validation.read_text())["aggregate"]
            if expected!=best_word:raise ValueError("Initial continuation no longer reproduces qualified source validation")
        save_checkpoint(out/"best.pt",model=network,optimizer=optimizer,epoch=0,global_step=steps,config=training,
            resolved_config=json.dumps(contract,sort_keys=True),architecture=network.architecture_identity,
            result_class=result_class,run_contract=contract,validation_metrics={"geometry":best_geometry,"words":best_word,"emitted":best_emitted,"emission_policy":best_policy})
        print(f"[{args.branch}] epoch0 raw_word_F1={best_word['alphanumeric_casefold']['f1']:.6f} emitted_word_F1={best_emitted['alphanumeric_casefold']['f1']:.6f} geometry_F1={best_geometry['proposal_f1_iou_50']:.6f}",flush=True)
    final_epoch=1 if args.smoke_steps else training.epochs
    for epoch in range(len(records)+1,final_epoch+1):
        effective=epoch-1;sampler.set_epoch(effective);tx.set_epoch(effective)
        rates=set_learning_rates(optimizer,config,epoch)
        counted=CountedBatches(loader,args.smoke_steps*branch["gradient_accumulation"] if args.smoke_steps else None)
        train=train_epoch(network,counted,optimizer,config,branch,device=device,max_steps=None);train.update(counted.counts)
        if not args.smoke_steps and train["samples"]!=len(training_indices):raise ValueError("Incomplete training epoch")
        directory=out/"validation"/f"epoch_{epoch:03d}"
        # Geometry is cheap enough to check every epoch. Word evaluation occurs
        # at the declared cadence, every new geometry best, and the last epoch.
        details=evaluate_geometry(network,val_loader,refs,config,shape,device=device)
        geometry=details["aggregate"]
        new_geometry=selection_key(geometry,epoch)>selection_key(geometry_best,geometry_best_epoch)
        do_word=args.branch=="pretrain" or bool(args.smoke_steps) or new_geometry or epoch%config["word_validation_interval"]==0 or epoch==final_epoch
        word=None
        if do_word:
            replay=evaluate_geometry(network,val_loader,refs,config,shape,device=device,save_dir=directory/"masks")
            if replay["aggregate"]!=geometry:raise RuntimeError("Validation replay changed")
            word=ocr.evaluate(replay,refs,manifest,root,shape,directory/"masks")
            write_json(directory/"geometry.json",replay);write_json(directory/"words.json",word)
        if new_geometry:geometry_best,geometry_best_epoch=geometry,epoch
        steps+=train["steps"]
        records.append({"epoch":epoch,"paired_augmentation_epoch":effective,"global_step":steps,
            "learning_rates":rates,"train":train,"validation_geometry":geometry,
            "validation_words":word["aggregate"] if word else None,
            "validation_emitted":word["emitted_aggregate"] if word else None,
            "emission_policy":word["emission_policy"] if word else None})
        args_checkpoint=dict(model=network,optimizer=optimizer,epoch=epoch,global_step=steps,config=training,
            resolved_config=json.dumps(contract,sort_keys=True),architecture=network.architecture_identity,
            result_class=result_class,run_contract=contract,train_metrics=train,
            validation_metrics={"geometry":geometry,"words":word["aggregate"] if word else None,
                "emitted":word["emitted_aggregate"] if word else None,"emission_policy":word["emission_policy"] if word else None})
        if word and (args.branch=="pretrain" or word_selection_key(word["aggregate"],word["emitted_aggregate"],geometry,epoch)>word_selection_key(best_word,best_emitted,best_geometry,best_epoch)):
            best_word,best_geometry,best_epoch=word["aggregate"],geometry,epoch
            best_emitted,best_policy=word["emitted_aggregate"],word["emission_policy"]
            save_checkpoint(out/"best.pt",**args_checkpoint)
        save_checkpoint(out/"last.pt",**args_checkpoint,training_selection_metrics={"records_json":json.dumps(records,sort_keys=True),
            "best_epoch":best_epoch,"best_word":best_word,"best_geometry":best_geometry,
            "best_emitted":best_emitted,"best_policy":best_policy,
            "geometry_best":geometry_best,"geometry_best_epoch":geometry_best_epoch})
        write_json(out/"epochs.json",records)
        print(f"[{args.branch}] {epoch}/{final_epoch} loss={train['loss']:.6f} geometry_F1={geometry['proposal_f1_iou_50']:.6f} "
            f"word_F1={word['aggregate']['alphanumeric_casefold']['f1'] if word else 'not_scheduled'} "
            f"best_raw_F1={best_word['alphanumeric_casefold']['f1']:.6f} best_emitted_F1={best_emitted['alphanumeric_casefold']['f1']:.6f} best_epoch={best_epoch} elapsed={time.monotonic()-start:.1f}s",flush=True)
    load_checkpoint(out/"best.pt",model=network,expected_architecture=network.architecture_identity,
        expected_run_contract=contract,expected_training_config=training)
    directory=out/"selected_validation"
    detailed=evaluate_geometry(network,val_loader,refs,config,shape,device=device,save_dir=directory/"masks")
    selected_word=ocr.evaluate(detailed,refs,manifest,root,shape,directory/"masks")
    if (detailed["aggregate"]!=best_geometry or selected_word["aggregate"]!=best_word
            or selected_word["emitted_aggregate"]!=best_emitted or selected_word["emission_policy"]!=best_policy):
        raise RuntimeError("Selected replay changed")
    write_json(directory/"geometry.json",detailed);write_json(directory/"words.json",selected_word)
    report={"status":"SUCCESS","reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":result_class,"training_complete":not bool(args.smoke_steps) and len(records)==training.epochs,
        "run_contract":contract,"completed_epochs":len(records),"training_population_images":len(training_indices),
        "selected_validation_words_sha256":file_sha256(directory/"words.json"),"checkpoint":{"path":"best.pt","epoch":best_epoch,"sha256":file_sha256(out/"best.pt")},
        "validation_words":best_word,"validation_geometry":best_geometry,"geometry_best_epoch":geometry_best_epoch,
        "validation_emitted":best_emitted,"emission_policy":best_policy,
        "runtime":runtime_environment(),"hostname":socket.gethostname(),"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
        "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None,
        "oom":False,"input_shape":[branch["physical_batch_size"],3,*branch["image_hw"]],"output_shape":[branch["physical_batch_size"],1,*[d//2 for d in branch["image_hw"]]],"validation_image_hw":shape,
        "gradient_accumulation":branch["gradient_accumulation"],"elapsed_seconds_this_invocation":time.monotonic()-start}
    write_json(out/"run_manifest.json",report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","data-root","synthetic-manifest","synthetic-image-root","synthetic-cache-root","source-checkpoint","source-report","source-validation","parseq-source",
                 "parseq-source-archive","parseq-checkpoint","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--branch",required=True);parser.add_argument("--device",required=True)
    parser.add_argument("--validation-batch-size",type=int,required=True)
    parser.add_argument("--smoke-steps",type=int);parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.output_dir.exists() and not args.resume:raise ValueError("Preserve existing output; exact-contract resume required")
    try:run(args)
    except Exception as error:
        initialized=torch.cuda.is_initialized()
        write_json(args.output_dir/"failure.json",{"status":"FAILED","reproduction_mode":"approximate","claim_label":"approximate reproduction",
            "error_type":type(error).__name__,"error":str(error),"branch":args.branch,"device":args.device,"precision":"float32",
            "oom":isinstance(error,torch.cuda.OutOfMemoryError),"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if initialized else None,
            "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved() if initialized else None})
        raise


if __name__=="__main__":main()
