"""Matched approximate continuations; selection uses fixed TRAIN validation only."""

import argparse
import json
from pathlib import Path
import socket
import time

import torch
from torch.utils.data import DataLoader, Subset

from segmentation.data import SegmentationDataset, file_sha256
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.icdar2015_improvements import OwnerDataset, make_transform
from segmentation.icdar2015_improvement_runtime import (
    evaluate_geometry, load_config, load_training_population, model, references,
    selection_key, source_hashes, train_epoch, training_config,
)
from segmentation.train import EpochShuffleSampler
from segmentation.training import (
    build_optimizer, load_checkpoint, runtime_environment, save_checkpoint, seed_everything,
)
from scripts.screen_icdar2015_unet_resolutions import select_larger
from scripts.train_icdar2015_totaltext_transfer import write_json


def run(args):
    config=load_config(args.config)
    branch=config["branches"][args.branch]
    training=training_config(config,branch["loss"])
    if args.validation_batch_size<1 or (args.smoke_steps is not None and args.smoke_steps<1):
        raise ValueError("Positive explicit execution batch size/steps required")
    screening=json.loads(args.screening_report.read_text())
    if (screening["status"]!="SUCCESS" or screening["run_contract"]["config"]!=config
            or screening["selected_larger_resolution"]!=select_larger(
                {k:{"aggregate":v} for k,v in screening["resolutions"].items()},config)):
        raise ValueError("Qualified TRAIN-only resolution screening required")
    resolution=screening["selected_larger_resolution"] if branch["resolution"]=="selected_larger" else branch["resolution"]
    shape=config["resolutions"][resolution]
    root=args.data_root.resolve()
    manifest,fit,val=load_training_population(config,root)
    if [manifest.samples[i].sample_id for i in val]!=screening["run_contract"]["validation_ids"]:
        raise ValueError("Screening/training validation assignment differs")
    SegmentationDataset(manifest,root,verify_checksums=True)
    evaluated=val[:training.batch_size] if args.smoke_steps else val
    refs=references(manifest,root,evaluated)
    result_class="smoke-test result" if args.smoke_steps else "approximate reproduction"
    contract={"config":config,"config_sha256":file_sha256(args.config),"branch":args.branch,
        "branch_config":branch,"resolution":resolution,"image_hw":shape,
        "source_sha256":source_hashes("scripts/train_icdar2015_unet_improvement.py",
            "scripts/screen_icdar2015_unet_resolutions.py"),
        "screening_report_sha256":file_sha256(args.screening_report),
        "source_checkpoint":str(args.source_checkpoint.resolve()),"data_root":str(root),
        "train_ids":[manifest.samples[i].sample_id for i in fit],
        "validation_ids":[manifest.samples[i].sample_id for i in val],
        "evaluated_ids":[manifest.samples[i].sample_id for i in evaluated],
        "device":args.device,"validation_batch_size":args.validation_batch_size,
        "precision":"float32","tf32":False,"smoke_steps":args.smoke_steps,"result_class":result_class}
    run_dir=args.output_dir.resolve();path=run_dir/"run_contract.json"
    if run_dir.exists():
        if not args.resume or not path.exists() or json.loads(path.read_text())!=contract:
            raise ValueError("Existing output requires exact-contract --resume")
        final_report=run_dir/"run_manifest.json"
        if final_report.exists():
            report=json.loads(final_report.read_text())
            if (report["status"]!="SUCCESS" or report["run_contract"]!=contract
                    or file_sha256(run_dir/"best.pt")!=report["checkpoint"]["sha256"]):
                raise ValueError("Completed run/checkpoint differs")
            details=json.loads((run_dir/"validation_geometry.json").read_text())
            if details["aggregate"]!=report["validation"]:raise ValueError("Completed validation differs")
            for row in details["per_image"]:
                if file_sha256(run_dir/"validation_masks"/row["mask_file"])!=row["mask_sha256"]:
                    raise ValueError("Completed validation mask changed")
            return report
    else:
        if args.resume:raise ValueError("Cannot resume missing output")
        write_json(path,contract);write_json(run_dir/"resolved_config.json",{"profile":config,"execution":contract})
    device=torch.device(args.device)
    seed_everything(training.seed,deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device.type=="cuda":_prepare_cuda_memory_tracking(device)
    network=model(config,branch["normalization"],source_checkpoint=args.source_checkpoint).to(device)
    optimizer=build_optimizer(network,training)
    tx=make_transform(config,shape,augment=True);vx=make_transform(config,shape,augment=False)
    train_data=OwnerDataset(SegmentationDataset(manifest,root,transform=tx,verify_checksums=False))
    val_data=OwnerDataset(SegmentationDataset(manifest,root,transform=vx,verify_checksums=False))
    sampler=EpochShuffleSampler(len(fit),seed=training.seed)
    options={"num_workers":config["training"]["num_workers"],"pin_memory":device.type=="cuda"}
    train_loader=DataLoader(Subset(train_data,fit),batch_size=training.batch_size,sampler=sampler,**options)
    val_loader=DataLoader(Subset(val_data,evaluated),batch_size=args.validation_batch_size,shuffle=False,**options)
    best_path,last_path=run_dir/"best.pt",run_dir/"last.pt"
    records=[];steps=0;best_epoch=0
    start=time.monotonic()
    if args.resume:
        payload=load_checkpoint(last_path,model=network,optimizer=optimizer,
            expected_architecture=network.architecture_identity,expected_run_contract=contract,
            expected_training_config=training,restore_rng=True)
        selection=payload["training_selection_metrics"]
        records=json.loads(selection["records_json"]);best_epoch=selection["best_epoch"]
        best_metrics=selection["best_metrics"];initial=selection["initial_metrics"];steps=payload["global_step"]
        if len(records)!=payload["epoch"]:raise ValueError("Resume history/checkpoint mismatch")
        best_payload=torch.load(best_path,map_location="cpu",weights_only=False)
        if (best_payload["run_contract"]!=contract or best_payload["epoch"]!=best_epoch
                or best_payload["validation_metrics"]!=best_metrics):raise ValueError("Best checkpoint differs")
        del payload,best_payload
    else:
        initial=evaluate_geometry(network,val_loader,refs,config,shape,device=device)["aggregate"]
        best_metrics=initial
        save_checkpoint(best_path,model=network,optimizer=optimizer,epoch=0,global_step=0,config=training,
            resolved_config=json.dumps(contract,sort_keys=True),architecture=network.architecture_identity,
            result_class=result_class,run_contract=contract,validation_metrics=initial)
        write_json(run_dir/"initial_validation.json",initial)
        print(f"[{args.branch}] epoch0 geometry F1={initial['proposal_f1_iou_50']:.6f}",flush=True)
    final_epoch=1 if args.smoke_steps else training.epochs
    for epoch in range(len(records)+1,final_epoch+1):
        global_epoch=config["source_checkpoint_epoch"]+epoch-1
        sampler.set_epoch(global_epoch);tx.set_epoch(global_epoch)
        metrics=train_epoch(network,train_loader,optimizer,config,loss_name=branch["loss"],
            device=device,max_steps=args.smoke_steps)
        if not args.smoke_steps and metrics["samples"]!=len(fit):raise ValueError("Incomplete training epoch")
        validation=evaluate_geometry(network,val_loader,refs,config,shape,device=device)["aggregate"]
        steps+=metrics["steps"];records.append({"epoch":epoch,"source_equivalent_epoch":global_epoch+1,
            "global_step":steps,"train":metrics,"validation":validation})
        checkpoint_args=dict(model=network,optimizer=optimizer,epoch=epoch,global_step=steps,config=training,
            resolved_config=json.dumps(contract,sort_keys=True),architecture=network.architecture_identity,
            result_class=result_class,run_contract=contract,train_metrics=metrics,validation_metrics=validation)
        if selection_key(validation,epoch)>selection_key(best_metrics,best_epoch):
            best_epoch,best_metrics=epoch,validation;save_checkpoint(best_path,**checkpoint_args)
        save_checkpoint(last_path,**checkpoint_args,training_selection_metrics={
            "records_json":json.dumps(records,sort_keys=True),"best_epoch":best_epoch,
            "best_metrics":best_metrics,"initial_metrics":initial})
        write_json(run_dir/"epochs.json",records)
        print(f"[{args.branch}] {epoch}/{final_epoch} loss={metrics['loss']:.6f} "
            f"val geometry F1={validation['proposal_f1_iou_50']:.6f} "
            f"recall={validation['word_recall_iou_50']:.6f} "
            f"best={best_epoch} elapsed={time.monotonic()-start:.1f}s",flush=True)
    load_checkpoint(best_path,model=network,expected_architecture=network.architecture_identity,
        expected_run_contract=contract,expected_training_config=training)
    detailed=evaluate_geometry(network,val_loader,refs,config,shape,device=device,save_dir=run_dir/"validation_masks")
    if detailed["aggregate"]!=best_metrics:raise RuntimeError("Best checkpoint validation replay differs")
    write_json(run_dir/"validation_geometry.json",detailed)
    report={"status":"SUCCESS","reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":result_class,"training_complete":not bool(args.smoke_steps) and len(records)==training.epochs,
        "completed_epochs":len(records),"run_contract":contract,"initial_validation":initial,
        "validation":detailed["aggregate"],"checkpoint":{"path":"best.pt","epoch":best_epoch,
            "sha256":file_sha256(best_path)},"normalization_migration":network.normalization_migration,
        "runtime":runtime_environment(),"hostname":socket.gethostname(),"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
        "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None,
        "oom":False,"input_shape":[training.batch_size,3,*shape],
        "output_shape":[training.batch_size,1,*[d//2 for d in shape]],
        "elapsed_seconds_this_invocation":time.monotonic()-start}
    write_json(run_dir/"run_manifest.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","data-root","source-checkpoint","screening-report","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--branch",required=True,choices=("control512","larger_bce","larger_instance","larger_instance_gn"))
    parser.add_argument("--validation-batch-size",type=int,required=True)
    parser.add_argument("--device",required=True)
    parser.add_argument("--smoke-steps",type=int)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        device=torch.device(args.device)
        write_json(args.output_dir/"failure.json",{"status":"FAILED","error_type":type(error).__name__,
            "error":str(error),"oom":isinstance(error,torch.cuda.OutOfMemoryError),"device":args.device,
            "branch":args.branch,"precision":"float32","configured_train_batch_size":8,
            "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" and torch.cuda.is_initialized() else None,
            "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" and torch.cuda.is_initialized() else None})
        raise


if __name__=="__main__":main()
