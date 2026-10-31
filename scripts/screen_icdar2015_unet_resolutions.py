"""Frozen-weight resolution screening on the fixed TRAIN validation100 only."""

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
    evaluate_geometry, load_config, load_training_population, model, references, source_hashes,
)
from segmentation.training import runtime_environment, seed_everything
from scripts.train_icdar2015_totaltext_transfer import write_json


def select_larger(rows, config):
    candidates=[name for name in rows if name!="square512"]
    return sorted(candidates,key=lambda name:(-rows[name]["aggregate"]["proposal_f1_iou_50"],
        -rows[name]["aggregate"]["word_recall_iou_50"],
        config["resolutions"][name][0]*config["resolutions"][name][1],name))[0]


def run(args):
    config=load_config(args.config)
    if args.batch_size<1:raise ValueError("Positive explicit validation batch size required")
    root=args.data_root.resolve()
    manifest,fit,val=load_training_population(config,root)
    # This rechecks every input checksum before screening.
    SegmentationDataset(manifest,root,verify_checksums=True)
    refs=references(manifest,root,val)
    contract={"config":config,"config_sha256":file_sha256(args.config),
        "source_sha256":source_hashes("scripts/screen_icdar2015_unet_resolutions.py"),
        "data_root":str(root),"checkpoint":str(args.source_checkpoint.resolve()),
        "validation_ids":[manifest.samples[i].sample_id for i in val],
        "validation_batch_size":args.batch_size,"device":args.device,"precision":"float32","tf32":False}
    path=args.output_dir/"run_contract.json"
    if path.exists():
        if json.loads(path.read_text())!=contract:raise ValueError("Screening contract changed")
    else:write_json(path,contract)
    seed_everything(config["training"]["seed"],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    network=model(config,"batchnorm",source_checkpoint=args.source_checkpoint).to(device)
    rows={}
    for name,shape in config["resolutions"].items():
        result=args.output_dir/name/"results.json"
        if result.exists():
            row=json.loads(result.read_text())
            if row["run_contract"]!=contract:raise ValueError("Cached screening differs")
            for image in row["per_image"]:
                if file_sha256(result.parent/"masks"/image["mask_file"])!=image["mask_sha256"]:
                    raise ValueError("Screening mask cache changed")
            rows[name]=row
            continue
        transform=make_transform(config,shape,augment=False)
        data=OwnerDataset(SegmentationDataset(manifest,root,transform=transform,verify_checksums=False))
        batches=DataLoader(Subset(data,val),batch_size=args.batch_size,shuffle=False,
            num_workers=config["training"]["num_workers"],pin_memory=device.type=="cuda")
        if device.type=="cuda":_prepare_cuda_memory_tracking(device)
        start=time.monotonic()
        row=evaluate_geometry(network,batches,refs,config,shape,device=device,save_dir=result.parent/"masks")
        row.update(status="SUCCESS",run_contract=contract,resolution=name,image_hw=shape,
            elapsed_seconds=time.monotonic()-start,transform=transform.metadata(),oom=False,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None)
        rows[name]=row;write_json(result,row)
        print(f"[resolution screen] {name}: {json.dumps(row['aggregate'],sort_keys=True)}",flush=True)
    selected=select_larger(rows,config)
    report={"status":"SUCCESS","reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":"TRAIN validation100 resolution screening", "run_contract":contract,
        "selected_larger_resolution":selected,"resolutions":{k:v["aggregate"] for k,v in rows.items()},
        "runtime":runtime_environment(),"host":socket.gethostname(),
        "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None}
    write_json(args.output_dir/"results.json",report)
    print(f"[resolution screen] selected larger: {selected}",flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","data-root","source-checkpoint","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--batch-size",type=int,required=True)
    parser.add_argument("--device",required=True)
    args=parser.parse_args()
    try:run(args)
    except Exception as error:
        device=torch.device(args.device)
        write_json(args.output_dir/"failure.json",{"status":"FAILED","error_type":type(error).__name__,
            "error":str(error),"oom":isinstance(error,torch.cuda.OutOfMemoryError),"device":args.device,
            "batch_size":args.batch_size,"precision":"float32",
            "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" and torch.cuda.is_initialized() else None,
            "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" and torch.cuda.is_initialized() else None})
        raise


if __name__=="__main__":main()
