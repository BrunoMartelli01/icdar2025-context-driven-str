"""Fit/decision-calibrate/evaluate Gap heads without using evaluation GT for choices."""

import argparse
from collections import Counter
import gc
from hashlib import sha256
import json
import math
from pathlib import Path
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from localize.adaspot_gap_preserve import decode_gap, preserve_support
from localize.adaspot_gap import gap_instances
from localize.word_multitask import parseq_word_proposals
from models.adaspot_gap import AdaSpotGapAGUNet
from models.adaspot_gap_context import sampled_patches, stratum_weights
from models.adaspot_gap_normalized import NormalizedGapHead
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE, ParseqApproxConfig, ParseqApproxRecognizer
from scripts.diagnose_adaspot_phase0 import REPO, check_hash, paired_changes, summarize, write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_adaspot_gap_context import load_validation, memory
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.adaspot_gap import AdaSpotGapDataset
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.training import seed_everything


def split_fit_calibration(rows, seed, count):
    ids=[r["sample_id"] for r in rows]
    if len(ids)!=len(set(ids)) or not 0<count<len(ids):raise ValueError("Unique TRAIN population and proper held-out size required")
    chosen=set(sorted(ids,key=lambda s:sha256(f"{seed}:gap-calibration:{s}".encode()).digest())[:count])
    return [s for s in ids if s not in chosen],[s for s in ids if s in chosen]


def counts_for(labels,item):
    maps=item["maps"];valid=maps["validity"]
    ignored=[int(i) for i in np.unique(labels[labels>0]) if not np.any((labels==i)&valid)]
    care=labels.copy();care[np.isin(care,ignored)]=0
    counts=instance_geometry(care,maps["owners"],maps["expected_owner_ids"].tolist())
    return counts,ignored


def aggregate(rows):
    total=Counter()
    for row in rows:total.update(row["counts"])
    return {**total,"word_f1":2*total["matched_gt_iou_ge_50"]/(total["raw_proposals"]+total["gt_words"])}


def candidate_key(candidate):
    a=candidate["aggregate"]
    return (a["word_f1"],-a["gt_words_touched_by_fused_components"],-candidate["epoch"],
        candidate["threshold"] if candidate["threshold"] is not None else 1.,candidate["decoder"]=="cut")


def extract_calibration(args,c,contract,manifest,calibration_ids,device):
    root,out=args.word_root.resolve(),args.output_root.resolve()
    by_id={s.sample_id:i for i,s in enumerate(manifest.samples)}
    transform=_transform(contract["resolved_preprocessing"],"letterbox",train=False)
    dataset=AdaSpotGapDataset(SegmentationDataset(manifest,root,transform=transform),**contract["resolved_config"]["targets"])
    loader=DataLoader(Subset(dataset,[by_id[s] for s in calibration_ids]),batch_size=c["image_batch_size"],
        shuffle=False,num_workers=c["num_workers"],timeout=60 if c["num_workers"] else 0)
    model=AdaSpotGapAGUNet("control",skip_floor=contract["resolved_config"]["skip_floor"],initial_gap_bias=c["initial_bias"])
    payload=torch.load(args.source_run/"control/best.pt",map_location="cpu",weights_only=False)
    if payload["run_contract"]!=contract:raise ValueError("Frozen source contract changed")
    model.load_state_dict(payload["model_state_dict"],strict=True);del payload
    model.eval().to(device)
    for p in model.parameters():p.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(device);items=[];receipts=[]
    for batch in loader:
        with torch.no_grad():evidence=model.forward_evidence(batch["image"].to(device))
        for i,sid in enumerate(batch["sample_id"]):
            sample=manifest.samples[by_id[sid]]
            with Image.open(root/sample.word_owner_annotation_path) as owner:
                ids=np.unique(np.asarray(owner));ids=ids[ids>0].astype(np.int32)
            maps={"probabilities":evidence["words"][i].sigmoid().cpu().numpy(),"canvas":batch["canvas"][i].numpy(),
                "validity":batch["validity"][i,0].numpy().astype(bool),"owners":batch["owners_native"][i].numpy(),
                "expected_owner_ids":ids,"gap_target":batch["gap_target"][i,0].numpy(),"text_target":batch["mask"][i,0].numpy().astype(bool)}
            folder=out/"calibration_features"/sid.replace(":","__");folder.mkdir(parents=True)
            feature=evidence["features"][i].detach().cpu().contiguous();torch.save(feature,folder/"features.pt")
            np.savez_compressed(folder/"maps.npz",**maps)
            baseline=gap_instances(maps["probabilities"],None,maps["canvas"],contract["resolved_config"]["postprocessing"],use_gap=False)[0]
            items.append({"sample_id":sid,"features":feature,"maps":maps,"baseline":baseline})
            receipts.append({"sample_id":sid,"features_path":str(folder/"features.pt"),"features_sha256":sha256_file(folder/"features.pt"),
                "maps_path":str(folder/"maps.npz"),"maps_sha256":sha256_file(folder/"maps.npz")})
        print(f"[head calibration features] {len(items)}/{len(calibration_ids)}",flush=True)
    measured=memory(device,"frozen Text+Kernel, calibration extraction only",[c["image_batch_size"],3,512,512])
    del model,evidence,dataset,loader;gc.collect();torch.cuda.empty_cache()
    write_json(out/"calibration_manifest.json",receipts)
    return items,measured


def calibrate(head,items,pp,c,epoch,device,directory):
    grid={(t,d):[] for t in c["thresholds"] for d in c["decoders"]}
    for item in items:
        maps=item["maps"];raw=maps["canvas"]&(maps["probabilities"][0]>=pp["text_threshold"])
        with torch.inference_mode():gap=head.native_logits(item["features"].to(device)).sigmoid().cpu().numpy()
        path=directory/"calibration_predictions"/str(epoch)/(item["sample_id"].replace(":","__")+".npz")
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,gap_probability=gap)
        for threshold in c["thresholds"]:
            cut=decode_gap(maps["probabilities"],gap,maps["canvas"],pp,threshold,"cut") if np.any(raw&(gap>threshold)) else item["baseline"]
            for decoder in c["decoders"]:
                labels=preserve_support(cut,raw) if decoder=="preserve" and not np.array_equal(cut>0,raw) else cut
                counts,_=counts_for(labels,item)
                grid[threshold,decoder].append({"sample_id":item["sample_id"],"counts":counts})
    return [{"epoch":epoch,"threshold":t,"decoder":d,"aggregate":aggregate(rows),"per_image":rows} for (t,d),rows in grid.items()]


def fit_arm(kind,x,classes,weights,mean,scale,calibration,pp,c,device,out):
    directory=out/kind;directory.mkdir()
    seed_everything(c["seed"],deterministic_algorithms=True)
    head=NormalizedGapHead(kind,mean,scale,hidden=c["hidden"],initial_bias=c["initial_bias"]).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"])
    rows=[{"sample_id":item["sample_id"],"counts":counts_for(item["baseline"],item)[0]} for item in calibration]
    best={"epoch":0,"threshold":None,"decoder":"cut","aggregate":aggregate(rows),"per_image":rows}
    best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};candidates=[best];history=[]
    generator=torch.Generator(device="cpu").manual_seed(c["seed"])
    labels=(classes<2).float();torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,c["epochs"]+1):
        cpu_order=torch.randperm(len(x),generator=generator);order=cpu_order.to(device)
        loss_sum,steps=0.,0
        for start in range(0,len(x),c["pixel_batch_size"]):
            indices=order[start:start+c["pixel_batch_size"]];optimizer.zero_grad(set_to_none=True)
            logits=head(x[indices]);loss=(F.binary_cross_entropy_with_logits(logits,labels[indices],reduction="none")*weights[classes[indices]]).mean()
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite normalized-head loss")
            output_layer=head.last if kind=="nonlinear" else head.first
            before=output_layer.weight.detach().clone() if epoch==1 and start==0 else None
            loss.backward();optimizer.step();steps+=1;loss_sum+=float(loss.detach())*len(indices)
            if before is not None:
                positions=torch.tensor([0,255,256,32896,65535]);feature=calibration[0]["features"]
                with torch.inference_mode():
                    point=head(sampled_patches(feature,positions).to(device));native=head.native_logits(feature.to(device)).flatten()[positions.to(device)]
                delta=float((output_layer.weight.detach()-before).abs().max());error=float((point-native).abs().max())
                if delta<=0 or error>1e-4 or logits.device.type!="cuda":raise RuntimeError("CUDA update/native equivalence failed")
                write_json(directory/"first_cuda_step.json",{"status":"PASS","result_class":"smoke-test result","device":str(logits.device),
                    "output_weight_update_linf":delta,"point_native_error":error,"parameters":sum(p.numel() for p in head.parameters())})
        record={"epoch":epoch,"loss":loss_sum/len(x),"cuda_steps":steps,"sample_order_sha256":sha256(cpu_order.numpy().tobytes()).hexdigest()}
        history.append(record);write_json(directory/"epochs"/f"{epoch:03d}.json",record)
        if epoch in c["checkpoint_epochs"]:
            current=calibrate(head,calibration,pp,c,epoch,device,directory);candidates.extend(current)
            candidate=max(current,key=candidate_key)
            if candidate_key(candidate)>candidate_key(best):
                best=candidate;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            write_json(directory/"calibration"/f"{epoch:03d}.json",current)
        torch.save({"kind":kind,"head":head.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,
            "generator_state":generator.get_state()},directory/"last.pt")
        print(f"[{kind}] epoch={epoch}/{c['epochs']} loss={record['loss']:.5f} selected_epoch={best['epoch']} calibration_F1={best['aggregate']['word_f1']:.5f} threshold={best['threshold']} decoder={best['decoder']}",flush=True)
    torch.save({"kind":kind,"head":best_state,"selection":best},directory/"best.pt")
    result={"completed_epochs":c["epochs"],"history":history,"candidates":candidates,"selection":best,
        "checkpoint_sha256":sha256_file(directory/"best.pt"),"parameters":sum(p.numel() for p in head.parameters()),
        "memory":memory(device,f"normalized {kind} head + fit patches",[c["pixel_batch_size"],576])}
    write_json(directory/"training_complete.json",result)
    return result


def run(args):
    config_path=args.config.resolve()
    if args.mode!="approximate" or REPO/"configs/approximate" not in config_path.parents:raise ValueError("Explicit APPROXIMATE config required")
    c=json.loads(config_path.read_text())
    if (c["reproduction_mode"]!=args.mode or c["arms"]!=["linear","nonlinear"] or c["epochs"]<1
            or any(not 0<t<1 for t in c["thresholds"]) or c["decoders"]!=["cut","preserve"]
            or any(e<1 or e>c["epochs"] for e in c["checkpoint_epochs"])):raise ValueError("Unsupported explicit normalized experiment")
    device=torch.device(args.device)
    if device.type!="cuda" or not torch.cuda.is_available():raise RuntimeError("GPU required; no CPU fallback")
    torch.cuda.init();torch.set_num_threads(c["cpu_threads"]);cv2.setNumThreads(1)
    seed_everything(c["seed"],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.output_root.resolve()
    if any(out==p.resolve() or p.resolve() in out.parents for p in (REPO,args.context_run,args.previous_probe,args.source_run,args.phase0_run,args.word_root)):
        raise ValueError("New external output root required")
    context=json.loads((args.context_run/"results.json").read_text())
    context_audit=json.loads((args.context_run/"independent_audit.json").read_text())
    check_hash(args.context_run/"results.json",context_audit["result_sha256"])
    if context["status"]!="SUCCESS" or context_audit["status"]!="PASS":raise ValueError("Qualified source context experiment required")
    for key in ("source_run","previous_probe","phase0_run","word_root"):
        if getattr(args,key).resolve()!=Path(context["protocol"][key]).resolve():raise ValueError("Pinned input root differs: "+key)
    hashes=dict(context["protocol"]["source_hashes"])
    for p,h in hashes.items():check_hash(Path(p),h)
    for p in (args.context_run/"results.json",args.context_run/"training_manifest.json",config_path):hashes[str(p.resolve())]=sha256_file(p)
    for name in ("models/adaspot_gap_normalized.py","localize/adaspot_gap_preserve.py","scripts/train_adaspot_gap_normalized.py"):
        hashes[str(REPO/name)]=sha256_file(REPO/name)
    contract=json.loads((args.source_run/"control/run_contract.json").read_text());sc=contract["resolved_config"];pp=sc["postprocessing"]
    manifest=load_segmentation_manifest(args.word_root/sc["train_manifest"],require_checksums=True)
    if manifest.dataset!="total_text" or manifest.split_role!="train":raise ValueError("TRAIN only")
    train_ids,val_ids,split_hash=_split_indices(manifest,c["seed"],sc["validation_count"])
    if split_hash!=sc["validation_ids_sha256"]:raise ValueError("Frozen evaluation split changed")
    tm=json.loads((args.context_run/"training_manifest.json").read_text())
    fit_ids,calib_ids=split_fit_calibration(tm["per_image"],c["seed"],c["calibration_images"])
    expected={manifest.samples[i].sample_id for i in train_ids};evaluation={manifest.samples[i].sample_id for i in val_ids}
    if set(fit_ids)|set(calib_ids)!=expected or (set(fit_ids)|set(calib_ids))&evaluation:raise ValueError("Fit/calibration/evaluation isolation failed")
    out.mkdir(parents=True,exist_ok=False);args._output_created=True
    protocol={"mode":"approximate","result_class":"approximate reproduction","resolved_config":c,"source_hashes":hashes,
        "fit_ids":fit_ids,"calibration_ids":calib_ids,"evaluation_ids":sorted(evaluation),"word_root":str(args.word_root.resolve()),
        "source_run":str(args.source_run.resolve()),"context_run":str(args.context_run.resolve()),"previous_probe":str(args.previous_probe.resolve()),
        "phase0_run":str(args.phase0_run.resolve()),"device":str(device),"gpu":torch.cuda.get_device_name(device),"dtype":"float32",
        "torch":str(torch.__version__),"cuda":torch.version.cuda,"selection_uses_evaluation":False,
        "parseq_source":str(args.parseq_source.resolve()),"parseq_checkpoint":str(args.parseq_checkpoint.resolve()),"parseq_archive":str(args.parseq_archive.resolve())}
    write_json(out/"protocol.json",protocol)
    check_hash(Path(tm["cache_path"]),tm["cache_sha256"])
    cache=torch.load(tm["cache_path"],map_location="cpu",weights_only=True)
    positions=[];offset=0
    for row in tm["per_image"]:
        n=sum(row["stratum_counts"])
        if row["sample_id"] in set(fit_ids):positions.append(torch.arange(offset,offset+n))
        offset+=n
    indices=torch.cat(positions);torch.save(indices,out/"fit_pixel_indices.pt")
    classes=cache["strata"][indices];weights,counts=stratum_weights(classes)
    x=cache["patches"][indices].to(device);del cache
    var,mean=torch.var_mean(x,dim=0,unbiased=False);scale=var.clamp_min(0).sqrt().clamp_min(c["normalization_std_floor"])
    mean,scale=mean.cpu(),scale.cpu()
    torch.save({"mean":mean,"scale":scale},out/"normalization.pt")
    write_json(out/"fit_manifest.json",{"fit_images":len(fit_ids),"calibration_images":len(calib_ids),"fit_points":len(indices),
        "stratum_counts":counts.tolist(),"stratum_weights":weights.tolist(),"fit_indices_sha256":sha256_file(out/"fit_pixel_indices.pt"),
        "normalization_sha256":sha256_file(out/"normalization.pt")})
    classes,weights=classes.to(device),weights.to(device)
    calibration,extraction=extract_calibration(args,c,contract,manifest,calib_ids,device)
    trained={kind:fit_arm(kind,x,classes,weights,mean,scale,calibration,pp,c,device,out) for kind in c["arms"]}
    if [r["sample_order_sha256"] for r in trained["linear"]["history"]]!=[r["sample_order_sha256"] for r in trained["nonlinear"]["history"]]:
        raise ValueError("Matched fit permutations differ")
    del x,classes,weights,calibration;gc.collect();torch.cuda.empty_cache()
    # Evaluation features and labels are loaded only AFTER all choices are frozen.
    cached=json.loads((args.previous_probe/"feature_manifest.json").read_text())
    validation=load_validation(cached["validation"],args.source_run)
    baseline=json.loads((args.phase0_run/"results.json").read_text())["arms"]["pred/axis/ar1"]
    by_gt={r["sample_id"]:r["ground_truth"] for r in baseline["per_image"]}
    frozen={"baseline":[]};native={};unique={}
    for kind in ["baseline"]+c["arms"]:
        samples=[];native_rows=[]
        if kind!="baseline":
            saved=torch.load(out/kind/"best.pt",map_location="cpu",weights_only=True)
            head=NormalizedGapHead(kind,mean,scale,hidden=c["hidden"],initial_bias=c["initial_bias"]).to(device)
            head.load_state_dict(saved["head"],strict=True);selection=saved["selection"]
        for item in validation:
            sid=item["sample_id"];maps=item["maps"]
            if kind=="baseline":gap=None;labels=decode_gap(maps["probabilities"],None,maps["canvas"],pp,None,"cut")
            else:
                with torch.inference_mode():gap=head.native_logits(item["features"].to(device)).sigmoid().cpu().numpy()
                labels=decode_gap(maps["probabilities"],gap,maps["canvas"],pp,selection["threshold"],selection["decoder"])
            counts,ignored=counts_for(labels,item);native_rows.append({"sample_id":sid,"counts":counts})
            with Image.open(item["proposal"]["image_path"]) as image:rgb=np.asarray(image.convert("RGB"))
            predictions=[]
            for p in parseq_word_proposals(labels,(rgb.shape[1],rgb.shape[0]),padding=0,min_area=0,max_instances=None):
                if p["instance_id"] in ignored:continue
                x0,y0,x1,y1=p["bbox"];crop=np.ascontiguousarray(rgb[y0:y1,x0:x1]);digest=crop_array_hash(crop)
                path=out/"crops"/(digest+".png");path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():Image.fromarray(crop).save(path)
                unique[digest]=path;predictions.append({**p,"crop_rgb_sha256":digest})
            samples.append({"sample_id":sid,"ground_truth":by_gt[sid],"predictions":predictions})
            path=out/kind/"evaluation_maps"/(sid.replace(":","__")+".npz");path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,labels=labels,gap_probability=gap if gap is not None else np.zeros_like(labels,dtype=np.float32))
        frozen[kind]=samples;native[kind]={"aggregate":aggregate(native_rows),"per_image":native_rows}
        write_json(out/kind/"frozen_ocr_inputs.json",samples)
    del head,validation;gc.collect();torch.cuda.empty_cache()
    recognizer=None;readings={}
    try:
        recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,checkpoint=args.parseq_checkpoint,
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=args.parseq_archive,device=str(device)))
        torch.cuda.reset_peak_memory_stats(device);jobs=sorted(unique.items())
        for start in range(0,len(jobs),c["image_batch_size"]):
            batch=jobs[start:start+c["image_batch_size"]];crops=[]
            for digest,path in batch:
                with Image.open(path) as image:crop=np.asarray(image.convert("RGB"))
                if crop_array_hash(crop)!=digest:raise ValueError("Frozen OCR crop changed")
                crops.append(crop)
            for (digest,_),p in zip(batch,recognizer.recognize_candidate_batch(crops)):readings[digest]=p.to_dict()
        ocr_memory=memory(device,"qualified PARSeq AR1 only",[c["image_batch_size"],3,32,128])
    finally:
        if recognizer is not None:recognizer.release()
    write_json(out/"ocr_readings.json",readings)
    results={};order={s["sample_id"]:i for i,s in enumerate(baseline["per_image"])}
    for kind,samples in frozen.items():
        for sample in samples:sample["predictions"]=[{**p,"text":readings[p["crop_rgb_sha256"]]["raw_text"]} for p in sample["predictions"]]
        samples.sort(key=lambda s:order[s["sample_id"]]);scored=summarize(samples)
        if scored["aggregate"]["gt_words"]!=baseline["aggregate"]["gt_words"]:raise ValueError("Evaluation denominator changed")
        results[kind]={"ocr":scored,"native":native[kind],"changes_vs_baseline":paired_changes(baseline,scored),**(trained[kind] if kind in trained else {})}
    if results["baseline"]["ocr"]["aggregate"]["exact_casefold_iou_ge_50"]!=baseline["aggregate"]["exact_casefold_iou_ge_50"]:
        raise ValueError("Fresh baseline OCR failed to replay")
    for p,h in hashes.items():check_hash(Path(p),h)
    for row in cached["validation"]:check_hash(Path(row["feature_path"]),row["feature_sha256"])
    for row in json.loads((out/"calibration_manifest.json").read_text()):
        check_hash(Path(row["features_path"]),row["features_sha256"]);check_hash(Path(row["maps_path"]),row["maps_sha256"])
    write_json(out/"results.json",{"status":"SUCCESS","result_class":"approximate reproduction","protocol":protocol,"arms":results,
        "extraction_memory":extraction,"ocr_memory":ocr_memory})
    lines=["# Normalized Gap training — approximate reproduction","","Head fit1030 / calibration100 / evaluation125; calibration-only choices, exploratory reused evaluation holdout. No official or independent-transfer claim.","",
        "| Arm | Chosen epoch | Threshold / decoder | Exact /983 | Localized RGB | Proposals | Hmean | Recovered / damaged |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for kind,r in results.items():
        a,ch=r["ocr"]["aggregate"],r["changes_vs_baseline"];s=r.get("selection",{"epoch":0,"threshold":None,"decoder":"off"})
        lines.append(f"| {kind} | {s['epoch']} | {s['threshold']} / {s['decoder']} | {a['exact_casefold_iou_ge_50']} | {a['geometry_iou_ge_50']} | {a['proposals']} | {a['hmean']:.4%} | {ch['recovered']} / {ch['damaged']} |")
    (out/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:{"selection":v.get("selection"),"ocr":v["ocr"]["aggregate"]} for k,v in results.items()},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=["approximate"],required=True)
    for name in ("config","context-run","previous-probe","source-run","phase0-run","word-root","output-root","parseq-source","parseq-checkpoint","parseq-archive"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",required=True);args=p.parse_args()
    try:run(args)
    except Exception as error:
        if getattr(args,"_output_created",False):
            failed={"status":"FAILED","error":str(error),"traceback":traceback.format_exc(),"oom":isinstance(error,torch.cuda.OutOfMemoryError)}
            if torch.cuda.is_available():failed.update(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(args.output_root/"failure.json",failed)
        raise
