"""Full official TRAIN Gap refit; frozen prior choices, separate TEST evaluation."""

import argparse
from collections import Counter
import gc
from hashlib import sha256
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from localize.adaspot_gap_preserve import decode_gap
from localize.word_multitask import parseq_word_proposals
from models.adaspot_gap import AdaSpotGapAGUNet
from models.adaspot_gap_decoder import TrainableGapDecoder
from scripts.diagnose_adaspot_phase0 import REPO, check_hash, paired_changes, write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash, _annotation_path, _ground_truth
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _tensor
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_adaspot_gap_context import memory
from scripts.select_adaspot_gap_ocr import recognize, score
from scripts.train_adaspot_gap_decoder import state_digest
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.adaspot_gap import AdaSpotGapDataset
from segmentation.adaspot_gap_decoder import decoder_loss
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.training import seed_everything


def validate_population(train, test, config):
    if train.dataset != "total_text" or train.split_role != "train" or test.dataset != "total_text" or test.split_role != "test":
        raise ValueError("Official Total-Text TRAIN and TEST roles required")
    if len(train.samples) != config["train_images"] or len(test.samples) != config["test_images"]:
        raise ValueError("Full official population required; no subset")
    a, b = [s.sample_id for s in train.samples], [s.sample_id for s in test.samples]
    if len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) & set(b):
        raise ValueError("Unique disjoint TRAIN/TEST IDs required")
    overlaps=[{"test_id":s.sample_id,"train_ids":[t.sample_id for t in train.samples if t.image_sha256==s.image_sha256],"sha256":s.image_sha256}
              for s in test.samples if s.image_sha256 in {t.image_sha256 for t in train.samples}]
    if overlaps!=config["known_official_rgb_overlap"]:
        raise ValueError("Unknown TRAIN/TEST RGB checksum overlap; inspect and explicitly document before execution")
    return a, b


def validate_epoch(seen, expected):
    if len(seen) != len(expected) or len(set(seen)) != len(seen) or set(seen) != set(expected):
        raise ValueError("Epoch omitted or duplicated TRAIN images")


def ignored_instances(labels, validity):
    if labels.shape!=validity.shape or validity.dtype!=bool:
        raise ValueError("Aligned labels and boolean native validity required")
    return [int(i) for i in np.unique(labels[labels > 0]) if not np.any((labels == i) & validity)]


def run(args):
    cp = args.config.resolve(); c = json.loads(cp.read_text()); device = torch.device(args.device)
    if args.mode != "approximate" or c["reproduction_mode"] != args.mode or REPO/"configs/approximate" not in cp.parents:
        raise ValueError("Explicit approximate configuration required")
    if (c["epochs"] < 1 or c["selected_epoch"] not in c["checkpoint_epochs"] or c["evaluation_epochs"] != [c["selected_epoch"],c["epochs"]]
            or not 0 < c["gap_threshold"] < 1 or c["decoder"] not in ("cut","preserve")):
        raise ValueError("Invalid fixed full-data protocol")
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required; no CPU fallback")
    torch.cuda.init(); torch.set_num_threads(c["cpu_threads"]); cv2.setNumThreads(1)
    seed_everything(c["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    prior=json.loads((args.previous_selection_run/"results.json").read_text());p=prior["protocol"]
    audit=json.loads((args.previous_selection_run/"independent_audit.json").read_text())
    check_hash(args.previous_selection_run/"results.json",audit["result_sha256"])
    if prior["status"]!="SUCCESS" or audit["status"]!="PASS":raise ValueError("Audited prior selection required")
    s=prior["selection"]
    if (s["epoch"],s["threshold"],s["decoder"])!=(c["selected_epoch"],c["gap_threshold"],c["decoder"]):
        raise ValueError("Prior TRAIN-calibration choices must remain fixed")
    for name in ("source_run","word_root","parseq_source","parseq_checkpoint","parseq_archive"):
        if getattr(args,name).resolve()!=Path(p[name]).resolve():raise ValueError("Qualified path differs: "+name)
    if args.source_checkpoint.resolve()!=(args.source_run/"control/best.pt").resolve():raise ValueError("Qualified source checkpoint required")
    expected_prior=Path(p["decoder_run"])/"decoder"/(s["checkpoint_source"]+".pt")
    if args.prior_gap_checkpoint.resolve()!=expected_prior.resolve():raise ValueError("Qualified previous Gap checkpoint required")
    hashes=dict(p["source_hashes"])
    for path,h in hashes.items():check_hash(Path(path),h)
    check_hash(args.test_manifest,c["test_manifest_sha256"])
    paths=[cp,Path(__file__),args.previous_selection_run/"results.json",args.previous_selection_run/"independent_audit.json",
           args.source_checkpoint,args.prior_gap_checkpoint,args.train_manifest,args.test_manifest]
    for path in paths:hashes[str(path.resolve())]=sha256_file(path)
    for path,h in hashes.items():check_hash(Path(path),h)
    contract=json.loads((args.source_run/"control/run_contract.json").read_text());sc=contract["resolved_config"]
    if args.train_manifest.resolve()!=(args.word_root/sc["train_manifest"]).resolve():raise ValueError("Qualified v4 TRAIN manifest required")
    train=load_segmentation_manifest(args.train_manifest,require_checksums=True)
    test=load_segmentation_manifest(args.test_manifest,require_checksums=True)
    fit,test_ids=validate_population(train,test,c)
    out=args.output_root.resolve()
    if any(out==x.resolve() or x.resolve() in out.parents for x in (REPO,args.previous_selection_run,args.source_run,args.word_root,args.test_root)):
        raise ValueError("New external output root required")
    out.mkdir(parents=True,exist_ok=False);args._output_created=True
    protocol={"mode":"approximate","result_class":"approximate reproduction","resolved_config":c,"source_hashes":hashes,
        "train_ids":fit,"test_ids":test_ids,"training_population":"all official TRAIN1255; no internal holdout",
        "test_used_for_selection":False,"test_historically_explored":True,"selected_epoch_fixed_before_test":c["selected_epoch"],
        **{name:str(getattr(args,name).resolve()) for name in ("previous_selection_run","source_run","source_checkpoint","prior_gap_checkpoint","word_root","train_manifest","test_root","test_manifest","parseq_source","parseq_checkpoint","parseq_archive")},
        "device":str(device),"gpu":torch.cuda.get_device_name(device),"dtype":"float32","torch":str(torch.__version__),"cuda":torch.version.cuda,
        "deterministic_algorithms":True,"tf32":False,"seed":c["seed"],"resolved_preprocessing":contract["resolved_preprocessing"],"source_postprocessing":sc["postprocessing"]}
    write_json(out/"protocol.json",protocol)
    transform=_transform(contract["resolved_preprocessing"],"letterbox",train=True)
    dataset=AdaSpotGapDataset(SegmentationDataset(train,args.word_root,transform=transform),**sc["targets"])
    # Check all TEST RGB/checksums now, without reading TEST transcripts.
    for sample in test.samples:check_hash(args.test_root/sample.image_path,sample.image_sha256)
    source=AdaSpotGapAGUNet("control",skip_floor=sc["skip_floor"],initial_gap_bias=c["initial_bias"])
    payload=torch.load(args.source_checkpoint,map_location="cpu",weights_only=False)
    if payload["run_contract"]!=contract:raise ValueError("Source checkpoint contract differs")
    source.load_state_dict(payload["model_state_dict"],strict=True);del payload
    model=TrainableGapDecoder(source,initial_bias=c["initial_bias"]).to(device)
    frozen_digest=state_digest(source)
    optimizer=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=c["learning_rate"],weight_decay=c["weight_decay"])
    generator=torch.Generator().manual_seed(c["seed"]);history=[];global_step=0
    directory=out/"decoder";directory.mkdir();torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, c["epochs"]+1):
        started = time.monotonic(); transform.set_epoch(epoch-1); model.train()
        loader = DataLoader(Subset(dataset, list(range(len(dataset)))), batch_size=c["image_batch_size"], shuffle=True,
                            generator=generator, num_workers=c["num_workers"], timeout=60 if c["num_workers"] else 0)
        totals, seen, steps = Counter(), [], 0
        for batch in loader:
            local = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            evidence = model.forward_evidence(local["image"])
            if evidence["gap"].device != device: raise RuntimeError("Training left requested CUDA device")
            loss, terms = decoder_loss(evidence["gap"], local["gap_target"], local["mask"], local["validity"], evidence["words"][:, :1].sigmoid(), c["loss"])
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite decoder loss")
            if global_step == 0:
                with torch.no_grad(): reference = model.source.forward_evidence(local["image"])["words"]
                teacher_error = float((reference-evidence["words"]).abs().max())
                if teacher_error != 0: raise RuntimeError("Qualified Text/Kernel forward changed")
            if local["validity"].any():
                before = model.gap.weight.detach().clone() if global_step == 1 else None
                loss.backward()
                if any(p.grad is not None for p in model.source.parameters()): raise RuntimeError("Frozen source received gradients")
                gradient = sum(float(p.grad.detach().abs().sum()) for p in model.decoder.parameters() if p.grad is not None) if global_step == 1 else None
                optimizer.step(); global_step += 1; steps += 1
                if before is not None:
                    change = float((model.gap.weight.detach()-before).abs().max())
                    if not gradient > 0 or not change > 0: raise RuntimeError("Decoder/head did not learn on CUDA")
                    write_json(directory/"first_cuda_step.json", {"status": "PASS", "result_class": "smoke-test result", "device": str(device),
                        "decoder_gradient_l1": gradient, "head_weight_update_linf": change, "teacher_max_error": teacher_error})
            for key, value in {"loss": loss, **terms}.items(): totals[key] += float(value.detach())*len(batch["sample_id"])
            seen.extend(batch["sample_id"])
            if steps == 1 or steps % 50 == 0: print(f"[Gap decoder] epoch={epoch} batch={steps}/{len(loader)} loss={float(loss.detach()):.5f}", flush=True)
        validate_epoch(seen,fit)
        record = {"epoch": epoch, "samples": len(seen), "sample_ids": seen, "cuda_steps": steps, "skipped_all_invalid_batches": len(loader)-steps,
                  "sample_order_sha256": sha256("".join(s+"\n" for s in seen).encode()).hexdigest(),
                  **{k: v/len(seen) for k, v in totals.items()}}
        if epoch in c["checkpoint_epochs"]:
            path=directory/"checkpoints"/f"{epoch:03d}.pt";path.parent.mkdir(exist_ok=True)
            torch.save({"branch":model.branch_state(),"epoch":epoch,"global_step":global_step},path)
        record["seconds"] = time.monotonic()-started
        record["memory"] = memory(device, "frozen ResNet50/Text/Kernel + trainable Gap attention decoder/AdamW", [c["image_batch_size"], 3, 512, 512])
        history.append(record); write_json(directory/"epochs"/f"{epoch:03d}.json", record)
        torch.save({"branch": model.branch_state(), "optimizer": optimizer.state_dict(), "epoch": epoch, "global_step": global_step,
                    "generator_state": generator.get_state(), "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device)}, directory/"last.pt")
        print(f"[Full TRAIN] epoch={epoch}/{c['epochs']} images={len(seen)} loss={record['loss']:.5f} CUDAsteps={steps} seconds={record['seconds']:.1f}",flush=True)
    if state_digest(source)!=frozen_digest:raise RuntimeError("Frozen Text/Kernel weights or BN changed")
    selected_checkpoint=directory/"checkpoints"/f"{c['selected_epoch']:03d}.pt"
    complete={"status":"SUCCESS","result_class":"approximate reproduction","completed_epochs":c["epochs"],"history":history,
        "global_cuda_steps":global_step,"train_images":len(fit),"trainable_parameters":sum(x.numel() for x in model.parameters() if x.requires_grad),
        "source_state_sha256_before_after":frozen_digest,"selection":"fixed from preceding calibration; no current validation or TEST selection",
        "selected_epoch":c["selected_epoch"],"selected_checkpoint_sha256":sha256_file(selected_checkpoint),"last_checkpoint_sha256":sha256_file(directory/"last.pt"),
        "saved_epoch_sha256":{str(e):sha256_file(directory/"checkpoints"/f"{e:03d}.pt") for e in c["checkpoint_epochs"]}}
    write_json(directory/"training_complete.json",complete)
    # Release optimizer/last loss graph before loading any recognition model.
    del optimizer,loss,terms,evidence,local,batch,reference,before
    gc.collect();torch.cuda.empty_cache()
    evaluated=evaluate_test(model,test,contract,c,protocol,args,device,out)
    del model,source,dataset;gc.collect();torch.cuda.empty_cache()
    frozen=evaluated.pop("_frozen");jobs=evaluated.pop("_jobs");readings={}
    evaluated["ocr_memory"]=recognize(jobs,readings,protocol,c,device,out,"test")
    arms={name:{"ocr":score(samples,readings)} for name,samples in frozen.items()}
    for name,arm in arms.items():
        if arm["ocr"]["aggregate"]["gt_words"]!=c["test_care_words"] or len(arm["ocr"]["per_image"])!=len(test_ids):
            raise ValueError("Full TEST denominator/population changed")
        arm["changes_vs_baseline"]=paired_changes(arms["baseline"]["ocr"],arm["ocr"])
    evaluated["arms"]=arms
    duplicate_ids={r["test_id"] for r in c["known_official_rgb_overlap"]}
    hash_disjoint={name:score([r for r in rows if r["sample_id"] not in duplicate_ids],readings) for name,rows in frozen.items()}
    evaluated["hash_disjoint_test_subset"]={"images":len(test_ids)-len(duplicate_ids),"excluded_exact_duplicate_test_ids":sorted(duplicate_ids),
        "arms":{name:{"ocr":r,"changes_vs_baseline":paired_changes(hash_disjoint["baseline"],r)} for name,r in hash_disjoint.items()}}
    for path,h in hashes.items():check_hash(Path(path),h)
    write_json(out/"results.json",{"status":"SUCCESS","result_class":"approximate reproduction","protocol":protocol,"training":complete,**evaluated})
    lines=["# Full TRAIN1255 Gap — approximate reproduction","","All1255official TRAIN images used each epoch, including former calibration/evaluation. Prior983word results are not a held-out evaluation of this refit.",
           "TEST300 was historically explored in this repository. All choices fixed before this run. Epoch1 is preselected; final10 is diagnostic, not TEST-selected. Bbox/casefold protocol is not official polygon scoring or Table5.","",
           "| Arm | RGB localized | Exact /2204 | Proposals | OCR Hmean | Recovered / damaged vs baseline |","| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name,arm in evaluated["arms"].items():
        a,ch=arm["ocr"]["aggregate"],arm["changes_vs_baseline"]
        lines.append(f"| {name} | {a['geometry_iou_ge_50']} | {a['exact_casefold_iou_ge_50']} | {a['proposals']} | {a['hmean']:.4%} | {ch['recovered']} / {ch['damaged']} |")
    lines.extend(["","Official TEST has2exact RGB duplicates in TRAIN (TEST578=TRAIN212, TEST656=TRAIN81). Primary table retains all300; secondary298below excludes these two explicitly. No independent TEST300 claim.","",
        "| Hash-disjoint TEST298 arm | Care words | Localized | Exact | Proposals | OCR Hmean |","| --- | ---: | ---: | ---: | ---: | ---: |"])
    for name,arm in evaluated["hash_disjoint_test_subset"]["arms"].items():
        a=arm["ocr"]["aggregate"];lines.append(f"| {name} | {a['gt_words']} | {a['geometry_iou_ge_50']} | {a['exact_casefold_iou_ge_50']} | {a['proposals']} | {a['hmean']:.4%} |")
    (out/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({name:arm['ocr']['aggregate'] for name,arm in evaluated['arms'].items()},indent=2))


def evaluate_test(model,test,contract,c,protocol,args,device,out):
    """Predict all arms before reading each image's validity/transcripts."""
    transform=_transform(contract["resolved_preprocessing"],"letterbox",train=False)
    pp=contract["resolved_config"]["postprocessing"]
    choices={"baseline":{"threshold":None,"decoder":"cut","epoch":0,"role":"frozen M/K control"},
        "prior_gap":{"threshold":c["gap_threshold"],"decoder":c["decoder"],"epoch":c["selected_epoch"],"role":"previous1030fit Gap, fixed prior choice"},
        "full_selected":{"threshold":c["gap_threshold"],"decoder":c["decoder"],"epoch":c["selected_epoch"],"role":"predeclared full-data checkpoint"},
        "full_last":{"threshold":c["gap_threshold"],"decoder":c["decoder"],"epoch":c["epochs"],"role":"final-epoch diagnostic; never TEST-selected"}}
    segmentation_memory={}
    # Sequential branch loading; predictions contain no TEST GT input.
    for arm in ("prior_gap","full_selected","full_last"):
        checkpoint=args.prior_gap_checkpoint if arm=="prior_gap" else out/"decoder/checkpoints"/f"{choices[arm]['epoch']:03d}.pt"
        payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
        recorded=payload["selection"]["epoch"] if arm=="prior_gap" else payload["epoch"]
        if recorded!=choices[arm]["epoch"]:raise ValueError("Loaded checkpoint epoch differs from fixed protocol")
        model.load_branch(payload["branch"]);del payload
        model.eval();torch.cuda.reset_peak_memory_stats(device)
        for i,sample in enumerate(test.samples):
            with Image.open(args.test_root/sample.image_path) as image:rgb=np.asarray(image.convert("RGB"))
            with torch.inference_mode():e=model.forward_evidence(_tensor(rgb,transform).to(device),teacher=arm=="prior_gap")
            folder=out/"test_predictions"/sample.sample_id.replace(":","__");folder.mkdir(parents=True,exist_ok=True)
            if arm=="prior_gap":
                np.savez_compressed(folder/"source.npz",probabilities=e["words"][0].sigmoid().cpu().numpy(),
                    canvas=letterbox_canvas_mask((rgb.shape[1],rgb.shape[0]),native_size=256))
            np.savez_compressed(folder/(arm+".npz"),gap_probability=e["gap"][0,0].sigmoid().cpu().numpy())
            if (i+1)%100==0:print(f"[TEST prediction {arm}] {i+1}/{len(test.samples)}",flush=True)
        segmentation_memory[arm]=memory(device,"frozen ResNet50/Text/Kernel + Gap inference",[1,3,512,512])
    del model,e;gc.collect();torch.cuda.empty_cache()
    # model parameter references are released by the caller before PARSeq below.
    frozen={name:[] for name in choices};jobs={};metadata=[];ignored_by_arm=Counter()
    for i,sample in enumerate(test.samples):
        sid=sample.sample_id;folder=out/"test_predictions"/sid.replace(":","__")
        with np.load(folder/"source.npz") as z:words,canvas=z["probabilities"],z["canvas"]
        with Image.open(args.test_root/sample.image_path) as image:rgb=np.asarray(image.convert("RGB"))
        # First freeze raw labels for EVERY arm. Care validity cannot alter decode.
        labels={}
        for arm,choice in choices.items():
            if arm=="baseline":gap=None
            else:
                with np.load(folder/(arm+".npz")) as z:gap=z["gap_probability"]
            labels[arm]=decode_gap(words,gap,canvas,pp,choice["threshold"],choice["decoder"])
            np.savez_compressed(folder/(arm+"_labels.npz"),labels=labels[arm])
        # Identical native validity transform; used only for declared ignore rule.
        validity_path=args.test_root/sample.validity_annotation_path
        check_hash(validity_path,sample.validity_annotation_sha256)
        with Image.open(validity_path) as image:
            _,_,validity=transform.transform_with_validity(Image.fromarray(rgb),Image.new("L",(rgb.shape[1],rgb.shape[0])),image.convert("L"),i)
        valid=validity[0].numpy().astype(bool)
        np.savez_compressed(folder/"validity.npz",validity=valid)
        current=[]
        for arm,raster in labels.items():
            ignored=ignored_instances(raster,valid)
            ignored_by_arm[arm]+=len(ignored);predictions=[]
            for p in parseq_word_proposals(raster,(rgb.shape[1],rgb.shape[0]),padding=0,min_area=0,max_instances=None):
                if p["instance_id"] in ignored:continue
                x0,y0,x1,y1=p["bbox"];crop=np.ascontiguousarray(rgb[y0:y1,x0:x1]);h=crop_array_hash(crop)
                path=out/"crops"/(h+".png");path.parent.mkdir(parents=True,exist_ok=True)
                if h not in jobs:Image.fromarray(crop).save(path);jobs[h]=path
                predictions.append({**p,"crop_rgb_sha256":h})
            current.append((arm,{"sample_id":sid,"predictions":predictions,"ignored_instance_ids":ignored}))
        gt,annotation_sha=_ground_truth(sample,args.test_root,(rgb.shape[1],rgb.shape[0]))
        annotation=_annotation_path(sample,args.test_root)
        metadata.append({"sample_id":sid,"image_sha256":sample.image_sha256,"annotation_path":str(annotation),"annotation_sha256":annotation_sha,
            "prediction_sha256":{str(path):sha256_file(path) for path in sorted(folder.glob('*.npz'))}})
        for arm,row in current:frozen[arm].append({**row,"ground_truth":gt})
        if (i+1)%100==0:print(f"[TEST frozen crops] {i+1}/{len(test.samples)} unique={len(jobs)}",flush=True)
    write_json(out/"test_inputs.json",frozen);write_json(out/"test_artifact_manifest.json",metadata)
    # All raw neural predictions and crops above are fixed. Drop CUDA branch owners.
    return {"_frozen":frozen,"_jobs":jobs,"choices":choices,"segmentation_memory":segmentation_memory,"ignored_predictions":dict(ignored_by_arm)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=["approximate"],required=True)
    for name in ("config","previous-selection-run","source-run","source-checkpoint","prior-gap-checkpoint","word-root","train-manifest","test-root","test-manifest","parseq-source","parseq-checkpoint","parseq-archive","output-root"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",required=True);args=p.parse_args()
    try:run(args)
    except Exception as error:
        if getattr(args,"_output_created",False):
            failure={"status":"FAILED","error":str(error),"traceback":traceback.format_exc(),"oom":isinstance(error,torch.cuda.OutOfMemoryError)}
            if torch.cuda.is_available():failure.update(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(args.output_root/"failure.json",failure)
        raise
