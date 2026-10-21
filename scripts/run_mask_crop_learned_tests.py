"""Isolated split/merge selection and recognition-supervised crop refinement.

Fit on frozen V3 TRAIN IDs, select only on the existing TRAIN holdout, then
freeze both choices before loading TEST annotations. All results APPROXIMATE.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from localize.mask_crop_hypotheses import candidate_features, hypotheses, select_groups
from scripts.analyze_native128_failures import box_matrices
from scripts.evaluate_totaltext_v1_findcontours_parseq import _gt
from scripts.run_mask_crop_budget import (REPO, CropOCR, aggregate, config, measure,
                                         read, setup, utc, write)
from segmentation.config import load_segmentation_config
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.evaluate import _model_from_config, _preprocess, validate_checkpoint
from segmentation.recognition_crop_refiner import (RecognitionCropRefiner, box_loss,
                                                   frozen_parseq_loss, sample_boxes)


def state(root, phase, **extra):
    write(root/"status.json", {"status":"RUNNING", "phase":phase, "pid":os.getpid(),
                              "claim_label":"approximate reproduction", "utc":utc(), **extra})
    print(f"[{phase}] "+json.dumps(extra), flush=True)


def memory(models, batch, shape, *, deterministic=True):
    return {"models":models, "device":"cuda:0", "precision":"float32", "batch_size":batch,
            "input_shape":shape,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(), "oom":False,
            "deterministic_algorithms":deterministic, "tf32":False}


def load_rgb(record):
    if file_sha256(Path(record["image_path"])) != record["image_sha256"]:
        raise ValueError("Image changed after proposal export")
    with Image.open(record["image_path"]) as im:
        return np.ascontiguousarray(im.convert("RGB"))


def load_probability(record):
    if file_sha256(Path(record["probability_path"])) != record["probability_sha256"]:
        raise ValueError("Probability export changed")
    with np.load(record["probability_path"], allow_pickle=False) as item:
        return item["probability"]


def export(c, root, split, ids, samples, data_root, model):
    """GT-free generation; exported binary masks must replay the old TEST masks."""
    records = {}
    for index, sid in enumerate(ids, 1):
        sample = samples[sid]; image_path = Path(data_root)/sample.image_path
        if file_sha256(image_path) != sample.image_sha256: raise ValueError("Manifest RGB differs")
        with Image.open(image_path) as opened: rgb_image = opened.convert("RGB")
        rgb = np.ascontiguousarray(rgb_image)
        with torch.inference_mode():
            logits = model(_preprocess(rgb_image).unsqueeze(0).cuda())
            if tuple(logits.shape) != (1,1,128,128) or not torch.isfinite(logits).all():
                raise ValueError("Invalid frozen U-Net output")
            probability = logits.sigmoid()[0,0].cpu().numpy()
        raw, pool, groups = hypotheses(probability, rgb, c["hypotheses"])
        mask_sha = sha256((probability >= .5).astype(np.uint8).tobytes()).hexdigest()
        if split == "test":
            old = read(Path(c["paths"]["source_ocr"])/"epoch-0447/proposals"/(sid.replace(":","__")+".json"))
            if mask_sha != old["native_mask_array_sha256"]:
                raise ValueError("Frozen V3 TEST mask did not replay: "+sid)
            fields = ("bbox_native","bbox","padded_bbox_original","contour_index")
            if [[p[k] for k in fields] for p in raw[:10]] != [[p[k] for k in fields] for p in old["variants"]["1"]["proposals"]]:
                raise ValueError("Frozen TEST baseline boxes did not replay")
        directory = root/"exports"/split; directory.mkdir(parents=True,exist_ok=True)
        npz = directory/(sid.replace(":","__")+".npz")
        np.savez_compressed(npz, probability=probability)
        record = {"sample_id":sid, "image_path":str(image_path), "image_sha256":sample.image_sha256,
                  "image_hw":list(rgb.shape[:2]), "probability_path":str(npz), "probability_sha256":file_sha256(npz),
                  "mask_array_sha256":mask_sha,"baseline":raw[:10],"pool":pool,"groups":groups,
                  "raw_component_count":len(raw)}
        write(directory/(sid.replace(":","__")+".json"), record)
        records[sid] = record
        if index%50 == 0 or index==len(ids):
            state(root,"EXPORT_"+split.upper(),images=index,total=len(ids))
    return records


def load_model(c):
    contract = read(c["paths"]["training_contract"])
    source_path = REPO/contract["config"]["source_config"]
    if file_sha256(source_path) != contract["source_config_sha256"]:
        raise ValueError("Original scientific segmentation config changed")
    source = load_segmentation_config(source_path)
    model = _model_from_config(source).eval()
    validate_checkpoint(Path(c["paths"]["checkpoint"]),model,c["sha256"]["checkpoint"],source.architecture.identity)
    model.requires_grad_(False)
    return model.cuda()


def recognize(records, ocr, root, split):
    for index, record in enumerate(records.values(),1):
        ocr.image(load_rgb(record),record["pool"])
        # Pool baseline objects were copied during generation: bind their exact crop IDs.
        by_id = {r["id"]:r for r in record["pool"]}
        record["baseline"] = [by_id[r["id"]] for r in record["baseline"]]
        write(root/"exports"/split/(record["sample_id"].replace(":","__")+".json"),record)
        if index%50 == 0 or index==len(records):
            state(root,"OCR_"+split.upper(),images=index,total=len(records),new_ocr=ocr.inferred)


def ground_truth(records, root, split):
    gt, sources = {}, {}
    for sid, record in records.items():
        image = Path(record["image_path"])
        path = image.parent.parent/"annotations"/(sid.partition(":")[2]+".txt")
        sources[str(path)] = file_sha256(path)
        gt[sid] = _gt(path,(record["image_hw"][1],record["image_hw"][0]))
    write(root/(split+"_ground_truth.json"),{"ground_truth":gt,"annotation_sha256":sources,
          "policy":"original care words; unchanged case/punctuation; no prediction ignore neutralization"})
    return gt


def baseline_metrics(records, gt, ocr):
    rows = [{"sample_id":sid,**measure(r["baseline"],gt[sid],ocr.rows)} for sid,r in records.items()]
    return {"aggregate":aggregate(rows),"per_image":rows}


def training_matrix(records, gt, ocr):
    features, labels = [], []
    for sid, record in records.items():
        p = load_probability(record)
        iou,_,_ = box_matrices(record["pool"],gt[sid])
        for i, row in enumerate(record["pool"]):
            text = ocr.rows[row["crop_array_sha256"]]["raw_text"].casefold()
            correct = any(iou[i,j]>=.5 and text==w["text"].casefold() for j,w in enumerate(gt[sid]))
            features.append(candidate_features(row,p,ocr.rows[row["crop_array_sha256"]]))
            labels.append(float(correct))
    x,y = torch.tensor(features,dtype=torch.float32),torch.tensor(labels,dtype=torch.float32)
    if not len(x) or not 0 < y.sum() < len(y):
        raise ValueError("Selector training requires nonempty positive and negative examples")
    return x,y


def fit_selector(records, gt, ocr, c, root, smoke):
    x,y = training_matrix(records,gt,ocr)
    mean,scale=x.mean(0),x.std(0).clamp_min(.01);z=(x-mean)/scale
    model=nn.Sequential(nn.Linear(x.shape[1],32),nn.ReLU(),nn.Linear(32,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=c["selector"]["learning_rate"],weight_decay=c["selector"]["weight_decay"])
    history=[]
    for epoch in range(1,(3 if smoke else c["selector"]["epochs"])+1):
        optimizer.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(model(z)[:,0],y)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite correctness selector loss")
        loss.backward();optimizer.step();history.append({"epoch":epoch,"loss":float(loss.detach())})
    checkpoint={"state_dict":model.state_dict(),"mean":mean,"scale":scale,"features":x.shape[1],
                "training_candidates":len(x),"positive_candidates":int(y.sum()),"history":history,
                "selection":"fixed final TRAIN epoch; no TEST/holdout optimizer input"}
    torch.save(checkpoint,root/"selector.pt")
    write(root/"selector_training.json",{k:v for k,v in checkpoint.items() if k not in ("state_dict","mean","scale")})
    return model.eval(),mean,scale


def score_records(records, ocr, fitted):
    model,mean,scale=fitted;scores={}
    for sid,record in records.items():
        p=load_probability(record)
        x=torch.tensor([candidate_features(r,p,ocr.rows[r["crop_array_sha256"]]) for r in record["pool"]],dtype=torch.float32)
        with torch.inference_mode():
            values=model((x-mean)/scale)[:,0].sigmoid().tolist() if len(x) else []
        scores[sid]={r["id"]:v for r,v in zip(record["pool"],values)}
    return scores


def selected_metrics(records,gt,ocr,scores,policy):
    rows=[];choices={}
    for sid,record in records.items():
        if policy["identity"]: selected,chosen=record["baseline"],[]
        else:
            selected,chosen=select_groups(record["baseline"],record["pool"],record["groups"],scores[sid],
                                           cost=policy["cost"],margin=policy["margin"])
        rows.append({"sample_id":sid,**measure(selected,gt[sid],ocr.rows)})
        choices[sid]={"selected_ids":[r["id"] for r in selected],"replacements":chosen}
    result=aggregate(rows);result["replacements"]=sum(len(r["replacements"]) for r in choices.values())
    return {"aggregate":result,"per_image":rows,"choices":choices,"policy":policy}


def select_policy(records,gt,ocr,scores,c,root):
    policies=[{"identity":True,"cost":None,"margin":None}]+[
        {"identity":False,"cost":cost,"margin":margin} for cost in c["selector"]["costs"] for margin in c["selector"]["margins"]]
    reports=[selected_metrics(records,gt,ocr,scores,p) for p in policies]
    # Earliest identity wins an exact F1/replacement tie.
    winner=max(range(len(reports)),key=lambda i:(reports[i]["aggregate"]["f1"],-reports[i]["aggregate"]["replacements"],-i))
    write(root/"selector_calibration.json",{"selected_index":winner,"policies":reports,
          "selection_split":"existing TRAIN holdout125; all GT/predictions count"})
    return policies[winner]


def context_geometry(crop, image_hw, factor):
    h,w=image_hw;x0,y0,x1,y1=crop
    cx,cy=(x0+x1)/2,(y0+y1)/2
    ww,hh=max(4,x1-x0)*factor,max(4,y1-y0)*factor
    return [max(0,int(np.floor(cx-ww/2))),max(0,int(np.floor(cy-hh/2))),
            min(w,int(np.ceil(cx+ww/2))),min(h,int(np.ceil(cy+hh/2)))]


def normalized_box(box, context):
    context=np.asarray(context,np.float32);scale=context[2:]-context[:2]
    return (np.asarray(box,np.float32)-np.tile(context[:2],2))/np.tile(scale,2)


def pixel_box(box, context, image_hw):
    context=np.asarray(context,np.float64);scale=np.tile(context[2:]-context[:2],2)
    value=np.asarray(box,np.float64)*scale+np.tile(context[:2],2)
    # Float32 normalization of integer edges must preserve the exact identity crop.
    value=np.where(np.abs(value-np.rint(value))<1e-3,np.rint(value),value)
    result=np.r_[np.floor(value[:2]),np.ceil(value[2:])].astype(int)
    result[[0,2]]=np.clip(result[[0,2]],0,image_hw[1]);result[[1,3]]=np.clip(result[[1,3]],0,image_hw[0])
    if np.any(result[2:]<=result[:2]):raise ValueError("Refiner produced an empty RGB crop")
    return result.tolist()


def contexts(records, examples, c):
    """RGB and probability support only; GT never enters context construction."""
    height,width=c["refiner"]["context_hw"]
    output=np.empty((len(examples),4,height,width),np.float32)
    by_image={}
    for i,e in enumerate(examples):by_image.setdefault(e["sample_id"],[]).append(i)
    for sid,indexes in by_image.items():
        record=records[sid];rgb=load_rgb(record);p=load_probability(record)
        full=cv2.resize(p,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
        for i in indexes:
            x0,y0,x1,y1=examples[i]["context"]
            output[i,:3]=cv2.resize(rgb[y0:y1,x0:x1],(width,height),interpolation=cv2.INTER_LINEAR).transpose(2,0,1).astype(np.float32)/127.5-1
            output[i,3]=cv2.resize(full[y0:y1,x0:x1],(width,height),interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(output)


def inference_examples(records,c):
    examples=[]
    for sid,record in records.items():
        for row in record["baseline"]:
            ctx=context_geometry(row["padded_bbox_original"],record["image_hw"],c["refiner"]["context_scale"])
            examples.append({"sample_id":sid,"proposal_id":row["id"],"context":ctx,
                             "base":normalized_box(row["padded_bbox_original"],ctx).tolist()})
    return examples


def training_examples(records,gt,c,system,smoke):
    base={(r["sample_id"],r["proposal_id"]):r for r in inference_examples(records,c)}
    examples=[];excluded=Counter()
    for sid,record in records.items():
        iou,_,_=box_matrices(record["baseline"],gt[sid])
        for i,row in enumerate(record["baseline"]):
            if not len(gt[sid]) or float(iou[i].max()) < c["refiner"]["minimum_training_iou"]:
                excluded["low_iou_or_no_care_gt"]+=1;continue
            j=int(iou[i].argmax());word=gt[sid][j];text=word["text"]
            if not text or len(text)>system.model.max_label_length:
                excluded["empty_or_excess_length"]+=1;continue
            if any(ch not in system.tokenizer._stoi for ch in text):
                excluded["unsupported_training_charset"]+=1;continue
            item=copy.deepcopy(base[sid,row["id"]]);ctx=item["context"]
            truth=normalized_box(word["bbox"],ctx)
            neighbors=[normalized_box(w["bbox"],ctx) for k,w in enumerate(gt[sid]) if k!=j]
            neighbors=[b for b in neighbors if min(b[2],1)>max(b[0],0) and min(b[3],1)>max(b[1],0)]
            if len(neighbors)>c["refiner"]["maximum_neighbors"]:
                excluded["too_many_neighbors"]+=1;continue
            item.update(text=text,truth=truth.tolist(),neighbors=[b.tolist() for b in neighbors],gt_index=j)
            examples.append(item)
    maximum=min(32,c["refiner"]["maximum_training_crops"]) if smoke else c["refiner"]["maximum_training_crops"]
    available=len(examples)
    order=np.random.default_rng(c["seed"]).permutation(available)[:maximum]
    selected=[examples[i] for i in order]
    if not selected:raise ValueError("No eligible TRAIN examples for the crop refiner")
    return selected,{"eligible":available,"selected":len(selected),"excluded":dict(excluded),
                     "subsampling":"seeded uniform eligible TRAIN proposals; evaluation denominators unchanged"}


def tensor_state_hash(model):
    digest=sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def refined_metrics(records,gt,ocr,model,examples,inputs,c):
    predicted=[];size=c["refiner"]["batch_size"]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(examples),size):
            batch=examples[start:start+size]
            boxes=torch.tensor([r["base"] for r in batch],dtype=torch.float32,device="cuda")
            predicted.extend(model(inputs[start:start+size].cuda(),boxes).cpu().tolist())
    by_image={sid:[] for sid in records}
    lookup={(sid,p["id"]):p for sid,r in records.items() for p in r["baseline"]}
    for item,box in zip(examples,predicted):
        sid=item["sample_id"];row=copy.deepcopy(lookup[sid,item["proposal_id"]])
        row["source_crop_array_sha256"]=row.pop("crop_array_sha256")
        row["source_padded_bbox_native"]=row.pop("padded_bbox_native")
        row["source_padded_bbox_original"]=row["padded_bbox_original"]
        row["padded_bbox_original"]=pixel_box(box,item["context"],records[sid]["image_hw"])
        # Detection bbox stays fixed: measure only the change in readable RGB.
        by_image[sid].append(row)
    rows=[]
    for sid,proposals in by_image.items():
        ocr.image(load_rgb(records[sid]),proposals)
        rows.append({"sample_id":sid,**measure(proposals,gt[sid],ocr.rows)})
    return {"aggregate":aggregate(rows),"per_image":rows,"proposals":by_image}


def gains(report,baseline):
    ref={r["sample_id"]:r for r in baseline["per_image"]};new=lost=0
    for row in report["per_image"]:
        a=set(ref[row["sample_id"]]["correct_gt_indices"]);b=set(row["correct_gt_indices"])
        new+=len(b-a);lost+=len(a-b)
    report["aggregate"].update(gained=new,lost=lost)
    return report


def fit_refiner(train,val,train_gt,val_gt,ocr,c,root,smoke):
    state(root,"REFINER_PREPARE")
    system=ocr.recognizer().model.eval().requires_grad_(False)
    before=tensor_state_hash(system)
    examples,eligibility=training_examples(train,train_gt,c,system,smoke)
    write(root/"refiner_training_examples.json",{"examples":examples,**eligibility})
    inputs=contexts(train,examples,c)
    val_examples=inference_examples(val,c);val_inputs=contexts(val,val_examples,c)
    np.savez_compressed(root/"refiner_training_contexts.npz",contexts=inputs.numpy())
    geometry=torch.tensor([r["base"] for r in examples],dtype=torch.float32)
    truth=torch.tensor([r["truth"] for r in examples],dtype=torch.float32)
    maximum=c["refiner"]["maximum_neighbors"]
    neighbors=torch.zeros(len(examples),maximum,4);valid=torch.zeros(len(examples),maximum,dtype=torch.bool)
    for i,item in enumerate(examples):
        n=len(item["neighbors"])
        if n: neighbors[i,:n]=torch.tensor(item["neighbors"]);valid[i,:n]=True
    model=RecognitionCropRefiner(translation=c["refiner"]["translation"],log_scale=c["refiner"]["log_scale"]).cuda()
    baseline=baseline_metrics(val,val_gt,ocr)
    identity=refined_metrics(val,val_gt,ocr,model,val_examples,val_inputs,c)
    if identity["aggregate"]!=baseline["aggregate"] or any(
        p["crop_array_sha256"]!=p["source_crop_array_sha256"] for rows in identity["proposals"].values() for p in rows):
        raise AssertionError("Identity refiner failed to replay original RGB crops and OCR")
    write(root/"refiner_identity_smoke.json",{"status":"PASS","aggregate":identity["aggregate"],
          "rgb_crop_hash_replays":sum(map(len,identity["proposals"].values()))})
    optimizer=torch.optim.AdamW(model.parameters(),lr=c["refiner"]["learning_rate"],weight_decay=c["refiner"]["weight_decay"])
    best=-1.;best_epoch=None;history=[];step_count=0
    size=c["refiner"]["batch_size"];epochs=1 if smoke else c["refiner"]["epochs"]
    generator=np.random.default_rng(c["seed"])
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,epochs+1):
        state(root,"REFINER_TRAIN",epoch=epoch,epochs=epochs,training_crops=len(examples))
        # PyTorch CUDA grid_sample backward is explicitly nondeterministic.
        torch.use_deterministic_algorithms(False)
        model.train();order=generator.permutation(len(examples));totals=Counter()
        for start in range(0,len(examples),size):
            indexes=order[start:start+size];idx=torch.from_numpy(indexes)
            batch=inputs[idx].cuda();base=geometry[idx].cuda()
            predicted=model(batch,base)
            rgb=sample_boxes(batch[:,:3],predicted)
            recognition=frozen_parseq_loss(system,rgb,[examples[i]["text"] for i in indexes])
            coverage,contamination,regression=box_loss(predicted,truth[idx].cuda(),neighbors[idx].cuda(),valid[idx].cuda())
            t=c["refiner"]
            loss=(t["recognition_weight"]*recognition+t["coverage_weight"]*coverage+
                  t["contamination_weight"]*contamination+t["regression_weight"]*regression)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite recognition-supervised crop loss")
            if step_count==0:
                gradient=torch.autograd.grad(recognition,predicted,retain_graph=True)[0]
                if not torch.isfinite(gradient).all() or not float(gradient.norm())>0:
                    raise RuntimeError("Frozen PARSeq recognition loss has no finite crop-coordinate gradient")
                write(root/"recognition_gradient_smoke.json",{"status":"PASS",
                      "recognition_to_crop_gradient_norm":float(gradient.norm()),
                      "frozen_recognizer_parameters":sum(p.numel() for p in system.parameters()),
                      "trainable_recognizer_parameters":sum(p.numel() for p in system.parameters() if p.requires_grad),
                      "sampler":"CUDA bilinear; nondeterministic backward explicitly enabled"})
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),t["gradient_clip"],error_if_nonfinite=True)
            if step_count==0 and not float(norm)>0:raise RuntimeError("PARSeq/geometry loss did not reach crop-refiner weights")
            optimizer.step();step_count+=1
            for key,value in [("loss",loss),("recognition",recognition),("coverage",coverage),("contamination",contamination),("regression",regression)]:
                totals[key]+=float(value.detach())*len(indexes)
        torch.use_deterministic_algorithms(True)
        measured=gains(refined_metrics(val,val_gt,ocr,model,val_examples,val_inputs,c),baseline)
        row={"epoch":epoch,"train":{k:v/len(examples) for k,v in totals.items()},"validation":measured["aggregate"]}
        history.append(row);write(root/"refiner_history.json",history)
        write(root/"refiner_validation"/f"epoch-{epoch:03d}.json",measured)
        if measured["aggregate"]["f1"]>best:
            best=measured["aggregate"]["f1"];best_epoch=epoch
            torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"epoch":epoch,
                        "config":c["refiner"],"validation":measured["aggregate"]},root/"refiner_best.pt")
        state(root,"REFINER_VALIDATION",**row)
    payload=torch.load(root/"refiner_best.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(payload["state_dict"],strict=True);model.eval()
    after=tensor_state_hash(system)
    if before!=after or any(p.grad is not None for p in system.parameters()):
        raise AssertionError("Frozen PARSeq weights/gradients changed during crop training")
    acceptance="learned" if best>baseline["aggregate"]["f1"] else "identity"
    report={"selected_learned_epoch":best_epoch,"selected_validation_f1":best,
            "accepted_policy":acceptance,"baseline_validation":baseline["aggregate"],
            "recognizer_tensor_sha256_before":before,"recognizer_tensor_sha256_after":after,
            "optimizer_steps":step_count,"training_eligibility":eligibility,
            "memory":memory(["crop refiner","frozen qualified PARSeq, input gradients"],size,[size,4,*c["refiner"]["context_hw"]],deterministic=False)}
    write(root/"refiner_selection.json",report)
    del inputs,val_inputs,neighbors,truth,geometry,optimizer
    gc.collect();torch.cuda.empty_cache()
    return model,report


def run(args):
    c=config(args.config,args.mode);root=args.output.resolve()
    if root.exists():raise FileExistsError("Fresh external experiment output required")
    if root.is_relative_to(REPO):raise ValueError("Keep experiment rasters/weights outside repository")
    root.mkdir(parents=True);started=time.monotonic();hardware=setup(c)
    contract=read(c["paths"]["training_contract"])
    train_ids,val_ids=list(contract["train_ids"]),list(contract["validation_ids"])
    if len(train_ids)!=1130 or len(val_ids)!=125 or set(train_ids)&set(val_ids):
        raise ValueError("Frozen V3 TRAIN/holdout split differs")
    train_manifest=load_segmentation_manifest(Path(c["paths"]["train_manifest"]),require_checksums=True)
    test_manifest=load_segmentation_manifest(Path(c["paths"]["test_manifest"]),require_checksums=True)
    if train_manifest.split_role!="train" or test_manifest.split_role!="test":raise ValueError("Dataset role mismatch")
    test_ids=[s.sample_id for s in test_manifest.samples]
    if len(test_ids)!=300:raise ValueError("Expected official TEST300")
    if args.smoke:train_ids,val_ids,test_ids=train_ids[:8],val_ids[:6],test_ids[:6]
    samples={s.sample_id:s for s in train_manifest.samples};test_samples={s.sample_id:s for s in test_manifest.samples}
    code=[Path(__file__),REPO/"scripts/run_mask_crop_budget.py",REPO/"localize/mask_crop_hypotheses.py",
          REPO/"segmentation/recognition_crop_refiner.py",REPO/"models/unet_adv.py",REPO/"recognition/parseq.py"]
    source=Path(c["paths"]["source_ocr"])
    write(root/"resolved_config.json",{"config":c,"config_sha256":file_sha256(args.config),"runtime":hardware,
          "reproduction_mode":"approximate","result_class":"smoke-test result" if args.smoke else "approximate reproduction; TRAIN-fitted exploratory TEST300",
          "splits":{"train":train_ids,"validation":val_ids,"test":test_ids},
          "code_sha256":{str(p):file_sha256(p) for p in code},
          "source_ocr_sha256":{str(source/n):file_sha256(source/n) for n in ("protocol.json","ocr.jsonl","ground_truth.json")},
          "metric":"maximum one-to-one unpadded bboxIoU>=.5 + exact Unicode casefold, punctuation retained, all predictions/care GT count",
          "test_selection":False,"no_strict_or_table5_claim":True})
    state(root,"EXPORT_TRAIN")
    model=load_model(c)
    train=export(c,root,"train",train_ids,samples,c["paths"]["train_data_root"],model)
    val=export(c,root,"validation",val_ids,samples,c["paths"]["train_data_root"],model)
    write(root/"segmentation_training_memory.json",memory(["frozen V3/447"],1,[1,3,256,256]))
    del model;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    ocr=CropOCR(c,root/"ocr_new.jsonl",source)
    try:
        recognize(train,ocr,root,"train");recognize(val,ocr,root,"validation")
        write(root/"recognition_training_memory.json",memory(["qualified PARSeq AR+refine1"],c["runtime"]["ocr_batch_size"],[c["runtime"]["ocr_batch_size"],3,32,128]))
        # Only TRAIN annotations are read for any optimizer or calibration choice.
        train_gt=ground_truth(train,root,"train");val_gt=ground_truth(val,root,"validation")
        if not args.smoke and sum(map(len,val_gt.values()))!=983:raise ValueError("Holdout983 care denominator differs")
        state(root,"FIT_SELECTOR")
        fitted=fit_selector(train,train_gt,ocr,c,root,args.smoke)
        val_scores=score_records(val,ocr,fitted)
        policy=select_policy(val,val_gt,ocr,val_scores,c,root)
        write(root/"selector_policy_frozen.json",{"policy":policy,"selected_on":"TRAIN holdout only","test_annotations_loaded":False})
        refiner,selection=fit_refiner(train,val,train_gt,val_gt,ocr,c,root,args.smoke)
        write(root/"choices_frozen_before_test.json",{"selector_policy":policy,"refiner":selection,
              "selector_checkpoint_sha256":file_sha256(root/"selector.pt"),"refiner_checkpoint_sha256":file_sha256(root/"refiner_best.pt"),
              "test_annotations_loaded":False,"utc":utc()})
        ocr.release();torch.cuda.reset_peak_memory_stats()
        model=load_model(c)
        test=export(c,root,"test",test_ids,test_samples,c["paths"]["test_data_root"],model)
        write(root/"segmentation_test_memory.json",memory(["frozen V3/447","crop refiner idle"],1,[1,3,256,256]))
        del model;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        recognize(test,ocr,root,"test")
        test_gt=ground_truth(test,root,"test")
        baseline=baseline_metrics(test,test_gt,ocr)
        if not args.smoke and (baseline["aggregate"]["gt_words"]!=2204 or baseline["aggregate"]["correct"]!=795 or baseline["aggregate"]["proposals"]!=2086):
            raise AssertionError("TEST baseline795/2086/2204 did not replay")
        scores=score_records(test,ocr,fitted)
        selected=gains(selected_metrics(test,test_gt,ocr,scores,policy),baseline)
        pool_rows=[{"sample_id":sid,**measure(r["pool"],test_gt[sid],ocr.rows)} for sid,r in test.items()]
        pool={"aggregate":aggregate(pool_rows),"per_image":pool_rows,"label":"candidate-pool diagnostic only; oracle matching is not an inference selector"}
        write(root/"test2_split_merge.json",{"baseline":baseline,"selected":selected,"pool_diagnostic":pool,"scores":scores})
        examples=inference_examples(test,c);inputs=contexts(test,examples,c)
        refined=gains(refined_metrics(test,test_gt,ocr,refiner,examples,inputs,c),baseline)
        accepted=refined if selection["accepted_policy"]=="learned" else gains(copy.deepcopy(baseline),baseline)
        write(root/"test3_crop_refiner.json",{"baseline":baseline,"learned":refined,"accepted":accepted,"selection":selection})
        used={r["crop_array_sha256"] for record in test.values() for r in record["pool"]}
        used.update(r["crop_array_sha256"] for rows in refined["proposals"].values() for r in rows)
        write(root/"ocr_test_used.json",{k:ocr.rows[k] for k in sorted(used)})
        report={"status":"SUCCESS","claim_label":"approximate reproduction","result_class":"smoke-test result" if args.smoke else "approximate reproduction; exploratory TEST300",
                "images":len(test),"baseline":baseline["aggregate"],"test2_selected":selected["aggregate"],"test2_pool_diagnostic":pool["aggregate"],
                "test3_learned":refined["aggregate"],"test3_accepted":accepted["aggregate"],"selector_policy":policy,
                "refiner_selected_epoch":selection["selected_learned_epoch"],"refiner_accepted_policy":selection["accepted_policy"],
                "elapsed_seconds":time.monotonic()-started,"new_ocr":ocr.inferred,
                "memory_test":memory(["crop refiner","qualified PARSeq AR+refine1"],c["runtime"]["ocr_batch_size"],[c["runtime"]["ocr_batch_size"],3,32,128])}
        write(root/"results.json",report)
        write(root/"status.json",{"status":"SUCCESS","phase":"COMPLETE","utc":utc(),"results":"results.json"})
        print(json.dumps(report),flush=True)
    finally:ocr.release()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",required=True,choices=["approximate"]);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    output_existed=args.output.exists()
    try:run(args)
    except Exception as error:
        ready=torch.cuda.is_initialized()
        if not output_existed:
            write(args.output/"status.json",{"status":"FAILED","phase":"FAILED","error":str(error),"utc":utc(),"traceback":traceback.format_exc(),
                  "oom":isinstance(error,torch.cuda.OutOfMemoryError),"peak_allocated_bytes":torch.cuda.max_memory_allocated() if ready else None,
                  "peak_reserved_bytes":torch.cuda.max_memory_reserved() if ready else None})
        raise
