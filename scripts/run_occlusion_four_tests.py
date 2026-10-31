"""Four APPROXIMATE component experiments; frozen segmentation/PARSeq, TRAIN-only selection.

Run from an immutable source snapshot. Artifacts are external, restartable and
hash-bound. These tests do not execute or claim the full context/DeepSolo pipeline.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn

from localize.occlusion_four_tests import (instance_pool, mser_pools, features, ocr_features,
    expand_box, row, same_reading_nms, correction_choice, select_groups, decode_instance_labels, bind_baseline)
from models.adaspot_gap import AdaSpotGapAGUNet
from models.adaspot_gap_decoder import TrainableGapDecoder
from scripts.run_mask_crop_budget import CropOCR, measure, aggregate, read, write, utc, setup
from scripts.analyze_native128_failures import box_matrices
from scripts.evaluate_totaltext_v1_findcontours_parseq import _gt
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _tensor
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.training import seed_everything

REPO=Path(__file__).resolve().parents[1]
BRANCHES=("mk","gap")
RGB_BRANCHES=("mser","ss_mser_inspired")


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def status(root, phase, **values):
    write(root/"status.json",dict(status="RUNNING",phase=phase,pid=os.getpid(),utc=utc(),**values))
    print(phase,json.dumps(values),flush=True)


def memory(label, batch, shape):
    torch.cuda.synchronize()
    return dict(models=label,precision="float32",device="cuda:0",batch_size=batch,input_shape=shape,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),oom=False)


def path_for(root,phase,sid):return root/"exports"/phase/(sid.replace(":","__")+".json")


def load_rgb(record):
    path=Path(record["image_path"])
    if file_sha256(path)!=record["image_sha256"]:raise ValueError("RGB identity changed")
    with Image.open(path) as image:return np.ascontiguousarray(image.convert("RGB"))


def checked_config(path,mode):
    path=path.resolve();c=read(path)
    if mode!="approximate" or REPO/"configs/approximate" not in path.parents or c["reproduction_mode"]!=mode:
        raise ValueError("Explicit isolated APPROXIMATE config required")
    if c["schema_version"]!=1 or c["experiment_kind"]!="occlusion_four_tests_v1":raise ValueError("Unknown contract")
    if c["metric_iou_threshold"]!=0.5:raise ValueError("Reused qualified metric requires explicit IoU0.5")
    for key,value in c["input_sha256"].items():
        if file_sha256(Path(c["paths"][key]))!=value:raise ValueError("Source changed: "+key)
    for name,value in c["bert"]["files_sha256"].items():
        if file_sha256(Path(c["paths"]["bert_snapshot"])/name)!=value:raise ValueError("BERT artifact changed: "+name)
    return c


def prepare(root,c,smoke):
    source=read(c["paths"]["source_contract"]);splits=read(c["paths"]["split_protocol"])
    train=load_segmentation_manifest(Path(c["paths"]["train_manifest"]),require_checksums=True)
    test=load_segmentation_manifest(Path(c["paths"]["test_manifest"]),require_checksums=True)
    train_by={s.sample_id:s for s in train.samples};test_by={s.sample_id:s for s in test.samples}
    ids={"fit":splits["fit_ids"],"calibration":splits["calibration_ids"],"evaluation":splits["evaluation_ids"],"test":list(test_by)}
    if [len(ids[k]) for k in ("fit","calibration","evaluation","test")]!=[1030,100,125,300]:raise ValueError("Population changed")
    if any(set(ids[a])&set(ids[b]) for a,b in (("fit","calibration"),("fit","evaluation"),("calibration","evaluation"))):
        raise ValueError("Leaked selector splits")
    if set(ids["fit"]+ids["calibration"]+ids["evaluation"])!=set(train_by):raise ValueError("TRAIN population drift")
    if smoke:ids={k:v[:c["smoke_images"][k]] for k,v in ids.items()}
    index={}
    for phase,sample_ids in ids.items():
        by=train_by if phase!="test" else test_by
        data_root=Path(c["paths"]["word_root"] if phase!="test" else c["paths"]["test_root"])
        index[phase]=[]
        for sid in sample_ids:
            s=by[sid];image_path=data_root/s.image_path
            index[phase].append(dict(sample_id=sid,image_path=str(image_path),image_sha256=s.image_sha256,
                annotation_path=str(image_path.parent.parent/"annotations"/(sid.split(":")[1]+".txt"))))
    # Fail on unknown pixel-identical TRAIN/TEST overlap. Known duplicates stay explicit.
    known=c["known_duplicate_test_ids"]
    overlap=[s.sample_id for s in test.samples if s.image_sha256 in {q.image_sha256 for q in train.samples}]
    if sorted(overlap)!=sorted(known):raise ValueError("Unexpected RGB overlap")
    code={str(p.relative_to(REPO)):file_sha256(p) for folder in c["code_roots"] for p in sorted((REPO/folder).rglob('*.py'))}
    protocol=dict(schema_version=1,reproduction_mode="approximate",claim_label="approximate reproduction",
        result_class="smoke-test result" if smoke else "exploratory approximate reproduction",
        resolved_config=c,config_sha256=digest(c),source_code_sha256=code,index=index,
        selection_uses_test=False,test_historically_explored=True,
        validation_note="125 evaluation images excluded from Gap fit; prior checkpoint selection used this historical cohort; not independent validation",
        component_scope="localized PARSeq ablations; not a complete BLIP2/MPNet/DeepSolo routing evaluation",
        metric="micro exact Unicode casefold retaining punctuation AND original unpadded bbox IoU>=0.5, maximum one-to-one; all care GT and all predictions including empty/duplicate readings; no prediction neutralization on ignored GT",
        runtime=setup(c),python=platform.python_version(),opencv=cv2.__version__,
        packages={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy','scipy','transformers','tokenizers','Pillow')})
    if (root/"protocol.json").exists():
        old=read(root/"protocol.json")
        for key in ("config_sha256","source_code_sha256","index","result_class"):
            if old[key]!=protocol[key]:raise ValueError("Cannot resume changed contract: "+key)
    else:
        root.mkdir(parents=True,exist_ok=True);write(root/"protocol.json",protocol);write(root/"resolved_config.json",c)
    return source,index


def export(root,c,contract,index):
    pending=[(phase,r) for phase,records in index.items() for r in records if not path_for(root,phase,r["sample_id"]).exists()]
    if not pending:return
    source=AdaSpotGapAGUNet("control",skip_floor=contract["resolved_config"]["skip_floor"],initial_gap_bias=c["initial_gap_bias"])
    checkpoint=torch.load(c["paths"]["source_checkpoint"],map_location="cpu",weights_only=False)
    if checkpoint["run_contract"]!=contract:raise ValueError("Source checkpoint/contract differ")
    source.load_state_dict(checkpoint["model_state_dict"],strict=True);del checkpoint
    model=TrainableGapDecoder(source,initial_bias=c["initial_gap_bias"])
    payload=torch.load(c["paths"]["gap_checkpoint"],map_location="cpu",weights_only=True)
    if payload["selection"]["epoch"]!=c["gap_epoch"]:raise ValueError("Unexpected preselected Gap epoch")
    model.load_branch(payload["branch"]);del payload
    model.requires_grad_(False);model.cuda().eval()
    transform=_transform(contract["resolved_preprocessing"],"letterbox",train=False)
    pp=contract["resolved_config"]["postprocessing"]
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    for number,(phase,entry) in enumerate(pending,1):
        rgb=load_rgb(entry)
        with torch.inference_mode():e=model.forward_evidence(_tensor(rgb,transform).cuda())
        probabilities=e["words"][0].sigmoid().cpu().numpy();gap=e["gap"][0,0].sigmoid().cpu().numpy()
        canvas=letterbox_canvas_mask((rgb.shape[1],rgb.shape[0]),native_size=probabilities.shape[1])
        record={**entry,"image_hw":list(rgb.shape[:2]),"arms":{},"features":{}}
        for arm in BRANCHES:
            labels=decode_instance_labels(probabilities,gap if arm=="gap" else None,canvas,pp,
                              c["gap_threshold"] if arm=="gap" else None,"preserve" if arm=="gap" else "cut")
            record["arms"][arm]=instance_pool(probabilities[0],labels,rgb,c["hypotheses"])
        for name,values in mser_pools(rgb,c["mser"]).items():
            record["arms"][name]={"baseline":values,"pool":values,"groups":[]}
        for arm,item in record["arms"].items():
            record["features"][arm]={r["id"]:features(r,rgb,None if arm in RGB_BRANCHES else probabilities[0],text_threshold=pp["text_threshold"]) for r in item["pool"]}
        views=[]
        for base in record["arms"]["mk"]["baseline"]:
            current=[];seen=set()
            for fraction in c["correction"]["expansions"]:
                box=expand_box(base["bbox"],rgb.shape[:2],fraction)
                if tuple(box) in seen:continue
                seen.add(tuple(box));current.append(row(box,rgb.shape[:2],f"view:{base['id']}:{fraction}","view"))
            views.append({"base_id":base["id"],"bbox":base["bbox"],"views":current})
        record["correction"]=views
        target=path_for(root,phase,entry["sample_id"]);target.parent.mkdir(parents=True,exist_ok=True)
        maps=target.with_suffix('.npz');np.savez_compressed(maps,probabilities=probabilities,gap=gap,canvas=canvas)
        record.update(maps_path=str(maps),maps_sha256=file_sha256(maps))
        write(target,record)
        if number%50==0 or number==len(pending):status(root,"EXPORT",images=number,total=len(pending))
    write(root/"segmentation_memory.json",{**memory("frozen M/K + previous1030fit Gap",1,[1,3,512,512]),"seconds":time.perf_counter()-tick})
    del source,model,e;gc.collect();torch.cuda.empty_cache()


def load_records(root,index):
    output={}
    for phase,entries in index.items():
        output[phase]={r["sample_id"]:read(path_for(root,phase,r["sample_id"])) for r in entries}
        for r in output[phase].values():
            if file_sha256(Path(r["maps_path"]))!=r["maps_sha256"]:raise ValueError("Probability artifact drift")
    return output


def ocr_all(root,c,records):
    cache=CropOCR(c,root/("ocr_"+str(len(list(root.glob('ocr_*.jsonl'))))+".jsonl"))
    for source in sorted(root.glob('ocr_*.jsonl')):
        for line in source.open():
            r=json.loads(line);key=r["crop_array_sha256"]
            if key in cache.rows and cache.rows[key]!=r:raise ValueError("OCR cache collision")
            cache.rows[key]=r
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    for phase,items in records.items():
        for number,r in enumerate(items.values(),1):
            rgb=load_rgb(r)
            for arm in r["arms"].values():
                cache.image(rgb,arm["pool"]);bind_baseline(arm)
            for trace in r["correction"]:cache.image(rgb,trace["views"])
            r["ocr_complete"]=True;write(path_for(root,phase,r["sample_id"]),r)
            if number%100==0 or number==len(items):status(root,"OCR_"+phase,images=number,total=len(items),new=cache.inferred)
    write(root/"ocr_memory.json",{**memory("qualified PARSeq AR/refine1",c["runtime"]["ocr_batch_size"],[c["runtime"]["ocr_batch_size"],3,32,128]),
                                   "seconds":time.perf_counter()-tick,"new_crops":cache.inferred,"cache_hits":cache.reused})
    cache.release();return cache


def context_all(root,c,records,cache):
    if (root/"context_complete.json").exists():return
    from transformers import AutoTokenizer,AutoModelForMaskedLM
    from recognition.bert_correction import BertWordScorer
    snapshot=Path(c["paths"]["bert_snapshot"])
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,use_fast=True)
    model,audit=AutoModelForMaskedLM.from_pretrained(snapshot,local_files_only=True,use_safetensors=True,
        torch_dtype=torch.float32,output_loading_info=True)
    if audit["missing_keys"] or audit["mismatched_keys"] or audit["error_msgs"] or set(audit["unexpected_keys"])-set(c["bert"]["allowed_unused_checkpoint_keys"]):
        raise RuntimeError("BERT weight loading audit failed: "+str(audit))
    model.cuda().eval();scorer=BertWordScorer(tokenizer,model,device="cuda:0",batch_size=c["bert"]["batch_size"],max_tokens=c["bert"]["max_input_tokens"])
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();count=0
    for phase,items in records.items():
        if phase=="fit":continue  # BERT does not fit a model or policy here.
        for number,r in enumerate(items.values(),1):
            if r.get("context_complete"):continue
            ordered=sorted(r["correction"],key=lambda v:((v["bbox"][1]+v["bbox"][3])/2,v["bbox"][0]))
            original=[cache.rows[t["views"][0]["crop_array_sha256"]]["raw_text"] for t in ordered]
            for i,trace in enumerate(ordered):
                candidates=sorted({cache.rows[v["crop_array_sha256"]]["raw_text"] for v in trace["views"]})
                if len(candidates)<2:trace["context"]={};continue
                candidates=[s for s in candidates if s and len(s)<=c["correction"]["maximum_context_word_length"]]
                if len(candidates)<2:trace["context"]={};continue
                left=original[max(0,i-c["correction"]["context_neighbors"]):i]
                right=original[i+1:i+1+c["correction"]["context_neighbors"]]
                values=scorer.score(candidates,left=left,right=right,caption=None)
                trace["context"]={s:v["mean_log_probability"] for s,v in zip(candidates,values)};count+=len(candidates)
            r["context_complete"]=True;write(path_for(root,phase,r["sample_id"]),r)
            if number%100==0 or number==len(items):status(root,"CONTEXT_"+phase,images=number,total=len(items))
    write(root/"context_complete.json",{**memory("pinned BERT local OCR context",c["bert"]["batch_size"],[c["bert"]["batch_size"],scorer.max_input_tokens_seen]),
          "seconds":time.perf_counter()-tick,"scored_candidates":count,"loading_audit":audit})
    del scorer,model;gc.collect();torch.cuda.empty_cache()


def labels_for(records):
    gt={};hashes={}
    for sid,r in records.items():
        path=Path(r["annotation_path"]);hashes[str(path)]=file_sha256(path)
        gt[sid]=_gt(path,tuple(r["image_hw"])[::-1])
    return gt,hashes


def correctness(pool,gt,readings,threshold):
    overlaps,_,_=box_matrices(pool,gt)
    return [float(any(overlaps[i,j]>=threshold and readings[r["crop_array_sha256"]]["raw_text"].casefold()==g["text"].casefold()
                      for j,g in enumerate(gt))) for i,r in enumerate(pool)]


def fit_classifier(records,gt,cache,c,arms,post,path):
    xs=[];ys=[]
    for sid,r in records.items():
        for arm in arms:
            pool=r["arms"][arm]["pool"]
            xs.extend(r["features"][arm][p["id"]]+(ocr_features(cache.rows[p["crop_array_sha256"]]) if post else []) for p in pool)
            ys.extend(correctness(pool,gt[sid],cache.rows,c["metric_iou_threshold"]))
    x=torch.tensor(xs,dtype=torch.float32);y=torch.tensor(ys,dtype=torch.float32)
    if not len(x) or not 0<y.sum()<len(y):raise ValueError("Classifier requires real positive/negative examples")
    mean=x.mean(0);scale=x.std(0).clamp_min(c["selector"]["minimum_feature_scale"]);z=(x-mean)/scale
    seed_everything(c["seed"],deterministic_algorithms=True)
    model=nn.Sequential(nn.Linear(x.shape[1],c["selector"]["hidden"]),nn.ReLU(),nn.Linear(c["selector"]["hidden"],1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=c["selector"]["learning_rate"],weight_decay=c["selector"]["weight_decay"])
    history=[]
    for epoch in range(c["selector"]["epochs"]):
        optimizer.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(model(z)[:,0],y)
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite selector loss")
        loss.backward();optimizer.step();history.append(float(loss.detach()))
    torch.save(dict(state_dict=model.state_dict(),mean=mean,scale=scale,post_ocr=post,arms=arms,features=x.shape[1]),path)
    write(path.with_suffix('.json'),dict(training_candidates=len(x),positives=int(y.sum()),loss=history,post_ocr=post,
        note="Frozen final epoch; no calibration/evaluation/TEST optimizer inputs"))
    return model.eval(),mean,scale


def score_record(r,arm,cache,fitted,post):
    model,mean,scale=fitted;pool=r["arms"][arm]["pool"]
    if not pool:return {}
    x=torch.tensor([r["features"][arm][p["id"]]+(ocr_features(cache.rows[p["crop_array_sha256"]]) if post else []) for p in pool],dtype=torch.float32)
    with torch.inference_mode():values=model((x-mean)/scale)[:,0].sigmoid().tolist()
    return dict(zip((p["id"] for p in pool),values))


def selected(r,arm,scores,policy,available=None):
    item=r["arms"][arm]
    base=item["baseline"];pool=item["pool"];groups=item["groups"]
    if available is not None:
        base=[p for p in base if p["id"] in available];pool=[p for p in pool if p["id"] in available]
        groups=[g for g in groups if set(g["parents"]+g["children"])<=available]
    if policy.get("identity"):return base
    return select_groups(base,pool,groups,scores,cost=policy["cost"],margin=policy["margin"])[0]


def measurements(records,gt,cache,predict):
    rows=[];outputs={}
    for sid,r in records.items():
        predictions=predict(sid,r);outputs[sid]=predictions
        rows.append({"sample_id":sid,**measure(predictions,gt[sid],cache.rows)})
    return dict(aggregate=aggregate(rows),per_image=rows),outputs


def changes(report,base):
    lookup={r["sample_id"]:set(r["correct_gt_indices"]) for r in base["per_image"]}
    return dict(recovered=sum(len(set(r["correct_gt_indices"])-lookup[r["sample_id"]]) for r in report["per_image"]),
                damaged=sum(len(lookup[r["sample_id"]]-set(r["correct_gt_indices"])) for r in report["per_image"]))


def calibrate_selector(records,gt,cache,arm,scores,c):
    policies=[{"identity":True}]+[dict(identity=False,cost=a,margin=b) for a in c["selector"]["costs"] for b in c["selector"]["margins"]]
    grid=[]
    for p in policies:
        report,_=measurements(records,gt,cache,lambda sid,r:selected(r,arm,scores[sid][arm],p))
        grid.append(dict(policy=p,aggregate=report["aggregate"]))
    winner=max(range(len(grid)),key=lambda i:(grid[i]["aggregate"]["f1"],-i))
    return grid[winner]["policy"],grid


def preselect(r,arm,scores,budget):
    # Limit unique unread RGB crops, not output count. Selection has no OCR argument.
    chosen=set();hashes=set()
    for p in sorted(r["arms"][arm]["pool"],key=lambda p:(-scores[p["id"]],p["id"])):
        key=p["crop_array_sha256"]
        if key not in hashes and len(hashes)>=budget:continue
        hashes.add(key);chosen.add(p["id"])
    return chosen


def evaluate(root,c,records,cache):
    fit_gt,fit_hashes=labels_for(records["fit"]);cal_gt,cal_hashes=labels_for(records["calibration"])
    write(root/"train_annotation_hashes.json",{**fit_hashes,**cal_hashes})
    fitted={"post":fit_classifier(records["fit"],fit_gt,cache,c,BRANCHES,True,root/"selector_post.pt"),
            "pre":fit_classifier(records["fit"],fit_gt,cache,c,BRANCHES,False,root/"selector_pre.pt"),
            "rgb":fit_classifier(records["fit"],fit_gt,cache,c,RGB_BRANCHES,True,root/"selector_rgb.pt")}
    scores={}
    for phase,items in records.items():
        if phase=="fit":continue
        scores[phase]={sid:{arm:score_record(r,arm,cache,fitted["post" if arm in BRANCHES else "rgb"],True) for arm in (*BRANCHES,*RGB_BRANCHES)} for sid,r in items.items()}
    policies={};grid={}
    for arm in BRANCHES:policies[arm],grid[arm]=calibrate_selector(records["calibration"],cal_gt,cache,arm,scores["calibration"],c)
    admissions={}
    for arm in RGB_BRANCHES:
        trials=[]
        for threshold in c["selector"]["admission_thresholds"]:
            def predict(sid,r):
                values=[p for p in r["arms"][arm]["pool"] if scores["calibration"][sid][arm][p["id"]]>=threshold]
                return same_reading_nms(values,cache.rows,scores["calibration"][sid][arm],c["nms_iou"])
            report,_=measurements(records["calibration"],cal_gt,cache,predict)
            trials.append(dict(threshold=threshold,aggregate=report["aggregate"]))
        admissions[arm]=max(trials,key=lambda t:(t["aggregate"]["f1"],t["threshold"]))["threshold"];grid[arm]=trials
    correction_policies=[]
    for weight in c["correction"]["context_weights"]:
        trials=[]
        for votes in c["correction"]["minimum_votes"]:
            for gain in c["correction"]["minimum_gains"]:
                p=dict(votes=votes,gain=gain,weight=weight)
                report,_=correction_metrics(records["calibration"],cal_gt,cache,p)
                trials.append(dict(policy=p,aggregate=report["aggregate"],changed_crops=report["changed_crops"]))
        best=max(range(len(trials)),key=lambda i:(trials[i]["aggregate"]["f1"],-trials[i]["changed_crops"],-i))
        correction_policies.append(trials[best]["policy"]);grid["correction_"+str(weight)]=trials
    selection=dict(selector=policies,admission=admissions,correction=correction_policies,
        classification="ASSUMPTION",uses="TRAIN calibration100 only; identity/KEEP are retained controls")
    write(root/"calibration.json",grid);write(root/"selection_frozen_before_evaluation.json",selection)
    frozen_selection_sha=file_sha256(root/"selection_frozen_before_evaluation.json")
    all_reports={};all_outputs={};label_hashes={}
    for phase in ("evaluation","test"):
        gt,hashes=labels_for(records[phase]);label_hashes.update(hashes)
        reports={};outputs={}
        for arm in BRANCHES:
            for name,p in (("baseline",dict(identity=True)),("selected",policies[arm])):
                reports[arm+"_"+name],outputs[arm+"_"+name]=measurements(records[phase],gt,cache,lambda sid,r:selected(r,arm,scores[phase][sid][arm],p))
        for arm in RGB_BRANCHES:
            def standalone(sid,r):
                values=[p for p in r["arms"][arm]["pool"] if scores[phase][sid][arm][p["id"]]>=admissions[arm]]
                return same_reading_nms(values,cache.rows,scores[phase][sid][arm],c["nms_iou"])
            reports[arm],outputs[arm]=measurements(records[phase],gt,cache,standalone)
            def augmented(sid,r):
                added=standalone(sid,r);base=r["arms"]["mk"]["baseline"]
                ranked={**{p["id"]:scores[phase][sid]["mk"][p["id"]] for p in base},**scores[phase][sid][arm]}
                return same_reading_nms([*base,*added],cache.rows,ranked,c["nms_iou"])
            reports[arm+"_augment_mk"],outputs[arm+"_augment_mk"]=measurements(records[phase],gt,cache,augmented)
            reports[arm+"_pool_ceiling"],_=measurements(records[phase],gt,cache,lambda sid,r:r["arms"][arm]["pool"])
            reports[arm+"_pool_ceiling"]["diagnostic_only"]="GT-assisted candidate coverage, not achieved selector accuracy"
        for p in correction_policies:
            name="correction_visual" if not p["weight"] else "correction_context_"+str(p["weight"])
            reports[name],outputs[name]=correction_metrics(records[phase],gt,cache,p)
        for budget in c["budgets"]:
            for arm in BRANCHES:
                pre={sid:score_record(r,arm,cache,fitted["pre"],False) for sid,r in records[phase].items()}
                for method in ("learned","geometry"):
                    def budgeted(sid,r):
                        ranks=pre[sid] if method=="learned" else {p["id"]:(p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]) for p in r["arms"][arm]["pool"]}
                        available=preselect(r,arm,ranks,budget)
                        return selected(r,arm,scores[phase][sid][arm],policies[arm],available)
                    name=f"budget_{arm}_{method}_{budget}"
                    reports[name],outputs[name]=measurements(records[phase],gt,cache,budgeted)
                    requested={sid:sorted(preselect(r,arm,pre[sid] if method=="learned" else
                        {p["id"]:(p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]) for p in r["arms"][arm]["pool"]},budget)) for sid,r in records[phase].items()}
                    reports[name]["ocr_requests_by_image"]=requested
                    reports[name]["requested_unique_crops"]=sum(len({p["crop_array_sha256"] for p in records[phase][sid]["arms"][arm]["pool"] if p["id"] in values}) for sid,values in requested.items())
                    reports[name]["budget_is_pre_ocr"]=True
        for name,report in reports.items():
            base=reports["gap_baseline" if name.startswith("gap_") or name.startswith("budget_gap_") else "mk_baseline"]
            report["changes_vs_baseline"]=changes(report,base)
        # Keep all original care words. Known duplicates are a separately labelled sensitivity only.
        secondary={name:dict(aggregate=aggregate([r for r in report["per_image"] if r["sample_id"] not in c["known_duplicate_test_ids"]]))
                   for name,report in reports.items()} if phase=="test" else {}
        all_reports[phase]=reports;all_outputs[phase]=outputs
        write(root/(phase+"_results.json"),reports);write(root/(phase+"_predictions.json"),outputs)
        write(root/(phase+"_annotation_hashes.json"),hashes)
        if secondary:write(root/"test_duplicate_sensitivity.json",secondary)
        status(root,"EVALUATED_"+phase,images=len(records[phase]),gt_words=reports["mk_baseline"]["aggregate"]["gt_words"])
    if frozen_selection_sha!=file_sha256(root/"selection_frozen_before_evaluation.json"):raise ValueError("Selection mutated")
    write(root/"results.json",dict(status="SUCCESS",reproduction_mode="approximate",claim_label="approximate reproduction",
        result_class=read(root/"protocol.json")["result_class"],selection=selection,selection_sha256=frozen_selection_sha,
        cohorts={phase:{name:report["aggregate"] for name,report in reports.items()} for phase,reports in all_reports.items()}))
    return all_reports,all_outputs


def correction_metrics(records,gt,cache,policy):
    extra={};predictions={};changed=0
    for sid,r in records.items():
        current=[];by_id={p["id"]:p for p in r["arms"]["mk"]["baseline"]}
        for trace in r["correction"]:
            text=correction_choice(trace,cache.rows,policy["votes"],policy["gain"],policy["weight"])
            changed+=text!=cache.rows[trace["views"][0]["crop_array_sha256"]]["raw_text"]
            key="correction:"+digest([sid,trace["base_id"],text]);extra[key]={"raw_text":text}
            current.append({**by_id[trace["base_id"]],"crop_array_sha256":key})
        predictions[sid]=current
    # Evaluation gets explicit corrected text identities; inference never receives GT.
    readings={**cache.rows,**extra};rows=[]
    for sid in records:rows.append({"sample_id":sid,**measure(predictions[sid],gt[sid],readings)})
    # Retain selected strings in the per-prediction artifact, not only opaque keys.
    for values in predictions.values():
        for p in values:p["corrected_text"]=readings[p["crop_array_sha256"]]["raw_text"]
    return dict(aggregate=aggregate(rows),per_image=rows,changed_crops=changed),predictions


def independent_audit(root,reports,outputs,records,cache):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    checks=0;threshold=read(root/"protocol.json")["resolved_config"]["metric_iou_threshold"]
    for phase,arms in reports.items():
        gt,_=labels_for(records[phase])
        for name,report in arms.items():
            if name.endswith("_pool_ceiling"):continue
            for scored in report["per_image"]:
                sid=scored["sample_id"];pred=outputs[phase][name][sid]
                overlaps,_,_=box_matrices(pred,gt[sid]);adj=overlaps>=threshold
                for i,p in enumerate(pred):
                    text=p.get("corrected_text",cache.rows.get(p["crop_array_sha256"],{}).get("raw_text"))
                    if text is None:raise ValueError("Audit lacks prediction text")
                    for j,g in enumerate(gt[sid]):adj[i,j]&=text.casefold()==g["text"].casefold()
                count=int((maximum_bipartite_matching(csr_matrix(adj),perm_type="column")>=0).sum()) if adj.size else 0
                if count!=scored["correct"]:raise ValueError("Independent matching disagrees")
                checks+=1
    write(root/"independent_audit.json",dict(status="PASS",image_arm_checks=checks,matching="SciPy maximum bipartite, independent of primary augment matcher",
        results_sha256=file_sha256(root/"results.json")))


def cold_budget_execution(root,c,reports,records,reference):
    """Execute the preselected TEST crops with an empty cache, measure actual stage cost."""
    measurements_out={}
    for name,report in reports.items():
        if not name.startswith("budget_"):continue
        arm=name.split('_')[1];out=root/"budget_execution"/name
        out.mkdir(parents=True,exist_ok=True)
        if (out/"receipt.json").exists():measurements_out[name]=read(out/"receipt.json");continue
        cache=CropOCR(c,out/"ocr.jsonl");torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();unique=0
        for sid,r in records.items():
            requested=set(report["ocr_requests_by_image"][sid])
            values=[p for p in r["arms"][arm]["pool"] if p["id"] in requested]
            cache.image(load_rgb(r),values);unique+=len({p["crop_array_sha256"] for p in values})
            for p in values:
                key=p["crop_array_sha256"]
                if cache.rows[key]["raw_text"]!=reference.rows[key]["raw_text"]:raise RuntimeError("Cold budget OCR text disagrees with replay")
        result={**memory("qualified PARSeq, cold preselected crops",c["runtime"]["ocr_batch_size"],[c["runtime"]["ocr_batch_size"],3,32,128]),
                "seconds_including_load":time.perf_counter()-tick,"requested_unique_crops_by_image":unique,"inferred_unique_crops":cache.inferred,
                "preselection_cpu_included_in_time":False,"all_readings_replayed":True}
        write(out/"receipt.json",result);measurements_out[name]=result;cache.release();status(root,"COLD_BUDGET",arm=name)
    write(root/"budget_execution.json",measurements_out)


def report_markdown(root):
    d=read(root/"results.json");rows=d["cohorts"]["test"]
    lines=["# Four occlusion tests — approximate reproduction","",
        "Exploratory component ablations, not independent validation, official polygon scoring or paper Table 5. No full DeepSolo routing evaluation.","",
        "| Arm | Correct | Outputs | Precision | Recall | F1 |","| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name,a in rows.items():
        lines.append(f"| {name} | {a['correct']} | {a['proposals']} | {a['precision']:.2%} | {a['recall']:.2%} | {a['f1']:.2%} |")
    lines.extend(["","Pool ceilings are GT-assisted diagnostics, not achieved inference results. All policy choices were frozen on TRAIN calibration before evaluation/TEST labels were loaded.",
        "Known pixel-identical TRAIN/TEST images are retained in primary results and separately excluded in test_duplicate_sensitivity.json. Historical checkpoint/parameter selection prevents independent generalization claims.",
        "Budget comparisons enforce unique pre-OCR crop counts; cold execution receipts include loading and exclude segmentation/proposal/preselection time. Do not interpret them as end-to-end latency."])
    (root/"comparison.md").write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['approximate']);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    try:
        c=checked_config(args.config,args.mode);contract,index=prepare(root,c,args.smoke)
        export(root,c,contract,index);records=load_records(root,index);cache=ocr_all(root,c,records)
        context_all(root,c,records,cache)
        reports,outputs=evaluate(root,c,records,cache)
        independent_audit(root,reports,outputs,records,cache)
        cold_budget_execution(root,c,reports['test'],records['test'],cache)
        report_markdown(root)
        for key,value in c['input_sha256'].items():
            if file_sha256(Path(c['paths'][key]))!=value:raise ValueError('Input changed during run: '+key)
        write(root/'status.json',dict(status='SUCCESS',utc=utc(),results_sha256=file_sha256(root/'results.json')))
        print('SUCCESS',str(root/'comparison.md'),flush=True)
    except Exception as error:
        root.mkdir(parents=True,exist_ok=True)
        write(root/('failure_'+str(time.time_ns())+'.json'),dict(status='FAILED',error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),oom=isinstance(error,torch.cuda.OutOfMemoryError),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None))
        write(root/'status.json',dict(status='FAILED',error=str(error),utc=utc()));raise


if __name__=='__main__':main()
