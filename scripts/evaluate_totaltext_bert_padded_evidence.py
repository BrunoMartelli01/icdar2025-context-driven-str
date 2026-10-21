"""Separate approximate padding experiment; immutable localization, expanded RGB.

No GT enters either inference stage. Cached BERT scores retain exact strings,
contexts and model identity; novel padded-native words get genuine new scores.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import math
from pathlib import Path
import re
import sys
import time

from recognition.bert_correction import BertWordScorer
from recognition.bert_visual_correction import ar_scores,select
from recognition.parseq import ParseqApproxConfig,ParseqApproxRecognizer
from recognition.scene_text_v1_evaluation import evaluate_images
from scripts.evaluate_totaltext_bert_correction import checked_file,digest,matched_gt,payload_digest,read_json,seal_json,shard_path
from scripts.evaluate_totaltext_bert_methods import runtime_common,setup
from scripts.evaluate_totaltext_bert_visual_correction import prepared as prior_prepared,checked_row as prior_row,ceiling


def expanded_box(box,size_wh,fraction,minimum):
    if (not math.isfinite(fraction) or fraction<=0 or type(minimum) is not int or minimum<1
            or len(box)!=4 or len(size_wh)!=2):
        raise ValueError("Invalid explicit padding geometry/size")
    x1,y1,x2,y2=box
    width,height=size_wh
    if not (0<=x1<x2<=width and 0<=y1<y2<=height):
        raise ValueError("Original bbox escapes RGB image")
    p=max(minimum,math.ceil(fraction*(y2-y1)))
    return [max(0,x1-p),max(0,y1-p),min(width,x2+p),min(height,y2+p)],p


def implementations():
    from scripts.evaluate_totaltext_bert_visual_correction import implementations as prior
    return {**prior(),"scripts/evaluate_totaltext_bert_padded_evidence.py":digest(Path(__file__))}


def config(path):
    cfg=read_json(path)
    if (path.resolve().parent.name!="approximate" or path.resolve().parent.parent.name!="configs"
            or cfg["experiment_kind"]!="totaltext_bert_padded_evidence"
            or cfg["reproduction_mode"]!="approximate" or cfg["claim_label"]!="approximate reproduction"):
        raise ValueError("Explicit configs/approximate/ padding experiment required")
    expanded_box([1,1,2,2],[3,3],cfg["padding"]["height_fraction"],cfg["padding"]["minimum_pixels"])
    if len({c["name"] for c in cfg["controls"]})!=len(cfg["controls"]):
        raise ValueError("Duplicate control names")
    return cfg


def prepare(args):
    cfg=config(args.config)
    p,r,inputs=prior_prepared(args.upstream_dir)
    previous=checked_file(args.upstream_dir/"results.json",cfg["upstream"]["results_sha256"])
    if digest(args.upstream_dir/"protocol.json")!=cfg["upstream"]["protocol_sha256"]:
        raise ValueError("Prior visual/BERT protocol differs")
    for key in ("source","model","caption","scorer","context_policy","parseq"):
        if cfg[key]!=p["resolved_config"][key]:
            raise ValueError("Cached BERT/model/source compatibility differs: "+key)
    labels=checked_file(args.upstream_dir/"evaluation_labels.json",r["evaluation_labels_sha256"])
    vsha,bsha=digest(args.upstream_dir/"visual_runtime.json"),digest(args.upstream_dir/"bert_runtime.json")
    observations=[]
    sources={}
    for sample in inputs["samples"]:
        v=prior_row(args.upstream_dir,"visual",sample,r,vsha)
        b=prior_row(args.upstream_dir,"bert",sample,r,bsha)
        if b["visual_sha256"]!=digest(shard_path(args.upstream_dir,"visual",sample["sample_id"])):
            raise ValueError("Cached BERT scores refer to different visual candidates")
        blocks=[]
        for block,vt,bt in zip(sample["blocks"],v["trace"],b["trace"]):
            if vt["candidate_pool"]!=bt["candidate_pool"] or block["insertion_anchor"]!=bt["insertion_anchor"]:
                raise ValueError("Cached BERT exact candidate/reference identity differs")
            blocks.append({**{k:block[k] for k in ("rank","bbox","crop_rgb_sha256","text","insertion_anchor")},
                "status":vt["status"],"candidate_pool":bt["candidate_pool"],"anchor_index":bt["anchor_index"],
                "context":bt["context"],"cached_bert_scores":bt["bert_scores"]})
        observations.append({**sample,"blocks":blocks})
        for stage in ("visual","bert"):
            path=shard_path(args.upstream_dir,stage,sample["sample_id"])
            sources[str(path)]=digest(path)
    protocol={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":cfg["result_class"],"resolved_config":cfg,"config_sha256":digest(args.config),
        "implementation_sha256":implementations(),"upstream_protocol_sha256":r["protocol_sha256"],
        "source_inference_sha256":sources,"input_change":"height-relative padded RGB, original evaluation boxes unchanged"}
    out=args.output_dir
    seal_json(out/"protocol.json",protocol)
    common={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction","protocol_sha256":digest(out/"protocol.json")}
    seal_json(out/"inputs.json",{**common,"samples":observations})
    seal_json(out/"evaluation_labels.json",{**common,"samples":labels["samples"]})
    seal_json(out/"baseline.json",{**common,"metrics":previous["baseline"]})
    seal_json(out/"prepared.json",{**common,**{n+"_sha256":digest(out/(n+".json")) for n in ("inputs","evaluation_labels","baseline")}})
    print("Prepared padded evidence",len(observations),flush=True)


def prepared(out):
    r=read_json(out/"prepared.json")
    p=checked_file(out/"protocol.json",r["protocol_sha256"])
    if p["implementation_sha256"]!=implementations():
        raise ValueError("Padding implementation changed; use a new output version")
    return p,r,checked_file(out/"inputs.json",r["inputs_sha256"])


def checked(out,stage,sample,r,runtime_sha):
    row=read_json(shard_path(out,stage,sample["sample_id"]))
    if (row["payload_sha256"]!=payload_digest({k:v for k,v in row.items() if k!="payload_sha256"})
            or row["sample_id"]!=sample["sample_id"] or row["image_sha256"]!=sample["image_sha256"]
            or row["runtime_sha256"]!=runtime_sha or row["inputs_sha256"]!=r["inputs_sha256"]
            or len(row["trace"])!=len(sample["blocks"])):
        raise ValueError("Padding shard payload/population/provenance differs")
    for a,b in zip(sample["blocks"],row["trace"]):
        if any(a[k]!=b[k] for k in ("rank","bbox","text","crop_rgb_sha256","insertion_anchor","context")):
            raise ValueError("Padding immutable localization/OCR/context differs")
    return row


def save(out,stage,sample,r,runtime_sha,trace,tick,torch,device,**extra):
    row={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "sample_id":sample["sample_id"],"image_sha256":sample["image_sha256"],"inputs_sha256":r["inputs_sha256"],
        "runtime_sha256":runtime_sha,"trace":trace,"seconds":time.perf_counter()-tick,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda:0" else None,
        "peak_reserved_bytes":torch.cuda.max_memory_reserved() if device=="cuda:0" else None,**extra}
    row["payload_sha256"]=payload_digest(row)
    seal_json(shard_path(out,stage,sample["sample_id"]),row)


def visual(args):
    p,r,inputs=prepared(args.output_dir)
    cfg=p["resolved_config"]
    torch=setup(cfg,args.device)
    import numpy as np
    from PIL import Image
    if digest(args.runtime_deps/"einops/einops.py")!=cfg["caption"]["einops_source_sha256"]:
        raise ValueError("Existing einops source differs")
    sys.path.insert(0,str(args.runtime_deps.resolve()))
    import einops
    if einops.__version__!=cfg["caption"]["einops_version"]:
        raise ValueError("Existing einops version differs")
    if (digest(args.parseq_checkpoint)!=cfg["parseq"]["checkpoint_sha256"]
            or digest(args.parseq_archive)!=cfg["parseq"]["source_archive_sha256"]):
        raise ValueError("Qualified PARSeq weights/archive differ")
    model=ParseqApproxRecognizer(ParseqApproxConfig(args.parseq_source,args.parseq_checkpoint,
        device=args.device,runtime_source_type=cfg["parseq"]["source_type"],runtime_source_archive=args.parseq_archive,
        decode_ar=cfg["parseq"]["decode_ar"],refine_iters=cfg["parseq"]["refine_iters"]))
    runtime={**runtime_common(cfg,r,args.device),"model_set":["PARSeq-S"],"model_provenance":model._provenance,
        "input_shape":[1,3,32,128],"score_batch_size":cfg["parseq"]["score_batch_size"],"padding":cfg["padding"],
        "runtime_dependency_path":str(args.runtime_deps.resolve())}
    seal_json(args.output_dir/"visual_runtime.json",runtime)
    runtime_sha=digest(args.output_dir/"visual_runtime.json")
    for number,sample in enumerate(inputs["samples"],1):
        if args.max_images is not None and number>args.max_images:break
        if shard_path(args.output_dir,"visual",sample["sample_id"]).exists():
            checked(args.output_dir,"visual",sample,r,runtime_sha);continue
        if digest(Path(sample["image_path"]))!=sample["image_sha256"]:
            raise ValueError("Frozen RGB image changed")
        with Image.open(sample["image_path"]) as opened:rgb=np.asarray(opened.convert("RGB"))
        tick=time.perf_counter();trace=[]
        if args.device=="cuda:0":torch.cuda.reset_peak_memory_stats()
        for block in sample["blocks"]:
            box,pixels=expanded_box(block["bbox"],[rgb.shape[1],rgb.shape[0]],cfg["padding"]["height_fraction"],cfg["padding"]["minimum_pixels"])
            a,b,c,d=block["bbox"]
            original=np.ascontiguousarray(rgb[b:d,a:c])
            if sha256(np.asarray(original.shape,dtype="<i8").tobytes()+original.tobytes()).hexdigest()!=block["crop_rgb_sha256"]:
                raise ValueError("Original crop differs from frozen RGB identity")
            a,b,c,d=box;crop=np.ascontiguousarray(rgb[b:d,a:c])
            crop_sha=sha256(np.asarray(crop.shape,dtype="<i8").tobytes()+crop.tobytes()).hexdigest()
            pool=[{**c,"sources":list(c["sources"])} for c in block["candidate_pool"]]
            native_index=0;native=None;scores=[]
            if block["status"]=="scored":
                native=model.recognize(crop,input_role="T3_localized_crop").to_dict()
                text=native["raw_text"]
                if re.fullmatch(r"[A-Za-z]+",text) and 2<=len(text)<=cfg["candidates"]["maximum_length"]:
                    indices=[i for i,x in enumerate(pool) if x["text"].casefold()==text.casefold()]
                    if indices:native_index=indices[0]
                    else:
                        native_index=len(pool)
                        from rapidfuzz.distance import Levenshtein
                        pool.append({"text":text,"sources":["padded_native_PARSeq"],"edit_distance":Levenshtein.distance(block["text"].casefold(),text.casefold())})
                tensor=model.transform(Image.fromarray(crop)).unsqueeze(0).to(args.device)
                with torch.inference_mode():
                    memory=model.model.model.encode(tensor)
                    scores=ar_scores(model.model.model,model.model.tokenizer,memory,[x["text"] for x in pool],cfg["parseq"]["score_batch_size"])
            trace.append({**block,"padded_bbox":box,"padding_pixels":pixels,"padded_crop_rgb_sha256":crop_sha,
                "candidate_pool":pool,"native":native,"native_index":native_index,"visual_scores":scores})
        if args.device=="cuda:0":torch.cuda.synchronize()
        save(args.output_dir,"visual",sample,r,runtime_sha,trace,tick,torch,args.device)
        if number%25==0 or number==len(inputs["samples"]) or args.max_images==number:print("Padded visual",number,"/",len(inputs["samples"]),flush=True)


def decisions(block,scores,cfg):
    output={}
    for control in cfg["controls"]:
        anchor=block["anchor_index"] if control.get("anchor")=="insertion" else 0
        if control.get("native"):
            d={"selected_index":block["native_index"],"reason":"padded_native_control","gain":0.0}
        elif control["keep_only"] or not scores:
            d={"selected_index":anchor,"reason":"control/protected_KEEP","gain":0.0}
        else:
            d=select(block["candidate_pool"],block["visual_scores"],scores.get(control["condition"]),anchor_index=anchor,
                visual_margin=control["visual_margin"],beta=control["beta"],minimum_gain=control["minimum_gain"])
        output[control["name"]]={**d,"text":block["candidate_pool"][d["selected_index"]]["text"]}
    return output


def bert(args):
    p,r,inputs=prepared(args.output_dir);cfg=p["resolved_config"]
    torch=setup(cfg,args.device)
    from transformers import AutoTokenizer,AutoModelForMaskedLM
    spec=cfg["scorer"]
    root=args.hf_cache/("models--"+spec["model_id"].replace("/","--"))/"snapshots"/spec["revision"]
    for name,value in spec["files_sha256"].items():
        if digest(root/name)!=value:raise ValueError("Pinned BERT file differs: "+name)
    tok=AutoTokenizer.from_pretrained(root,local_files_only=True,use_fast=True)
    model,audit=AutoModelForMaskedLM.from_pretrained(root,local_files_only=True,use_safetensors=True,
        torch_dtype=torch.float32,output_loading_info=True)
    if (audit["missing_keys"] or audit["mismatched_keys"] or audit["error_msgs"]
            or set(audit["unexpected_keys"])-set(spec["allowed_unused_checkpoint_keys"])):
        raise RuntimeError(f"BERT checkpoint incompatibility: {audit}")
    model.to(args.device).eval()
    shapes=set()
    def hook(module,args,kwargs):shapes.add(tuple(kwargs["input_ids"].shape))
    handle=model.register_forward_pre_hook(hook,with_kwargs=True)
    vsha=digest(args.output_dir/"visual_runtime.json")
    runtime={**runtime_common(cfg,r,args.device),"model_set":[spec["model_id"]],"model_files_sha256":spec["files_sha256"],
        "loading_audit":audit,"visual_runtime_sha256":vsha,"score_batch_size":spec["batch_size"],
        "cache_rule":"exact candidate text/context/model identity; new padded-native candidates freshly scored"}
    seal_json(args.output_dir/"bert_runtime.json",runtime);runtime_sha=digest(args.output_dir/"bert_runtime.json")
    scorer=BertWordScorer(tok,model,device=args.device,batch_size=spec["batch_size"],max_tokens=spec["max_input_tokens"])
    for number,sample in enumerate(inputs["samples"],1):
        if args.max_images is not None and number>args.max_images:break
        if shard_path(args.output_dir,"bert",sample["sample_id"]).exists():
            checked(args.output_dir,"bert",sample,r,runtime_sha);continue
        v=checked(args.output_dir,"visual",sample,r,vsha)
        tick=time.perf_counter();trace=[];shapes.clear()
        if args.device=="cuda:0":torch.cuda.reset_peak_memory_stats()
        for original,block in zip(sample["blocks"],v["trace"]):
            n=len(original["candidate_pool"])
            if block["candidate_pool"][:n]!=original["candidate_pool"]:
                raise ValueError("Cached exact-string candidate prefix changed")
            scores={condition:list(values) for condition,values in original["cached_bert_scores"].items()}
            extra=[x["text"] for x in block["candidate_pool"][n:]]
            freshly_scored=len(extra)
            if block["status"]=="scored" and len(block["candidate_pool"])>1 and not scores:
                # No old likelihood exists for a singleton. Score the actual
                # reference together with every new candidate; never invent it.
                ctx=original["context"]
                texts=[c["text"] for c in block["candidate_pool"]]
                freshly_scored=len(texts)
                for condition,cap in (("local_ocr",None),("local_ocr_caption",ctx["caption"])):
                    scores[condition]=scorer.score(texts,left=ctx["left"],right=ctx["right"],caption=cap)
            elif extra:
                ctx=original["context"]
                for condition,cap in (("local_ocr",None),("local_ocr_caption",ctx["caption"])):
                    scores[condition].extend(scorer.score(extra,left=ctx["left"],right=ctx["right"],caption=cap))
            trace.append({**block,"bert_scores":scores,"newly_scored_candidates":freshly_scored,"decisions":decisions(block,scores,cfg)})
        if args.device=="cuda:0":torch.cuda.synchronize()
        save(args.output_dir,"bert",sample,r,runtime_sha,trace,tick,torch,args.device,
            visual_sha256=digest(shard_path(args.output_dir,"visual",sample["sample_id"])),
            observed_input_shapes=[list(s) for s in sorted(shapes)])
        if number%25==0 or number==len(inputs["samples"]) or args.max_images==number:print("Padded BERT",number,"/",len(inputs["samples"]),flush=True)
    handle.remove()


def evaluate(args):
    p,r,inputs=prepared(args.output_dir);cfg=p["resolved_config"]
    gt=checked_file(args.output_dir/"evaluation_labels.json",r["evaluation_labels_sha256"])["samples"]
    baseline=checked_file(args.output_dir/"baseline.json",r["baseline_sha256"])["metrics"]
    vsha,bsha=digest(args.output_dir/"visual_runtime.json"),digest(args.output_dir/"bert_runtime.json")
    if read_json(args.output_dir/"bert_runtime.json")["visual_runtime_sha256"]!=vsha:raise ValueError("BERT/visual runtime link differs")
    rows={c["name"]:[] for c in cfg["controls"]};details={k:[] for k in rows};oracle=0
    new_scores=0;files={}
    for sample,label in zip(inputs["samples"],gt):
        if sample["sample_id"]!=label["sample_id"]:raise ValueError("Label population/order differs")
        v=checked(args.output_dir,"visual",sample,r,vsha);b=checked(args.output_dir,"bert",sample,r,bsha)
        if b["visual_sha256"]!=digest(shard_path(args.output_dir,"visual",sample["sample_id"])):raise ValueError("BERT visual shard link differs")
        for original,vt,bt in zip(sample["blocks"],v["trace"],b["trace"]):
            if (bt["candidate_pool"]!=vt["candidate_pool"] or bt["visual_scores"]!=vt["visual_scores"]
                    or bt["decisions"]!=decisions(bt,bt["bert_scores"],cfg)):
                raise ValueError("Padded control/candidate/score trace differs")
            for c,values in original["cached_bert_scores"].items():
                if bt["bert_scores"][c][:len(values)]!=values:raise ValueError("Cached exact BERT scores changed")
            new_scores+=bt["newly_scored_candidates"]
        before=matched_gt(sample["blocks"],label["ground_truth"])
        for name in rows:
            predictions=[{"rank":t["rank"],"bbox":t["bbox"],"text":t["decisions"][name]["text"]} for t in b["trace"]]
            after=matched_gt(predictions,label["ground_truth"])
            rows[name].append({"ground_truth":label["ground_truth"],"predictions":predictions})
            details[name].append({"sample_id":sample["sample_id"],"ground_truth":label["ground_truth"],"predictions":predictions,
                "recovered_gt_indices":sorted(after-before),"damaged_gt_indices":sorted(before-after)})
        oracle+=ceiling(sample["blocks"],label["ground_truth"],[t["candidate_pool"] for t in b["trace"]])
        for stage in ("visual","bert"):
            path=shard_path(args.output_dir,stage,sample["sample_id"]);files[str(path)]=digest(path)
    controls={}
    for name,row in rows.items():
        metrics=evaluate_images(row,text_available=True)
        rec=sum(len(d["recovered_gt_indices"]) for d in details[name]);damage=sum(len(d["damaged_gt_indices"]) for d in details[name])
        if (metrics["geometry"]!=baseline["geometry"] or metrics["recognition"]["exact_casefold_spatial_one_to_one"]
                -baseline["recognition"]["exact_casefold_spatial_one_to_one"]!=rec-damage):
            raise ValueError("Padded accounting or frozen geometry differs")
        controls[name]={"metrics":metrics,"recovered":rec,"damaged":damage,"net":rec-damage,"per_image":details[name]}
        print(name,"exact",metrics["recognition"]["exact_casefold_spatial_one_to_one"],"recovered",rec,"damaged",damage,flush=True)
    seal_json(args.output_dir/"results.json",{"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":cfg["result_class"],"protocol_sha256":r["protocol_sha256"],"baseline":baseline,"controls":controls,
        "diagnostic_GT_oracle":{"label":"GT-dependent upper bound; never achieved recognition","count":oracle},
        "inference_files_sha256":files,"newly_scored_candidates":new_scores,"failed_samples":0,"excluded_samples":0,
        "test_loaded":False,"training_performed":False,"independent_validation":False})
    print("Padded pool GT oracle",oracle,"new candidate scores",new_scores,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest="stage",required=True)
    a=subs.add_parser("prepare")
    for n in ("config","upstream-dir","output-dir"):a.add_argument("--"+n,type=Path,required=True)
    a=subs.add_parser("visual")
    for n in ("output-dir","parseq-source","parseq-checkpoint","parseq-archive","runtime-deps"):a.add_argument("--"+n,type=Path,required=True)
    a.add_argument("--device",choices=("cpu","cuda:0"),required=True);a.add_argument("--max-images",type=int)
    a=subs.add_parser("bert")
    for n in ("output-dir","hf-cache"):a.add_argument("--"+n,type=Path,required=True)
    a.add_argument("--device",choices=("cpu","cuda:0"),required=True);a.add_argument("--max-images",type=int)
    a=subs.add_parser("evaluate");a.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    try:
        if getattr(args,"max_images",None) is not None and args.max_images<1:raise ValueError("max-images must be positive")
        {"prepare":prepare,"visual":visual,"bert":bert,"evaluate":evaluate}[args.stage](args)
    except Exception as error:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/("error-"+args.stage+".log")).write_text(type(error).__name__+": "+str(error)+"\n")
        raise


if __name__=="__main__":main()
