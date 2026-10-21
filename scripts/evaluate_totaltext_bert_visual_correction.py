"""Six frozen TRAIN-development controls for approximate PARSeq/BERT correction.

Prepare isolates evaluation labels. Visual/bert stages read observable inputs only.
Candidate oracles are produced exclusively by evaluate, never used for selection.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import math
from pathlib import Path
import sys
import time

from recognition.bert_correction import BertWordScorer,local_context
from recognition.bert_visual_correction import CandidatePolicy,append_beam,ar_beam,ar_scores,candidate_pool,eligible,select
from recognition.neuspell_correction import SpellPolicy
from recognition.scene_text_v1_evaluation import box_iou,evaluate_images,maximum_matching
from scripts.evaluate_totaltext_bert_correction import checked_file,digest,matched_gt,payload_digest,read_json,seal_json,shard_path
from scripts.evaluate_totaltext_bert_methods import check_head,prepared as upstream_prepared,runtime_common,setup
from scripts.generate_totaltext_bert_captions import bind_rgb


def implementations():
    root=Path(__file__).resolve().parents[1]
    names=("scripts/evaluate_totaltext_bert_visual_correction.py","recognition/bert_visual_correction.py",
        "recognition/bert_correction.py","recognition/bert_method_selection.py","recognition/neuspell_correction.py",
        "recognition/parseq.py","recognition/scene_text_v1_evaluation.py",
        "scripts/evaluate_totaltext_bert_correction.py","scripts/evaluate_totaltext_bert_methods.py",
        "scripts/generate_totaltext_bert_captions.py")
    return {n:digest(root/n) for n in names}


def config(path):
    if path.resolve().parent.name!="approximate" or path.resolve().parent.parent.name!="configs":
        raise ValueError("Visual BERT experiment requires configs/approximate/")
    cfg=read_json(path)
    if (cfg["reproduction_mode"]!="approximate" or cfg["claim_label"]!="approximate reproduction"
            or cfg["experiment_kind"]!="totaltext_bert_visual_correction"):
        raise ValueError("Expected explicit approximate visual BERT experiment")
    policy=CandidatePolicy(**cfg["candidates"])
    if not 2<=cfg["context_candidates"]<=policy.maximum_candidates:
        raise ValueError("Invalid shared visual shortlist bound")
    if len({c["name"] for c in cfg["controls"]})!=len(cfg["controls"]):
        raise ValueError("Duplicate control name")
    for c in cfg["controls"]:
        if c["anchor"] not in {"original","insertion"} or c["condition"] not in {None,"local_ocr","local_ocr_caption"}:
            raise ValueError("Invalid explicit visual/context control")
        if (type(c["keep_only"]) is not bool or bool(c["condition"]) != bool(c["beta"])
                or any(not math.isfinite(c[k]) or c[k]<0 for k in ("beta","visual_margin","minimum_gain"))):
            raise ValueError("Invalid control parameters")
    return cfg


def prepare(args):
    cfg=config(args.config)
    base=args.upstream_dir
    p,r,inputs=upstream_prepared(base)
    result=checked_file(base/"results.json",cfg["upstream"]["results_sha256"])
    if digest(base/"protocol.json")!=cfg["upstream"]["protocol_sha256"] or p["resolved_config"]["source"]!=cfg["source"]:
        raise ValueError("Frozen upstream protocol/source differs")
    vocab=checked_file(args.asset_dir/"vocabulary.json",cfg["vocabulary_sha256"])
    words=[]
    seen=set()
    import re
    for i,t in enumerate(vocab["labels"]):
        word=t.casefold()
        if (i not in vocab["special_ids"] and re.fullmatch(r"[a-z]+",word)
                and cfg["candidates"]["minimum_length"]<=len(word)<=cfg["candidates"]["maximum_length"]
                and word not in seen):
            words.append(word)
            seen.add(word)
    rgb=bind_rgb(inputs,args.rgb_manifest,args.data_root,cfg["caption"]["manifest_sha256"])
    policy=CandidatePolicy(**cfg["candidates"])
    observations=[]
    head_sha=digest(base/"head_runtime.json")
    for sample,image in zip(inputs["samples"],rgb):
        heads={v["name"]+"/"+c:check_head(base,sample,c,v["name"],r,head_sha,p["resolved_config"])
            for v in p["resolved_config"]["casing_variants"] for c in p["resolved_config"]["conditions"]}
        blocks=[]
        for index,b in enumerate(sample["blocks"]):
            sources={name:h["trace"][index]["neural"].get("top_labels",[]) for name,h in heads.items()}
            anchor=heads["published_lowercase/word_only"]["trace"][index]["method_selections"]["oov_insertions"]["text"]
            pools=candidate_pool(b["text"],sources,sample["caption"],words,anchor,policy)
            if anchor not in [c["text"] for c in pools]:
                raise ValueError("Insertion reference missing from initial candidate pool")
            blocks.append({**b,"insertion_anchor":anchor,"candidates":pools})
        observations.append({**sample,**image,"blocks":blocks})
    labels=checked_file(base/"evaluation_labels.json",r["evaluation_labels_sha256"])
    if [s["sample_id"] for s in labels["samples"]]!=[s["sample_id"] for s in observations]:
        raise ValueError("Evaluation-label population/order differs")
    protocol={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":cfg["result_class"],"resolved_config":cfg,"config_sha256":digest(args.config),
        "implementation_sha256":implementations(),"upstream_dir":str(base.resolve()),
        "upstream_protocol_sha256":digest(base/"protocol.json"),"upstream_inputs_sha256":r["inputs_sha256"],
        "upstream_results_sha256":digest(base/"results.json"),"upstream_head_runtime_sha256":head_sha,
        "rgb_manifest_sha256":digest(args.rgb_manifest),"vocabulary_sha256":digest(args.asset_dir/"vocabulary.json"),
        "vocabulary_words":len(words),"annotation_use":"prepare/evaluate only; no GT in visual/bert",
        "candidate_generation":"observable OCR/head/RGB/caption and external pretrained vocabulary"}
    out=args.output_dir
    seal_json(out/"protocol.json",protocol)
    common={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "protocol_sha256":digest(out/"protocol.json")}
    seal_json(out/"inputs.json",{**common,"samples":observations})
    seal_json(out/"evaluation_labels.json",{**common,"samples":labels["samples"]})
    seal_json(out/"baseline.json",{**common,"metrics":result["baseline"]})
    seal_json(out/"prepared.json",{**common,**{k+"_sha256":digest(out/(k+".json"))
        for k in ("inputs","evaluation_labels","baseline")}})
    print("Prepared",len(observations),"images; external vocabulary",len(words),flush=True)


def prepared(out):
    record=read_json(out/"prepared.json")
    p=checked_file(out/"protocol.json",record["protocol_sha256"])
    if p["implementation_sha256"]!=implementations():
        raise ValueError("Visual BERT implementation changed; use a new output version")
    inputs=checked_file(out/"inputs.json",record["inputs_sha256"])
    if inputs["protocol_sha256"]!=record["protocol_sha256"]:
        raise ValueError("Visual BERT input/protocol differs")
    return p,record,inputs


def checked_row(out,stage,sample,record,runtime_sha):
    path=shard_path(out,stage,sample["sample_id"])
    row=read_json(path)
    if (row["payload_sha256"]!=payload_digest({k:v for k,v in row.items() if k!="payload_sha256"})
            or row["inputs_sha256"]!=record["inputs_sha256"] or row["runtime_sha256"]!=runtime_sha
            or row["sample_id"]!=sample["sample_id"] or row["image_sha256"]!=sample["image_sha256"]
            or len(row["trace"])!=len(sample["blocks"])):
        raise ValueError("Visual BERT shard checksum/provenance/population differs")
    for b,t in zip(sample["blocks"],row["trace"]):
        if any(b[k]!=t[k] for k in ("rank","bbox","crop_rgb_sha256","text","insertion_anchor")):
            raise ValueError("Visual BERT frozen crop identity differs")
    return row


def save_row(out,stage,sample,record,runtime_sha,trace,tick,torch,device,**extra):
    row={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "sample_id":sample["sample_id"],"image_sha256":sample["image_sha256"],
        "inputs_sha256":record["inputs_sha256"],"runtime_sha256":runtime_sha,"trace":trace,
        "seconds":time.perf_counter()-tick,"peak_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda:0" else None,
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
        raise ValueError("Existing isolated einops source differs")
    sys.path.insert(0,str(args.runtime_deps.resolve()))
    import einops
    if einops.__version__!=cfg["caption"]["einops_version"]:
        raise ValueError("Existing einops version differs")
    from recognition.parseq import ParseqApproxConfig,ParseqApproxRecognizer
    if (digest(args.parseq_checkpoint)!=cfg["parseq"]["checkpoint_sha256"]
            or digest(args.parseq_archive)!=cfg["parseq"]["source_archive_sha256"]):
        raise ValueError("Qualified PARSeq checkpoint/archive differs")
    load=time.perf_counter()
    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
        checkpoint=args.parseq_checkpoint,device=args.device,runtime_source_type=cfg["parseq"]["source_type"],
        runtime_source_archive=args.parseq_archive,decode_ar=cfg["parseq"]["decode_ar"],refine_iters=cfg["parseq"]["refine_iters"]))
    model,tokenizer=recognizer.model.model,recognizer.model.tokenizer
    policy=CandidatePolicy(**cfg["candidates"])
    if policy.maximum_length>model.max_label_length:
        raise ValueError("Candidate maximum exceeds actual PARSeq character limit")
    runtime={**runtime_common(cfg,r,args.device),"model_set":["PARSeq-S"],"model_provenance":recognizer._provenance,
        "parameters":sum(t.numel() for t in recognizer.model.parameters()),"input_shape":[1,3,32,128],
        "score":"causal AR sum including EOS; not native iterative refinement confidence",
        "beam_width":policy.beam_width,"score_batch_size":cfg["parseq"]["score_batch_size"],
        "runtime_dependency_path":str(args.runtime_deps.resolve()),"einops":einops.__version__}
    seal_json(args.output_dir/"visual_runtime.json",runtime)
    runtime_sha=digest(args.output_dir/"visual_runtime.json")
    load_seconds=time.perf_counter()-load
    for number,sample in enumerate(inputs["samples"],1):
        if args.max_images is not None and number>args.max_images:
            break
        if shard_path(args.output_dir,"visual",sample["sample_id"]).exists():
            checked_row(args.output_dir,"visual",sample,r,runtime_sha)
            continue
        if digest(Path(sample["image_path"]))!=sample["image_sha256"]:
            raise ValueError("Frozen RGB image changed before scoring")
        with Image.open(sample["image_path"]) as opened:
            rgb=np.asarray(opened.convert("RGB"))
        if args.device=="cuda:0":
            torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter()
        trace=[]
        for block in sample["blocks"]:
            x1,y1,x2,y2=block["bbox"]
            if not (0<=x1<x2<=rgb.shape[1] and 0<=y1<y2<=rgb.shape[0]):
                raise ValueError("Frozen bbox escapes actual RGB image")
            crop=np.ascontiguousarray(rgb[y1:y2,x1:x2])
            crop_sha=sha256(np.asarray(crop.shape,dtype="<i8").tobytes()+crop.tobytes()).hexdigest()
            if crop_sha!=block["crop_rgb_sha256"]:
                raise ValueError("Crop RGB bytes differ from frozen PARSeq input")
            if not eligible(block["text"],policy):
                trace.append({**block,"status":"protected_shape_KEEP","beam":[],"visual_scores":[],
                    "full_candidates":block["candidates"],"candidate_pool":block["candidates"],"anchor_index":0})
                continue
            tensor=recognizer.transform(Image.fromarray(crop)).unsqueeze(0).to(args.device)
            with torch.inference_mode():
                memory=model.encode(tensor)
                beam=ar_beam(model,tokenizer,memory,policy.beam_width,policy.maximum_length)
                candidates=append_beam(block["candidates"],beam,block["text"],policy)
                scores=ar_scores(model,tokenizer,memory,[c["text"] for c in candidates],cfg["parseq"]["score_batch_size"])
                beam_scores=ar_scores(model,tokenizer,memory,[b["text"] for b in beam],cfg["parseq"]["score_batch_size"])
            if any(not math.isclose(b["sum_log_probability"],s["sum_log_probability"],abs_tol=2e-4,rel_tol=1e-5)
                   for b,s in zip(beam,beam_scores)):
                raise RuntimeError("Beam probability differs from independent complete teacher-forced score")
            anchor=next(i for i,c in enumerate(candidates) if c["text"]==block["insertion_anchor"])
            indices=list(dict.fromkeys([0,anchor]))
            for index in sorted(range(len(candidates)),key=lambda i:(-scores[i]["sum_log_probability"],i)):
                if index not in indices and len(indices)<cfg["context_candidates"]:
                    indices.append(index)
            trace.append({**block,"status":"scored","beam":beam,"beam_teacher_scores":beam_scores,
                "full_candidates":candidates,"full_visual_scores":scores,"shortlist_indices":indices,
                "candidate_pool":[candidates[i] for i in indices],"visual_scores":[scores[i] for i in indices],
                "anchor_index":indices.index(anchor),"input_shape":list(tensor.shape),"memory_shape":list(memory.shape)})
        if args.device=="cuda:0":
            torch.cuda.synchronize()
        save_row(args.output_dir,"visual",sample,r,runtime_sha,trace,tick,torch,args.device,model_loading_seconds=load_seconds)
        if number%10==0 or number==len(inputs["samples"]) or args.max_images==number:
            print("Visual",number,"/",len(inputs["samples"]),flush=True)


def bert(args):
    p,r,inputs=prepared(args.output_dir)
    cfg=p["resolved_config"]
    torch=setup(cfg,args.device)
    from transformers import AutoModelForMaskedLM,AutoTokenizer
    spec=cfg["scorer"]
    snapshot=args.hf_cache/("models--"+spec["model_id"].replace("/","--"))/"snapshots"/spec["revision"]
    for name,value in spec["files_sha256"].items():
        if digest(snapshot/name)!=value:
            raise ValueError("Pinned BERT model/config/tokenizer file differs: "+name)
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,use_fast=True)
    model,audit=AutoModelForMaskedLM.from_pretrained(snapshot,local_files_only=True,use_safetensors=True,
        torch_dtype=torch.float32,output_loading_info=True)
    if (audit["missing_keys"] or audit["mismatched_keys"] or audit["error_msgs"]
            or set(audit["unexpected_keys"])-set(spec["allowed_unused_checkpoint_keys"])):
        raise RuntimeError(f"BERT trained-head loading failed: {audit}")
    model.to(args.device).eval()
    shapes=set()
    def hook(module,args,kwargs):
        shapes.add(tuple(kwargs["input_ids"].shape))
    handle=model.register_forward_pre_hook(hook,with_kwargs=True)
    visual_sha=digest(args.output_dir/"visual_runtime.json")
    runtime={**runtime_common(cfg,r,args.device),"model_set":[spec["model_id"]],"loading_audit":audit,
        "model_files_sha256":spec["files_sha256"],"visual_runtime_sha256":visual_sha,
        "score_batch_size":spec["batch_size"],"max_input_tokens":spec["max_input_tokens"],
        "conditions":["local_ocr","local_ocr_caption"],"context":"original OCR neighbours; caption separate segment"}
    seal_json(args.output_dir/"bert_runtime.json",runtime)
    runtime_sha=digest(args.output_dir/"bert_runtime.json")
    scorer=BertWordScorer(tokenizer,model,device=args.device,batch_size=spec["batch_size"],max_tokens=spec["max_input_tokens"])
    policy=SpellPolicy(**cfg["context_policy"])
    for number,sample in enumerate(inputs["samples"],1):
        if args.max_images is not None and number>args.max_images:
            break
        if shard_path(args.output_dir,"bert",sample["sample_id"]).exists():
            checked_row(args.output_dir,"bert",sample,r,runtime_sha)
            continue
        v=checked_row(args.output_dir,"visual",sample,r,visual_sha)
        if args.device=="cuda:0":
            torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter()
        shapes.clear()
        trace=[]
        for index,block in enumerate(v["trace"]):
            left,right=local_context(sample["blocks"],index,policy)
            caption=" ".join(sample["caption"].split()[:policy.max_caption_words])
            scores={}
            if block["status"]=="scored" and len(block["candidate_pool"])>1:
                for condition,cap in (("local_ocr",None),("local_ocr_caption",caption)):
                    scores[condition]=scorer.score([c["text"] for c in block["candidate_pool"]],
                        left=left,right=right,caption=cap)
            decisions={}
            for control in cfg["controls"]:
                anchor=block["anchor_index"] if control["anchor"]=="insertion" else 0
                if control["keep_only"] or not scores:
                    choice={"selected_index":anchor,"reason":"control_KEEP" if control["keep_only"] else "protected/no_alternative_KEEP","gain":0.0}
                else:
                    choice=select(block["candidate_pool"],block["visual_scores"],scores.get(control["condition"]),
                        anchor_index=anchor,visual_margin=control["visual_margin"],beta=control["beta"],minimum_gain=control["minimum_gain"])
                decisions[control["name"]]={**choice,"text":block["candidate_pool"][choice["selected_index"]]["text"]}
            trace.append({**{k:block[k] for k in ("rank","bbox","crop_rgb_sha256","text","insertion_anchor","status","candidate_pool","anchor_index")},
                "context":{"left":left,"right":right,"caption":caption},"bert_scores":scores,"decisions":decisions})
        if args.device=="cuda:0":
            torch.cuda.synchronize()
        save_row(args.output_dir,"bert",sample,r,runtime_sha,trace,tick,torch,args.device,
            visual_sha256=digest(shard_path(args.output_dir,"visual",sample["sample_id"])),
            observed_input_shapes=[list(s) for s in sorted(shapes)])
        if number%10==0 or number==len(inputs["samples"]) or args.max_images==number:
            print("BERT",number,"/",len(inputs["samples"]),flush=True)
    handle.remove()


def ceiling(blocks,gt,pools):
    edges=[(i,j) for i,b in enumerate(blocks) for j,g in enumerate(gt)
        if box_iou(b["bbox"],g["bbox"])>=0.5 and g["text"].casefold() in {c["text"].casefold() for c in pools[i]}]
    return len(maximum_matching(len(blocks),len(gt),edges))


def evaluate(args):
    p,r,inputs=prepared(args.output_dir)
    cfg=p["resolved_config"]
    labels=checked_file(args.output_dir/"evaluation_labels.json",r["evaluation_labels_sha256"])["samples"]
    baseline=checked_file(args.output_dir/"baseline.json",r["baseline_sha256"])["metrics"]
    vsha,bsha=digest(args.output_dir/"visual_runtime.json"),digest(args.output_dir/"bert_runtime.json")
    if read_json(args.output_dir/"bert_runtime.json")["visual_runtime_sha256"]!=vsha:
        raise ValueError("BERT/visual runtime link differs")
    rows={c["name"]:[] for c in cfg["controls"]}
    details={name:[] for name in rows}
    oracles={"initial_candidates":0,"expanded_candidates":0,"visual_shortlist":0}
    per_image_oracles=[]
    files={}
    for sample,label in zip(inputs["samples"],labels):
        if label["sample_id"]!=sample["sample_id"]:
            raise ValueError("Label/sample ordering differs")
        v=checked_row(args.output_dir,"visual",sample,r,vsha)
        b=checked_row(args.output_dir,"bert",sample,r,bsha)
        if b["visual_sha256"]!=digest(shard_path(args.output_dir,"visual",sample["sample_id"])):
            raise ValueError("BERT scores bound to different visual candidates")
        for block,vt,bt in zip(sample["blocks"],v["trace"],b["trace"]):
            if bt["candidate_pool"]!=vt["candidate_pool"] or bt["anchor_index"]!=vt["anchor_index"]:
                raise ValueError("Shared contextual candidate pool/anchor changed")
            for control in cfg["controls"]:
                anchor=bt["anchor_index"] if control["anchor"]=="insertion" else 0
                if control["keep_only"] or not bt["bert_scores"]:
                    expected={"selected_index":anchor,"reason":"control_KEEP" if control["keep_only"] else "protected/no_alternative_KEEP","gain":0.0}
                else:
                    expected=select(vt["candidate_pool"],vt["visual_scores"],bt["bert_scores"].get(control["condition"]),
                        anchor_index=anchor,visual_margin=control["visual_margin"],beta=control["beta"],minimum_gain=control["minimum_gain"])
                expected["text"]=vt["candidate_pool"][expected["selected_index"]]["text"]
                if expected!=bt["decisions"][control["name"]]:
                    raise ValueError("Frozen control decision differs from actual scores")
        before=matched_gt(sample["blocks"],label["ground_truth"])
        for name in rows:
            preds=[{"rank":block["rank"],"bbox":block["bbox"],"text":bt["decisions"][name]["text"]}
                for block,bt in zip(sample["blocks"],b["trace"])]
            after=matched_gt(preds,label["ground_truth"])
            rows[name].append({"ground_truth":label["ground_truth"],"predictions":preds})
            details[name].append({"sample_id":sample["sample_id"],"ground_truth":label["ground_truth"],
                "predictions":preds,"recovered_gt_indices":sorted(after-before),"damaged_gt_indices":sorted(before-after)})
        values={"initial_candidates":ceiling(sample["blocks"],label["ground_truth"],[x["candidates"] for x in sample["blocks"]]),
            "expanded_candidates":ceiling(sample["blocks"],label["ground_truth"],[t["full_candidates"] for t in v["trace"]]),
            "visual_shortlist":ceiling(sample["blocks"],label["ground_truth"],[t["candidate_pool"] for t in v["trace"]])}
        for key,count in values.items():
            oracles[key]+=count
        per_image_oracles.append({"sample_id":sample["sample_id"],**values})
        for stage in ("visual","bert"):
            path=shard_path(args.output_dir,stage,sample["sample_id"])
            files[str(path)]=digest(path)
    controls={}
    errors=baseline["geometry"]["iou_ge_0_50_one_to_one"]-baseline["recognition"]["exact_casefold_spatial_one_to_one"]
    for name,value in rows.items():
        metrics=evaluate_images(value,text_available=True)
        recovered=sum(len(d["recovered_gt_indices"]) for d in details[name])
        damaged=sum(len(d["damaged_gt_indices"]) for d in details[name])
        if (metrics["geometry"]!=baseline["geometry"] or metrics["recognition"]["exact_casefold_spatial_one_to_one"]
                -baseline["recognition"]["exact_casefold_spatial_one_to_one"]!=recovered-damaged):
            raise ValueError("Correction geometry or recovered/damaged accounting differs")
        controls[name]={"metrics":metrics,"recovered":recovered,"damaged":damaged,"net":recovered-damaged,
            "baseline_geometry_errors":errors,"recovery_fraction_geometry_errors":recovered/errors if errors else 0.0,
            "per_image":details[name]}
        print(name,"exact",metrics["recognition"]["exact_casefold_spatial_one_to_one"],"recovered",recovered,"damaged",damaged,flush=True)
    result={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":cfg["result_class"],"protocol_sha256":r["protocol_sha256"],"baseline":baseline,"controls":controls,
        "diagnostic_GT_oracle":{"label":"diagnostic upper bound; not achieved prediction accuracy","counts":oracles,"per_image":per_image_oracles},
        "inference_files_sha256":files,"test_loaded":False,"training_performed":False,"independent_validation":False,
        "failed_samples":0,"excluded_samples":0,"official_polygon_protocol":False}
    seal_json(args.output_dir/"results.json",result)
    print("GT-only candidate ceilings",oracles,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="stage",required=True)
    a=subs.add_parser("prepare")
    for name in ("config","upstream-dir","asset-dir","rgb-manifest","data-root","output-dir"):
        a.add_argument("--"+name,type=Path,required=True)
    a=subs.add_parser("visual")
    for name in ("output-dir","parseq-source","parseq-checkpoint","parseq-archive","runtime-deps"):
        a.add_argument("--"+name,type=Path,required=True)
    a.add_argument("--device",choices=("cpu","cuda:0"),required=True)
    a.add_argument("--max-images",type=int)
    a=subs.add_parser("bert")
    for name in ("output-dir","hf-cache"):
        a.add_argument("--"+name,type=Path,required=True)
    a.add_argument("--device",choices=("cpu","cuda:0"),required=True)
    a.add_argument("--max-images",type=int)
    a=subs.add_parser("evaluate")
    a.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    try:
        if getattr(args,"max_images",None) is not None and args.max_images<1:
            raise ValueError("max-images must be positive")
        {"prepare":prepare,"visual":visual,"bert":bert,"evaluate":evaluate}[args.stage](args)
    except Exception as error:
        if hasattr(args,"output_dir"):
            args.output_dir.mkdir(parents=True,exist_ok=True)
            (args.output_dir/("error-"+args.stage+".log")).write_text(type(error).__name__+": "+str(error)+"\n")
        raise


if __name__=="__main__":
    main()
