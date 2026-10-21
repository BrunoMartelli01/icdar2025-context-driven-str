"""Approximate, frozen Total-Text development battery: casing, edits, PLL, scene context.

Ground truth is isolated from head inference and MLM reranking. Existing sealed
v2/MLM implementations and results are imported without modification.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import platform
import random
import time

from recognition.bert_correction import BertWordScorer, local_context
from recognition.bert_method_selection import MethodPolicy, bounded_candidates, insertion_selection, pll_selection
from recognition.neuspell_correction import NeuSpellWordHead, SpellPolicy, correct_blocks, select_spelling
from recognition.scene_text_v1_evaluation import evaluate_images
from scripts.evaluate_totaltext_bert_correction import (
    checked_file,digest,load_captions,matched_gt,payload_digest,read_json,seal_json,shard_path,source_inputs)
from scripts.evaluate_totaltext_neuspell import load_word_head, verify_assets
from scripts.generate_totaltext_bert_captions import load_config


def implementation_hashes():
    root=Path(__file__).resolve().parents[1]
    names=("scripts/evaluate_totaltext_bert_methods.py","recognition/bert_method_selection.py",
        "recognition/bert_correction.py","recognition/neuspell_correction.py",
        "recognition/scene_text_v1_evaluation.py","scripts/evaluate_totaltext_bert_correction.py",
        "scripts/evaluate_totaltext_neuspell.py","scripts/generate_totaltext_bert_captions.py")
    return {n:digest(root/n) for n in names}


def config(path):
    cfg=load_config(path)
    SpellPolicy(**cfg["policy"])
    MethodPolicy(**cfg["method_policy"])
    if (cfg["methods"] != ["conservative","oov_insertions","bert_pll_rerank"]
            or cfg["conditions"] != ["word_only","local_ocr","local_ocr_caption"]
            or len(cfg["casing_variants"]) != 2
            or len({v["name"] for v in cfg["casing_variants"]}) != 2
            or any(type(v["do_lower_case"]) is not bool or type(v["explicit_override"]) is not bool
                   for v in cfg["casing_variants"])):
        raise ValueError("Unsupported explicit BERT methods/casing/condition protocol")
    return cfg


def prepare(args):
    cfg=config(args.config)
    if args.conditions:
        if len(set(args.conditions)) != len(args.conditions) or any(c not in cfg["conditions"] for c in args.conditions):
            raise ValueError("Requested BERT condition subset differs from the declared battery")
        if args.conditions != [c for c in cfg["conditions"] if c in args.conditions] or args.conditions[0] != "word_only":
            raise ValueError("BERT controls require word_only first and declared condition order")
        cfg["conditions"]=args.conditions
    observations,labels,source=source_inputs(args.source_dir.resolve(),cfg,[p.resolve() for p in args.reuse_ocr])
    caption_source=None
    if "local_ocr_caption" in cfg["conditions"]:
        if args.captions is None:
            raise ValueError("Scene condition requires complete SHA-bound captions; select explicit text-only conditions for controls")
        metadata=read_json(args.captions)
        if (metadata["model_id"] != cfg["caption"]["model_id"] or metadata["revision"] != cfg["caption"]["revision"]
                or metadata["generation"] != cfg["caption"]["generation"]
                or metadata["model_files_sha256"] != cfg["caption"]["files_sha256"]):
            raise ValueError("Caption model/revision/generation differs from the declared BERT context experiment")
        cap=load_captions(args.captions,observations)
        for sample in observations:
            sample["caption"]=cap[sample["sample_id"]]
        caption_source={"file":str(args.captions.resolve()),"sha256":digest(args.captions),
                        "provenance":{k:v for k,v in read_json(args.captions).items() if k != "samples"}}
    common={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":cfg["result_class"]}
    protocol={**common,"resolved_config":cfg,"config_sha256":digest(args.config),
        "implementation_sha256":implementation_hashes(),"source_dir":str(args.source_dir.resolve()),
        "source_files_sha256":source["files_sha256"],"upstream_protocol":source["upstream_protocol"],
        "captions":caption_source,"reuse_ocr_directories":[str(p.resolve()) for p in args.reuse_ocr]}
    out=args.output_dir
    seal_json(out/"protocol.json",protocol)
    bound={**common,"protocol_sha256":digest(out/"protocol.json")}
    seal_json(out/"inputs.json",{**bound,"samples":observations})
    seal_json(out/"evaluation_labels.json",{**bound,"samples":labels})
    seal_json(out/"baseline.json",{**bound,"metrics":source["baseline"]})
    seal_json(out/"prepared.json",{**bound,**{n+"_sha256":digest(out/(n+".json"))
                  for n in ("inputs","evaluation_labels","baseline")}})
    print("Prepared BERT methods",len(observations),"images; conditions",cfg["conditions"],flush=True)


def prepared(out):
    record=read_json(out/"prepared.json")
    protocol=checked_file(out/"protocol.json",record["protocol_sha256"])
    if protocol["implementation_sha256"] != implementation_hashes():
        raise ValueError("BERT method implementation changed; prepare a new output version")
    inputs=checked_file(out/"inputs.json",record["inputs_sha256"])
    if inputs["protocol_sha256"] != record["protocol_sha256"]:
        raise ValueError("BERT method input/protocol differs")
    return protocol,record,inputs


def setup(cfg,device):
    import numpy as np
    import torch
    if device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("BERT methods CUDA unavailable; no automatic CPU substitution")
    if device not in {"cpu","cuda:0"}:
        raise ValueError("Select CPU or CUDA:0 explicitly")
    seed=cfg["model"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg["model"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    if device == "cuda:0":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    return torch


def runtime_common(cfg,record,device):
    import numpy as np
    import torch
    import transformers
    return {"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "inputs_sha256":record["inputs_sha256"],"device":device,"precision":"float32",
        "seed":cfg["model"]["seed"],"cpu_threads":cfg["model"]["cpu_threads"],
        "deterministic_algorithms":True,"tf32":False,"python":platform.python_version(),
        "torch":torch.__version__,"transformers":transformers.__version__,"numpy":np.__version__}


def identity(row,sample,condition,variant,record,runtime_sha):
    if row.get("payload_sha256") != payload_digest({k:v for k,v in row.items() if k != "payload_sha256"}):
        raise ValueError("BERT method shard payload checksum differs")
    if (row["sample_id"] != sample["sample_id"] or row["image_sha256"] != sample["image_sha256"]
            or row["condition"] != condition or row["variant"] != variant
            or row["inputs_sha256"] != record["inputs_sha256"] or row["runtime_sha256"] != runtime_sha
            or len(row["trace"]) != len(sample["blocks"])):
        raise ValueError("BERT method shard provenance/population differs")
    for original,trace in zip(sample["blocks"],row["trace"]):
        if any(original[k] != trace[k] for k in ("rank","bbox","crop_rgb_sha256")) or original["text"] != trace["original_text"]:
            raise ValueError("BERT method crop identity changed")


def head_path(out,variant,condition,sid):
    return shard_path(out/"head"/variant,condition,sid)


def check_head(out,sample,condition,variant,record,runtime_sha,cfg):
    path=head_path(out,variant,condition,sample["sample_id"])
    row=read_json(path)
    identity(row,sample,condition,variant,record,runtime_sha)
    spell,method=SpellPolicy(**cfg["policy"]),MethodPolicy(**cfg["method_policy"])
    if condition == "word_only":
        candidate_neural=[t["neural"] for t in row["trace"]]
    else:
        base=check_head(out,sample,"word_only",variant,record,runtime_sha,cfg)
        candidate_neural=[t["neural"] for t in base["trace"]]
        if row["candidate_source_sha256"] != digest(head_path(out,variant,"word_only",sample["sample_id"])):
            raise ValueError("Frozen word-only candidate source changed")
    for i,(trace,neural) in enumerate(zip(row["trace"],candidate_neural)):
        original=trace["original_text"]
        left,right=([],[]) if condition=="word_only" else local_context(sample["blocks"],i,spell)
        caption=" ".join(sample["caption"].split()[:spell.max_caption_words]) if condition=="local_ocr_caption" else None
        if trace["context"] != {"left":left,"right":right,"caption":caption}:
            raise ValueError("BERT method context differs from frozen OCR/caption inputs")
        if (trace["method_selections"]["conservative"] != select_spelling(original,trace["neural"],spell)["conservative"]
                or trace["method_selections"]["oov_insertions"] != insertion_selection(original,trace["neural"],method)
                or trace["candidates"] != bounded_candidates(original,neural,method,spell)):
            raise ValueError("BERT method head selection/candidate trace differs")
    return row


def infer_head(args):
    from transformers import AutoTokenizer
    protocol,record,inputs=prepared(args.output_dir)
    cfg=protocol["resolved_config"]
    if "word_only" not in cfg["conditions"]:
        raise ValueError("Frozen candidate generation requires word_only control")
    torch=setup(cfg,args.device)
    snapshot,vocabulary,assets=verify_assets(cfg,args.hf_cache.resolve(),args.asset_dir.resolve())
    start=time.perf_counter()
    model,audit=load_word_head(snapshot,device=args.device,expected_labels=cfg["model"]["label_count"],
                               allowed_unused_keys=cfg["model"]["allowed_unused_checkpoint_keys"])
    tokenizers={}
    for variant in cfg["casing_variants"]:
        kwargs={"do_lower_case":variant["do_lower_case"]} if variant["explicit_override"] else {}
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,use_fast=True,**kwargs)
        if tokenizer.do_lower_case != variant["do_lower_case"]:
            raise ValueError("BERT method explicit casing variant differs from resolved tokenizer")
        tokenizers[variant["name"]]=tokenizer
    runtime={**runtime_common(cfg,record,args.device),"model_id":cfg["model"]["model_id"],
        "revision":cfg["model"]["revision"],"model_set":[cfg["model"]["model_id"]],
        "model_files_sha256":assets["model_files_sha256"],"assets_sha256":digest(args.asset_dir/"assets.json"),
        "loading_audit":audit,"casing_variants":cfg["casing_variants"],"batch_size":cfg["model"]["batch_size"],
        "max_input_tokens":cfg["model"]["max_input_tokens"],"candidate_pool_source":"word_only for every condition"}
    seal_json(args.output_dir/"head_runtime.json",runtime)
    runtime_sha=digest(args.output_dir/"head_runtime.json")
    load_seconds=time.perf_counter()-start
    spell,method=SpellPolicy(**cfg["policy"]),MethodPolicy(**cfg["method_policy"])
    for name,tokenizer in tokenizers.items():
        decoder=NeuSpellWordHead(tokenizer,model,vocabulary,device=args.device,batch_size=cfg["model"]["batch_size"],
            max_input_tokens=cfg["model"]["max_input_tokens"],saved_top_k=cfg["model"]["saved_top_k"])
        for condition in cfg["conditions"]:
            for i,sample in enumerate(inputs["samples"],1):
                path=head_path(args.output_dir,name,condition,sample["sample_id"])
                if path.exists():
                    check_head(args.output_dir,sample,condition,name,record,runtime_sha,cfg)
                    continue
                if args.device == "cuda:0":
                    torch.cuda.reset_peak_memory_stats()
                tick=time.perf_counter()
                trace=correct_blocks(sample["blocks"],condition=condition,caption=sample["caption"],decoder=decoder,policy=spell)
                if condition == "word_only":
                    candidate_neural=[t["neural"] for t in trace]
                    source_sha=None
                else:
                    frozen=check_head(args.output_dir,sample,"word_only",name,record,runtime_sha,cfg)
                    candidate_neural=[t["neural"] for t in frozen["trace"]]
                    source_sha=digest(head_path(args.output_dir,name,"word_only",sample["sample_id"]))
                for row,neural in zip(trace,candidate_neural):
                    row["method_selections"]={"conservative":row["selections"]["conservative"],
                        "oov_insertions":insertion_selection(row["original_text"],row["neural"],method)}
                    row["candidates"]=bounded_candidates(row["original_text"],neural,method,spell)
                if args.device == "cuda:0":
                    torch.cuda.synchronize()
                row={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
                    "inputs_sha256":record["inputs_sha256"],"runtime_sha256":runtime_sha,"variant":name,"condition":condition,
                    "sample_id":sample["sample_id"],"image_sha256":sample["image_sha256"],"trace":trace,
                    "candidate_source_sha256":source_sha,"seconds":time.perf_counter()-tick,"model_loading_seconds":load_seconds,
                    "input_shapes":[list(s) for s in sorted(decoder.input_shapes)],
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated() if args.device=="cuda:0" else None,
                    "peak_reserved_bytes":torch.cuda.max_memory_reserved() if args.device=="cuda:0" else None}
                row["payload_sha256"]=payload_digest(row)
                seal_json(path,row)
                if i%25==0 or i==len(inputs["samples"]):
                    print(f"[head {name}/{condition}] {i}/{len(inputs['samples'])}",flush=True)


def pll_path(out,variant,condition,sid):
    return shard_path(out/"pll"/variant,condition,sid)


def check_pll(out,sample,condition,variant,record,runtime_sha,cfg,head):
    row=read_json(pll_path(out,variant,condition,sample["sample_id"]))
    identity(row,sample,condition,variant,record,runtime_sha)
    if row["head_sha256"] != digest(head_path(out,variant,condition,sample["sample_id"])):
        raise ValueError("PLL head/candidate source checksum differs")
    policy=MethodPolicy(**cfg["method_policy"])
    for trace,source in zip(row["trace"],head["trace"]):
        if (trace["candidates"] != source["candidates"] or trace["context"] != source["context"]
                or trace["selection"] != pll_selection(trace["candidates"],trace["scores"],policy)):
            raise ValueError("PLL selection/context/candidate trace differs")
    return row


def infer_pll(args):
    from transformers import AutoModelForMaskedLM,AutoTokenizer
    protocol,record,inputs=prepared(args.output_dir)
    cfg=protocol["resolved_config"]
    torch=setup(cfg,args.device)
    spec=cfg["scorer"]
    snapshot=args.hf_cache/("models--"+spec["model_id"].replace("/","--"))/"snapshots"/spec["revision"]
    if digest(snapshot/"model.safetensors") != spec["weight_sha256"]:
        raise ValueError("BERT MLM scorer checkpoint checksum differs")
    model,loading=AutoModelForMaskedLM.from_pretrained(snapshot,local_files_only=True,use_safetensors=True,
        torch_dtype=torch.float32,output_loading_info=True)
    if (model.config.model_type != "bert" or any(loading[k] for k in ("missing_keys","mismatched_keys","error_msgs"))
            or set(loading["unexpected_keys"]) - set(spec["allowed_unused_checkpoint_keys"])):
        raise RuntimeError(f"BERT MLM scorer checkpoint compatibility failure: {loading}")
    model.to(args.device).eval()
    tokenizer=AutoTokenizer.from_pretrained(snapshot,use_fast=True,local_files_only=True)
    files=("config.json","model.safetensors","tokenizer.json","tokenizer_config.json","vocab.txt")
    runtime={**runtime_common(cfg,record,args.device),"model_id":spec["model_id"],"revision":spec["revision"],
        "model_set":[spec["model_id"]],"head_runtime_sha256":digest(args.output_dir/"head_runtime.json"),
        "loading_audit":loading,"model_files_sha256":{n:digest(snapshot/n) for n in files},
        "mask_task_batch_size":spec["batch_size"],"max_input_tokens":spec["max_input_tokens"],
        "scoring":spec["scoring"],"method_policy":cfg["method_policy"]}
    seal_json(args.output_dir/"pll_runtime.json",runtime)
    runtime_sha=digest(args.output_dir/"pll_runtime.json")
    head_sha=digest(args.output_dir/"head_runtime.json")
    if read_json(args.output_dir/"head_runtime.json")["inputs_sha256"] != record["inputs_sha256"]:
        raise ValueError("Head runtime/input checksum differs")
    scorer=BertWordScorer(tokenizer,model,device=args.device,batch_size=spec["batch_size"],max_tokens=spec["max_input_tokens"])
    shapes=set()
    def observe_shape(module,positional,keywords):
        ids=keywords.get("input_ids",positional[0] if positional else None)
        if ids is not None:
            shapes.add(tuple(ids.shape))
    hook=model.register_forward_pre_hook(observe_shape,with_kwargs=True)
    policy=MethodPolicy(**cfg["method_policy"])
    for variant in cfg["casing_variants"]:
        name=variant["name"]
        for condition in cfg["conditions"]:
            for i,sample in enumerate(inputs["samples"],1):
                head=check_head(args.output_dir,sample,condition,name,record,head_sha,cfg)
                path=pll_path(args.output_dir,name,condition,sample["sample_id"])
                if path.exists():
                    check_pll(args.output_dir,sample,condition,name,record,runtime_sha,cfg,head)
                    continue
                shapes.clear()
                scorer.max_input_tokens_seen=0
                if args.device=="cuda:0":
                    torch.cuda.reset_peak_memory_stats()
                tick=time.perf_counter()
                trace=[]
                for row in head["trace"]:
                    candidates=row["candidates"]
                    context=row["context"]
                    scores=scorer.score([c["text"] for c in candidates],left=context["left"],right=context["right"],
                        caption=context["caption"]) if len(candidates)>1 else []
                    trace.append({**{k:row[k] for k in ("rank","bbox","crop_rgb_sha256","original_text")},
                        "context":context,"candidates":candidates,"scores":scores,
                        "selection":pll_selection(candidates,scores,policy)})
                if args.device=="cuda:0":
                    torch.cuda.synchronize()
                row={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
                    "inputs_sha256":record["inputs_sha256"],"runtime_sha256":runtime_sha,"variant":name,"condition":condition,
                    "sample_id":sample["sample_id"],"image_sha256":sample["image_sha256"],"trace":trace,
                    "head_sha256":digest(head_path(args.output_dir,name,condition,sample["sample_id"])),
                    "seconds":time.perf_counter()-tick,"input_shapes":[list(s) for s in sorted(shapes)],
                    "input_tokens_max":scorer.max_input_tokens_seen,
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated() if args.device=="cuda:0" else None,
                    "peak_reserved_bytes":torch.cuda.max_memory_reserved() if args.device=="cuda:0" else None}
                row["payload_sha256"]=payload_digest(row)
                seal_json(path,row)
                if i%25==0 or i==len(inputs["samples"]):
                    print(f"[PLL {name}/{condition}] {i}/{len(inputs['samples'])}",flush=True)
    hook.remove()


def evaluate(args):
    protocol,record,inputs=prepared(args.output_dir)
    cfg=protocol["resolved_config"]
    labels=checked_file(args.output_dir/"evaluation_labels.json",record["evaluation_labels_sha256"])["samples"]
    baseline=checked_file(args.output_dir/"baseline.json",record["baseline_sha256"])["metrics"]
    if [s["sample_id"] for s in labels] != [s["sample_id"] for s in inputs["samples"]]:
        raise ValueError("BERT method evaluation label population/order differs")
    head_sha=digest(args.output_dir/"head_runtime.json")
    pll_sha=digest(args.output_dir/"pll_runtime.json")
    if read_json(args.output_dir/"pll_runtime.json")["head_runtime_sha256"] != head_sha:
        raise ValueError("PLL/head runtime differs")
    summaries={}
    for variant in cfg["casing_variants"]:
        name=variant["name"]
        for condition in cfg["conditions"]:
            heads=[check_head(args.output_dir,s,condition,name,record,head_sha,cfg) for s in inputs["samples"]]
            plls=[check_pll(args.output_dir,s,condition,name,record,pll_sha,cfg,h) for s,h in zip(inputs["samples"],heads)]
            for method in cfg["methods"]:
                rows,details=[],[]
                recovered=damaged=changed=0
                for sample,label,head,pll in zip(inputs["samples"],labels,heads,plls):
                    choices=[t["selection"] for t in pll["trace"]] if method=="bert_pll_rerank" else [t["method_selections"][method] for t in head["trace"]]
                    predictions=[{**{k:b[k] for k in ("rank","bbox")},"text":choice["text"]} for b,choice in zip(sample["blocks"],choices)]
                    before,after=matched_gt(sample["blocks"],label["ground_truth"]),matched_gt(predictions,label["ground_truth"])
                    got,lost=sorted(after-before),sorted(before-after)
                    recovered+=len(got)
                    damaged+=len(lost)
                    changed+=sum(b["text"] != p["text"] for b,p in zip(sample["blocks"],predictions))
                    rows.append({"ground_truth":label["ground_truth"],"predictions":predictions})
                    details.append({"sample_id":sample["sample_id"],"ground_truth":label["ground_truth"],
                        "predictions":predictions,"recovered_gt_indices":got,"damaged_gt_indices":lost})
                metric=evaluate_images(rows,text_available=True)
                net=metric["recognition"]["exact_casefold_spatial_one_to_one"]-baseline["recognition"]["exact_casefold_spatial_one_to_one"]
                if metric["geometry"] != baseline["geometry"] or net != recovered-damaged:
                    raise ValueError("BERT methods changed geometry or inconsistent recovery totals")
                key=name+"/"+condition+"/"+method
                summaries[key]={"metrics":metric,"recovered":recovered,"damaged":damaged,"net":net,
                    "changed_crops":changed,"per_image":details,"head_seconds_total":sum(h["seconds"] for h in heads),
                    "pll_seconds_total":sum(p["seconds"] for p in plls) if method=="bert_pll_rerank" else 0}
                print(key,"exact",metric["recognition"]["exact_casefold_spatial_one_to_one"],"recovered",recovered,"damaged",damaged,flush=True)
    result={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
        "result_class":cfg["result_class"],"protocol_sha256":record["protocol_sha256"],"baseline":baseline,
        "head_runtime_sha256":head_sha,"pll_runtime_sha256":pll_sha,"conditions":summaries,
        "scene_context_evaluated":"local_ocr_caption" in cfg["conditions"],"test_loaded":False,"training_performed":False,
        "thresholds_selected_on_test":False,"independent_validation":False,"official_polygon_protocol":False}
    seal_json(args.output_dir/"results.json",result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="stage",required=True)
    a=sub.add_parser("prepare")
    for name in ("config","source-dir","output-dir"):
        a.add_argument("--"+name,type=Path,required=True)
    a.add_argument("--reuse-ocr",type=Path,action="append",default=[])
    a.add_argument("--captions",type=Path)
    a.add_argument("--conditions",nargs="+",choices=("word_only","local_ocr","local_ocr_caption"))
    a=sub.add_parser("head")
    for name in ("output-dir","hf-cache","asset-dir"):
        a.add_argument("--"+name,type=Path,required=True)
    a.add_argument("--device",choices=("cpu","cuda:0"),required=True)
    a=sub.add_parser("pll")
    for name in ("output-dir","hf-cache"):
        a.add_argument("--"+name,type=Path,required=True)
    a.add_argument("--device",choices=("cpu","cuda:0"),required=True)
    a=sub.add_parser("evaluate")
    a.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    try:
        {"prepare":prepare,"head":infer_head,"pll":infer_pll,"evaluate":evaluate}[args.stage](args)
    except Exception as e:
        seal_json(args.output_dir/"errors"/(str(time.time_ns())+".json"),
            {"error_type":type(e).__name__,"error":str(e),"stage":args.stage,"reproduction_mode":"approximate",
             "claim_label":"approximate reproduction","result_class":"failed BERT methods development experiment"})
        raise


if __name__=="__main__":
    main()
