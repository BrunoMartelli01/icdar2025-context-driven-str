#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, time
from collections import Counter
from pathlib import Path

MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
REVISION="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

INSTRUCTION="""You are a structural OCR grouping controller.
You are NOT allowed to transcribe, correct, replace, normalize, or invent text.

You receive OCR crops from ONE image in frozen reading order.
Each crop has an integer index, bounding box, and OCR string.

Allowed actions:
KEEP i
MERGE i j [k ...]
SPLIT i p [q ...]
MERGE_SPLIT i j [k ...] | p [q ...]

SPLIT boundaries are 1-based character boundaries after p characters.
MERGE concatenates the listed OCR strings exactly in the listed order.
MERGE_SPLIT concatenates exactly, then splits at the listed character boundaries.

Rules:
- Every crop index must appear EXACTLY ONCE across all actions.
- Never reuse or omit a crop.
- Never output OCR words or reconstructed text.
- No explanations, JSON, bullets, markdown, labels, or commentary.
- Do not change characters.
- Prefer KEEP when uncertain.
Return actions only, one per line."""

def lexical_counts(predicted, expected):
    pred=Counter(" ".join(predicted).casefold().split())
    gt=Counter(x.casefold() for x in expected)
    m=sum((pred&gt).values()); p=sum(pred.values()); g=sum(gt.values())
    precision=m/p if p else 0.; recall=m/g if g else 0.
    return {"matches":m,"predicted_tokens":p,"gt_words":g,
            "precision":precision,"recall":recall,
            "f1":2*precision*recall/(precision+recall) if precision+recall else 0.,
            "surplus":sum((pred-gt).values()),"missing":sum((gt-pred).values())}

def make_prompt(sample):
    obs=[{"index":i,"bbox":b["bbox"],"ocr":b["text"]} for i,b in enumerate(sample["blocks"])]
    return INSTRUCTION+"\n\nOBSERVATIONS:\n"+json.dumps(obs,ensure_ascii=False,separators=(",",":"))+"\n\nACTIONS:\n"

def apply_split(text, pos):
    if not pos or pos!=sorted(set(pos)): raise ValueError("bad split boundaries")
    if pos[0]<=0 or pos[-1]>=len(text): raise ValueError("split boundary outside string")
    out=[]; start=0
    for p in pos: out.append(text[start:p]); start=p
    out.append(text[start:])
    if any(x=="" for x in out): raise ValueError("empty split token")
    return out

def parse_actions(raw,sample):
    lines=[x.strip() for x in raw.strip().splitlines() if x.strip()]
    if not lines: raise ValueError("empty output")
    n=len(sample["blocks"]); used=set(); actions=[]; output=[]
    def take(ids):
        if not ids or len(set(ids))!=len(ids): raise ValueError("bad crop list")
        if any(i<0 or i>=n for i in ids): raise ValueError("unknown crop id")
        if any(i in used for i in ids): raise ValueError("crop reused")
        used.update(ids)
    for line in lines:
        if re.fullmatch(r"KEEP \d+",line):
            i=int(line.split()[1]); take([i]); t=sample["blocks"][i]["text"]
            output.append(t); actions.append({"action":"KEEP","ids":[i],"output":[t]})
        elif line.startswith("MERGE_SPLIT "):
            m=re.fullmatch(r"MERGE_SPLIT ([0-9 ]+) \| ([0-9 ]+)",line)
            if not m: raise ValueError("invalid MERGE_SPLIT")
            ids=[int(x) for x in m.group(1).split()]; pos=[int(x) for x in m.group(2).split()]
            if len(ids)<2: raise ValueError("MERGE_SPLIT needs >=2 crops")
            take(ids); text="".join(sample["blocks"][i]["text"] for i in ids); toks=apply_split(text,pos)
            output.extend(toks); actions.append({"action":"MERGE_SPLIT","ids":ids,"positions":pos,"output":toks})
        elif re.fullmatch(r"MERGE(?: \d+){2,}",line):
            ids=[int(x) for x in line.split()[1:]]; take(ids)
            text="".join(sample["blocks"][i]["text"] for i in ids)
            output.append(text); actions.append({"action":"MERGE","ids":ids,"output":[text]})
        elif line.startswith("SPLIT "):
            parts=line.split()
            if len(parts)<3 or any(not x.isdigit() for x in parts[1:]): raise ValueError("invalid SPLIT")
            i=int(parts[1]); pos=[int(x) for x in parts[2:]]; take([i])
            toks=apply_split(sample["blocks"][i]["text"],pos)
            output.extend(toks); actions.append({"action":"SPLIT","ids":[i],"positions":pos,"output":toks})
        else:
            raise ValueError("unknown/free-text action")
    if used!=set(range(n)): raise ValueError("not all crops consumed exactly once")
    return actions,output

def load_inputs(root):
    x=json.loads((Path(root)/"inputs.json").read_text()); samples=x["samples"]
    if len(samples)!=229 or sum(len(s["blocks"]) for s in samples)!=1203: raise RuntimeError("frozen population mismatch")
    return samples

def load_gt(path):
    a=json.loads(Path(path).read_text()); gt={}
    for r in a["per_word"]: gt.setdefault(str(r["sample_id"]),[]).append(r["gt_text"])
    if sum(map(len,gt.values()))!=849: raise RuntimeError("GT denominator mismatch")
    return gt

def gen_path(root,sid): return Path(root)/"generations"/f"{int(sid):04d}.json"

def infer(a):
    samples=load_inputs(a.source_root)
    snapshot=Path(a.hf_cache)/("models--"+MODEL_ID.replace("/","--"))/"snapshots"/REVISION
    if not snapshot.exists(): raise FileNotFoundError(snapshot)
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
    device=torch.device(a.device)
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    if not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 required")
    out=Path(a.output_dir); (out/"generations").mkdir(parents=True,exist_ok=True)
    pending=[s for s in samples if not gen_path(out,s["sample_id"]).exists()]
    print(f"[infer] cached={len(samples)-len(pending)} pending={len(pending)}",flush=True)
    if not pending:return
    torch.cuda.set_device(device); torch.manual_seed(0); torch.backends.cudnn.benchmark=False
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to(device).eval()
    cfg=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=256,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    completed=len(samples)-len(pending)
    for s in pending:
        sid=str(s["sample_id"]); prompt=make_prompt(s)
        ids=tok.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,return_tensors="pt")
        if ids.shape[-1]>8192: raise RuntimeError(f"prompt too long {sid}")
        torch.cuda.reset_peak_memory_stats(device); t0=time.perf_counter()
        with torch.inference_mode(): generated=model.generate(ids.to(device),generation_config=cfg)
        torch.cuda.synchronize(device)
        new=generated[0,ids.shape[-1]:].detach().cpu(); raw=tok.decode(new,skip_special_tokens=True)
        try:
            actions,output=parse_actions(raw,s); status="valid"; error=None
        except Exception as e:
            actions=[{"action":"KEEP","ids":[i],"output":[b["text"]]} for i,b in enumerate(s["blocks"])]
            output=[b["text"] for b in s["blocks"]]; status="fallback_v1"; error=str(e)
        payload={"sample_id":sid,"model":{"id":MODEL_ID,"revision":REVISION,"dtype":"bfloat16"},
                 "prompt":prompt,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                 "raw_output":raw,"status":status,"error":error,"input_tokens":int(ids.shape[-1]),
                 "generated_tokens":int(len(new)),"latency_seconds":time.perf_counter()-t0,
                 "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
                 "peak_reserved_bytes":torch.cuda.max_memory_reserved(device),
                 "actions":actions,"output_tokens":output}
        p=gen_path(out,sid); tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n"); tmp.replace(p)
        completed+=1; print(f"[{completed}/229] sample={sid} {status} actions={len(actions)} out_tokens={len(output)}",flush=True)

def evaluate(a):
    samples=load_inputs(a.source_root); gt=load_gt(a.audit); out=Path(a.output_dir)
    base=Counter(); sel=Counter(); statuses=Counter(); ops=Counter(); details=[]
    for s in samples:
        sid=str(s["sample_id"]); expected=gt[sid]; baseline=[b["text"] for b in s["blocks"]]
        b=lexical_counts(baseline,expected)
        for k in ("matches","predicted_tokens","gt_words","surplus","missing"): base[k]+=b[k]
        g=json.loads(gen_path(out,sid).read_text())
        try: actions,selected=parse_actions(g["raw_output"],s); status="valid"
        except Exception:
            actions=[{"action":"KEEP","ids":[i],"output":[x]} for i,x in enumerate(baseline)]; selected=baseline; status="fallback_v1"
        statuses[status]+=1
        for x in actions: ops[x["action"]]+=1
        q=lexical_counts(selected,expected)
        for k in ("matches","predicted_tokens","gt_words","surplus","missing"): sel[k]+=q[k]
        details.append({"sample_id":sid,"baseline_matches":b["matches"],"selector_matches":q["matches"],"delta":q["matches"]-b["matches"],"status":status,"actions":actions,"baseline_tokens":baseline,"selector_tokens":selected,"gt":expected})
    if base["matches"]!=503 or base["gt_words"]!=849 or base["predicted_tokens"]!=1203: raise RuntimeError(f"baseline control failed {dict(base)}")
    def finish(c):
        m,p,g=c["matches"],c["predicted_tokens"],c["gt_words"]; pr=m/p if p else 0.; rc=m/g if g else 0.
        return {**dict(c),"precision":pr,"recall":rc,"f1":2*pr*rc/(pr+rc) if pr+rc else 0.}
    B=finish(base); S=finish(sel)
    report={"classification":"PRACTICAL_APPROXIMATION_TRAIN_ONLY","scope":{"images":229,"gt_words":849,"crops":1203},
            "model":{"id":MODEL_ID,"revision":REVISION,"dtype":"bfloat16"},"caption":False,"free_text_correction":False,
            "baseline":B,"selector":S,"net_match_gain":S["matches"]-B["matches"],
            "oracle_reference":{"feasible_merge_split":627,"gain_vs_v1":124},
            "fraction_of_oracle_net_gain_captured":(S["matches"]-503)/124,
            "statuses":dict(statuses),"operations":dict(ops),
            "images_improved":sum(d["delta"]>0 for d in details),"images_worsened":sum(d["delta"]<0 for d in details),
            "images_equal":sum(d["delta"]==0 for d in details),"details":details}
    out.mkdir(parents=True,exist_ok=True); rp=out/"results.json"; rp.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("\n=== QWEN STRUCTURAL SELECTOR ===")
    print(f"V1 matches              = {B['matches']}/849")
    print(f"selector matches        = {S['matches']}/849")
    print(f"net gain                = {report['net_match_gain']:+d}")
    print(f"selector predicted toks = {S['predicted_tokens']}")
    print(f"precision               = {S['precision']:.4f}")
    print(f"recall                  = {S['recall']:.4f}")
    print(f"f1                      = {S['f1']:.4f}")
    print("oracle reference        = 627/849")
    print(f"oracle gain captured    = {report['fraction_of_oracle_net_gain_captured']:.4f}")
    print(f"statuses                = {dict(statuses)}")
    print(f"operations              = {dict(ops)}")
    print(f"images +/-/=            = {report['images_improved']}/{report['images_worsened']}/{report['images_equal']}")
    print("JSON",rp)

def self_test():
    s={"blocks":[{"bbox":[0,0,1,1],"text":"F"},{"bbox":[1,0,2,1],"text":"lu"},{"bbox":[2,0,3,1],"text":"or"},{"bbox":[3,0,4,1],"text":"ide"},{"bbox":[0,2,4,3],"text":"USEONLY"}]}
    _,o=parse_actions("MERGE 0 1 2 3\nSPLIT 4 3",s); assert o==["Fluoride","USE","ONLY"]
    s2={"blocks":[{"bbox":[0,0,1,1],"text":"AB"},{"bbox":[1,0,2,1],"text":"CD"}]}
    _,o2=parse_actions("MERGE_SPLIT 0 1 | 1 3",s2); assert o2==["A","BC","D"]
    for bad in ("KEEP 0\nKEEP 0\nKEEP 1\nKEEP 2\nKEEP 3\nKEEP 4","MERGE 0 1 -> Flu"):
        try: parse_actions(bad,s); raise AssertionError("bad output accepted")
        except ValueError: pass
    print("SELF-TEST OK")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["infer","evaluate","self-test"])
    ap.add_argument("--source-root",default="artifacts/diagnostics/geometry-context-reconstruction-train-v1")
    ap.add_argument("--audit",default="artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json")
    ap.add_argument("--output-dir",default="artifacts/diagnostics/qwen-structural-selector-train-v1")
    ap.add_argument("--hf-cache",default="/extra1/bmartelli/context-str-artifacts/huggingface"); ap.add_argument("--device",default="cuda:0")
    a=ap.parse_args()
    {"infer":infer,"evaluate":evaluate,"self-test":lambda x:self_test()}[a.stage](a)

if __name__=="__main__": main()
