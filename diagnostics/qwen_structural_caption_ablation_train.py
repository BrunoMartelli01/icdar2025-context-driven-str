#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, time
from collections import Counter
from pathlib import Path

MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
REVISION="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TASKS=("split","merge")
CAPTIONS=("none","instructblip","florence2_base_ft","blip2_vitl")

SPLIT_INSTRUCTION="""You are a conservative OCR word-boundary detector.

You receive OCR crops from ONE image.
Most crops are already one complete word and MUST remain unchanged.
Your ONLY task is to identify a crop that clearly contains TWO OR MORE consecutive words with the spaces missing.

You are NOT allowed to:
- correct, replace, remove, reorder, normalize, or invent characters
- merge different crops
- output reconstructed words
- explain your decision

Allowed output:
SPLIT i p [q ...]

where i is the crop index and p/q are 1-based character boundaries AFTER those characters.

If no crop clearly needs splitting, output exactly:
NONE

Use SPLIT only when the boundary is clear from the OCR string itself.
The scene caption, when present, is FALLIBLE context only:
- it may help decide whether a split is plausible
- it is NOT a transcription
- never introduce characters or words from the caption
- never split only because the caption suggests a phrase

Examples:

CROPS:
[{"index":0,"ocr":"USEONLY"},{"index":1,"ocr":"OPEN"}]
OUTPUT:
SPLIT 0 3

CROPS:
[{"index":0,"ocr":"DONOTENTER"}]
OUTPUT:
SPLIT 0 2 5

CROPS:
[{"index":0,"ocr":"OPEN"},{"index":1,"ocr":"A12B"},{"index":2,"ocr":"Bookshop"}]
OUTPUT:
NONE

CROPS:
[{"index":0,"ocr":"NOENTRY"},{"index":1,"ocr":"EXIT"}]
OUTPUT:
SPLIT 0 2

Return only SPLIT lines or NONE.
No JSON. No bullets. No explanation."""

MERGE_INSTRUCTION="""You are a conservative OCR crop-merging controller.

You receive OCR crops from ONE image in frozen reading order.
Most crops already represent complete words and MUST remain unchanged.

Your ONLY task is to identify cases where TWO OR MORE separate OCR crops are clearly fragments of ONE word.

You are NOT allowed to:
- correct, replace, remove, normalize, or invent characters
- split a crop
- output reconstructed words
- explain your decision

Allowed output:
MERGE i j [k ...]

The system will concatenate the listed OCR strings EXACTLY in the order you provide.
All crops not mentioned in a MERGE are automatically kept unchanged.

If no merge is clearly needed, output exactly:
NONE

The scene caption, when present, is FALLIBLE context only:
- it may help decide whether fragments plausibly belong to one word
- it is NOT a transcription
- never introduce characters or words from the caption
- never merge two complete words merely because the caption suggests a phrase

Important:
- Use each crop in at most one MERGE.
- Merge only fragments that clearly belong to one word.
- Do not merge two complete neighboring words.
- Prefer NONE when uncertain.

Examples:

CROPS:
[{"index":0,"ocr":"F"},{"index":1,"ocr":"lu"},{"index":2,"ocr":"or"},{"index":3,"ocr":"ide"}]
OUTPUT:
MERGE 0 1 2 3

CROPS:
[{"index":0,"ocr":"OPEN"},{"index":1,"ocr":"EXIT"}]
OUTPUT:
NONE

CROPS:
[{"index":0,"ocr":"CONDI"},{"index":1,"ocr":"TIONED"},{"index":2,"ocr":"BEER"}]
OUTPUT:
MERGE 0 1

CROPS:
[{"index":0,"ocr":"NO"},{"index":1,"ocr":"ENTRY"}]
OUTPUT:
NONE

Return only MERGE lines or NONE.
No JSON. No bullets. No explanation."""

def lexical_counts(predicted, expected):
    pred=Counter(" ".join(predicted).casefold().split())
    gt=Counter(x.casefold() for x in expected)
    m=sum((pred&gt).values()); p=sum(pred.values()); g=sum(gt.values())
    pr=m/p if p else 0.; rc=m/g if g else 0.
    return {"matches":m,"predicted_tokens":p,"gt_words":g,"precision":pr,"recall":rc,
            "f1":2*pr*rc/(pr+rc) if pr+rc else 0.,
            "surplus":sum((pred-gt).values()),"missing":sum((gt-pred).values())}

def caption_text(sample, name):
    if name=="none":
        return None
    caps=sample.get("captions",{})
    aliases={
        "instructblip":["instructblip","InstructBLIP"],
        "florence2_base_ft":["florence2_base_ft","Florence-2-base-ft","florence2-base-ft"],
        "blip2_vitl":["blip2_vitl","blip2-vitl","BLIP2 pretrain_vitL","blip2-pretrain-vitL"],
    }
    for k in aliases[name]:
        if k in caps:
            return caps[k]
    raise KeyError(f"caption {name} not found; available={sorted(caps)}")

def make_prompt(sample,task,caption):
    obs=[{"index":i,"bbox":b["bbox"],"ocr":b["text"]} for i,b in enumerate(sample["blocks"])]
    inst=SPLIT_INSTRUCTION if task=="split" else MERGE_INSTRUCTION
    payload="CROPS:\n"+json.dumps(obs,ensure_ascii=False,separators=(",",":"))
    c=caption_text(sample,caption)
    if c is not None:
        payload += "\n\nFALLIBLE SCENE CAPTION:\n"+c
    return inst+"\n\n"+payload+"\n\nOUTPUT:\n"

def apply_split(text,pos):
    if not pos or pos!=sorted(set(pos)): raise ValueError("bad split boundaries")
    if pos[0]<=0 or pos[-1]>=len(text): raise ValueError("split boundary outside string")
    out=[]; start=0
    for p in pos:
        out.append(text[start:p]); start=p
    out.append(text[start:])
    if any(x=="" for x in out): raise ValueError("empty token")
    return out

def parse_split(raw,sample):
    text=raw.strip()
    if text=="NONE":
        return [],[b["text"] for b in sample["blocks"]]
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    if not lines: raise ValueError("empty output")
    seen=set(); splits={}
    for line in lines:
        parts=line.split()
        if len(parts)<3 or parts[0]!="SPLIT" or any(not x.isdigit() for x in parts[1:]):
            raise ValueError("invalid SPLIT line")
        i=int(parts[1]); pos=[int(x) for x in parts[2:]]
        if i<0 or i>=len(sample["blocks"]) or i in seen: raise ValueError("bad/reused crop id")
        seen.add(i); splits[i]=apply_split(sample["blocks"][i]["text"],pos)
    out=[]; actions=[]
    for i,b in enumerate(sample["blocks"]):
        if i in splits:
            out.extend(splits[i]); actions.append({"action":"SPLIT","id":i,"output":splits[i]})
        else:
            out.append(b["text"])
    return actions,out

def parse_merge(raw,sample):
    text=raw.strip()
    if text=="NONE":
        return [],[b["text"] for b in sample["blocks"]]
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    if not lines: raise ValueError("empty output")
    n=len(sample["blocks"]); used=set(); merges=[]
    for line in lines:
        if not re.fullmatch(r"MERGE(?: \d+){2,}",line):
            raise ValueError("invalid MERGE line")
        ids=[int(x) for x in line.split()[1:]]
        if len(set(ids))!=len(ids): raise ValueError("duplicate id in merge")
        if any(i<0 or i>=n for i in ids): raise ValueError("unknown crop id")
        if any(i in used for i in ids): raise ValueError("crop reused")
        if ids!=sorted(ids): raise ValueError("merge ids not in reading order")
        used.update(ids); merges.append(ids)
    start={ids[0]:ids for ids in merges}; members={i for ids in merges for i in ids}
    out=[]; actions=[]; i=0
    while i<n:
        if i in start:
            ids=start[i]; token="".join(sample["blocks"][j]["text"] for j in ids)
            out.append(token); actions.append({"action":"MERGE","ids":ids,"output":[token]}); i+=1
        elif i in members:
            i+=1
        else:
            out.append(sample["blocks"][i]["text"]); i+=1
    return actions,out

def parse_output(raw,sample,task):
    return parse_split(raw,sample) if task=="split" else parse_merge(raw,sample)

def load_inputs(root):
    x=json.loads((Path(root)/"inputs.json").read_text()); s=x["samples"]
    if len(s)!=229 or sum(len(x["blocks"]) for x in s)!=1203: raise RuntimeError("frozen population mismatch")
    # Fail fast if caption keys are not present.
    for name in CAPTIONS[1:]:
        caption_text(s[0],name)
    return s

def load_gt(path):
    a=json.loads(Path(path).read_text()); gt={}
    for r in a["per_word"]: gt.setdefault(str(r["sample_id"]),[]).append(r["gt_text"])
    if sum(map(len,gt.values()))!=849: raise RuntimeError("GT denominator mismatch")
    return gt

def gen_path(root,task,caption,sid):
    return Path(root)/"generations"/task/caption/f"{int(sid):04d}.json"

def infer(a):
    samples=load_inputs(a.source_root)
    snapshot=Path(a.hf_cache)/("models--"+MODEL_ID.replace("/","--"))/"snapshots"/REVISION
    if not snapshot.exists(): raise FileNotFoundError(snapshot)
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
    device=torch.device(a.device)
    if device.type!="cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    if not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 required")
    out=Path(a.output_dir)
    requests=[(s,t,c) for t in TASKS for c in CAPTIONS for s in samples]
    pending=[x for x in requests if not gen_path(out,x[1],x[2],x[0]["sample_id"]).exists()]
    print(f"[infer] total={len(requests)} cached={len(requests)-len(pending)} pending={len(pending)}",flush=True)
    if not pending:return
    torch.cuda.set_device(device); torch.manual_seed(0); torch.backends.cudnn.benchmark=False
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to(device).eval()
    cfg=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=96,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    done=len(requests)-len(pending)
    for s,task,caption in pending:
        sid=str(s["sample_id"]); prompt=make_prompt(s,task,caption)
        chat=tok.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,tokenize=False)
        enc=tok(chat,return_tensors="pt",add_special_tokens=False)
        if enc["input_ids"].shape[-1]>8192: raise RuntimeError(f"prompt too long {task}/{caption}/{sid}")
        t0=time.perf_counter()
        with torch.inference_mode():
            g=model.generate(input_ids=enc["input_ids"].to(device),attention_mask=enc["attention_mask"].to(device),generation_config=cfg)
        torch.cuda.synchronize(device)
        new=g[0,enc["input_ids"].shape[-1]:].detach().cpu(); raw=tok.decode(new,skip_special_tokens=True)
        try:
            actions,output=parse_output(raw,s,task); status="valid"; error=None
        except Exception as e:
            actions=[]; output=[b["text"] for b in s["blocks"]]; status="fallback_v1"; error=str(e)
        payload={"sample_id":sid,"task":task,"caption":caption,"raw_output":raw,"status":status,"error":error,
                 "actions":actions,"output_tokens":output,
                 "model":{"id":MODEL_ID,"revision":REVISION,"dtype":"bfloat16"},
                 "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                 "input_tokens":int(enc["input_ids"].shape[-1]),"generated_tokens":int(len(new)),
                 "latency_seconds":time.perf_counter()-t0}
        p=gen_path(out,task,caption,sid); p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n"); tmp.replace(p)
        done+=1
        print(f"[{done}/{len(requests)}] {task}/{caption} sample={sid} {status} actions={len(actions)}",flush=True)

def evaluate(a):
    samples=load_inputs(a.source_root); gt=load_gt(a.audit); out=Path(a.output_dir)
    base=Counter()
    for s in samples:
        sid=str(s["sample_id"]); b=lexical_counts([x["text"] for x in s["blocks"]],gt[sid])
        for k in ("matches","predicted_tokens","gt_words","surplus","missing"): base[k]+=b[k]
    if base["matches"]!=503 or base["predicted_tokens"]!=1203 or base["gt_words"]!=849:
        raise RuntimeError(f"baseline control failed {dict(base)}")
    results={}
    for task in TASKS:
        for caption in CAPTIONS:
            agg=Counter(); statuses=Counter(); actions_n=0; improved=worsened=equal=0
            for s in samples:
                sid=str(s["sample_id"]); baseline=[x["text"] for x in s["blocks"]]
                b=lexical_counts(baseline,gt[sid])
                g=json.loads(gen_path(out,task,caption,sid).read_text())
                try:
                    actions,selected=parse_output(g["raw_output"],s,task); status="valid"
                except Exception:
                    actions=[]; selected=baseline; status="fallback_v1"
                statuses[status]+=1; actions_n+=len(actions)
                q=lexical_counts(selected,gt[sid])
                for k in ("matches","predicted_tokens","gt_words","surplus","missing"): agg[k]+=q[k]
                d=q["matches"]-b["matches"]
                improved+=d>0; worsened+=d<0; equal+=d==0
            m,p,g=agg["matches"],agg["predicted_tokens"],agg["gt_words"]
            pr=m/p if p else 0.; rc=m/g if g else 0.; f1=2*pr*rc/(pr+rc) if pr+rc else 0.
            oracle=577 if task=="split" else 551
            gain_oracle=74 if task=="split" else 48
            results[f"{task}/{caption}"]={
                "matches":m,"predicted_tokens":p,"precision":pr,"recall":rc,"f1":f1,
                "net_gain":m-503,"oracle_reference":oracle,
                "oracle_gain_captured":(m-503)/gain_oracle,
                "statuses":dict(statuses),"actions":actions_n,
                "images_improved":improved,"images_worsened":worsened,"images_equal":equal}
    report={"classification":"PRACTICAL_APPROXIMATION_TRAIN_ONLY","baseline":{"matches":503,"predicted_tokens":1203,"gt_words":849},"results":results}
    out.mkdir(parents=True,exist_ok=True); rp=out/"results.json"; rp.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("\n=== STRUCTURAL CAPTION ABLATION ===")
    print("condition                         matches   gain  tokens    F1     actions  valid/fallback  +/-/=")
    for key,v in results.items():
        st=v["statuses"]
        print(f"{key:32s} {v['matches']:3d}/849 {v['net_gain']:+5d} {v['predicted_tokens']:6d} {v['f1']:.4f} {v['actions']:8d} {st.get('valid',0):3d}/{st.get('fallback_v1',0):3d} {v['images_improved']:3d}/{v['images_worsened']:3d}/{v['images_equal']:3d}")
    print("JSON",rp)

def inspect(a):
    s=load_inputs(a.source_root)[0]
    print("caption keys:",sorted(s.get("captions",{})))
    for c in CAPTIONS:
        x=caption_text(s,c)
        print(c,":",None if x is None else x[:160].replace("\n"," "))

def self_test():
    s={"blocks":[{"bbox":[0,0,1,1],"text":"USEONLY"},{"bbox":[1,0,2,1],"text":"OPEN"}],"captions":{"InstructBLIP":"x","Florence-2-base-ft":"y","BLIP2 pretrain_vitL":"z"}}
    _,o=parse_split("SPLIT 0 3",s); assert o==["USE","ONLY","OPEN"]
    _,o=parse_merge("MERGE 0 1",s); assert o==["USEONLYOPEN"]
    assert caption_text(s,"instructblip")=="x"
    print("SELF-TEST OK")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["infer","evaluate","inspect","self-test"])
    ap.add_argument("--source-root",default="artifacts/diagnostics/geometry-context-reconstruction-train-v1")
    ap.add_argument("--audit",default="artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json")
    ap.add_argument("--output-dir",default="artifacts/diagnostics/structural-caption-ablation-train-v1")
    ap.add_argument("--hf-cache",default="/extra1/bmartelli/context-str-artifacts/huggingface"); ap.add_argument("--device",default="cuda:0")
    a=ap.parse_args()
    if a.stage=="self-test": self_test()
    elif a.stage=="inspect": inspect(a)
    elif a.stage=="infer": infer(a)
    else: evaluate(a)

if __name__=="__main__": main()
