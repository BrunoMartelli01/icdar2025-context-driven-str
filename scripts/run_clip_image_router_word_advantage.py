"""Fixed native/CLIP word-versus-background whole-image router.

Efficient inference of the same signal tested in the complete router battery.
Reuse is explicit and byte-checked; other CLIP features are never substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from recognition.clip_crop_admission import percentile
from recognition.clip_image_router import accepted, choose_threshold, partition, summarize
from recognition.clip_word_verifier import unit
from recognition.group_word_selection_runtime import square_canvas
from scripts import run_clip_image_router as battery
from scripts.crop_clip_runtime import _embeddings, _cuda_peak, sha, write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _correct
from scripts.run_clip_high_recall_admission import payload_digest, seal
from scripts.run_clip_word_verification import model_identity, paths_profile

REPO = battery.REPO
FAMILY = "native_word_advantage"


class WordAdvantageScorer:
    def __init__(self,model,processor,cfg,device):
        self.model,self.processor,self.cfg,self.device=model,processor,cfg,device
        self.texts={}
        prompts=cfg["textness_prompts"]
        v=_embeddings(model,processor,images=None,texts=prompts["positive"]+prompts["negative"],
                      device=device,batch_size=cfg["execution"]["batch_size"])
        self.negative=unit(v[len(prompts["positive"]):].mean(axis=0))

    def score(self,rows,crops):
        if len(rows)!=len(crops):raise ValueError("crop population differs")
        missing=sorted({r["raw_text"] for r in rows if r["raw_text"].strip()}-self.texts.keys())
        prompts=[t.format(text=w) for w in missing for t in self.cfg["word_templates"]]
        if prompts:
            v=_embeddings(self.model,self.processor,images=None,texts=prompts,device=self.device,
                          batch_size=self.cfg["execution"]["batch_size"])
            n=len(self.cfg["word_templates"])
            self.texts.update({w:unit(v[i*n:(i+1)*n].mean(axis=0)) for i,w in enumerate(missing)})
        active=[i for i,r in enumerate(rows) if r["raw_text"].strip()]
        def processor(*,images,return_tensors):
            return self.processor(images=images,return_tensors=return_tensors,input_data_format="channels_last")
        images=[square_canvas(crops[i]) for i in active]
        v=_embeddings(self.model,processor,images=images,texts=None,device=self.device,
                      batch_size=self.cfg["execution"]["batch_size"]) if images else []
        output=dict.fromkeys(range(len(rows)))
        for i,vector in zip(active,v):
            output[i]=float(vector@(self.texts[rows[i]["raw_text"]]-self.negative))
        return [output[i] for i in range(len(rows))]


def run(args):
    cfg,paths,observed=battery.load_inputs(args)
    if cfg.get("fixed_rule")!=FAMILY:raise ValueError("explicit fixed-rule config required")
    paths["output_root"]=args.output_root.resolve() if args.output_root else Path(cfg["output_root"]).resolve()
    if paths["output_root"].is_relative_to(REPO) or any(paths["output_root"].is_relative_to(paths[k]) for k in ("source_run","segmentation_run")):
        raise ValueError("isolated external output root required")
    spec={**json.loads((REPO/cfg["model_config"]).read_text())["models"][args.model],"weight_sha256":cfg["model_weight_sha256"][args.model]}
    identity=model_identity(paths_profile(paths["model_paths_profile"])["models"][args.model],spec)
    if args.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable: select CPU explicitly or run on GPU host")
    reuse=None
    if args.reuse_score_run:
        reuse_plan=json.loads((args.reuse_score_run/"resolved_configuration.json").read_text())
        if reuse_plan["model"]!=identity or reuse_plan["device"]!=args.device:
            raise ValueError("reuse must have identical model and execution device")
        for field in ("source_recognition_sha256","source_geometry_sha256","source_checkpoint_sha256","word_templates","textness_prompts","execution"):
            if reuse_plan["config"][field]!=cfg[field]:raise ValueError("reuse scientific input differs: "+field)
        for name,digest in reuse_plan["code_sha256"].items():
            if sha(REPO/name)!=digest:raise ValueError("reuse source code changed: "+name)
        reuse={"root":str(args.reuse_score_run.resolve()),"plan_sha256":sha(args.reuse_score_run/"resolved_configuration.json")}
    seed=cfg["execution"]["seed"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(cfg["execution"]["cpu_threads"])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.device.startswith("cuda"):torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats(args.device)
    ids=[r["sample_id"] for r in observed]
    split=partition(ids,cfg["calibration_images"],seed) if not args.limit else {"calibration":ids,"validation":[]}
    plan={"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
          "result_class":"smoke-test result" if args.limit else "partial-dataset result","table_5_claim":False,
          "config":cfg,"device":args.device,"model":identity,"split":split,"reuse":reuse,
          "code_sha256":{n:sha(REPO/n) for n in (*battery.CODE,"scripts/run_clip_image_router_word_advantage.py")},
          "input_digest":hashlib.sha256(json.dumps(observed,sort_keys=True).encode()).hexdigest(),
          "software":{"torch":torch.__version__,"numpy":np.__version__,"pillow":Image.__version__,"transformers":__import__("transformers").__version__},
          "gpu":torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
          "cuda_build":torch.version.cuda,"deterministic_algorithms":True,"tf32":False}
    root=paths["output_root"]/args.model/hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    seal(root/"resolved_configuration.json",plan);plan_sha=sha(root/"resolved_configuration.json")
    scorer,model=None,None
    shards=[]
    try:
        for i,s in enumerate(observed,1):
            path=root/"scores"/(s["sample_id"].replace(":","__")+".json")
            if path.exists():
                shard=json.loads(path.read_text())
                if shard["plan_sha256"]!=plan_sha or shard["payload_sha256"]!=payload_digest(shard):raise ValueError("cached score contract changed")
            else:
                start=time.monotonic();old_path=args.reuse_score_run/"scores"/path.name if args.reuse_score_run else None
                origin=None
                if old_path and old_path.exists():
                    old=json.loads(old_path.read_text())
                    if old["plan_sha256"]!=reuse["plan_sha256"] or old["payload_sha256"]!=payload_digest(old):raise ValueError("reused score payload differs")
                    if [{k:v for k,v in r.items() if k!='clip'} for r in old["rows"]]!=s["rows"]:raise ValueError("reused exact OCR/image crops differ")
                    values=[r["clip"]["clip_word_advantage"] if r["clip"] else None for r in old["rows"]]
                    origin={"path":str(old_path),"sha256":sha(old_path)}
                else:
                    if scorer is None:
                        from transformers import CLIPModel,CLIPProcessor
                        model,info=CLIPModel.from_pretrained(identity["local_path"],local_files_only=True,
                            use_safetensors=spec["weight_file"].endswith("safetensors"),output_loading_info=True)
                        if any(info.values()):raise ValueError("CLIP checkpoint key mismatch: "+str(info))
                        model=model.eval().float().to(args.device)
                        scorer=WordAdvantageScorer(model,CLIPProcessor.from_pretrained(identity["local_path"],local_files_only=True),cfg,args.device)
                    if sha(Path(s["image_path"]))!=s["image_sha256"]:raise ValueError("source original RGB changed")
                    with Image.open(s["image_path"]) as opened:rgb=np.asarray(opened.convert("RGB"))
                    crops=[]
                    for row in s["rows"]:
                        x1,y1,x2,y2=row["bbox"];crop=np.ascontiguousarray(rgb[y1:y2,x1:x2])
                        if crop_array_hash(crop)!=row["crop_rgb_sha256"]:raise ValueError("exact RGB crop changed")
                        crops.append(crop)
                    values=scorer.score(s["rows"],crops)
                if any(v is not None and not np.isfinite(v) for v in values):raise ValueError("nonfinite word contrast")
                shard={**s,"rows":[{**r,"clip_word_advantage":v} for r,v in zip(s["rows"],values)],"plan_sha256":plan_sha,
                       "reuse":origin,"ground_truth_in_scoring":False,"runtime":{"seconds":time.monotonic()-start,"cuda_peak":_cuda_peak(args.device),
                       "device":args.device,"precision":"float32","batch_size":cfg["execution"]["batch_size"],"processed_shape":[3,224,224],"model_set":[spec["model_id"]] if origin is None else []}}
                shard["payload_sha256"]=payload_digest(shard);seal(path,shard)
            shards.append(shard)
            if i%25==0 or i==len(observed):print(f"[fixed CLIP router {args.model}] {i}/{len(observed)}",flush=True)
        del model,scorer
        if args.device.startswith("cuda"):torch.cuda.empty_cache()
        cal=[r for s in shards if s["sample_id"] in split["calibration"] for r in s["rows"] if r["clip_word_advantage"] is not None]
        reference={"native":sorted(r["confidence"] for r in cal),"word_advantage":sorted(r["clip_word_advantage"] for r in cal)}
        scores={s["sample_id"]:{FAMILY:min((percentile(reference['native'],r['confidence'])*percentile(reference['word_advantage'],r['clip_word_advantage']))**.5 for r in s['rows'])
                               if s["rows"] and all(r["raw_text"].strip() and r["clip_word_advantage"] is not None for r in s["rows"]) else None} for s in shards}
        seal(root/"scores_by_image.json",{"scores":scores,"rank_reference":reference,"ground_truth_in_scores":False,"fit_sample_ids":split["calibration"]})
        report=battery.checked(paths["source_run"]/"recognition_results.json",cfg["source_recognition_sha256"])
        evaluated=[]
        for r in report["branches"][cfg["source_variant"]][cfg["source_method"]]["per_image"]:
            if r["sample_id"] not in ids:continue
            care=[p for p in r["predictions_all"] if p["instance_id"] not in r["ignored_prediction_ids"]]
            matched=_correct(care,r["ground_truth"],normalize=False)
            if len(matched)!=r["counts"]["exact_casefold_iou_ge_50"]:raise ValueError("baseline replay differs")
            evaluated.append(dict(sample_id=r["sample_id"],tp=len(matched),words=len(care),gt_words=len(r["ground_truth"]),outputs_all=len(r["predictions_all"])))
        chosen=choose_threshold([r for r in evaluated if r["sample_id"] in split["calibration"]],scores,FAMILY,cfg["target_accuracy"])
        threshold=chosen["threshold"]
        chosen.update(validation=summarize([r for r in evaluated if r["sample_id"] in split["validation"]],scores,FAMILY,threshold),
                      all_images=summarize(evaluated,scores,FAMILY,threshold))
        baseline=summarize(evaluated,scores,None,None)
        if not args.limit and (baseline["correct_words"]!=cfg["expected_correct_words"] or baseline["detected_words"]!=cfg["expected_care_outputs"] or baseline["accepted_gt_words"]!=cfg["expected_gt_words"]):raise ValueError("frozen source denominator changed")
        result={"status":"SUCCESS","claim_label":"approximate reproduction","reproduction_mode":"approximate","table_5_claim":False,
                "plan_sha256":plan_sha,"model":args.model,"baseline":baseline,"split":split,"selected_family":FAMILY,"choices":{FAMILY:chosen},
                "selection_uses_validation_labels":False,"family_selected_after_b32_validation":True,"test_loaded":False,
                "per_image":evaluated,"reused_images":sum(s["reuse"] is not None for s in shards),
                "result_class":plan["result_class"],"elapsed_seconds":sum(s["runtime"]["seconds"] for s in shards)}
        seal(root/"results.json",result)
        routes=[{**s,"action":"accept_parseq" if accepted(scores[s["sample_id"]][FAMILY],threshold) else "deepsolo"} for s in observed]
        seal(root/"routing_requests.json",{"plan_sha256":plan_sha,"result_sha256":sha(root/"results.json"),"selected_family":FAMILY,"families":{FAMILY:routes},
                                           "thresholds":{FAMILY:threshold},"fallback_status":"NOT_EXECUTED","selection_uses_validation_labels":False,
                                           "family_selected_after_b32_validation":True})
        print(json.dumps({"report":str(root/"results.json"),"threshold":threshold,"validation":chosen["validation"],"all_images":chosen["all_images"]}),flush=True)
        return root
    except Exception as error:
        write_json(root/"failure.json",{"status":"FAILED","error":str(error),"error_type":type(error).__name__,"device":args.device,
                   "out_of_memory":isinstance(error,(MemoryError,torch.cuda.OutOfMemoryError)),"cuda_peak":_cuda_peak(args.device)})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",required=True,choices=["approximate"])
    p.add_argument("--config",type=Path,default=REPO/"configs/approximate/clip_image_router_word_advantage_v1.json")
    p.add_argument("--paths",type=Path,default=battery.PATHS)
    p.add_argument("--model",choices=["vit_b32","vit_b16","vit_l14"],default="vit_l14")
    p.add_argument("--device",choices=["cpu","cuda:0"],default="cuda:0")
    p.add_argument("--reuse-score-run",type=Path);p.add_argument("--output-root",type=Path);p.add_argument("--limit",type=int)
    run(p.parse_args())


if __name__=="__main__":main()
