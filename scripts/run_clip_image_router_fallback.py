"""Execute only whole-image DeepSolo requests from an audited CLIP router run.

Uses the qualified isolated host runtime, preserving all raw detections. This
phase does not pretend that selecting one DeepSolo string is scene evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

from deepsolo.adapter import DeepSoloSubprocessAdapter
from deepsolo.candidates import verify_host_source_identity
from deepsolo.contracts import DeepSoloResponse
from scripts.crop_clip_runtime import sha, write_json
from scripts.run_clip_high_recall_admission import seal

REPO = Path(__file__).resolve().parents[1]


def validate_response(response, cfg):
    value = DeepSoloResponse.from_dict(response).to_dict()
    expected = {"device":cfg["device"],"score_threshold":cfg["score_threshold"],"lexicon_type":None}
    if (not value["success"] or value["artifact"]["checkpoint_sha256"] != cfg["checkpoint_sha256"]
        or value["artifact"]["candidate"]["artifact_id"] != cfg["candidate"]
        or value["artifact"]["inference_options"] != expected
        or value["placement"]["device"] != cfg["device"]
        or value["reduction_policy"] != "reading_order_space_join"):
        raise RuntimeError("DeepSolo returned failed or incompatible response")
    return value


def run(args):
    if args.config.resolve().parent != REPO/"configs/approximate" or args.paths.resolve().parent != REPO/"configs/approximate":
        raise ValueError("isolated APPROXIMATE fallback config/paths required")
    cfg, paths = [json.loads(p.read_text()) for p in (args.config,args.paths)]
    if (args.mode != "approximate" or cfg["reproduction_mode"] != "approximate"
        or cfg["claim_label"] != "approximate reproduction" or cfg["table_5_claim"] is not False
        or cfg["device"] != "cuda:0" or not 0 <= cfg["score_threshold"] <= 1):
        raise ValueError("invalid fallback protocol")
    router = args.router_run.resolve()
    plan_path, request_path = router/"resolved_configuration.json",router/"routing_requests.json"
    plan, requests = [json.loads(p.read_text()) for p in (plan_path,request_path)]
    report_path = router/"results.json"
    if requests["plan_sha256"] != sha(plan_path) or requests["result_sha256"] != sha(report_path):
        raise RuntimeError("router decision evidence changed")
    if plan["reproduction_mode"] != "approximate" or plan["result_class"] == "smoke-test result":
        raise ValueError("use a complete APPROXIMATE router run")
    family = args.family or requests["selected_family"]
    if family in requests["families"]:
        rows = requests["families"][family]
    else:
        # Every family threshold was selected on calibration only and preserved.
        from recognition.clip_image_router import accepted
        report = json.loads(report_path.read_text())
        if family not in report["choices"]:
            raise ValueError("unknown calibrated routing family")
        image_scores = json.loads((router/"scores_by_image.json").read_text())["scores"]
        threshold = report["choices"][family]["threshold"]
        rows = [{**r,"action":"accept_parseq" if accepted(image_scores[r["sample_id"]][family],threshold) else "deepsolo"}
                for r in requests["families"][requests["selected_family"]]]
    expected_ids = set(plan["split"]["calibration"]+plan["split"]["validation"])
    if len(rows) != len(expected_ids) or {r["sample_id"] for r in rows} != expected_ids:
        raise ValueError("routing image population differs")
    if any(r["action"] not in ("accept_parseq","deepsolo") for r in rows):
        raise ValueError("invalid routing action")
    for r in rows:
        if sha(Path(r["image_path"])) != r["image_sha256"]:
            raise RuntimeError("original routed image changed")
    runtime_path = Path(paths["deepsolo_runtime_manifest"]).resolve()
    if sha(runtime_path) != cfg["runtime_sha256"]:
        raise RuntimeError("qualified DeepSolo runtime identity changed")
    source = Path(paths["deepsolo_source_root"]).resolve()
    runtime = verify_host_source_identity(runtime_path,source)
    candidate = runtime["candidates"][cfg["candidate"]]
    if candidate["checkpoint_sha256"] != cfg["checkpoint_sha256"] or sha(Path(candidate["checkpoint_path"])) != cfg["checkpoint_sha256"]:
        raise RuntimeError("pinned DeepSolo candidate checkpoint differs")
    fallback_plan = {"schema_version":1,"reproduction_mode":"approximate","claim_label":"approximate reproduction",
                     "table_5_claim":False,"config":cfg,"paths":paths,"router_plan_sha256":sha(plan_path),
                     "routing_requests_sha256":sha(request_path),"family":family,
                     "code_sha256":{n:sha(REPO/n) for n in ("scripts/run_clip_image_router_fallback.py",
                        "recognition/clip_image_router.py","deepsolo/adapter.py","deepsolo/candidates.py",
                        "deepsolo/contracts.py","deepsolo/worker.py","deepsolo/reduction.py")},
                     "routes":[{"sample_id":r["sample_id"],"action":r["action"],"image_sha256":r["image_sha256"]} for r in rows]}
    root = args.output_root.resolve()/hashlib.sha256(json.dumps(fallback_plan,sort_keys=True).encode()).hexdigest()[:12]
    if root.is_relative_to(router) or root.is_relative_to(REPO):
        raise ValueError("use an isolated external fallback output root")
    if args.check:
        print(json.dumps({"status":"PREFLIGHT_PASS","family":family,"accept_images":sum(r["action"]=="accept_parseq" for r in rows),
                          "deepsolo_images":sum(r["action"]=="deepsolo" for r in rows),"output":str(root)}))
        return
    seal(root/"resolved_configuration.json",fallback_plan)
    adapter = DeepSoloSubprocessAdapter(Path(paths["deepsolo_python"]).absolute(),runtime_path,
                                       host_source_root=source,repository_root=REPO)
    completed = []
    for i,r in enumerate(rows,1):
        if r["action"] == "accept_parseq":
            completed.append({"sample_id":r["sample_id"],"source":"parseq",
                              "predictions":[{"bbox":w["bbox"],"text":w["raw_text"]} for w in r["rows"]]})
            continue
        path = root/"deepsolo"/(r["sample_id"].replace(":","__")+".json")
        contract = {"plan_sha256":sha(root/"resolved_configuration.json"),"sample_id":r["sample_id"],"image_sha256":r["image_sha256"]}
        try:
            if path.exists():
                record = json.loads(path.read_text())
                if record["contract"] != contract:
                    raise RuntimeError("fallback cached contract differs")
                response = DeepSoloResponse.from_dict(record["response"]).to_dict()
            else:
                response = adapter.infer(image_path=Path(r["image_path"]),candidate_id=cfg["candidate"],
                    inference_options={"device":cfg["device"],"score_threshold":cfg["score_threshold"],"lexicon_type":None},
                    reduction_policy="reading_order_space_join").to_dict()
                seal(path,{"contract":contract,"response":response})
            response = validate_response(response,cfg)
            completed.append({"sample_id":r["sample_id"],"source":"deepsolo","raw_response_path":str(path),"raw_response_sha256":sha(path)})
            print(f"[whole-image DeepSolo] {i}/{len(rows)} {r['sample_id']}",flush=True)
        except Exception as error:
            write_json(root/"failure.json",{"status":"FAILED","sample_id":r["sample_id"],"error":str(error),
                       "traceback":traceback.format_exc(),"completed_images":len(completed),
                       "worker_response":error.response.to_dict() if getattr(error,"response",None) is not None else None})
            raise
    seal(root/"completed_outputs.json",{"status":"SUCCESS","claim_label":"approximate reproduction",
         "family":family,"images":completed,"fallback_executed":True,
         "end_to_end_accuracy":None,"limitation":"Raw full-scene outputs preserved; polygon coordinate/ignore normalization evaluation required before combined accuracy claims"})
    print(json.dumps({"status":"SUCCESS","output":str(root/"completed_outputs.json")}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",required=True,choices=["approximate"])
    p.add_argument("--router-run",type=Path,required=True); p.add_argument("--family")
    p.add_argument("--config",type=Path,default=REPO/"configs/approximate/clip_image_router_fallback_v1.json")
    p.add_argument("--paths",type=Path,default=REPO/"configs/approximate/clip_image_router_fallback_local_paths_v1.json")
    p.add_argument("--output-root",type=Path,required=True); p.add_argument("--check",action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
