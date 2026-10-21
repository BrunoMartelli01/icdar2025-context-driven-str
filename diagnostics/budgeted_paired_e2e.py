"""Frozen InstructBLIP paired routing: CPU preparation/evaluation, explicit PAPER export.

No segmentation/PARSeq inference, threshold search, training or default changes.
Word accuracy is image-local exact casefold token-multiset recall, not Table 5.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import time

from diagnostics.caption_routing_train import checked, seal_write, route_scores
from diagnostics.joined_t3 import checked_shard, join_crop_transcriptions
from diagnostics.lavis_captioner_candidates import digest
from scripts.evaluate_icdar13_e2e_faithful_vs_balanced import _sha as sha
from scoring.budgeted_paired import route

TRAIN_REPORT_SHA = "389219133531817df915413a14ef01f727f0e6d710dc1d1eb177ad843e43b46b"
TRAIN_GT_SHA = "38afc5fbb93bf6c6cc9970ad90454509f1089d6249a56447e6f60aecfbf2cc1b"
CALIBRATION_SHA = "2ee15ee137fdcf68605f001fa05f523a4c17b42673ff9848172dc25668f1a9c2"
TEST_REPORT_SHA = "10ebcef601ed07ebdc0708c84d4d79fc4b6d9fdad0989cbcddb2d80cfc162529"
DEEP_SHA = "ac2a01b3d8df0ee42e4632ff57a0d3a20a5706fb0ac1c5ae33b2455d0d90da68"
RUNTIME_SHA = "c52a65b7d844cd35bcf6e38e3cb58c5194851c8327cbdaba9b313869ec2c7ffa"
SEGMENTATION_SHA = "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2"


def read(path):
    return json.loads(Path(path).read_text())


def require_hash(path, expected):
    if sha(path) != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}")


def validate_deep(response):
    if response.get("success") is not True or response.get("error") is not None:
        raise RuntimeError("failed DeepSolo response")
    artifact = response["artifact"]
    if artifact["checkpoint_sha256"] != DEEP_SHA:
        raise RuntimeError("wrong DeepSolo checkpoint")
    if artifact["inference_options"] != {"device": "cuda:0", "lexicon_type": None, "score_threshold": None}:
        raise RuntimeError("DeepSolo inference options changed")
    if response["reduction_policy"] != "highest_score":
        raise RuntimeError("highest_score control required; all raw strings are evaluated separately")
    if not isinstance(response["recognized_strings"], list) or not all(
            isinstance(t, str) for t in response["recognized_strings"]):
        raise RuntimeError("invalid DeepSolo strings")
    if not isinstance(response["selected_output"], str):
        raise RuntimeError("invalid DeepSolo selected output")
    return response


def match_words(text, words):
    """Duplicates capped, omissions count as errors; no spatial/oracle assignment."""
    if any(not isinstance(w, str) or len(w.split()) != 1 for w in words):
        raise ValueError("requires one nonempty token per GT word")
    expected = Counter(w.casefold() for w in words)
    predicted = Counter(text.casefold().split())
    correct = sum((expected & predicted).values())
    return {"gt_words": len(words), "correct_words": correct,
            "predicted_tokens": sum(predicted.values()),
            "extra_wrong_tokens": sum(predicted.values()) - correct,
            "all_gt_recovered": correct == len(words),
            "exact_image_word_multiset": predicted == expected}


def aggregate(records):
    count = len(records)
    totals = {k: sum(r[k] for r in records) for k in
              ("gt_words", "correct_words", "predicted_tokens", "extra_wrong_tokens",
               "all_gt_recovered", "exact_image_word_multiset")}
    return {"images": count, **totals,
            "word_accuracy": totals["correct_words"]/totals["gt_words"] if totals["gt_words"] else None,
            "word_precision": totals["correct_words"]/totals["predicted_tokens"] if totals["predicted_tokens"] else None,
            "complete_image_accuracy": totals["exact_image_word_multiset"]/count if count else None}


def validate_route(row, tau):
    args = {k: row[k] for k in ("T1", "T2", "T3", "S1", "S3", "L1", "L3")}
    decision = route(tau=tau, **args).to_dict()
    if abs(decision["C"]-row["C"]) > 1e-12 or decision["selected_output"] != row["selected_output"]:
        raise RuntimeError("cached paired decision mismatch")
    return decision


def prepare(a):
    require_hash(a.calibration, CALIBRATION_SHA)
    calibration = read(a.calibration)
    config = read(a.config)
    for key in ("tau", "alpha", "beta", "caption_profile", "caption_comparison", "strategy"):
        if config[key] != calibration["configuration"][key]:
            raise RuntimeError(f"frozen calibration/config mismatch: {key}")
    if config["reproduction_mode"] != "approximate" or calibration["scope"] != {
            "split": "TRAIN", "images": 229, "gt_words": 849}:
        raise RuntimeError("wrong calibration scope/mode")
    require_hash(a.train_root/"results.json", TRAIN_REPORT_SHA)
    source = checked(a.train_root/"results.json")
    protocol = checked(a.train_root/"protocol.json")
    if protocol["payload_sha256"] != source["contract_sha256"]:
        raise RuntimeError("TRAIN protocol identity mismatch")
    train_inputs = checked(a.train_root/"inputs.json", source["contract_sha256"])
    if digest(train_inputs["samples"]) != protocol["inputs_sha256"]:
        raise RuntimeError("TRAIN input payload digest mismatch")
    train_by_id = {r["sample_id"]: r for r in train_inputs["samples"]}
    if set(train_by_id) != {str(i) for i in range(100,329)} or len(train_inputs["samples"]) != 229:
        raise RuntimeError("TRAIN population mismatch")
    train = []
    for r in calibration["per_image"]:
        sid = r["sample_id"]
        inp = train_by_id[sid]
        image = Path(inp["image_path"])
        if not image.resolve().is_relative_to(a.data_root.resolve()):
            raise RuntimeError("TRAIN image escapes dataset")
        require_hash(image, inp["image_sha256"])
        path = a.train_root/"routes/instructblip/whole_caption"/f"{int(sid):04d}.json"
        require_hash(path, calibration["route_file_sha256"][str(path)])
        cached = checked(path, source["contract_sha256"])
        decision = validate_route(cached["image_routes"]["selected_candidate_paired"], config["tau"])
        if decision != {k:r[k] for k in decision}:
            raise RuntimeError("calibrated TRAIN routing changed")
        train.append({"sample_id": sid, "image_path": str(image), "image_sha256": inp["image_sha256"],
                      "decision": decision})
    if len(train) != 229 or len({r["sample_id"] for r in train}) != 229:
        raise RuntimeError("TRAIN route population mismatch")
    require_hash(a.train_gt_report, TRAIN_GT_SHA)
    gt_source = read(a.train_gt_report)
    gt_train = {sid: [] for sid in train_by_id}
    for w in gt_source["per_word"]:
        gt_train[w["sample_id"]].append(w["text"])
    if sum(map(len,gt_train.values())) != 849:
        raise RuntimeError("TRAIN GT count mismatch")

    require_hash(a.test_root/"results.json", TEST_REPORT_SHA)
    suite = read(a.test_root/"results.json")
    contract = suite["contract_sha256"]
    if suite["scope"] != {"split":"TEST", "images":233, "gt_words":1095}:
        raise RuntimeError("TEST population mismatch")
    if any(v != SEGMENTATION_SHA for v in (suite["protocol"]["inputs"]["checkpoint"], protocol["mask_checkpoint_sha256"])):
        raise RuntimeError("segmentation checkpoint mismatch")
    require_hash(a.recognition_manifest, suite["protocol"]["inputs"]["recognition_manifest"])
    if read(a.test_root/"artifacts/report-receipt.json") != {"contract_sha256":contract,"report_sha256":TEST_REPORT_SHA}:
        raise RuntimeError("TEST source receipt mismatch")
    stages = ("01-proposals", "02-parseq", "05-deepsolo")
    seals = {s:read(a.test_root/"artifacts/seals"/f"{s}.json") for s in stages}
    for seal in seals.values():
        if seal["contract_sha256"] != contract or set(seal["sha256"]) != {f"{i:04d}.json" for i in range(1,234)}:
            raise RuntimeError("TEST stage incomplete")
    from recognition.icdar13_annotations import load_icdar13_recognition_samples
    samples, _ = load_icdar13_recognition_samples(a.recognition_manifest, a.data_root)
    test, gt_test, deep_test, refs = [], {}, {}, {}
    for sample in samples:
        sid = sample.sample_id
        parts = {stage:checked_shard(a.test_root,stage,sid,sample.image_sha256,contract,seals) for stage in stages}
        proposal, ocr = parts["01-proposals"], parts["02-parseq"]
        if ocr["proposal_shard_sha256"] != seals["01-proposals"]["sha256"][f"{int(sid):04d}.json"]:
            raise RuntimeError("PARSeq crop source mismatch")
        crops = [{"candidate_id":p["candidate_id"], "bbox":p["bbox"],
                  "text":ocr["crops"][p["ocr_key"]]["raw_text"]} for p in proposal["v1_anchors"]]
        image = a.data_root/sample.image_path
        test.append({"sample_id":sid,"image_path":str(image.resolve()),"image_sha256":sample.image_sha256,
                     "T1":ocr["T1"]["raw_text"],"joined":join_crop_transcriptions(crops),"proposals":crops})
        gt_test[sid] = [w.transcription for w in sample.words]
        deep_test[sid] = validate_deep(parts["05-deepsolo"]["response"])
        refs[sid] = {s:seals[s]["sha256"][f"{int(sid):04d}.json"] for s in stages}
    if sum(map(len,gt_test.values())) != 1095:
        raise RuntimeError("TEST GT count mismatch")
    runtime = a.assets/"deepsolo-host/qualification/runtime.json"
    require_hash(runtime,RUNTIME_SHA)
    source_files = [Path(__file__),Path("scoring/budgeted_paired.py"),Path("scoring/strategies.py"),
                    Path("scoring/semantic.py"),Path("diagnostics/joined_t3.py"),
                    Path("diagnostics/caption_routing_train.py"),Path("localize/practical_candidate_selection_v2.py"),
                    *sorted(Path("deepsolo").glob("*.py"))]
    record = {"mode":"approximate","classification":"PRACTICAL_APPROXIMATION", "config":config,
              "calibration_sha256":CALIBRATION_SHA,"config_sha256":sha(a.config),
              "instructblip_spec":protocol["specs"]["instructblip"],
              "train_report_sha256":TRAIN_REPORT_SHA,"test_report_sha256":TEST_REPORT_SHA,
              "deep_runtime_sha256":RUNTIME_SHA,"assets":str(a.assets.resolve()),
              "data_root":str(a.data_root.resolve()),"dependencies":{str(p):sha(p) for p in source_files},
              "train_inputs_digest":digest(train),"test_inputs_digest":digest(test),"test_source_shards":refs,
              "gt_digest":digest({"TRAIN":gt_train,"TEST":gt_test}),
              "deep_test_digest":digest(deep_test),
              "word_protocol":"all raw DeepSolo strings joined; casefold whitespace multiset per image; duplicates capped; omissions wrong; no spatial matching",
              "legacy_control":"highest_score selected_output also reported separately; no prediction selected using GT",
              "table5_accuracy":None,"test_threshold_calibration":False}
    new_contract = digest(record)
    seal_write(a.output_dir/"protocol.json",record)
    seal_write(a.output_dir/"inputs.json",{"contract_sha256":new_contract,"TRAIN":train,"TEST":test})
    seal_write(a.output_dir/"gt.json",{"contract_sha256":new_contract,"TRAIN":gt_train,"TEST":gt_test})
    for row in test:
        seal_write(shard(a,"TEST","deepsolo",row),{"contract_sha256":new_contract,
                   "sample_id":row["sample_id"],"image_sha256":row["image_sha256"],
                   "response":deep_test[row["sample_id"]],"source":"verified frozen TEST suite"})
    print("[prepare] TRAIN routes 229/229; TEST v1/PARSeq/DeepSolo 233/233 verified",flush=True)
    status(a)


def shard(a, split, stage, row):
    return a.output_dir/"cache"/split/stage/f"{int(row['sample_id']):04d}.json"


def load(a):
    p = checked(a.output_dir/"protocol.json")
    for path, expected in p["dependencies"].items():
        require_hash(Path(path),expected)
    contract = p["payload_sha256"]
    inputs = checked(a.output_dir/"inputs.json",contract)
    for split in ("TRAIN","TEST"):
        if digest(inputs[split]) != p[f"{split.lower()}_inputs_digest"]:
            raise RuntimeError("frozen inputs changed")
    return p,contract,inputs


def cached(a, split, stage, row, contract):
    path = shard(a,split,stage,row)
    if not path.exists():
        return None
    value = checked(path,contract)
    if value["sample_id"] != row["sample_id"] or value["image_sha256"] != row["image_sha256"]:
        raise RuntimeError("cached output image/split mismatch")
    if stage == "deepsolo":
        validate_deep(value["response"])
    return value


def status(a):
    _,contract,inputs = load(a)
    missing = {f"{split}/{stage}":sum(cached(a,split,stage,r,contract) is None for r in inputs[split])
               for split,stage in (("TRAIN","deepsolo"),("TEST","deepsolo"),("TEST","caption"))}
    print("[missing]",json.dumps(missing),flush=True)


def export_deep(a):
    p,contract,inputs = load(a)
    pending = [r for r in inputs["TRAIN"] if cached(a,"TRAIN","deepsolo",r,contract) is None]
    if not pending:
        print("[reuse] TRAIN DeepSolo 229/229",flush=True)
        return
    if a.device != "cuda:0":
        raise ValueError("missing DeepSolo requires explicit PAPER --device cuda:0")
    from deepsolo.adapter import DeepSoloSubprocessAdapter
    from deepsolo.candidates import QUALIFIED_CANDIDATE_ID
    assets = Path(p["assets"])
    runtime = assets/"deepsolo-host/qualification/runtime.json"
    require_hash(runtime,p["deep_runtime_sha256"])
    adapter = DeepSoloSubprocessAdapter(a.deepsolo_python,runtime,
               host_source_root=assets/"deepsolo-host/DeepSolo-upstream/DeepSolo")
    for i,row in enumerate(pending,230-len(pending)):
        image = Path(row["image_path"])
        require_hash(image,row["image_sha256"])
        print(f"[DeepSolo TRAIN] {i}/229 sample={row['sample_id']} starting",flush=True)
        response = adapter.infer(image_path=image,candidate_id=QUALIFIED_CANDIDATE_ID,
                   inference_options={"device":"cuda:0","lexicon_type":None,"score_threshold":None},
                   reduction_policy="highest_score").to_dict()
        validate_deep(response)
        seal_write(shard(a,"TRAIN","deepsolo",row),{"contract_sha256":contract,
                   "sample_id":row["sample_id"],"image_sha256":row["image_sha256"],"response":response,
                   "purpose":"standalone TRAIN control, not online routing invocation count"})
        print(f"[DeepSolo TRAIN] {i}/229 saved, detections={len(response['recognized_strings'])}",flush=True)


def export_captions(a):
    p,contract,inputs = load(a)
    pending = [r for r in inputs["TEST"] if cached(a,"TEST","caption",r,contract) is None]
    if not pending:
        print("[reuse] InstructBLIP TEST 233/233",flush=True)
        return
    if a.device != "cuda:0":
        raise ValueError("missing captions require explicit PAPER --device cuda:0")
    import torch
    from PIL import Image
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("PAPER CUDA unavailable")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    spec = p["instructblip_spec"]
    snapshot = Path(p["assets"])/"huggingface"/("models--"+spec["model"].replace("/","--"))/"snapshots"/spec["revision"]
    for name,expected in spec["weights_sha256"].items():
        require_hash(snapshot/name,expected)
    processor = InstructBlipProcessor.from_pretrained(str(snapshot),local_files_only=True)
    start = time.perf_counter()
    model,loading = InstructBlipForConditionalGeneration.from_pretrained(str(snapshot),local_files_only=True,
                torch_dtype=torch.float16,low_cpu_mem_usage=True,output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
        raise RuntimeError(f"InstructBLIP strict load failure: {loading}")
    model = model.to(a.device).eval()
    metadata = {"contract_sha256":contract,"model_spec":spec,"parameters":sum(x.numel() for x in model.parameters()),
                "python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,
                "gpu":torch.cuda.get_device_name(0),"device":a.device,"dtype":"float16","batch_size":1,
                "seed":0,"cudnn_benchmark":torch.backends.cudnn.benchmark,
                "cudnn_deterministic":torch.backends.cudnn.deterministic,"loading_info":loading,
                "classification":"HARDWARE_DEVIATION: sequential FP16 GPU inference, as frozen TRAIN caption profile"}
    model_path = a.output_dir/"cache/TEST/caption/model.json"
    seal_write(model_path,metadata)
    print(f"[InstructBLIP] loaded in {time.perf_counter()-start:.2f}s",flush=True)
    with torch.inference_mode():
        for i,row in enumerate(pending,234-len(pending)):
            image = Path(row["image_path"])
            require_hash(image,row["image_sha256"])
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with Image.open(image) as source:
                encoded = processor(images=source.convert("RGB"),text=spec["prompt"],return_tensors="pt")
            encoded = {k:v.to(a.device,dtype=torch.float16) if k=="pixel_values" else v.to(a.device) for k,v in encoded.items()}
            try:
                generated = model.generate(**encoded,**spec["generation"])
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(f"InstructBLIP OOM sample={row['sample_id']}; prior completed shards preserved") from error
            text = processor.batch_decode(generated,skip_special_tokens=True)[0].strip()
            if not text:
                raise RuntimeError("empty InstructBLIP caption")
            torch.cuda.synchronize()
            seal_write(shard(a,"TEST","caption",row),{"contract_sha256":contract,
                       "sample_id":row["sample_id"],"image_sha256":row["image_sha256"],"text":text,
                       "model_metadata_sha256":sha(model_path),"latency_seconds":time.perf_counter()-start,
                       "input_shape":list(encoded["pixel_values"].shape),
                       "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes":torch.cuda.max_memory_reserved()})
            print(f"[InstructBLIP TEST] {i}/233 sample={row['sample_id']}: {text}",flush=True)


def evaluate(a):
    p,contract,inputs = load(a)
    gt = checked(a.output_dir/"gt.json",contract)
    if digest({s:gt[s] for s in ("TRAIN","TEST")}) != p["gt_digest"]:
        raise RuntimeError("GT changed")
    rows = inputs[a.split]
    # Refuse partial denominators or silently assuming successful fallback.
    for r in rows:
        if cached(a,a.split,"deepsolo",r,contract) is None:
            raise RuntimeError(f"missing {a.split} DeepSolo sample={r['sample_id']}; run PAPER export")
        if a.split == "TEST" and cached(a,a.split,"caption",r,contract) is None:
            raise RuntimeError(f"missing TEST caption sample={r['sample_id']}; run PAPER export")
    scorer = None
    per_image = []
    try:
        for i,row in enumerate(rows,1):
            if a.split == "TRAIN":
                decision = validate_route(row["decision"],p["config"]["tau"])
            else:
                cap = cached(a,"TEST","caption",row,contract)
                require_hash(a.output_dir/"cache/TEST/caption/model.json",cap["model_metadata_sha256"])
                cache = cached(a,"TEST","route",row,contract)
                if cache is None:
                    if scorer is None:
                        from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
                        scorer = MPNetSemanticSimilarity(MPNetApproxConfig(device="cpu"))
                    scores = route_scores(row["T1"],row["joined"],row["proposals"],cap["text"],scorer,"whole_caption")
                    s1,s3 = scores[:2]
                    decision = route(tau=p["config"]["tau"],T1=row["T1"],T2=cap["text"],T3=row["joined"]["text"],
                                     S1=s1["semantic_similarity"],S3=s3["semantic_similarity"],
                                     L1=s1["lexical_similarity"],L3=s3["lexical_similarity"]).to_dict()
                    cache = {"contract_sha256":contract,"sample_id":row["sample_id"],"image_sha256":row["image_sha256"],
                             "caption_sha256":sha(shard(a,"TEST","caption",row)),"decision":decision,"scorer":scorer.metadata}
                    seal_write(shard(a,"TEST","route",row),cache)
                require_hash(shard(a,"TEST","caption",row),cache["caption_sha256"])
                decision = validate_route(cache["decision"],p["config"]["tau"])
            deep = cached(a,a.split,"deepsolo",row,contract)["response"]
            words = gt[a.split][row["sample_id"]]
            texts = {"parseq_selected":decision["selected_output"],"deepsolo_all_strings":" ".join(deep["recognized_strings"]),
                     "deepsolo_highest_score":deep["selected_output"]}
            for policy in ("all_strings","highest_score"):
                texts[f"routed_{policy}"] = texts[f"deepsolo_{policy}"] if decision["needs_fallback"] else texts["parseq_selected"]
            per_image.append({"sample_id":row["sample_id"],"image_sha256":row["image_sha256"],"decision":decision,
                              "gt_words":words,"texts":texts,"metrics":{k:match_words(t,words) for k,t in texts.items()},
                              "deepsolo_sha256":sha(shard(a,a.split,"deepsolo",row))})
            print(f"[CPU {a.split}] {i}/{len(rows)} sample={row['sample_id']} fallback={decision['needs_fallback']}",flush=True)
    finally:
        if scorer is not None:
            scorer.release()
    metrics = {k:aggregate([r["metrics"][k] for r in per_image]) for k in per_image[0]["metrics"]}
    rejected = [r for r in per_image if r["decision"]["needs_fallback"]]
    accepted = [r for r in per_image if not r["decision"]["needs_fallback"]]
    report = {"contract_sha256":contract,"classification":"PRACTICAL_APPROXIMATION","claim_label":"approximate reproduction",
              "split":a.split,"tau":p["config"]["tau"],"threshold_recalibrated":False,"metrics":metrics,
              "fallback_images":len(rejected),"fallback_image_fraction":len(rejected)/len(rows),
              "accepted_parseq":aggregate([r["metrics"]["parseq_selected"] for r in accepted]),
              "fallback_deepsolo":aggregate([r["metrics"]["deepsolo_all_strings"] for r in rejected]),
              "control_inference_images":len(rows),"online_conditional_fallback_images":len(rejected),
              "accuracy_definition":p["word_protocol"],"highest_score_control":p["legacy_control"],
              "complete_image_definition":"exact word multiset: every GT word recovered and no extra tokens; order ignored",
              "test_previously_explored":True,"table5_accuracy":None,"per_image":per_image}
    path = a.output_dir/f"{a.split.lower()}-results.json"
    seal_write(path,report)
    print(json.dumps({"split":a.split,"tau":report["tau"],"fallback_images":len(rejected),"metrics":metrics},indent=2),flush=True)
    print("JSON",path,"SHA256",sha(path),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","status","export-deepsolo","export-captions","evaluate"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--split",choices=("TRAIN","TEST"),default="TRAIN")
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    for name in ("calibration","config","train-root","test-root","train-gt-report","data-root","assets","recognition-manifest","deepsolo-python"):
        parser.add_argument("--"+name,type=Path)
    a = parser.parse_args()
    if a.action not in ("export-deepsolo","export-captions"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if a.action == "prepare":
        for name in ("calibration","config","train_root","test_root","train_gt_report","data_root","assets","recognition_manifest"):
            if getattr(a,name) is None:
                parser.error(f"prepare requires --{name.replace('_','-')}")
        prepare(a)
    elif a.action == "export-deepsolo":
        if a.deepsolo_python is None:
            parser.error("export-deepsolo requires --deepsolo-python")
        export_deep(a)
    elif a.action == "export-captions":
        export_captions(a)
    elif a.action == "evaluate":
        protocol,_,_ = load(a)
        os.environ["HF_HUB_CACHE"] = str(Path(protocol["assets"])/"huggingface")
        import torch
        torch.set_num_threads(4)
        evaluate(a)
    else:
        status(a)


if __name__ == "__main__":
    main()
