"""Independent full-population, fixed-selection and TEST overlap/count audit."""

import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from localize.adaspot_gap_preserve import decode_gap
from scripts.audit_adaspot_phase0 import count,digest
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from segmentation.data import SegmentationDataset,load_segmentation_manifest


def check_ocr(value,ids,readings):
    assert len(value["per_image"])==len(ids)
    assert {s["sample_id"] for s in value["per_image"]}==ids
    total=Counter()
    for s in value["per_image"]:
        pred,gt=s["predictions"],s["ground_truth"]
        metrics={"gt_words":len(gt),"proposals":len(pred),"geometry_iou_ge_50":count(pred,gt,False),"exact_casefold_iou_ge_50":count(pred,gt,True)}
        assert all(s["counts"][k]==v for k,v in metrics.items())
        assert all(p["text"]==readings[p["crop_rgb_sha256"]]["raw_text"] for p in pred)
        total.update(metrics)
    assert all(value["aggregate"][k]==v for k,v in total.items())
    assert value["aggregate"]["hmean"]==2*total["exact_casefold_iou_ge_50"]/(total["gt_words"]+total["proposals"])
    return total


def audit(root):
    if not __debug__:raise RuntimeError("Assertions required; no Python -O")
    torch.set_num_threads(2)
    result=json.loads((root/"results.json").read_text());assert result["status"]=="SUCCESS"
    assert result["result_class"]=="approximate reproduction"
    p=result["protocol"];c=p["resolved_config"];t=result["training"]
    assert p["mode"]=="approximate" and not p["test_used_for_selection"] and p["test_historically_explored"]
    for path,h in p["source_hashes"].items():assert digest(Path(path))==h,path
    train=load_segmentation_manifest(p["train_manifest"],require_checksums=True)
    test=load_segmentation_manifest(p["test_manifest"],require_checksums=True)
    # Recheck actual TRAIN RGB/target/owner/validity files after training.
    SegmentationDataset(train,p["word_root"],verify_checksums=True)
    for sample in test.samples:
        assert digest(Path(p["test_root"])/sample.image_path)==sample.image_sha256
    assert train.split_role=="train" and test.split_role=="test"
    train_ids,test_ids={s.sample_id for s in train.samples},{s.sample_id for s in test.samples}
    assert len(train_ids)==c["train_images"]==1255 and len(test_ids)==c["test_images"]==300
    assert train_ids==set(p["train_ids"]) and test_ids==set(p["test_ids"]) and not train_ids&test_ids
    grouped=defaultdict(list)
    for s in train.samples:grouped[s.image_sha256].append(s.sample_id)
    duplicates=[{"test_id":s.sample_id,"train_ids":grouped[s.image_sha256],"sha256":s.image_sha256} for s in test.samples if s.image_sha256 in grouped]
    assert duplicates==c["known_official_rgb_overlap"] and len(duplicates)==2
    assert t["completed_epochs"]==c["epochs"]==10
    assert [r["epoch"] for r in t["history"]]==list(range(1,11))
    for r in t["history"]:
        assert r["samples"]==len(r["sample_ids"])==len(set(r["sample_ids"]))==1255
        assert set(r["sample_ids"])==train_ids
        assert sha256("".join(s+"\n" for s in r["sample_ids"]).encode()).hexdigest()==r["sample_order_sha256"]
        assert r["cuda_steps"]==314 and r["skipped_all_invalid_batches"]==0
        assert r["memory"]["device"].startswith("cuda") and not r["memory"]["oom"]
    assert t["global_cuda_steps"]==3140
    assert p["selected_epoch_fixed_before_test"]==t["selected_epoch"]==c["selected_epoch"]==1
    old=json.loads((Path(p["previous_selection_run"])/"selection.json").read_text())
    assert (old["epoch"],old["threshold"],old["decoder"])==(c["selected_epoch"],c["gap_threshold"],c["decoder"])
    for epoch,h in t["saved_epoch_sha256"].items():
        path=root/"decoder/checkpoints"/f"{int(epoch):03d}.pt";assert digest(path)==h
    assert digest(root/"decoder/checkpoints/001.pt")==t["selected_checkpoint_sha256"]
    assert digest(root/"decoder/last.pt")==t["last_checkpoint_sha256"]
    smoke=json.loads((root/"decoder/first_cuda_step.json").read_text())
    assert smoke["status"]=="PASS" and smoke["device"].startswith("cuda") and smoke["teacher_max_error"]==0
    assert smoke["decoder_gradient_l1"]>0 and smoke["head_weight_update_linf"]>0
    source=torch.load(p["source_checkpoint"],map_location="cpu",weights_only=False)["model_state_dict"]
    h=sha256()
    for name,value in sorted(source.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    assert h.hexdigest()==t["source_state_sha256_before_after"];del source
    metadata=json.loads((root/"test_artifact_manifest.json").read_text())
    assert len(metadata)==300 and {r["sample_id"] for r in metadata}==test_ids
    for row in metadata:
        assert digest(Path(row["annotation_path"]))==row["annotation_sha256"]
        for path,h in row["prediction_sha256"].items():assert digest(Path(path))==h
    readings=json.loads((root/"test_readings.json").read_text())
    for h in readings:
        with Image.open(root/"crops"/(h+".png")) as image:assert crop_array_hash(np.asarray(image.convert("RGB")))==h
    frozen=json.loads((root/"test_inputs.json").read_text())
    choices=result["choices"];assert set(result["arms"])==set(choices)=={"baseline","prior_gap","full_selected","full_last"}
    assert choices["full_selected"]["epoch"]==1 and choices["full_last"]["epoch"]==10
    pp=p["source_postprocessing"];label_cases=0;exact_by_arm={}
    test_by_id={s.sample_id:s for s in test.samples}
    for name,arm in result["arms"].items():
        total=check_ocr(arm["ocr"],test_ids,readings);assert total["gt_words"]==c["test_care_words"]==2204
        assert arm["changes_vs_baseline"]["net"]==total["exact_casefold_iou_ge_50"]-result["arms"]["baseline"]["ocr"]["aggregate"]["exact_casefold_iou_ge_50"]
        exact_by_arm[name]=total["exact_casefold_iou_ge_50"]
        assert len(frozen[name])==300 and {s["sample_id"] for s in frozen[name]}==test_ids
        for raw,scored in zip(frozen[name],arm["ocr"]["per_image"]):
            sid=raw["sample_id"];assert sid==scored["sample_id"] and raw["ground_truth"]==scored["ground_truth"]
            assert raw["predictions"]==[{k:v for k,v in q.items() if k!="text"} for q in scored["predictions"]]
            folder=root/"test_predictions"/sid.replace(":","__")
            with np.load(folder/"source.npz") as z:words,canvas=z["probabilities"],z["canvas"]
            with np.load(folder/(name+"_labels.npz")) as z:labels=z["labels"]
            with np.load(folder/"validity.npz") as z:valid=z["validity"]
            if name=="baseline":gap=None
            else:
                with np.load(folder/(name+".npz")) as z:gap=z["gap_probability"]
            choice=choices[name]
            assert np.array_equal(labels,decode_gap(words,gap,canvas,pp,choice["threshold"],choice["decoder"]))
            expected_ignored=[int(v) for v in np.unique(labels[labels>0]) if not np.any((labels==v)&valid)]
            assert expected_ignored==raw["ignored_instance_ids"]
            expected_ids=set(int(v) for v in np.unique(labels[labels>0]))-set(expected_ignored)
            assert {q["instance_id"] for q in raw["predictions"]}==expected_ids
            with Image.open(Path(p["test_root"])/test_by_id[sid].image_path) as image:rgb=np.asarray(image.convert("RGB"))
            for q in raw["predictions"]:
                x0,y0,x1,y1=q["bbox"];assert crop_array_hash(np.ascontiguousarray(rgb[y0:y1,x0:x1]))==q["crop_rgb_sha256"]
            label_cases+=1
    subset=result["hash_disjoint_test_subset"];excluded={r["test_id"] for r in duplicates}
    assert subset["images"]==298 and set(subset["excluded_exact_duplicate_test_ids"])==excluded
    clean_ids=test_ids-excluded;clean_exact={}
    for name,arm in subset["arms"].items():
        total=check_ocr(arm["ocr"],clean_ids,readings)
        duplicate_words=sum(len(r["ground_truth"]) for r in result["arms"][name]["ocr"]["per_image"] if r["sample_id"] in excluded)
        assert total["gt_words"]==2204-duplicate_words
        clean_exact[name]=total["exact_casefold_iou_ge_50"]
    report={"status":"PASS","method":"Full1255-ID epoch audit; fixed prior selection; exact duplicate reporting; independent SciPy Hungarian fullTEST300/secondary298; mask replay and RGB crop/source hashes",
        "completed_cuda_epochs":10,"cuda_updates":3140,"train_samples_per_epoch":1255,"full_test_image_arm_cases":1200,
        "hash_disjoint_subset_image_arm_cases":1192,"mask_replay_cases":label_cases,"exact_by_arm":exact_by_arm,"hash_disjoint_exact_by_arm":clean_exact,
        "known_train_test_exact_duplicates":duplicates,"result_sha256":digest(root/"results.json"),"auditor_sha256":digest(Path(__file__))}
    text=json.dumps(report,indent=2,sort_keys=True)+"\n";path=root/"independent_audit.json"
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    print(text)


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("--run-root",type=Path,required=True);audit(a.parse_args().run_root)
