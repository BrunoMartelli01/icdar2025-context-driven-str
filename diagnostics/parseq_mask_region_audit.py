"""CPU-only, post-hoc TEST audit: native mask quality versus frozen v1 OCR errors.

No model is loaded, no threshold/localizer is changed. Bins are descriptive,
not causal attribution or a parameter-selection protocol.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from diagnostics.caption_routing_train import checked, seal_write
from diagnostics.budgeted_paired_e2e import TEST_REPORT_SHA, SEGMENTATION_SHA, require_hash, sha
from diagnostics.joined_t3 import checked_shard
from diagnostics.repository_contours_train import _iou, _inter
from pipeline.e2e_comparison_diagnostic import word_crop_diagnostic
from recognition.icdar13_annotations import load_icdar13_recognition_samples
from recognition.segmentation_word_topology import _native_word_box, _exclusive_gt_box
from segmentation.data import load_segmentation_manifest, icdar13_task22_rgb_to_binary_mask
from segmentation.evaluate import resize_binary_target_to_native_logits


def pixel_metrics(target, prediction):
    target, prediction = np.asarray(target,dtype=bool), np.asarray(prediction,dtype=bool)
    if target.shape != prediction.shape or target.ndim != 2 or not target.size:
        raise ValueError("aligned nonempty 2D masks required")
    tp = int(np.count_nonzero(target & prediction))
    fp = int(np.count_nonzero(~target & prediction))
    fn = int(np.count_nonzero(target & ~prediction))
    return {"target_pixels":tp+fn,"predicted_pixels":tp+fp,"tp":tp,"fp":fp,"fn":fn,
            "pixel_recall":tp/(tp+fn) if tp+fn else None,
            "pixel_precision":tp/(tp+fp) if tp+fp else None,
            "pixel_iou":tp/(tp+fp+fn) if tp+fp+fn else None,
            "pixel_f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def failure_bin(metrics, best_iou, exact):
    if exact:
        return "OCR_CORRECT"
    if metrics["target_pixels"] == 0:
        return "NO_NATIVE_GT_FOREGROUND"
    if metrics["pixel_recall"] < .5:
        return "LOW_MASK_RECALL_LT_50"
    if metrics["pixel_recall"] < .8:
        return "PARTIAL_MASK_RECALL_50_80"
    if metrics["pixel_precision"] < .8:
        return "HIGH_RECALL_LOW_MASK_PRECISION"
    if best_iou < .5:
        return "HIGH_MASK_RECALL_PRECISION_BUT_POOR_CROP"
    return "HIGH_MASK_RECALL_PRECISION_AND_GOOD_BOX_OCR_UNRESOLVED"


def distribution(values):
    values = [v for v in values if v is not None]
    return {"count":len(values),"p25":float(np.quantile(values,.25)) if values else None,
            "median":float(np.median(values)) if values else None,
            "p75":float(np.quantile(values,.75)) if values else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("suite-root","data-root","prediction-dir","segmentation-manifest","recognition-manifest","output-dir"):
        p.add_argument("--"+name,type=Path,required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError("refusing to overwrite previous diagnostic")
    require_hash(a.suite_root/"results.json",TEST_REPORT_SHA)
    suite = json.loads((a.suite_root/"results.json").read_text())
    contract = suite["contract_sha256"]
    require_hash(a.recognition_manifest,suite["protocol"]["inputs"]["recognition_manifest"])
    require_hash(a.segmentation_manifest,suite["protocol"]["inputs"]["segmentation_manifest"])
    samples,_ = load_icdar13_recognition_samples(a.recognition_manifest,a.data_root)
    segmentation = {s.sample_id:s for s in load_segmentation_manifest(a.segmentation_manifest).samples}
    stages = ("01-proposals","02-parseq")
    seals = {s:json.loads((a.suite_root/"artifacts/seals"/f"{s}.json").read_text()) for s in stages}
    for seal in seals.values():
        if seal["contract_sha256"] != contract or len(seal["sha256"]) != 233:
            raise RuntimeError("wrong source seal")
    rows, image_rows, source_hashes, controls_p, controls_gt = [], [], {}, [], []
    for index,sample in enumerate(samples,1):
        sid = sample.sample_id
        prop = checked_shard(a.suite_root,"01-proposals",sid,sample.image_sha256,contract,seals)
        ocr = checked_shard(a.suite_root,"02-parseq",sid,sample.image_sha256,contract,seals)
        if ocr["proposal_shard_sha256"] != seals["01-proposals"]["sha256"][f"{int(sid):04d}.json"]:
            raise RuntimeError("crop OCR source mismatch")
        if prop["segmentation"]["checkpoint"]["sha256"] != SEGMENTATION_SHA:
            raise RuntimeError("wrong checkpoint")
        if prop["segmentation"]["threshold"] != .5:
            raise RuntimeError("wrong prediction threshold")
        prediction_path = a.prediction_dir/f"{sid}.png"
        with Image.open(prediction_path) as image:
            values = np.asarray(image)
            if values.shape != (128,128) or not set(np.unique(values)) <= {0,255}:
                raise RuntimeError("native prediction PNG must be binary 128x128")
            pred = (values>0).astype(np.uint8)
        native_sha = hashlib.sha256(pred.tobytes()).hexdigest()
        if native_sha != prop["segmentation"]["native_mask_sha256"]:
            raise RuntimeError(f"prediction differs from actual PARSeq pipeline sample={sid}")
        target_path = a.data_root/segmentation[sid].mask_path
        require_hash(target_path,segmentation[sid].mask_sha256)
        with Image.open(target_path) as image:
            full_target = icdar13_task22_rgb_to_binary_mask(image)
        full_binary = np.asarray(full_target)>0
        target = resize_binary_target_to_native_logits(full_target).numpy().astype(np.uint8)
        _,pred_labels,_,_ = cv2.connectedComponentsWithStats(pred,connectivity=8)
        _,target_labels,_,_ = cv2.connectedComponentsWithStats(target,connectivity=8)
        crops = [{"candidate_id":r["candidate_id"],"bbox":r["bbox"],
                  "text":ocr["crops"][r["ocr_key"]]["raw_text"]} for r in prop["v1_anchors"]]
        controls_p.append(crops)
        controls_gt.append([{"bbox":list(w.bbox_xyxy),"text":w.transcription} for w in sample.words])
        image_rows.append({"sample_id":sid,**pixel_metrics(target,pred),"native_sha256":native_sha})
        source_hashes[sid] = {"prediction_png":sha(prediction_path),"target":sha(target_path),
                              "proposal":seals["01-proposals"]["sha256"][f"{int(sid):04d}.json"],
                              "ocr":seals["02-parseq"]["sha256"][f"{int(sid):04d}.json"]}
        for wi,word in enumerate(sample.words,1):
            box = _exclusive_gt_box(word.bbox_xyxy,sample.image_size_wh)
            nb = _native_word_box(box,sample.image_size_wh)
            x1,y1,x2,y2 = nb
            metrics = pixel_metrics(target[y1:y2,x1:x2],pred[y1:y2,x1:x2])
            best = max(crops,key=lambda c:_iou(c["bbox"],box),default=None)
            iou = _iou(best["bbox"],box) if best else 0.
            available = best is not None and _inter(best["bbox"],box)>0
            exact = available and best["text"].casefold()==word.transcription.casefold()
            any_exact = any(c["text"].casefold()==word.transcription.casefold() and _inter(c["bbox"],box)>0 for c in crops)
            r = {"sample_id":sid,"word_index":wi,"gt_text":word.transcription,"gt_bbox":box,"native_bbox":nb,
                 "source_target_pixels":int(full_binary[box[1]:box[3],box[0]:box[2]].sum()),
                 "target_component_count":len(set(np.unique(target_labels[y1:y2,x1:x2]))-{0}),
                 "predicted_component_count":len(set(np.unique(pred_labels[y1:y2,x1:x2]))-{0}),
                 **metrics,"best_crop_iou":iou,"crop_available":available,
                 "crop_id":best["candidate_id"] if available else None,
                 "crop_bbox":best["bbox"] if available else None,"parseq_text":best["text"] if available else None,
                 "best_crop_exact":exact,"any_overlapping_crop_exact":any_exact,
                 "diagnostic_bin":failure_bin(metrics,iou,exact)}
            rows.append(r)
        if index%25==0 or index==233:
            print(f"[CPU mask/OCR audit] {index}/233",flush=True)
    historical = word_crop_diagnostic(controls_p,controls_gt)
    if historical != suite["crop_diagnostics"]["v1_repository"]:
        raise RuntimeError("historical v1 OCR/geometry control mismatch")
    if len(rows)!=1095:
        raise RuntimeError("wrong word denominator")
    groups = {"correct":[r for r in rows if r["best_crop_exact"]],
              "incorrect_or_missing":[r for r in rows if not r["best_crop_exact"]]}
    summary = {name:{"words":len(rs),"pixel_recall":distribution([r["pixel_recall"] for r in rs]),
                      "pixel_precision":distribution([r["pixel_precision"] for r in rs]),
                      "pixel_iou":distribution([r["pixel_iou"] for r in rs]),
                      "crop_iou":distribution([r["best_crop_iou"] for r in rs]),
                      "bins":dict(Counter(r["diagnostic_bin"] for r in rs))} for name,rs in groups.items()}
    totals = {key:sum(r[key] for r in image_rows) for key in ("tp","fp","fn")}
    micro = {"iou":totals["tp"]/(totals["tp"]+totals["fp"]+totals["fn"]),
             "f1":2*totals["tp"]/(2*totals["tp"]+totals["fp"]+totals["fn"])}
    report = {"classification":"APPROXIMATE_DIAGNOSTIC","scope":{"split":"TEST","images":233,"gt_words":1095},
              "checkpoint_sha256":SEGMENTATION_SHA,"source_report_sha256":TEST_REPORT_SHA,
              "native_hash_matches":233,"models_rerun":False,"tuning":False,"table5_accuracy":None,
              "target_protocol":"exact white background; binary GT bilinear resize128 then >=.5; all colors foreground including Dont Care as existing native TEST evaluator; exact TEST task2 ignore labels unavailable",
              "mapping":"inclusive GT xyxy converted to half-open by adding 1 to maxima; floor/ceil to native128; raw 8-connected components",
              "historical_coordinate_control":"old suite treated GT maxima as exclusive in IoU; its complete metrics reproduced separately, not overwritten",
              "bin_definition":"posthoc only: recall <.5; .5<=recall<.8; recall>=.8 split by precision .8 and crop IoU .5; not proof of causal OCR failure",
              "limitations":["GT diagnostic crop association is not inference selection; repeated same-text words may share a crop",
                              "High mask recall/precision and bbox IoU do not guarantee an OCR-suitable crop",
                              "No new GT-box OCR control; recognition versus residual crop-content error is unresolved"],
              "summary":summary,"historical_control":historical,
              "corrected_geometry_counts":{str(t):sum(r["best_crop_iou"]>=t for r in rows) for t in (.25,.5,.75)},
              "any_overlapping_crop_exact":sum(r["any_overlapping_crop_exact"] for r in rows),
              "incorrect_source_foreground_lost_in_native_target":sum(
                  not r["best_crop_exact"] and r["target_pixels"]==0 and r["source_target_pixels"]>0 for r in rows),
              "native_global_micro":micro,"source_hashes":source_hashes,
              "per_image":image_rows,"per_word":rows}
    a.output_dir.mkdir(parents=True)
    out = a.output_dir/"results.json"
    seal_write(out,report)
    csv_path = a.output_dir/"per-word.csv"
    with csv_path.open("x",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({"summary":summary,"geometry":report["corrected_geometry_counts"],"micro":micro},indent=2))
    print("JSON",out,"SHA256",sha(out));print("CSV",csv_path,"SHA256",sha(csv_path))


if __name__=="__main__":
    main()
