#!/usr/bin/env python3
"""First-ten visual and CURRENT-vs-NATIVE_CC diagnostic; no parameter selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from rapidfuzz.fuzz import ratio

from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK, BlockParam, localize_mask
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter, FrozenSegmentationConfig, backproject_binary_mask
from recognition.icdar13_annotations import load_icdar13_recognition_samples
from recognition.parseq import PARSEQ_ARCHIVE_SOURCE, ParseqApproxConfig, ParseqApproxRecognizer
from segmentation.data import icdar13_task22_rgb_to_binary_mask, load_segmentation_manifest
from segmentation.evaluate import resize_binary_target_to_native_logits


PARAMS = BlockParam(min_area=150, pad=6, connectivity=8, tie_break=APPROXIMATE_TIE_BREAK)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.0


def _geometry(boxes: list[tuple[int, int, int, int]], gt: list[tuple[int, int, int, int]]) -> dict[str, int]:
    best = [max((_iou(word, box) for box in boxes), default=0.0) for word in gt]
    return {"iou_ge_0_25": sum(v >= .25 for v in best), "iou_ge_0_50": sum(v >= .5 for v in best), "iou_ge_0_75": sum(v >= .75 for v in best)}


def _native_cc(mask: np.ndarray, rgb: np.ndarray) -> tuple[int, list[dict]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, PARAMS.connectivity)
    height, width = rgb.shape[:2]
    candidates = []
    for component_id in range(1, count):
        component = (labels == component_id).astype(np.uint8)
        projected = backproject_binary_mask(component, (height, width))
        mapped_area = int(projected.sum())
        if mapped_area < PARAMS.min_area:
            continue
        ys, xs = np.nonzero(projected)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1))
        padded = (max(0,bbox[0]-PARAMS.pad), max(0,bbox[1]-PARAMS.pad), min(width,bbox[2]+PARAMS.pad), min(height,bbox[3]+PARAMS.pad))
        candidates.append((component_id, mapped_area, bbox, padded))
    candidates.sort(key=lambda v: (-v[1], v[2][1], v[2][0], v[0]))
    return count - 1, [
        {"rank": rank, "component_id": item[0], "mapped_foreground_area": item[1], "bbox": item[2], "padded_bbox": item[3], "crop": rgb[item[3][1]:item[3][3],item[3][0]:item[3][2]].copy()}
        for rank, item in enumerate(candidates[:10], 1)
    ]


def _metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    tp, fp, fn = int((gt & pred).sum()), int(((1-gt) & pred).sum()), int((gt & (1-pred)).sum())
    return (tp/(tp+fp+fn) if tp+fp+fn else 1.0, 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0)


def _title(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height+24), "white")
    canvas.paste(image, (0,24)); ImageDraw.Draw(canvas).text((5,5), text, fill="black")
    return canvas


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8)*255, "L").convert("RGB").resize((512,512), Image.Resampling.NEAREST)


def _save_visuals(out: Path, sample_id: str, rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, iou: float, f1: float) -> Image.Image:
    prefix = out / f"sample-{int(sample_id):04d}"
    original = Image.fromarray(rgb).convert("RGB"); original.thumbnail((512,512))
    gt_img, pred_img = _mask_image(gt), _mask_image(pred)
    overlay = np.zeros((128,128,3), dtype=np.uint8)
    overlay[(gt==1)&(pred==1)] = (255,255,255); overlay[(gt==1)&(pred==0)] = (0,0,255); overlay[(gt==0)&(pred==1)] = (255,0,0)
    error = np.zeros((128,128,3), dtype=np.uint8)
    error[(gt==0)&(pred==1)] = (255,0,0); error[(gt==1)&(pred==0)] = (0,0,255)
    panels = [
        _title(original, f"sample {sample_id} RGB IoU={iou:.3f} F1={f1:.3f}"),
        _title(gt_img, f"sample {sample_id} GT native 128"),
        _title(pred_img, f"sample {sample_id} PRED native 128"),
        _title(Image.fromarray(overlay).resize((512,512), Image.Resampling.NEAREST), "overlay white=TP red=FP blue=FN"),
        _title(Image.fromarray(error).resize((512,512), Image.Resampling.NEAREST), "error red=FP blue=FN"),
    ]
    names = ("rgb", "gt-native-128", "pred-native-128", "overlay", "error-map")
    for name, panel in zip(names, panels): panel.save(prefix.with_name(prefix.name+f"-{name}.png"))
    height=max(p.height for p in panels); sheet=Image.new("RGB",(sum(p.width for p in panels),height),"white")
    x=0
    for panel in panels: sheet.paste(panel,(x,0));x+=panel.width
    sheet.save(prefix.with_name(prefix.name+"-row.png")); return sheet


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recognition-manifest",type=Path,required=True); p.add_argument("--segmentation-manifest",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True); p.add_argument("--evaluation-config",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--trace-index",type=Path,required=True); p.add_argument("--parseq-source",type=Path,required=True); p.add_argument("--parseq-source-archive",type=Path,required=True); p.add_argument("--parseq-checkpoint",type=Path,required=True)
    p.add_argument("--gpu-reference-report",type=Path,required=True)
    p.add_argument("--device",default="cuda:0"); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--zip-output",type=Path,required=True)
    return p.parse_args()


def main() -> int:
    a=parse_args(); out=a.output_dir.resolve(); archive=a.zip_output.resolve()
    if out.exists() or archive.exists(): raise FileExistsError("refusing to overwrite diagnostic output")
    samples,_=load_icdar13_recognition_samples(a.recognition_manifest,a.data_root); samples=samples[:10]
    seg_manifest=load_segmentation_manifest(a.segmentation_manifest); seg_by_id={s.sample_id:s for s in seg_manifest.samples}
    traces=json.loads(a.trace_index.read_text()); trace_by_id={str(s["sample_id"]):s for s in traces["samples"]}
    reference=json.loads(a.gpu_reference_report.read_text())
    reference_native={str(v["sample_id"]):v for v in reference["frozen_test"]["frozen_segmentation"]["mask_hashes"]}
    out.mkdir(parents=True); segmenter=FrozenAuthorProtocolSegmenter(FrozenSegmentationConfig(a.evaluation_config,a.checkpoint,a.device))
    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=a.parseq_source,checkpoint=a.parseq_checkpoint,device=a.device,runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=a.parseq_source_archive))
    rows=[]; visual_rows=[]; gt_total=0
    try:
        for number,sample in enumerate(samples,1):
            print(f"[diagnostic] {number}/10 sample_id={sample.sample_id}",flush=True)
            rgb=np.asarray(Image.open(sample.image_path).convert("RGB")); result=segmenter(rgb); pred=result["mask_native"]
            expected=reference_native[sample.sample_id]
            if result["metadata"]["native_mask_sha256"] != expected["native_mask_sha256"] or result["metadata"]["output_mask_sha256"] != expected["projected_mask_sha256"]:
                raise RuntimeError(f"GPU mask hash drift for sample {sample.sample_id}")
            seg_sample=seg_by_id[sample.sample_id]
            binary=icdar13_task22_rgb_to_binary_mask(Image.open((a.data_root/seg_sample.mask_path).resolve()))
            gt=resize_binary_target_to_native_logits(binary,size=(128,128)).numpy().astype(np.uint8)
            iou,f1=_metrics(gt,pred); visual_rows.append(_save_visuals(out,sample.sample_id,rgb,gt,pred,iou,f1))
            gt_boxes=[(w.bbox_xyxy[0],w.bbox_xyxy[1],w.bbox_xyxy[2]+1,w.bbox_xyxy[3]+1) for w in sample.words]; gt_total+=len(gt_boxes)
            current=localize_mask(result["mask"],rgb,PARAMS); native_raw,native=_native_cc(pred,rgb)
            trace=json.loads(Path(trace_by_id[sample.sample_id]["result"]).read_text()); current_texts=[v["raw_text"] for v in trace["T3"]["all"]]
            if len(current_texts)!=len(current.retained): raise RuntimeError("frozen CURRENT trace crop count mismatch")
            for item in native:
                ocr=recognizer.recognize(item.pop("crop"),input_role="T3_localized_crop"); item["parseq_text"]=ocr.raw_text
                best=max(range(len(gt_boxes)),key=lambda idx:_iou(item["padded_bbox"],gt_boxes[idx]),default=None)
                item["best_gt_index"]=None if best is None else best+1; item["best_gt_text"]=None if best is None else sample.words[best].transcription
                item["casefold_exact"]=best is not None and item["parseq_text"].casefold()==sample.words[best].transcription.casefold()
                item["rapidfuzz_casefold_ge_0_8"]=best is not None and ratio(item["parseq_text"].casefold(),sample.words[best].transcription.casefold())>=80
            current_ocr=[]
            for block,text in zip(current.retained,current_texts):
                best=max(range(len(gt_boxes)),key=lambda idx:_iou(block.padded_bbox,gt_boxes[idx]),default=None)
                current_ocr.append((best is not None and text.casefold()==sample.words[best].transcription.casefold(),best is not None and ratio(text.casefold(),sample.words[best].transcription.casefold())>=80))
            rows.append({"sample_id":sample.sample_id,"native_iou":iou,"native_f1":f1,"gt_words":len(gt_boxes),"CURRENT":{"raw_component_count":len(current.retained)+len(current.rejected),"retained_crops":len(current.retained),"geometry":_geometry([b.padded_bbox for b in current.retained],gt_boxes),"parseq_exact":sum(v[0] for v in current_ocr),"parseq_fuzzy_ge_0_8":sum(v[1] for v in current_ocr)},"NATIVE_CC":{"raw_component_count":native_raw,"retained_crops":len(native),"geometry":_geometry([v["padded_bbox"] for v in native],gt_boxes),"parseq_exact":sum(v["casefold_exact"] for v in native),"parseq_fuzzy_ge_0_8":sum(v["rapidfuzz_casefold_ge_0_8"] for v in native),"crops":native}})
    finally:
        recognizer.release(); segmenter.release()
    width=max(v.width for v in visual_rows); sheet=Image.new("RGB",(width,sum(v.height for v in visual_rows)),"white"); y=0
    for row in visual_rows: sheet.paste(row,(0,y));y+=row.height
    sheet.save(out/"contact-sheet-0001-0010.png")
    aggregate={}
    for method in ("CURRENT","NATIVE_CC"):
        aggregate[method]={key:sum(r[method][key] for r in rows) for key in ("raw_component_count","retained_crops","parseq_exact","parseq_fuzzy_ge_0_8")}
        aggregate[method].update({key:sum(r[method]["geometry"][key] for r in rows) for key in ("iou_ge_0_25","iou_ge_0_50","iou_ge_0_75")})
    report={"schema_version":1,"classification":"APPROXIMATE_DIAGNOSTIC","new_segmentation_inference":True,"sample_count":10,"gt_words":gt_total,"parameters":{"min_area":150,"padding":6,"connectivity":8,"nmax":10},"checkpoint_sha256":_sha(a.checkpoint),"aggregate":aggregate,"per_sample":rows}
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    shutil.make_archive(str(archive.with_suffix("")),"zip",root_dir=out)
    print(json.dumps(aggregate,indent=2),flush=True); print(archive,flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
