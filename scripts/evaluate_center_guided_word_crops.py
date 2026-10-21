"""Paired, frozen two-U-Net masks/vectors -> crops -> optional real PARSeq study."""

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from diagnostics.render_train_visual_sheets import fitted
from localize.center_guided_word_crops import center_guided_masks, center_guided_crops, validate_refinement
from localize.word_center_offsets import split_text_with_center_offsets
from localize.word_guided_contours import findcontour_labels
from scripts.evaluate_word_multitask_crops import write_json, ratios, score_predictions, recognize
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_word_multitask import experiment_environment
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from segmentation.data import load_segmentation_manifest
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask, remap_letterbox_native_box
from segmentation.word_multitask_evaluation import aggregate_word_metrics


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/approximate/center_guided_word_crops_v1.json"
METHODS = ["baseline", "local_centers", "global_centers", "global_centers_expand"]


def read_config(path, mode):
    path = Path(path).resolve()
    if mode != "approximate" or REPO / "configs/approximate" not in path.parents:
        raise ValueError("Joint center/mask crops require explicit APPROXIMATE config; no namespace fallback")
    cfg = json.loads(path.read_text())
    if (cfg["schema_version"] != 1 or cfg["reproduction_mode"] != mode
        or cfg["result_class"] != "approximate reproduction" or cfg["methods"] != METHODS):
        raise ValueError("Unsupported paired-crop protocol")
    validate_refinement(cfg["refinement"])
    if cfg["crop"] != {"padding": 0, "min_area": 0, "max_instances": None}:
        raise ValueError("This matched v1 diagnostic requires pad0/no area filter/no cap for all branches")
    recognition = cfg["recognition"]
    if (recognition["decode_ar"] is not True or recognition["refine_iters"] != 1
        or recognition["casefold"] is not True or recognition["symbol_normalized_secondary_metric"] is not True
        or type(recognition["batch_size"]) is not int or recognition["batch_size"] < 1):
        raise ValueError("Unsupported pinned PARSeq decoding/scoring protocol")
    return cfg


def source_report(root):
    report = json.loads((root / "run_manifest.json").read_text())
    validation = json.loads((root / "validation.json").read_text())
    if (report["status"] != "SUCCESS" or report["result_class"] != "approximate reproduction"
        or report["run_contract"] != validation["run_contract"]
        or sha256_file(root / report["checkpoint"]["path"]) != report["checkpoint"]["sha256"]):
        raise ValueError("Incomplete, changed or inconsistent source run: " + str(root))
    return report, validation


def load_sources(args, cfg):
    base_root, offset_root = args.base_run.resolve(), args.offset_run.resolve()
    base, bv = source_report(base_root); offset, ov = source_report(offset_root)
    bc, oc = base["run_contract"], offset["run_contract"]
    if (bc["variant"] != "text" or bc["config"]["input_size"] != [512, 512]
        or bc["config"]["native_mask_size"] != [256, 256] or oc["base_sha256"] != base["checkpoint"]["sha256"]
        or bc["resolved_preprocessing"] != oc["resolved_preprocessing"]
        or bc["validation_ids_sha256"] != oc["validation_ids_sha256"]
        or offset["validation"] != ov["aggregate"] or bv["aggregate"] != base["validation_word_geometry"]
        or bc["config"]["postprocessing"]["text_threshold"] != cfg["refinement"]["text_threshold"]):
        raise ValueError("Sources are not a matched single-head mask / auxiliary vector pair")
    if offset["completed_epochs"] != oc["training_config"]["epochs"]:
        raise ValueError("Auxiliary training is incomplete")
    root = args.target_root.resolve()
    manifest_path = root / bc["config"]["train_manifest"]
    if sha256_file(manifest_path) != bc["config"]["train_manifest_sha256"]:
        raise ValueError("Qualified Total-Text dataset manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("This joint study uses the existing TRAIN holdout only")
    _, indices, fingerprint = _split_indices(manifest, bc["config"]["seed"], bc["config"]["validation_count"])
    ids = [manifest.samples[index].sample_id for index in indices]
    if (fingerprint != bc["validation_ids_sha256"] or ids != [r["sample_id"] for r in bv["per_image"]]
        or ids != [r["sample_id"] for r in ov["per_image"]] or len(ids) != 125):
        raise ValueError("Frozen validation identity/order differs")
    return {"base": base, "offset": offset, "base_validation": bv, "offset_validation": ov,
            "manifest": manifest, "ids": ids}


def prepare_recipes(p, offsets, canvas, image_wh, cfg, local_voting):
    """Inference boundary: no owner maps, annotations, validity or OCR arguments."""
    joint = center_guided_masks(p, offsets, canvas, cfg["refinement"])
    baseline, baseline_audit = findcontour_labels(joint["core"])
    local, local_signal = split_text_with_center_offsets(p, offsets, canvas, local_voting)
    labels = {"baseline": baseline, "local_centers": local,
              "global_centers": joint["preserved_labels"], "global_centers_expand": joint["expanded_labels"]}
    audits = {"baseline": baseline_audit, "local_centers": local_signal["audit"],
              "global_centers": joint["audit"], "global_centers_expand": joint["audit"]}
    recipes = {method: {"labels": value, "proposals": center_guided_crops(value, image_wh, cfg["crop"]),
                         "audit": audits[method]} for method, value in labels.items()}
    return recipes, joint


def render(rgb, recipes, joint, path):
    panels = []
    for method in ("baseline", "global_centers", "global_centers_expand"):
        panel = Image.fromarray(rgb).copy(); draw = ImageDraw.Draw(panel)
        for i, proposal in enumerate(recipes[method]["proposals"], 1):
            x1, y1, x2, y2 = proposal["bbox"]
            draw.rectangle((x1, y1, x2-1, y2-1), outline="#00ee55", width=2)
            draw.text((x1, y1), str(i), fill="#ffff00")
        panels.append((method+": "+str(len(recipes[method]["proposals"])), panel))
    density = np.log1p(joint["density"])
    mapped = np.rint(255*density/density.max()).astype(np.uint8) if density.max() else np.zeros(density.shape, np.uint8)
    heatmap = cv2.cvtColor(cv2.applyColorMap(mapped, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    panel = Image.fromarray(heatmap); draw = ImageDraw.Draw(panel)
    for center in joint["centers"]:
        x, y = center["vote_peak_xy_native"]
        draw.ellipse((x-2, y-2, x+2, y+2), outline="white", width=1)
    panels.append(("Predicted center votes: "+str(len(joint["centers"])), panel))
    palette = np.array([[0,0,0], [90,220,100], [100,160,255], [255,160,70], [230,80,180]], np.uint8)
    for key in ("preserved_labels", "expanded_labels"):
        labels = joint[key]
        colors = palette[1+(labels % (len(palette)-1))]; colors[labels == 0] = 0
        panels.append((key, Image.fromarray(colors)))
    sheet = Image.new("RGB", (320*3, 344*2), "white"); text = ImageDraw.Draw(sheet)
    for i, (title, panel) in enumerate(panels):
        x, y = (i%3)*320, (i//3)*344
        sheet.paste(fitted(panel, (320,320)), (x,y+24)); text.text((x+5,y+5),title,fill="black")
    sheet.save(path)


def verify_exports(result, output):
    for branch in result["branches"].values():
        for part in branch.values():
            for row in part["per_image"]:
                for name, digest in row["exports"].items():
                    if sha256_file(output / name) != digest:
                        raise ValueError("Completed fusion export changed: " + name)


def build_geometry(args, cfg, sources):
    output, target = args.output_root.resolve(), args.target_root.resolve()
    modules = ("scripts/evaluate_center_guided_word_crops.py", "localize/center_guided_word_crops.py",
        "localize/word_center_offsets.py", "localize/word_multitask.py", "localize/word_guided_contours.py",
        "localize/crop_recovery.py", "segmentation/totaltext_word_kernel.py", "segmentation/totaltext_image_frequency_augment.py",
        "scripts/evaluate_word_multitask_crops.py", "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py")
    protocol = {"config": cfg, "source_seed": sources["base"]["run_contract"]["config"]["seed"],
        "base_report_sha256": sha256_file(args.base_run / "run_manifest.json"),
        "auxiliary_report_sha256": sha256_file(args.offset_run / "run_manifest.json"),
        "base_validation_sha256": sha256_file(args.base_run / "validation.json"),
        "auxiliary_validation_sha256": sha256_file(args.offset_run / "validation.json"),
        "validation_ids_sha256": sources["base"]["run_contract"]["validation_ids_sha256"],
        "source_sha256": {name: sha256_file(REPO/name) for name in modules},
        "sources": {"base": sources["base"]["checkpoint"], "offset": sources["offset"]["checkpoint"]},
        "execution": "CPU postprocessing of archived RTX3090 FP32 two-network outputs; no segmentation rerun"}
    write_json(output / "geometry_protocol.json", protocol)
    path = output / "geometry_results.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["status"] != "SUCCESS" or saved["protocol"] != protocol:
            raise ValueError("Completed geometry protocol changed; select a new output directory")
        verify_exports(saved, output)
        return saved
    by_id = {sample.sample_id: sample for sample in sources["manifest"].samples}
    parts = {method: {"per_image": [], "native_totals": Counter(), "original_totals": Counter()} for method in METHODS}
    source_rows = zip(sources["base_validation"]["per_image"], sources["offset_validation"]["per_image"])
    local_voting = sources["offset"]["run_contract"]["resolved_config"]["voting"]
    for position, (base_row, offset_row) in enumerate(source_rows, 1):
        sample_id = base_row["sample_id"]; key = sample_id.replace(":", "__")
        base_dir = args.base_run / "validation_exports" / key
        offset_dir = args.offset_run / "validation_exports" / key
        if (sha256_file(base_dir/"maps.npz") != base_row["exports"]["maps.npz"]
            or sha256_file(offset_dir/"offsets.npz") != offset_row["export_sha256"]):
            raise ValueError("Frozen map/vector checksum changed: " + sample_id)
        sample = by_id[sample_id]; image_path = target / sample.image_path
        if sha256_file(image_path) != sample.image_sha256:
            raise ValueError("Original RGB changed: " + sample_id)
        with Image.open(image_path) as image: rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
        with np.load(base_dir/"maps.npz",allow_pickle=False) as b, np.load(offset_dir/"offsets.npz",allow_pickle=False) as o:
            if (b["probabilities"].shape != (1,256,256) or o["normalized_offsets"].shape != (2,256,256)
                or not np.array_equal(b["owners"],o["owners"]) or not np.array_equal(b["validity"],o["validity"])
                or not np.array_equal(b["canvas"],letterbox_canvas_mask((rgb.shape[1],rgb.shape[0]),native_size=256))):
                raise ValueError("Cached native coordinate/target alignment differs: " + sample_id)
            recipes, joint = prepare_recipes(b["probabilities"][0],o["normalized_offsets"],b["canvas"],
                (rgb.shape[1],rgb.shape[0]),cfg,local_voting)
            directory = output/"samples"/key; directory.mkdir(parents=True,exist_ok=True)
            for center in joint["centers"]:
                x,y = center["vote_peak_xy_native"]
                center["peak_pixel_box_original"] = remap_letterbox_native_box([x,y,x+1,y+1],
                    (rgb.shape[1],rgb.shape[0]),native_size=256)
            for method, recipe in recipes.items():
                for proposal in recipe["proposals"]:
                    x1,y1,x2,y2 = proposal["bbox"]
                    crop = np.ascontiguousarray(rgb[y1:y2,x1:x2])
                    proposal["crop_rgb_sha256"] = crop_array_hash(crop)
                    crop_path = directory/method/(str(proposal["instance_id"])+".png")
                    crop_path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(crop).save(crop_path)
                write_json(directory/(method+"_crops.json"), {"sample_id":sample_id,"result_class":"approximate reproduction",
                    "predicted_words_raw":len(recipe["proposals"]),"proposals":recipe["proposals"],"audit":recipe["audit"]})
            arrays = {method:recipe["labels"] for method,recipe in recipes.items()}
            np.savez_compressed(directory/"fusion.npz",**arrays,vote_density=joint["density"],seeds=joint["seeds"])
            render(rgb,recipes,joint,directory/"overview.png")
            write_json(directory/"centers.json", {"result_class":"approximate reproduction","centers":joint["centers"],"audit":joint["audit"]})
            # All crop recipes have now been frozen without GT.
            gt, annotation_sha = _train_gt(sample,target,(rgb.shape[1],rgb.shape[0]))
            for method, recipe in recipes.items():
                labels = recipe["labels"]
                ignored = [int(label) for label in np.unique(labels[labels>0]) if not np.any((labels == label)&b["validity"])]
                care_labels = labels.copy(); care_labels[np.isin(labels,ignored)] = 0
                native = instance_geometry(care_labels,b["owners"],b["expected_owner_ids"])
                care = [p for p in recipe["proposals"] if p["instance_id"] not in ignored]
                original = score_predictions(care,gt,text_available=False)
                files = [directory/(method+"_crops.json"), directory/"fusion.npz", directory/"overview.png", directory/"centers.json"]
                files += [directory/method/(str(p["instance_id"])+".png") for p in recipe["proposals"]]
                row = {"sample_id":sample_id,"image_path":str(image_path),"image_sha256":sample.image_sha256,
                    "proposals":recipe["proposals"],"audit":recipe["audit"],"ground_truth":gt,"annotation_sha256":annotation_sha,
                    "ignored_prediction_ids":ignored,"native_geometry":native,"original_geometry":original,
                    "predicted_words_raw":len(recipe["proposals"]),"predicted_care_count":len(care),
                    "absolute_count_error":abs(len(care)-len(gt)),"exports":{str(file.relative_to(output)):sha256_file(file) for file in files}}
                part = parts[method];part["per_image"].append(row)
                part["native_totals"].update(native);part["original_totals"].update(original)
        if position % 25 == 0: print(f"[joint crops] {position}/125",flush=True)
    for method, part in parts.items():
        if part["original_totals"]["gt_words"] != 983 or len(part["per_image"]) != 125:
            raise ValueError("Complete validation denominator changed")
        part["native"] = aggregate_word_metrics(dict(part.pop("native_totals")))
        part["original"] = ratios(dict(part.pop("original_totals")))
        part["count_mae"] = sum(row["absolute_count_error"] for row in part["per_image"])/125
    result = {"status":"SUCCESS","result_class":"approximate reproduction","reproduction_mode":"approximate",
              "protocol":protocol,"branches":{"two_unets":parts},"segmentation_inference":"none; aligned frozen outputs reused"}
    result["environment"] = experiment_environment(torch.device("cpu"))
    result["memory"] = {"model_set":"none; paired segmentation exports cached", "device":"cpu",
        "source_precision":"float32", "native_shape":[256,256], "cpu_threads":args.cpu_threads,
        "gpu_peak_allocated_bytes":None, "gpu_peak_reserved_bytes":None, "oom":False}
    write_json(path,result)
    return result


def gallery(output, geometry):
    parts = geometry["branches"]["two_unets"]
    lines = ["<!doctype html><meta charset='utf-8'><title>Two U-Nets: center guided crops</title>",
             "<style>body{font:16px system-ui;background:#fafafa;margin:24px}img{max-width:100%}section{margin:24px 0}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}</style>",
             "<h1>Approximate diagnostic: two U-Nets</h1><p>125 held-out TRAIN images; predicted masks/vectors determine crops. No official TEST claim.</p>",
             "<table><tr><th>Method</th><th>Crop geometry F1</th><th>Count MAE</th></tr>"]
    for method, part in parts.items():
        lines.append(f"<tr><td>{method}</td><td>{part['original']['geometry_f1']:.4f}</td><td>{part['count_mae']:.3f}</td></tr>")
    lines.append("</table>")
    for i, row in enumerate(parts["baseline"]["per_image"]):
        key = row["sample_id"].replace(":","__")
        counts = "; ".join(method+": "+str(part["per_image"][i]["predicted_words_raw"]) for method,part in parts.items())
        lines.append(f"<section><h2>{row['sample_id']}</h2><p>{counts}; care GT: {len(row['ground_truth'])}</p><a href='samples/{key}/overview.png'><img loading='lazy' src='samples/{key}/overview.png'></a><p><a href='samples/{key}/global_centers_expand/'>Expanded RGB crops</a></p></section>")
    (output/"index.html").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",required=True,choices=("approximate","strict"))
    parser.add_argument("--config",type=Path,default=CONFIG)
    for name in ("base-run","offset-run","target-root","output-root"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--action",choices=("geometry","all"),default="all")
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    parser.add_argument("--cpu-threads",type=int,default=2)
    for name in ("parseq-source","parseq-source-archive","parseq-checkpoint"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--reuse-ocr",type=Path,action="append",default=[])
    args = parser.parse_args()
    if args.cpu_threads < 1: raise ValueError("Positive CPU thread count required")
    torch.set_num_threads(args.cpu_threads)
    cfg = read_config(args.config,args.mode)
    output = args.output_root.resolve()
    for source in (args.base_run.resolve(),args.offset_run.resolve(),args.target_root.resolve(),REPO):
        if output == source or output in source.parents or (source != REPO and source in output.parents):
            raise ValueError("Use an isolated new output folder outside the dataset/source runs")
    sources = load_sources(args,cfg)
    output.mkdir(parents=True,exist_ok=True)
    with (output/"runner.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            geometry = build_geometry(args,cfg,sources)
            gallery(output,geometry)
            result = recognize(args,cfg,geometry) if args.action == "all" else geometry
        except Exception as error:
            from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json
            _atomic_json(output/"failure.json",{"status":"FAILED","result_class":"approximate reproduction",
                "type":type(error).__name__,"message":str(error),"oom":isinstance(error,torch.cuda.OutOfMemoryError),
                "device":args.device,"model_set":"cached two-network outputs; PARSeq alone if requested","precision":"float32"})
            raise
    for method, part in result["branches"]["two_unets"].items():
        print(method,json.dumps(part.get("aggregate",part.get("original"))),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
