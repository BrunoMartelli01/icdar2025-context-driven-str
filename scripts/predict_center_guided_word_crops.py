"""Experimental fresh two-U-Net inference on one RGB image, without annotations."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode, functional as TF

from models.unet_adv import PracticalTotalTextHighResAGUNet
from models.word_center_offsets import WordCenterOffsetAGUNet
from scripts.evaluate_center_guided_word_crops import CONFIG, REPO, prepare_recipes, read_config, render
from scripts.evaluate_word_multitask_crops import write_json
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_word_multitask import experiment_environment
from segmentation.totaltext_image_frequency_augment import letterbox_geometry, letterbox_canvas_mask, remap_letterbox_native_box
from segmentation.training import load_checkpoint, seed_everything


def image_tensor(image, preprocessing):
    """Exactly replay the qualified validation RGB transform, with no target."""
    if preprocessing["input_size"] != [512,512] or preprocessing["native_mask_size"] != [256,256]:
        raise ValueError("Fresh paired v1 inference requires input512/native256")
    fill = preprocessing["variants"]["letterbox"]["parameters"]["canvas_fill_rgb"]
    padded = letterbox_geometry(image.convert("RGB"),fill=tuple(fill))
    resized = TF.resize(padded,[512,512],interpolation=InterpolationMode.BILINEAR,antialias=True)
    return TF.normalize(TF.to_tensor(resized),mean=[.485,.456,.406],std=[.229,.224,.225])


def qualify_pair(base_root, offset_root):
    reports = []
    for root in (base_root,offset_root):
        report = json.loads((root/"run_manifest.json").read_text())
        if (report["status"] != "SUCCESS" or report["result_class"] != "approximate reproduction"
            or sha256_file(root/report["checkpoint"]["path"]) != report["checkpoint"]["sha256"]):
            raise ValueError("Only completed qualified checkpoint pairs are accepted")
        reports.append(report)
    base,offset = reports
    bc,oc = base["run_contract"],offset["run_contract"]
    if (bc["variant"] != "text" or bc["reproduction_mode"] != "approximate"
        or oc["claim_label"] != "approximate reproduction" or oc["base_sha256"] != base["checkpoint"]["sha256"]
        or bc["resolved_preprocessing"] != oc["resolved_preprocessing"]
        or bc["config"]["skip_floor"] != oc["resolved_config"]["skip_floor"]
        or offset["completed_epochs"] != oc["training_config"]["epochs"]):
        raise ValueError("Mismatched base/offset architecture or preprocessing provenance")
    for report in reports:
        for name in ("models/unet_adv.py","models/word_center_offsets.py","segmentation/transforms.py",
                     "segmentation/totaltext_image_frequency_augment.py","segmentation/training.py"):
            if name in report["run_contract"]["source_sha256"]:
                if sha256_file(REPO/name) != report["run_contract"]["source_sha256"][name]:
                    raise ValueError("Qualified inference source changed: "+name)
    return base,offset


def predict_maps(tensor, reports, roots, device):
    """Strict FP32 loading, one model resident at a time; errors propagate."""
    outputs,memory = [],[]
    for cls,report,root,activation in zip(
            (PracticalTotalTextHighResAGUNet,WordCenterOffsetAGUNet),reports,roots,(torch.sigmoid,torch.tanh)):
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        model = cls(skip_floor=reports[0]["run_contract"]["config"]["skip_floor"],encoder_pretrained=False)
        try:
            saved = load_checkpoint(root/report["checkpoint"]["path"],model=model,
                expected_architecture=model.architecture_identity,expected_run_contract=report["run_contract"])
            del saved
            model.to(device).eval()
            with torch.inference_mode():
                output = activation(model(tensor.unsqueeze(0).to(device))).squeeze(0).cpu().numpy()
            expected = (1 if cls is PracticalTotalTextHighResAGUNet else 2,256,256)
            if output.shape != expected or not np.isfinite(output).all():
                raise RuntimeError("Unexpected paired-network output shape or finite values")
            outputs.append(output)
            memory.append({"model_set":cls.__name__,"device":str(device),"dtype":"float32","batch_size":1,
                "input_shape":[1,3,512,512],"sequential_models":True,"oom":False,
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
        finally:
            del model
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    return outputs,memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",required=True,choices=("approximate","strict"))
    parser.add_argument("--config",type=Path,default=CONFIG)
    for name in ("base-run","offset-run","image","output-root"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    parser.add_argument("--cpu-threads",type=int,default=2)
    args = parser.parse_args()
    cfg = read_config(args.config,args.mode)
    if args.cpu_threads < 1: raise ValueError("Positive CPU thread count required")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly choose CPU or execute on the GPU host")
    roots = (args.base_run.resolve(),args.offset_run.resolve())
    reports = qualify_pair(*roots)
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Fresh-image inference requires a NEW empty output directory")
    if output == REPO or output in REPO.parents or any(output == root or root in output.parents or output in root.parents for root in roots):
        raise ValueError("Use an isolated new output directory")
    output.mkdir(parents=True,exist_ok=True)
    seed = reports[0]["run_contract"]["config"]["seed"]
    seed_everything(seed,deterministic_algorithms=True)
    sources = ("scripts/predict_center_guided_word_crops.py","scripts/evaluate_center_guided_word_crops.py",
        "localize/center_guided_word_crops.py","localize/word_center_offsets.py","localize/word_multitask.py",
        "localize/word_guided_contours.py","localize/crop_recovery.py")
    protocol = {"config":cfg,"seed":seed,"image_path":str(args.image.resolve()),"image_sha256":sha256_file(args.image),
        "checkpoint_sha256":[r["checkpoint"]["sha256"] for r in reports],
        "source_reports_sha256":[sha256_file(root/"run_manifest.json") for root in roots],
        "source_sha256":{name:sha256_file(REPO/name) for name in sources},
        "preprocessing":reports[0]["run_contract"]["resolved_preprocessing"],
        "hardware_classification":"HARDWARE_DEVIATION: batch1, sequential FP32 models on explicit device"}
    write_json(output/"protocol.json",protocol)
    start = time.perf_counter()
    try:
        with Image.open(args.image) as image:
            image = image.convert("RGB");rgb = np.ascontiguousarray(np.asarray(image));tensor = image_tensor(image,protocol["preprocessing"])
        (p,offsets),memory = predict_maps(tensor,reports,roots,device)
        image_wh = (rgb.shape[1],rgb.shape[0])
        recipes,joint = prepare_recipes(p[0],offsets,letterbox_canvas_mask(image_wh).numpy(),image_wh,cfg,
            reports[1]["run_contract"]["resolved_config"]["voting"])
        for center in joint["centers"]:
            x,y = center["vote_peak_xy_native"]
            center["peak_pixel_box_original"] = remap_letterbox_native_box([x,y,x+1,y+1],image_wh,native_size=256)
        exports = []
        for method,recipe in recipes.items():
            for proposal in recipe["proposals"]:
                x1,y1,x2,y2 = proposal["bbox"];crop = np.ascontiguousarray(rgb[y1:y2,x1:x2])
                proposal["crop_rgb_sha256"] = crop_array_hash(crop)
                path = output/method/(str(proposal["instance_id"])+".png")
                path.parent.mkdir(exist_ok=True);Image.fromarray(crop).save(path);exports.append(path)
        render(rgb,recipes,joint,output/"overview.png");exports.append(output/"overview.png")
        np.savez_compressed(output/"predictions.npz",text_probability=p,normalized_offsets=offsets,
            **{method:recipe["labels"] for method,recipe in recipes.items()},vote_density=joint["density"])
        exports.append(output/"predictions.npz")
        result = {"status":"SUCCESS","result_class":"approximate reproduction","reproduction_mode":"approximate",
            "protocol":protocol,"environment":experiment_environment(device),"memory":memory,
            "elapsed_seconds":time.perf_counter()-start,"centers":joint["centers"],"audit":joint["audit"],
            "methods":{method:{"predicted_word_hypotheses":len(recipe["proposals"]),"proposals":recipe["proposals"]} for method,recipe in recipes.items()},
            "exports_sha256":{str(path.relative_to(output)):sha256_file(path) for path in exports}}
        write_json(output/"result.json",result)
    except Exception as error:
        failure = {"status":"FAILED","result_class":"approximate reproduction","type":type(error).__name__,
            "message":str(error),"oom":isinstance(error,torch.cuda.OutOfMemoryError),"device":str(device),
            "model_set":"sequential base and auxiliary U-Nets","dtype":"float32","batch_size":1,"input_shape":[1,3,512,512]}
        if device.type == "cuda":
            failure.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(output/"failure.json",failure)
        raise
    print(json.dumps({method:row["predicted_word_hypotheses"] for method,row in result["methods"].items()}),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
