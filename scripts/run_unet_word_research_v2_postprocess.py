"""Three GT-blind extraction methods on the SAME trained v1 embedding maps."""

import gc
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from localize.unet_word_research import embedding_instances
from localize.unet_word_research_v2 import soft_embedding_instances
from localize.word_multitask import word_instances, parseq_word_proposals
from models.unet_word_research import WordResearchAGUNet
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_unet_word_research_v2 import memory, backend_state, experiment_environment
from scripts.train_totaltext_v4_highres_ablation import _json, _epoch_log
from scripts.train_totaltext_letterbox_word_f1_separator_head import _validation_reference
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.training import seed_everything
from segmentation.totaltext_word_kernel import instance_geometry
from segmentation.word_multitask_evaluation import aggregate_word_metrics


def load_embedding_source(root, config):
    directory = root / "embedding"
    receipt = directory / "run_manifest.json"
    checkpoint = directory / "last.pt"
    if (sha256_file(receipt) != config["embedding_run_manifest_sha256"]
            or sha256_file(checkpoint) != config["embedding_checkpoint_sha256"]):
        raise ValueError("Qualified epoch50 embedding source changed")
    report = json.loads(receipt.read_text()); history = _epoch_log(directory / "epochs.jsonl")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (report.get("status") != "SUCCESS" or not report.get("training_complete")
            or not history or history[-1]["epoch"] != config["embedding_epoch"]
            or history[-1]["last_checkpoint_sha256"] != config["embedding_checkpoint_sha256"]
            or payload.get("architecture") != config["embedding_architecture"]
            or payload.get("epoch") != config["embedding_epoch"]
            or payload.get("run_contract") != report.get("run_contract")
            or payload.get("reproduction_mode") != "approximate"):
        raise ValueError("Embedding epoch50 checkpoint provenance mismatch")
    return payload["model_state_dict"], report["run_contract"]


def run_postprocess(args, config, dataset, val_ids, manifest, transform, split_sha):
    device = torch.device(args.device)
    source_path = args.v1_root.resolve() / "embedding/last.pt"
    state, source_contract = load_embedding_source(args.v1_root.resolve(), config)
    seed_everything(config["seed"], deterministic_algorithms=True)
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    model = WordResearchAGUNet("embedding", skip_floor=config["skip_floor"],
                               embedding_dimensions=config["embedding_dimensions"])
    model.load_state_dict(state, strict=True); del state
    model.to(device).eval()
    references = _validation_reference(manifest, args.word_root.resolve(), val_ids, transform)
    loader = DataLoader(Subset(dataset, val_ids), batch_size=config["batch_size"],
                        num_workers=config["num_workers"], shuffle=False)
    source_files = ("scripts/run_unet_word_research_v2_postprocess.py", "scripts/run_unet_word_research_v2.py",
                    "localize/unet_word_research_v2.py", "localize/unet_word_research.py", "localize/word_multitask.py",
                    "models/unet_word_research.py", "segmentation/unet_word_research.py",
                    "segmentation/totaltext_word_kernel.py", "localize/crop_recovery.py")
    contract = {"reproduction_mode": "approximate", "result_class": "approximate reproduction",
        "execution_kind": "postprocessing_fixed_checkpoint", "resolved_config": config,
        "source_checkpoint_sha256": config["embedding_checkpoint_sha256"], "source_epoch": config["embedding_epoch"],
        "source_run_contract": source_contract, "validation_ids_sha256": split_sha,
        "execution_device": str(device), "backend": backend_state(),
        "runtime_inputs": {"word_root": str(args.word_root.resolve()), "glyph_root": str(args.glyph_root.resolve()),
                           "v1_root": str(args.v1_root.resolve())},
        "source_sha256": {name: sha256_file(Path(name)) for name in source_files}}
    output = args.output_root.resolve()
    rows = {name: [] for name in config["embedding_arms"]}
    totals = {name: {} for name in rows}; pixel = []
    try:
        for name in rows:
            directory = output / name; directory.mkdir(exist_ok=True)
            path = directory / "run_contract.json"
            if path.exists() and json.loads(path.read_text()) != contract:
                raise ValueError("Fixed-map extraction contract changed; use a new output directory")
            _json(path, contract)
        with torch.inference_mode():
            for batch in loader:
                evidence = model.forward_evidence(batch["image"].to(device))
                probabilities = evidence["words"].sigmoid().cpu().numpy()
                embeddings = evidence["embedding"].cpu().numpy()
                for i, sample_id in enumerate(batch["sample_id"]):
                    p, e = probabilities[i], embeddings[i]
                    owners, expected, canvas = references[sample_id]
                    valid = batch["validity"][i, 0].numpy().astype(bool)
                    target = batch["mask"][i, 0].numpy().astype(bool)
                    owners = owners.copy(); owners[~target | ~valid] = 0
                    hard = embedding_instances(p, e, canvas, config["postprocessing"])
                    standard = word_instances(p, canvas, config["postprocessing"])
                    soft = soft_embedding_instances(p, e, canvas, config["postprocessing"], config["soft_embedding"])
                    foreground = hard[0] > 0
                    tp, fp, fn = (int(np.count_nonzero(m & valid)) for m in
                        (foreground & target, foreground & ~target, ~foreground & target))
                    pixel.append((tp / (tp + fp + fn) if tp + fp + fn else 1.,
                                  2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.))
                    with Image.open(batch["image_path"][i]) as image:
                        image_wh = image.size
                    for name, prediction in zip(rows, (hard, standard[:2], soft)):
                        labels, audit = prediction
                        if not np.array_equal(labels > 0, foreground):
                            raise RuntimeError("Extraction variants changed foreground support")
                        care = labels.copy()
                        ignored = [int(v) for v in np.unique(labels[labels > 0]) if not np.any((labels == v) & valid)]
                        care[np.isin(care, ignored)] = 0
                        metrics = instance_geometry(care, owners, expected)
                        for key, value in metrics.items():
                            totals[name][key] = totals[name].get(key, 0) + value
                        directory = output / name / "validation_exports" / sample_id.replace(":", "__")
                        directory.mkdir(parents=True, exist_ok=True)
                        arrays = {"probabilities": p, "embedding": e, "labels": labels, "owners": owners,
                            "validity": valid, "canvas": canvas, "text_target": target,
                            "expected_owner_ids": np.asarray(expected, np.int32)}
                        path = directory / "maps.npz"
                        if path.exists():
                            with np.load(path, allow_pickle=False) as saved:
                                if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                                    raise ValueError("Previously exported fixed maps differ")
                        else:
                            np.savez_compressed(path, **arrays)
                        proposals = parseq_word_proposals(labels, image_wh, padding=0, min_area=0, max_instances=None)
                        _json(directory / "proposals.json", {"sample_id": sample_id,
                            "image_path": batch["image_path"][i], "image_sha256": sha256_file(Path(batch["image_path"][i])),
                            "maps_sha256": sha256_file(path), "proposals": proposals, "quality_scores": {},
                            "result_class": "approximate reproduction", "reproduction_mode": "approximate"})
                        rows[name].append({"sample_id": sample_id, **metrics, "audit": audit,
                            "ignored_instance_ids": ignored, "maps_sha256": sha256_file(path),
                            "proposals_sha256": sha256_file(directory / "proposals.json")})
                print(f"[embedding extraction] {len(rows[config['embedding_arms'][0]])}/{len(val_ids)}", flush=True)
        for name in rows:
            if len(rows[name]) != len(val_ids) or len({x["sample_id"] for x in rows[name]}) != len(val_ids):
                raise ValueError("Incomplete fixed-map validation")
            aggregate = aggregate_word_metrics(totals[name])
            if aggregate["gt_words"] != config["expected_validation_words"]:
                raise ValueError("Fixed-map validation denominator changed")
            aggregate.update(foreground_iou=float(np.mean([x[0] for x in pixel])),
                             foreground_f1_macro=float(np.mean([x[1] for x in pixel])))
            _json(output / name / "validation.json", {"aggregate": aggregate, "per_image": rows[name],
                "run_contract": contract, "checkpoint_sha256": config["embedding_checkpoint_sha256"],
                "result_class": "approximate reproduction", "memory": memory(device, config)})
            receipt = {"status": "SUCCESS", "training_complete": True, "completed_epochs": 0,
                "execution_kind": "postprocessing_fixed_checkpoint", "result_class": "approximate reproduction",
                "run_contract": contract, "checkpoint": {"epoch": config["embedding_epoch"], "path": str(source_path),
                    "sha256": config["embedding_checkpoint_sha256"]}, "validation_word_geometry": aggregate,
                "memory": memory(device, config), "environment": experiment_environment(device)}
            _json(output / name / "run_manifest.json", receipt)
        return {"status": "PASS", "action": "postprocess", "arms": {name: aggregate_word_metrics(totals[name]) for name in rows}}
    except Exception as error:
        _json(output / "postprocess_failure.json", {"status": "FAILED", "error_type": type(error).__name__,
            "message": str(error), "run_contract": contract,
            "memory": memory(device, config, oom=isinstance(error, torch.cuda.OutOfMemoryError))})
        raise
    finally:
        del model; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
