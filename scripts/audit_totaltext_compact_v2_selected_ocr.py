"""Independent NumPy/SciPy matching and numeric-area audit of selected-v2 OCR."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from rapidfuzz.fuzz import ratio


def match_count(edges):
    if min(edges.shape) == 0 or not edges.any():
        return 0
    return int(np.count_nonzero(maximum_bipartite_matching(csr_matrix(edges), perm_type="column") >= 0))


def independent_metrics(predictions, words, bbox_key, ocr):
    boxes = np.asarray([p[bbox_key] for p in predictions], dtype=float).reshape(-1, 4)
    gt = np.asarray([g["bbox"] for g in words], dtype=float).reshape(-1, 4)
    lower = np.maximum(boxes[:, None, :2], gt[None, :, :2])
    upper = np.minimum(boxes[:, None, 2:], gt[None, :, 2:])
    intersection = np.prod(np.maximum(upper-lower, 0), axis=-1)
    a = np.prod(np.maximum(boxes[:, 2:]-boxes[:, :2], 0), axis=-1)
    b = np.prod(np.maximum(gt[:, 2:]-gt[:, :2], 0), axis=-1)
    union = a[:, None] + b[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    texts = [ocr[p["crop_array_sha256"]]["raw_text"] for p in predictions]
    exact = np.asarray([[s.casefold() == g["text"].casefold() for g in words] for s in texts], dtype=bool).reshape(len(texts), len(words))
    strict = np.asarray([[s == g["text"] for g in words] for s in texts], dtype=bool).reshape(len(texts), len(words))
    fuzzy = np.asarray([[ratio(s.casefold(), g["text"].casefold()) >= 80 for g in words] for s in texts], dtype=bool).reshape(len(texts), len(words))
    return {"gt_instances": len(words), "retained_proposals": len(predictions),
        "nonempty_parseq_outputs": sum(bool(s.strip()) for s in texts),
        "iou_ge_0_25_one_to_one": match_count(iou >= .25),
        "iou_ge_0_50_one_to_one": match_count(iou >= .5),
        "iou_ge_0_75_one_to_one": match_count(iou >= .75),
        "exact_casefold_spatial_one_to_one": match_count((iou >= .5) & exact),
        "exact_case_sensitive_spatial_one_to_one": match_count((iou >= .5) & strict),
        "rapidfuzz_casefold_ge_0_8_spatial_one_to_one": match_count((iou >= .5) & fuzzy),
        "image_level_lexical_exact_multiset_matches": sum((Counter(s.casefold() for s in texts if s.strip())
            & Counter(g["text"].casefold() for g in words)).values())}


def check_distribution(actual, values):
    a = np.asarray(values, dtype=float)
    assert actual["count"] == len(a)
    if len(a):
        expected = {"mean": np.mean(a), "median": np.median(a), "p05": np.quantile(a, .05),
            "p95": np.quantile(a, .95), "min": min(a), "max": max(a)}
        assert all(np.isclose(actual[k], v, rtol=0, atol=1e-12) for k, v in expected.items())
    else:
        assert all(v is None for k, v in actual.items() if k != "count")


def audit(directory, replay_images=0):
    from segmentation.data import file_sha256
    protocol = json.loads((directory / "protocol.json").read_text())
    result = json.loads((directory / "results.json").read_text())
    assert result["status"] == "SUCCESS" and result["protocol_sha256"] == file_sha256(directory / "protocol.json")
    for name, digest in protocol["code_sha256"].items():
        assert file_sha256(Path(name)) == digest, name
    for name, digest in protocol["plan"]["source_sha256"].items():
        assert file_sha256(Path(protocol["source_run"]) / name) == digest, name
    ocr = {}
    for line in (directory / "ocr.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert row["protocol_sha256"] == result["protocol_sha256"]
        assert row["crop_array_sha256"] not in ocr
        ocr[row["crop_array_sha256"]] = row
    gt_data = json.loads((directory / "ground_truth.json").read_text())
    gt = gt_data["ground_truth"]
    assert sum(map(len, gt.values())) == result["care_words_per_checkpoint"]
    for row in gt_data["annotation_provenance"]:
        assert file_sha256(Path(row["path"])) == row["sha256"]
    all_shards, per_sample, used_hashes = {}, [], set()
    for epoch, checkpoint in result["checkpoints"].items():
        shards = [json.loads(p.read_text()) for p in sorted((directory / f"epoch-{int(epoch):04d}/proposals").glob("*.json"))]
        assert len(shards) == result["images_per_checkpoint"]
        assert len({s["sample_id"] for s in shards}) == len(shards)
        assert {s["sample_id"] for s in shards} == set(protocol["sample_ids"]) == set(gt)
        all_shards[epoch] = {s["sample_id"]: s for s in shards}
        summaries = checkpoint["mask_summary"]
        check_distribution(summaries["foreground_pixels_per_image"], [s["mask_measurements"]["foreground_pixels"] for s in shards])
        check_distribution(summaries["foreground_fraction_per_image"], [s["mask_measurements"]["foreground_pixels"]/16384 for s in shards])
        components = [c for s in shards for c in s["mask_measurements"]["components_8_connected"]]
        check_distribution(summaries["all_component_foreground_area_native"], [c["foreground_area_native"] for c in components])
        for s in shards:
            m = s["mask_measurements"]
            assert sum(c["foreground_area_native"] for c in m["components_8_connected"]) == m["foreground_pixels"]
            assert m["foreground_fraction"] == m["foreground_pixels"] / 16384
        for pad in protocol["plan"]["padding_native_pixels"]:
            rows = [p for s in shards for p in s["variants"][str(pad)]["proposals"]]
            used_hashes.update(p["crop_array_sha256"] for p in rows)
            sizes = summaries["retained_by_padding"][str(pad)]
            assert sizes["proposals"] == len(rows)
            for name, field in [("component_foreground_area_native", "foreground_area_native"), ("bbox_area_native", "bbox_area_native")]:
                check_distribution(sizes[name], [p[field] for p in rows])
            for stem, field in [("bbox", "bbox_native"), ("padded_crop", "padded_bbox_native")]:
                for axis, dimension in enumerate(["width", "height"]):
                    check_distribution(sizes[f"{stem}_{dimension}_native"], [p[field][axis+2]-p[field][axis] for p in rows])
            for kind, box in [("unpadded_detection_box", "bbox"), ("padded_crop_box", "padded_bbox_original")]:
                totals = Counter()
                for shard in shards:
                    values = independent_metrics(shard["variants"][str(pad)]["proposals"], gt[shard["sample_id"]], box, ocr)
                    totals.update(values)
                    per_sample.append({"epoch": int(epoch), "padding": pad, "box_protocol": kind,
                        "sample_id": shard["sample_id"], **values})
                metrics = checkpoint["summary"][str(pad)][kind]
                for key, value in totals.items():
                    target = metrics["geometry"] if key in metrics["geometry"] else metrics["recognition"]
                    assert target[key] == value, (epoch, pad, kind, key, target[key], value)
                assert len(checkpoint["correct_gt_ids"][str(pad)][kind]) == totals["exact_casefold_spatial_one_to_one"]
    assert used_hashes == set(ocr) and len(ocr) == result["unique_rgb_crops"]
    replay = None
    if replay_images:
        from hashlib import sha256
        from PIL import Image
        import torch
        from segmentation.config import load_segmentation_config
        from segmentation.evaluate import _preprocess, _model_from_config, validate_checkpoint
        from segmentation.training import seed_everything
        from segmentation.data import load_segmentation_manifest
        manifest = load_segmentation_manifest(Path(protocol["test_manifest"]), require_checksums=True)
        samples_by_id = {s.sample_id: s for s in manifest.samples}
        assert 1 <= replay_images <= len(protocol["sample_ids"])
        torch.set_num_threads(protocol["runtime"]["cpu_threads"])
        seed_everything(protocol["runtime"]["seed"], deterministic_algorithms=True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        config = load_segmentation_config(Path(__file__).resolve().parents[1] / protocol["source_run_contract"]["config"]["source_config"])
        selected = json.loads((Path(protocol["source_run"]) / "checkpoint_selection.json").read_text())
        rechecked = 0
        for epoch, shards in all_shards.items():
            chosen = next(r for r in selected.values() if r["epoch"] == int(epoch))
            model = _model_from_config(config).eval()
            validate_checkpoint(Path(protocol["source_run"])/"checkpoints"/chosen["checkpoint"], model, chosen["checkpoint_sha256"], config.architecture.identity)
            model.cuda()
            for sid in protocol["sample_ids"][:replay_images]:
                shard = shards[sid]
                sample = samples_by_id[sid]
                image = Path(protocol["data_root"])/sample.image_path
                assert file_sha256(image) == shard["image_sha256"]
                with Image.open(image) as im:
                    tensor = _preprocess(im.convert("RGB")).unsqueeze(0).cuda()
                with torch.inference_mode():
                    native = (model(tensor).sigmoid()[0, 0] >= protocol["plan"]["probability_threshold"]).cpu().numpy().astype(np.uint8)
                assert sha256(native.tobytes()).hexdigest() == shard["native_mask_array_sha256"]
                labels, n = ndimage.label(native, structure=np.ones((3, 3), dtype=np.uint8))
                areas = np.bincount(labels.ravel())
                scipy_components = []
                for i, slices in enumerate(ndimage.find_objects(labels), 1):
                    ys, xs = slices
                    scipy_components.append(([xs.start, ys.start, xs.stop, ys.stop], int(areas[i])))
                original_components = [(c["bbox_native"], c["foreground_area_native"]) for c in shard["mask_measurements"]["components_8_connected"]]
                assert sorted(scipy_components) == sorted(original_components)
                assert int(native.sum()) == shard["mask_measurements"]["foreground_pixels"]
                rechecked += 1
            del model, tensor
            gc.collect()
            torch.cuda.empty_cache()
        replay = {"images_replayed": rechecked, "scipy_8_connected_components": "PASS", "native_array_hashes": "PASS",
            "device": "cuda:0", "precision": "float32", "segmentation_batch_size": 1,
            "input_shape": [1, 3, 256, 256], "model_set": ["AuthorProtocolAGUNet"],
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    unexpected = [str(p) for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".pt", ".npy", ".npz"]]
    assert not unexpected
    report = {"status": "PASS", "claim_label": "approximate reproduction", "images_per_checkpoint": result["images_per_checkpoint"],
        "care_words_per_checkpoint": result["care_words_per_checkpoint"], "checkpoint_count": len(result["checkpoints"]),
        "independent_per_image_evaluations": len(per_sample), "independent_metrics": "NumPy IoU + SciPy maximum bipartite matching + Counter lexical counts",
        "size_distributions": "PASS", "native_replay": replay, "saved_rasters_or_weights": 0, "per_sample": per_sample}
    (directory / "audit.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "per_sample"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-directory", required=True, type=Path)
    parser.add_argument("--replay-images", type=int, default=0)
    args = parser.parse_args()
    audit(args.result_directory, args.replay_images)
