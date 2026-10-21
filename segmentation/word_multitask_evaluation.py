"""Validation and visual exports built on the existing geometry/pixel metrics."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from diagnostics.render_train_visual_sheets import fitted
from localize.word_multitask import word_instances, parseq_word_proposals
from segmentation.metrics import foreground_iou_f1_per_image_macro, foreground_iou_f1_global_micro
from segmentation.totaltext_word_kernel import instance_geometry


def aggregate_word_metrics(totals):
    result = dict(totals)
    tp, proposals, words = [totals[k] for k in ("matched_gt_iou_ge_50", "raw_proposals", "gt_words")]
    result.update(proposal_precision_iou_50=tp / proposals if proposals else 0.0,
                  word_recall_iou_50=tp / words if words else 0.0,
                  proposal_f1_iou_50=2 * tp / (proposals + words) if proposals + words else 0.0,
                  merge_rate=totals["gt_words_touched_by_fused_components"] / words if words else 0.0,
                  split_rate=totals["gt_words_split_across_predicted_components"] / words if words else 0.0,
                  merged_proposal_rate=totals["predicted_components_touching_multiple_gt_words"] / proposals if proposals else 0.0)
    return result


def render_word_maps(rgb, maps, proposals, path):
    """Aligned RGB boxes and up to three native probability/target map panels."""
    panels = [("Word boxes", rgb.copy())]
    draw = ImageDraw.Draw(panels[0][1])
    for proposal in proposals:
        x1, y1, x2, y2 = proposal["bbox"]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline="#00ee44", width=2)
        draw.text((x1, y1), str(proposal["instance_id"]), fill="#ffdd00")
    for name, values in zip(("Text", "Kernel", "Center"), maps):
        array = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
        panels.append((name, Image.fromarray(array)))
    sheet = Image.new("RGB", (320 * len(panels), 344), "white")
    caption = ImageDraw.Draw(sheet)
    for index, (name, panel) in enumerate(panels):
        sheet.paste(fitted(panel, (320, 320)), (index * 320, 24))
        caption.text((index * 320 + 6, 5), name, fill="black")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def evaluate_multitask(model, batches, references, device, config, *, export_dir=None, image_paths=None,
                      compare_postprocessing=False):
    """Native bbox one-to-one IoU matching; ignore predictions only if fully invalid.

    Geometry uses entire predicted instance boxes (no GT clipping/shrinking).
    Owner IDs are clipped to valid targets only for metrics; GT cannot seed
    inference. Lost expected words remain in the recall/merge/split denominator.
    """
    model.eval()
    totals, rows, predictions, targets, validities = {}, [], [], [], []
    alternatives = {method: {"totals": {}, "per_image": []} for method in
                    ("connected_components", "kernel_bfs", "watershed")} if compare_postprocessing else {}
    with torch.inference_mode():
        for batch in batches:
            probabilities = model(batch["image"].to(device)).sigmoid().cpu().numpy()
            for index, sample_id in enumerate(batch["sample_id"]):
                owners, expected, canvas = references[sample_id]
                target, valid = [batch[key][index, 0].numpy().astype(bool) for key in ("mask", "validity")]
                if probabilities[index].shape[-2:] != target.shape:
                    raise ValueError("Validation resolution mismatch")
                labels, audit, foreground = word_instances(probabilities[index], canvas, config["postprocessing"])
                owners = owners.copy()
                owners[~target | ~valid] = 0
                care_labels = labels.copy()
                ignored = []
                for label in np.unique(labels[labels > 0]):
                    if not np.any((labels == label) & valid):
                        care_labels[labels == label] = 0
                        ignored.append(int(label))
                geometry = instance_geometry(care_labels, owners, expected)
                for key, value in geometry.items():
                    totals[key] = totals.get(key, 0) + value
                rows.append({"sample_id": sample_id, **geometry, "audit": audit,
                             "ignored_prediction_ids": ignored})
                for method, alternative in alternatives.items():
                    post = {**config["postprocessing"], "method": method}
                    candidate, candidate_audit, _ = word_instances(probabilities[index], canvas, post)
                    for label in np.unique(candidate[candidate > 0]):
                        if not np.any((candidate == label) & valid):
                            candidate[candidate == label] = 0
                    metrics = instance_geometry(candidate, owners, expected)
                    for key, value in metrics.items():
                        alternative["totals"][key] = alternative["totals"].get(key, 0) + value
                    alternative["per_image"].append({"sample_id": sample_id, **metrics, "audit": candidate_audit})
                predictions.append(torch.from_numpy(foreground)[None, None])
                targets.append(torch.from_numpy(target)[None, None])
                validities.append(torch.from_numpy(valid)[None, None])
                if export_dir is not None:
                    from scripts.prune_segmentation_checkpoints import sha256_file
                    from scripts.train_totaltext_v4_highres_ablation import _json
                    path = Path(export_dir) / sample_id.replace(":", "__")
                    path.mkdir(parents=True, exist_ok=True)
                    if image_paths is None:
                        raise ValueError("RGB image paths required for original-coordinate exports")
                    with Image.open(image_paths[sample_id]) as opened:
                        rgb = opened.convert("RGB")
                    post = config["postprocessing"]
                    proposals = parseq_word_proposals(labels, rgb.size, padding=post["padding"],
                        min_area=post["min_area"], max_instances=post["max_instances"])
                    arrays = dict(probabilities=probabilities[index],
                        labels=labels, text_target=target, kernel_target=batch["kernel"][index, 0].numpy(),
                        center_target=batch["center"][index, 0].numpy(), validity=valid, owners=owners,
                        canvas=canvas, expected_owner_ids=np.asarray(expected, dtype=np.int32))
                    if (path / "maps.npz").exists():
                        with np.load(path / "maps.npz", allow_pickle=False) as cached:
                            if set(cached.files) != set(arrays) or any(not np.array_equal(cached[k], v) for k, v in arrays.items()):
                                raise ValueError("Existing exported maps differ: " + sample_id)
                    else:
                        np.savez_compressed(path / "maps.npz", **arrays)
                    render_word_maps(rgb, probabilities[index], proposals, path / "predictions.png")
                    target_maps = np.stack([target, batch["kernel"][index, 0].numpy(), batch["center"][index, 0].numpy()])
                    render_word_maps(rgb, target_maps, [], path / "targets.png")
                    _json(path / "proposals.json", {"sample_id": sample_id,
                        "result_class": "approximate reproduction", "reproduction_mode": "approximate",
                        "image_path": str(image_paths[sample_id]), "image_sha256": sha256_file(image_paths[sample_id]),
                        "maps_sha256": sha256_file(path / "maps.npz"), "proposals": proposals,
                        "geometry": geometry, "audit": audit})
                    rows[-1]["exports"] = {name: sha256_file(path / name) for name in
                        ("maps.npz", "predictions.png", "targets.png", "proposals.json")}
    if not rows:
        raise ValueError("Empty validation loader")
    pred, target, valid = [torch.cat(x) for x in (predictions, targets, validities)]
    options = {"empty_foreground_policy": "perfect_if_both_empty", "valid_mask": valid}
    macro = foreground_iou_f1_per_image_macro(pred, target, **options).to_dict()
    micro = foreground_iou_f1_global_micro(pred, target, **options).to_dict()
    aggregate = aggregate_word_metrics(totals)
    aggregate.update(foreground_iou=macro["foreground_iou"], foreground_f1_macro=macro["foreground_f1"],
                     foreground_iou_micro=micro["foreground_iou"], foreground_f1_micro=micro["foreground_f1"])
    return {"aggregate": aggregate, "per_image": rows, "pixel_macro": macro, "pixel_micro": micro,
            "postprocessing_ablation": {method: {"aggregate": aggregate_word_metrics(r["totals"]),
                                                 "per_image": r["per_image"]} for method, r in alternatives.items()}}
