"""Experimental frozen CLIP verification of PARSeq words on a fixed image.

APPROXIMATE only. Cosines and margins are evidence, not calibrated correctness
probabilities. Candidate generation and selection never receive annotations.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from recognition.group_word_selection_runtime import square_canvas
from scripts.crop_clip_runtime import _embeddings, crop_from_row


def word_pools(prepared: dict, ocr: dict) -> list[dict]:
    """Threshold-only raw PARSeq strings; keep the baseline box for every arm."""
    readings = {row["candidate_id"]: row for row in ocr["rows"]}
    if len(readings) != len(ocr["rows"]):
        raise ValueError("duplicate PARSeq candidate ID")
    if set(readings) != {row["candidate_id"] for row in prepared["candidates"]}:
        raise ValueError("PARSeq/prepared candidate coverage differs")
    anchors: dict[int, dict] = {}
    for row in prepared["candidates"]:
        if row["baseline"]:
            if row["anchor_rank"] in anchors:
                raise ValueError("duplicate baseline anchor")
            anchors[row["anchor_rank"]] = {
                "anchor_rank": row["anchor_rank"], "reference": row,
                "baseline_text": readings[row["candidate_id"]]["raw_text"],
                "baseline_parseq": readings[row["candidate_id"]]["mean_token_log_probability"], "words": {}}
    for row in prepared["candidates"]:
        if not any(origin["padding_native_pixels"] == 0 for origin in row["origins"]):
            continue
        pool = anchors[row["anchor_rank"]]
        reading = readings[row["candidate_id"]]
        text, confidence = reading["raw_text"], reading["mean_token_log_probability"]
        if confidence is not None and (not math.isfinite(confidence) or confidence > 0):
            raise ValueError("invalid native PARSeq mean token log probability")
        entry = pool["words"].setdefault(text, {"source_ids": [], "parseq": None})
        entry["source_ids"].append(row["candidate_id"])
        if confidence is not None:
            entry["parseq"] = max(confidence, entry["parseq"]) if entry["parseq"] is not None else confidence
    pools = [anchors[key] for key in sorted(anchors)]
    for pool in pools:
        base = pool["baseline_text"]
        if base not in pool["words"]:
            raise ValueError("baseline missing from word pool")
        pool["words"] = dict(sorted(pool["words"].items(), key=lambda item: (item[0] != base, item[0])))
    return pools


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("degenerate CLIP feature or context residual")
    return vector / norm


def context_prompts(tokenizer: Any, words: list[str], caption: str, *,
                    template: str, max_tokens: int, max_caption_words: int) -> tuple[dict, dict]:
    """Shorten only the shared caption prefix; never truncate a candidate word."""
    if not words or max_tokens < 1 or max_caption_words < 1:
        raise ValueError("invalid CLIP context budget")
    parts = caption.split()[:max_caption_words]
    while True:
        context = " ".join(parts)
        prompts = {word: template.format(text=word, caption=context) for word in words}
        tokens = tokenizer(list(prompts.values()), padding=False, truncation=False)["input_ids"]
        if all(len(value) <= max_tokens for value in tokens):
            return prompts, {"caption_prefix": context, "original_words": len(caption.split()),
                             "used_words": len(parts), "prompt_token_lengths": [len(v) for v in tokens]}
        if not parts:
            raise ValueError("complete candidate word exceeds CLIP context; refusing word truncation")
        parts.pop()


def image_views(image: np.ndarray, pool: dict, expansion: float) -> tuple[dict, list[int]]:
    if not math.isfinite(expansion) or expansion < 1:
        raise ValueError("surround expansion must be at least one")
    crop = crop_from_row(image, pool["reference"])
    x1, y1, x2, y2 = pool["reference"]["bbox"]
    dx, dy = (x2 - x1) * (expansion - 1) / 2, (y2 - y1) * (expansion - 1) / 2
    box = [max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
           min(image.shape[1], math.ceil(x2 + dx)), min(image.shape[0], math.ceil(y2 + dy))]
    sx1, sy1, sx2, sy2 = box
    surround = np.ascontiguousarray(image[sy1:sy2, sx1:sx2])
    masked = surround.copy()
    # Same surrounding pixels, removing the target's written letters.
    masked[y1-sy1:y2-sy1, x1-sx1:x2-sx1] = np.median(surround.reshape(-1, 3), axis=0).astype(np.uint8)
    return {"crop": square_canvas(crop), "surround": square_canvas(surround),
            "masked_surround": square_canvas(masked)}, box


def score_words(model: Any, processor: Any, image: np.ndarray, pools: list[dict], *,
                caption: str, shuffled_caption: str, config: dict, device: str) -> dict:
    """All alternatives in an anchor share identical image vectors in each view."""
    words = sorted({word for pool in pools for word in pool["words"] if word})
    if not words:
        return {"pools": pools, "scores": {str(p["anchor_rank"]): {} for p in pools},
                "prompts": {}, "context_budget": {}}
    templates, execution = config["templates"], config["execution"]
    context, budget = context_prompts(processor.tokenizer, words, caption,
        template=config["context_template"], max_tokens=model.config.text_config.max_position_embeddings,
        max_caption_words=config["max_caption_words"])
    shuffled, shuffled_budget = context_prompts(processor.tokenizer, words, shuffled_caption,
        template=config["context_template"], max_tokens=model.config.text_config.max_position_embeddings,
        max_caption_words=config["max_caption_words"])
    # Match the neutral prompt syntax exactly when removing the caption component.
    neutral = config["context_template"].format(text="", caption=budget["caption_prefix"])
    prompts = sorted({template.format(text=word) for word in words for template in templates.values()}
                     | set(context.values()) | set(shuffled.values()) | {neutral})
    vectors = _embeddings(model, processor, images=None, texts=prompts, device=device,
                          batch_size=execution["batch_size"])
    text_vectors = dict(zip(prompts, vectors))
    representations = {}
    for word in words:
        base = {name: text_vectors[template.format(text=word)] for name, template in templates.items()}
        representations[word] = {
            "raw": base["raw"], "written": base["written"],
            "ensemble": unit(np.mean([base[name] for name in config["ensemble_templates"]], axis=0)),
            "context": text_vectors[context[word]],
            "context_residual": unit(text_vectors[context[word]] - text_vectors[neutral]),
            "shuffled_context": text_vectors[shuffled[word]]}
    images = [square_canvas(image)]
    boxes = {}
    for pool in pools:
        views, box = image_views(image, pool, config["surround_expansion"])
        images.extend(views.values())
        boxes[str(pool["anchor_rank"])] = box

    def rgb_processor(*, images, return_tensors):
        return processor(images=images, return_tensors=return_tensors, input_data_format="channels_last")

    image_vectors = _embeddings(model, rgb_processor, images=images, texts=None, device=device,
                               batch_size=execution["batch_size"])
    scores, neutral_scores = {}, {}
    for index, pool in enumerate(pools):
        view_vectors = {"full": image_vectors[0], "crop": image_vectors[1+3*index],
                        "surround": image_vectors[2+3*index], "masked_surround": image_vectors[3+3*index]}
        key = str(pool["anchor_rank"])
        scores[key] = {word: {f"{view}_{prompt}": float(iv @ tv)
                             for view, iv in view_vectors.items() for prompt, tv in representations[word].items()}
                       for word in pool["words"] if word}
        neutral_scores[key] = {view: float(iv @ text_vectors[neutral]) for view, iv in view_vectors.items()}
    if not all(math.isfinite(v) for anchor in scores.values() for word in anchor.values() for v in word.values()):
        raise RuntimeError("nonfinite CLIP score")
    return {"pools": pools, "scores": scores, "prompts": {"contexts": context, "shuffled": shuffled,
            "templates": templates, "neutral": neutral}, "context_budget": budget,
            "shuffled_context_budget": shuffled_budget, "surround_boxes": boxes,
            "caption_only_scores": neutral_scores}


def best_word(pool: dict, scores: dict[str, float], *, margin: float = 0) -> str:
    """Strict gain; ties preserve baseline. Empty strings cannot replace a word."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("invalid margin")
    base = pool["baseline_text"]
    if not scores:
        return base
    if any(not word or not math.isfinite(score) for word, score in scores.items()):
        raise ValueError("selection requires finite scores for nonempty words")
    winner = max(scores, key=lambda word: (scores[word], word == base, -list(pool["words"]).index(word)))
    if winner != base and scores[winner] > scores.get(base, -math.inf) + margin:
        return winner
    return base


def selections(scored: dict, config: dict) -> dict:
    arms: dict[str, dict[str, str]] = {"baseline": {}, "parseq_confidence": {},
                                    "rank_fusion": {}, "agreement_gate": {}}
    features = [f"{view}_{prompt}" for view in ("full", "crop", "surround", "masked_surround")
                for prompt in ("raw", "written", "ensemble", "context", "context_residual", "shuffled_context")]
    arms.update({feature: {} for feature in features})
    for pool in scored["pools"]:
        key, base = str(pool["anchor_rank"]), pool["baseline_text"]
        arms["baseline"][key] = base
        native = {word: entry["parseq"] for word, entry in pool["words"].items()
                  if word and entry["parseq"] is not None}
        native_winner = best_word(pool, native)
        arms["parseq_confidence"][key] = native_winner
        by_word = scored["scores"][key]
        methods = sorted({name for entry in by_word.values() for name in entry})
        for method in methods:
            vals = {word: entry[method] for word, entry in by_word.items()}
            arms.setdefault(method, {})[key] = best_word(pool, vals)
        visual = {word: entry["crop_ensemble"] for word, entry in by_word.items()}
        eligible = list(native.keys() & visual.keys())
        # Midranks give tied evidence equal influence. No fitted scale/weight.
        ranks = {word: -sum(sum(v > source[word] for v in source.values())
                               + .5 * sum(v == source[word] for v in source.values())
                               for source in (native, visual)) for word in eligible}
        arms["rank_fusion"][key] = best_word(pool, ranks)
        surround = {word: entry["surround_ensemble"] for word, entry in by_word.items()}
        visual_winner = best_word(pool, visual, margin=config["agreement_margin"])
        # Agreement with PARSeq is a conservative proposal, still uncalibrated.
        arms["agreement_gate"][key] = (visual_winner if visual_winner == native_winner
            and visual_winner == best_word(pool, surround) else base)
    # Empty pools retain baseline in all arms, rather than disappearing from metrics.
    for arm in arms.values():
        for pool in scored["pools"]:
            arm.setdefault(str(pool["anchor_rank"]), pool["baseline_text"])
    return arms


def auc(labels: list[bool], scores: list[float]) -> float | None:
    """Pairwise AUROC via midranks; undefined with one class or no observations."""
    if len(labels) != len(scores) or any(not math.isfinite(s) for s in scores):
        raise ValueError("invalid AUROC inputs")
    positive, negative = sum(labels), len(labels) - sum(labels)
    if not positive or not negative:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum, start = 0., 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        rank_sum += ((start + 1 + stop) / 2) * sum(labels[i] for i in order[start:stop])
        start = stop
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)
