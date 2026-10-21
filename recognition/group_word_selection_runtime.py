"""Frozen PARSeq likelihood and CLIP scoring for arbitrary crop groups."""
from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.crop_clip_runtime import _embeddings


def square_canvas(image: np.ndarray) -> Image.Image:
    """Preserve the entire field of view before CLIP's square preprocessing."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
        raise ValueError("expected nonempty RGB uint8 image")
    height, width = image.shape[:2]
    side = max(height, width)
    fill = np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)
    canvas = np.empty((side, side, 3), np.uint8)
    canvas[:] = fill
    x, y = (side - width) // 2, (side - height) // 2
    canvas[y:y + height, x:x + width] = image
    return Image.fromarray(canvas)


def clip_group_scores(model, processor, image: np.ndarray, evidence: list[dict], *,
                      templates: dict[str, str], device: str, batch_size: int) -> dict:
    """Every competing word sees the SAME scene/group image embedding.

    Full scene is encoded once. A group uses its fixed representative source
    bbox, without ranking crop images by their own unrelated similarities.
    """
    if batch_size < 1 or not templates or any(template.count("{text}") != 1 for template in templates.values()):
        raise ValueError("invalid CLIP batch or prompt templates")
    active = [group for group in evidence if group["words"]]
    images = [square_canvas(image)]
    for group in active:
        x1, y1, x2, y2 = group["reference_bbox"]
        images.append(square_canvas(np.ascontiguousarray(image[y1:y2, x1:x2])))
    # CLIP's image processor can infer (3, 3, 3) as channels-first for tiny
    # word-group crops. PIL RGB inputs are always HWC; declare that explicitly.
    def process_rgb_images(*, images, return_tensors):
        return processor(images=images, return_tensors=return_tensors,
                         input_data_format="channels_last")

    image_vectors = _embeddings(model, process_rgb_images, images=images, texts=None,
                                device=device, batch_size=batch_size)
    prompts = sorted({template.format(text=word) for group in active for word in group["words"]
                      for template in templates.values()})
    text_vectors = _embeddings(model, processor, images=None, texts=prompts,
                               device=device, batch_size=batch_size)
    vectors = dict(zip(prompts, text_vectors))
    scores = {group["group_id"]: {} for group in evidence}
    for index, group in enumerate(active, 1):
        for word in sorted(group["words"]):
            values = {}
            for name, template in templates.items():
                text = vectors[template.format(text=word)]
                values[f"clip_full_{name}"] = float(image_vectors[0] @ text)
                values[f"clip_group_{name}"] = float(image_vectors[index] @ text)
            if not all(np.isfinite(value) for value in values.values()):
                raise RuntimeError("nonfinite CLIP score")
            scores[group["group_id"]][word] = values
    return scores


def forced_likelihood_batch(recognizer, images: list[np.ndarray], texts: list[str],
                            image_keys: list[str]) -> list[dict]:
    """Causal teacher-forced P(raw text + EOS | crop), with frozen weights.

    Encode each distinct image once within the batch. Candidate-dependent prefix
    decoding is essential: logits from a different greedy history are invalid.
    PAD steps are excluded, EOS included, score normalized by raw length + 1.
    """
    import torch
    if not texts or len(images) != len(texts) or len(texts) != len(image_keys):
        raise ValueError("invalid paired likelihood batch")
    system, tokenizer = recognizer.model, recognizer.model.tokenizer
    model = system.model
    if any(not text for text in texts):
        raise ValueError("empty PARSeq candidate")
    supported = [i for i, text in enumerate(texts) if len(text) <= model.max_label_length]
    if len(supported) != len(texts):
        # Greedy decoding can emit max_label_length + 1 characters without EOS.
        # The learned positional queries cannot score that entire text AND EOS.
        # Preserve the reading, report the architectural limit, never truncate.
        scored = forced_likelihood_batch(recognizer, [images[i] for i in supported],
            [texts[i] for i in supported], [image_keys[i] for i in supported]) if supported else []
        by_index = dict(zip(supported, scored))
        return [by_index[i] if i in by_index else {
            "raw_text": text, "status": "UNSUPPORTED_LENGTH",
            "max_label_length": int(model.max_label_length),
            "mean_log_probability_with_eos": None,
            "reason": "No learned query for complete raw text plus EOS; no truncation; other selectors retain reading"
        } for i, text in enumerate(texts)]
    unique, indices = {}, []
    tensors = []
    for image, key in zip(images, image_keys):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
            raise ValueError("invalid likelihood RGB crop")
        if key not in unique:
            unique[key] = len(tensors)
            tensors.append(recognizer.transform(Image.fromarray(image)))
        indices.append(unique[key])
    device = recognizer.config.device
    with torch.inference_mode():
        memory = model.encode(torch.stack(tensors).to(device))
        memory = memory[torch.tensor(indices, device=device)]
        target = tokenizer.encode(texts, device=device)
        tgt_in, tgt_out = target[:, :-1], target[:, 1:]
        content_mask, query_mask = system.generate_attn_masks(torch.arange(target.shape[1], device=device))
        padding = (tgt_in == tokenizer.pad_id) | (tgt_in == tokenizer.eos_id)
        decoded = model.decode(tgt_in, memory, tgt_mask=content_mask,
                               tgt_padding_mask=padding, tgt_query_mask=query_mask)
        log_probabilities = model.head(decoded).log_softmax(-1)
        valid = tgt_out != tokenizer.pad_id
        safe_targets = tgt_out.masked_fill(~valid, tokenizer.eos_id)
        logs = log_probabilities.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        lengths = valid.sum(-1)
        means = ((logs * valid).sum(-1) / lengths).cpu().tolist()
        character_logs = [row[mask].cpu().tolist() for row, mask in zip(logs, valid)]
    result = []
    for text, mean, token_logs in zip(texts, means, character_logs):
        if len(token_logs) != len(text) + 1 or not np.isfinite(mean) or mean > 0:
            raise RuntimeError("invalid teacher-forced token alignment or likelihood")
        result.append({"raw_text": text, "status": "SCORED", "mean_log_probability_with_eos": float(mean),
                       "token_log_probabilities_with_eos": token_logs,
                       "scoring": "causal teacher forcing; EOS included; PAD excluded"})
    return result
