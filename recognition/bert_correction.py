"""Bounded BERT post-OCR correction, exclusively for approximate experiments.

No ground truth, image identifiers, or evaluation results enter this module.
The original PARSeq transcription always participates as the KEEP candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

from rapidfuzz.distance import Levenshtein
from rapidfuzz import process


WORD = re.compile(r"^[a-z]+$")


@dataclass(frozen=True)
class CorrectionPolicy:
    min_word_length: int
    long_word_length: int
    short_max_edits: int
    long_max_edits: int
    max_relative_edit: float
    max_candidates: int
    protect_min_token_probability: float
    edit_penalty: float
    minimum_gain: float
    neighbours_per_side: int
    min_vertical_overlap: float
    max_horizontal_gap_heights: float
    max_caption_words: int

    def __post_init__(self):
        integers = (self.min_word_length, self.long_word_length,
                    self.short_max_edits, self.long_max_edits,
                    self.max_candidates, self.neighbours_per_side,
                    self.max_caption_words)
        if any(type(v) is not int or v < 1 for v in integers):
            raise ValueError("correction integer parameters must be positive integers")
        if self.long_word_length < self.min_word_length:
            raise ValueError("long_word_length must be >= min_word_length")
        if not 0 < self.max_relative_edit <= 1:
            raise ValueError("max_relative_edit must be in (0,1]")
        if not 0 < self.min_vertical_overlap <= 1:
            raise ValueError("min_vertical_overlap must be in (0,1]")
        if not 0 <= self.protect_min_token_probability <= 1:
            raise ValueError("protect_min_token_probability must be in [0,1]")
        for value in (self.edit_penalty, self.minimum_gain, self.max_horizontal_gap_heights):
            if not math.isfinite(value) or value < 0:
                raise ValueError("gate and geometry parameters must be finite and nonnegative")


def eligible_word(text: str, policy: CorrectionPolicy) -> bool:
    # Preserve numbers, punctuation, multiword crops and internal mixed case.
    return (len(text) >= policy.min_word_length
            and WORD.fullmatch(text.lower()) is not None
            and (text.isupper() or text.islower() or text.istitle()))


def _restore_case(word: str, original: str) -> str:
    if original.isupper():
        return word.upper()
    if original.istitle():
        return word.title()
    return word


def candidate_words(original: str, vocabulary: Sequence[str], evidence: Sequence[str],
                    policy: CorrectionPolicy) -> list[dict[str, Any]]:
    """Bounded lexical candidates; never use a dataset-derived lexicon."""
    keep = {"text": original, "source": "PARSeq_KEEP", "edit_distance": 0}
    if not eligible_word(original, policy):
        return [keep]
    base = original.lower()
    cutoff = (policy.long_max_edits if len(base) >= policy.long_word_length
              else policy.short_max_edits)
    observed = {w.lower() for text in evidence for w in re.findall(r"[A-Za-z]+", text)}
    # The model vocabulary is passed once, sorted and excluding continuation tokens.
    matches = process.extract(base, vocabulary, scorer=Levenshtein.distance,
                              score_cutoff=cutoff, limit=None)
    pool = {word: (distance, "BERT_VOCABULARY") for word, distance, _ in matches}
    for word in observed:
        distance = Levenshtein.distance(base, word, score_cutoff=cutoff)
        if distance <= cutoff:
            pool[word] = (distance, "OBSERVED_CONTEXT")
    ranked = []
    for word, (distance, origin) in pool.items():
        if (word == base or not WORD.fullmatch(word)
                or len(word) < policy.min_word_length
                or distance / max(len(base), len(word)) > policy.max_relative_edit):
            continue
        ranked.append((distance, word not in observed, word, origin))
    ranked.sort()
    return [keep] + [
        {"text": _restore_case(word, original), "source": origin, "edit_distance": distance}
        for distance, _, word, origin in ranked[:policy.max_candidates - 1]
    ]


def local_context(blocks: Sequence[dict], target: int, policy: CorrectionPolicy
                  ) -> tuple[list[str], list[str]]:
    """Conservative horizontal-line neighbours, based solely on crop geometry."""
    tx1, ty1, tx2, ty2 = blocks[target]["bbox"]
    left, right = [], []
    for index, block in enumerate(blocks):
        if index == target or not block["text"].strip():
            continue
        x1, y1, x2, y2 = block["bbox"]
        height = min(ty2 - ty1, y2 - y1)
        overlap = max(0, min(ty2, y2) - max(ty1, y1))
        if height <= 0 or overlap / height < policy.min_vertical_overlap:
            continue
        max_gap = policy.max_horizontal_gap_heights * max(ty2 - ty1, y2 - y1)
        if x2 <= tx1 and tx1 - x2 <= max_gap:
            left.append((tx1 - x2, x1, index, block["text"]))
        elif x1 >= tx2 and x1 - tx2 <= max_gap:
            right.append((x1 - tx2, x1, index, block["text"]))
    selected_left = sorted(left)[:policy.neighbours_per_side]
    selected_right = sorted(right)[:policy.neighbours_per_side]
    return ([v[3] for v in sorted(selected_left, key=lambda v: (v[1], v[2]))],
            [v[3] for v in sorted(selected_right, key=lambda v: (v[1], v[2]))])


def choose_candidate(candidates: Sequence[dict], scores: Sequence[dict],
                     policy: CorrectionPolicy) -> dict:
    """Strict gain gate; ties preserve KEEP; these scores are not confidence."""
    if not candidates or len(candidates) != len(scores):
        raise ValueError("candidate and score cardinality mismatch")
    reference = scores[0].get("mean_log_probability")
    if reference is None:
        return {"selected_index": 0, "reason": scores[0]["status"], "gain": None}
    if not math.isfinite(reference):
        raise ValueError("non-finite BERT reference score")
    best, best_gain = 0, policy.minimum_gain
    for index, (candidate, score) in enumerate(zip(candidates[1:], scores[1:]), 1):
        value = score.get("mean_log_probability")
        if value is None:
            continue
        if not math.isfinite(value):
            raise ValueError("non-finite BERT candidate score")
        gain = value - reference - policy.edit_penalty * candidate["edit_distance"]
        if gain > best_gain:
            best, best_gain = index, gain
    return {"selected_index": best,
            "reason": "gain_above_threshold" if best else "KEEP",
            "gain": best_gain if best else 0.0}


class BertWordScorer:
    """Target-only, mean PLL-word-l2r with fixed neighbouring OCR and caption.

    Tokens from the target word at and to the right of the scored token are
    masked. Left target subwords remain visible, following Kauf/Ivanova 2023.
    This target-only mean is an experimental adaptation of sentence-sum PLL.
    """

    def __init__(self, tokenizer, model, *, device: str, batch_size: int, max_tokens: int):
        if not tokenizer.is_fast or tokenizer.mask_token_id is None:
            raise ValueError("BERT requires a fast tokenizer with a MASK token")
        if batch_size < 1 or not 3 <= max_tokens <= model.config.max_position_embeddings:
            raise ValueError("invalid BERT batch size or input token limit")
        self.tokenizer, self.model = tokenizer, model.eval()
        self.device, self.batch_size, self.max_tokens = device, batch_size, max_tokens
        self.vocabulary = tuple(sorted(t for t in tokenizer.get_vocab() if WORD.fullmatch(t)))
        self.max_input_tokens_seen = 0

    def masked_tasks(self, candidate: str, left: Sequence[str], right: Sequence[str],
                     caption: str | None) -> tuple[list[dict], str]:
        prefix = " ".join(left)
        line = (prefix + " " if prefix else "") + candidate
        start, end = len(line) - len(candidate), len(line)
        if right:
            line += " " + " ".join(right)
        if caption is None:
            encoded = self.tokenizer(line, return_offsets_mapping=True, truncation=False)
            target_sequence = 0
        else:
            encoded = self.tokenizer(caption, line, return_offsets_mapping=True, truncation=False)
            target_sequence = 1
        ids = encoded["input_ids"]
        self.max_input_tokens_seen = max(self.max_input_tokens_seen, len(ids))
        if len(ids) > self.max_tokens:
            return [], "input_token_limit"
        positions = [i for i, ((a, b), seq) in enumerate(
            zip(encoded["offset_mapping"], encoded.sequence_ids()))
            if seq == target_sequence and a < end and b > start]
        if not positions or any(ids[p] in self.tokenizer.all_special_ids for p in positions):
            return [], "unknown_or_special_target"
        tasks = []
        for offset, position in enumerate(positions):
            masked = list(ids)
            for future in positions[offset:]:
                masked[future] = self.tokenizer.mask_token_id
            tasks.append({"input_ids": masked, "attention_mask": encoded["attention_mask"],
                          "token_type_ids": encoded["token_type_ids"],
                          "position": position, "label": ids[position]})
        return tasks, "scored"

    def score(self, candidates: Sequence[str], *, left: Sequence[str], right: Sequence[str],
              caption: str | None) -> list[dict]:
        import torch

        all_tasks, output = [], []
        for index, candidate in enumerate(candidates):
            tasks, status = self.masked_tasks(candidate, left, right, caption)
            output.append({"status": status, "token_log_probabilities": [],
                           "sum_log_probability": None, "mean_log_probability": None,
                           "token_count": len(tasks)})
            all_tasks.extend((index, task) for task in tasks)
        for start in range(0, len(all_tasks), self.batch_size):
            group = all_tasks[start:start + self.batch_size]
            features = [{key: task[key] for key in
                         ("input_ids", "attention_mask", "token_type_ids")} for _, task in group]
            padded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            padded = {k: v.to(self.device) for k, v in padded.items()}
            with torch.inference_mode():
                logits = self.model(**padded).logits
                positions = torch.tensor([t["position"] for _, t in group], device=self.device)
                labels = torch.tensor([t["label"] for _, t in group], device=self.device)
                selected = logits[torch.arange(len(group), device=self.device), positions].float()
                logps = selected.log_softmax(-1).gather(1, labels[:, None]).squeeze(1).cpu().tolist()
            if any(not math.isfinite(v) for v in logps):
                raise RuntimeError("BERT produced non-finite token scores")
            for (index, _), value in zip(group, logps):
                output[index]["token_log_probabilities"].append(value)
        for row in output:
            if row["status"] == "scored":
                values = row["token_log_probabilities"]
                if len(values) != row["token_count"]:
                    raise RuntimeError("BERT token score cardinality mismatch")
                row["sum_log_probability"] = sum(values)
                row["mean_log_probability"] = sum(values) / len(values)
        return output


def correct_blocks(blocks: Sequence[dict], *, condition: str, caption: str | None,
                   scorer: BertWordScorer, policy: CorrectionPolicy) -> list[dict]:
    if condition not in {"word_only", "local_ocr", "local_ocr_caption"}:
        raise ValueError("unknown BERT condition")
    if condition == "local_ocr_caption" and not caption:
        raise ValueError("caption condition requires an image-bound caption")
    trace = []
    for index, block in enumerate(blocks):
        neighbours_left, neighbours_right = local_context(blocks, index, policy)
        caption_budget = " ".join(caption.split()[:policy.max_caption_words]) if caption else None
        # Every condition scores the same candidate pool. Otherwise a caption
        # could improve candidate availability and confound the context ablation.
        evidence = [*neighbours_left, *neighbours_right] + ([caption_budget] if caption_budget else [])
        left, right = ([], []) if condition == "word_only" else (neighbours_left, neighbours_right)
        scene = caption_budget if condition == "local_ocr_caption" else None
        candidates = candidate_words(block["text"], scorer.vocabulary, evidence, policy)
        visual_minimum = block["minimum_token_probability"]
        if visual_minimum is not None and not (math.isfinite(visual_minimum) and 0 <= visual_minimum <= 1):
            raise ValueError("Invalid native PARSeq minimum token probability")
        if visual_minimum is not None and visual_minimum >= policy.protect_min_token_probability:
            scores = []
            decision = {"selected_index": 0, "reason": "visual_confidence_protected", "gain": None}
        elif len(candidates) == 1:
            scores = []
            decision = {"selected_index": 0, "reason": "no_eligible_alternative", "gain": None}
        else:
            scores = scorer.score([c["text"] for c in candidates], left=left, right=right, caption=scene)
            decision = choose_candidate(candidates, scores, policy)
        chosen = candidates[decision["selected_index"]]["text"]
        trace.append({"rank": block["rank"], "bbox": block["bbox"],
                      "crop_rgb_sha256": block["crop_rgb_sha256"],
                      "original_text": block["text"], "text": chosen,
                      "parseq_minimum_token_probability": visual_minimum,
                      "left_context": left, "right_context": right, "caption": scene,
                      "candidates": candidates, "scores": scores, "decision": decision})
    return trace
