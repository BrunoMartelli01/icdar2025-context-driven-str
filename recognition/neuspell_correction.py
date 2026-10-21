"""Frozen NeuSpell BERT word-head decoding for approximate post-OCR studies.

The original noisy word remains in the encoder input. Average its subword
states before the trained word-classification head, following the author
snippet. No MLM head, masked-word replacement or dataset lexicon is used.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha1, sha256
import io
import math
from pathlib import Path
import pickle
import re
from typing import Sequence

from rapidfuzz.distance import Levenshtein

from recognition.bert_correction import eligible_word, local_context


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Vocabulary contains executable/global object: {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Vocabulary contains a persistent object reference")


def git_blob_digest(data: bytes) -> str:
    return sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def convert_vocabulary(path: Path, *, expected_blob: str, expected_labels: int) -> dict:
    """Authenticate the original data-only pickle and preserve classifier order."""
    data = path.read_bytes()
    if git_blob_digest(data) != expected_blob:
        raise ValueError("NeuSpell vocabulary Git blob hash mismatch")
    obj = DataOnlyUnpickler(io.BytesIO(data)).load()
    if not isinstance(obj, dict) or not isinstance(obj.get("idx2token"), dict):
        raise ValueError("Expected the authoritative NeuSpell vocabulary dictionary")
    reverse, forward = obj["idx2token"], obj.get("token2idx")
    if (not isinstance(forward, dict) or len(reverse) != expected_labels
            or len(forward) != expected_labels or set(reverse) != set(range(expected_labels))
            or any(type(k) is not int for k in reverse)):
        raise ValueError("Vocabulary size or contiguous classifier indices differ")
    labels = [reverse[i] for i in range(expected_labels)]
    if (any(not isinstance(word, str) or not word for word in labels)
            or len(set(labels)) != expected_labels
            or any(type(forward.get(word)) is not int or forward[word] != i
                   for i, word in enumerate(labels))):
        raise ValueError("NeuSpell token2idx/idx2token mapping is inconsistent")
    special_ids = []
    for key in ("pad_token", "unk_token"):
        token = obj.get(key)
        if token not in forward:
            raise ValueError(f"Vocabulary lacks its declared {key}")
        special_ids.append(forward[token])
    if "eos_token" in obj:
        if obj["eos_token"] not in forward:
            raise ValueError("Vocabulary EOS mapping differs")
        special_ids.append(forward[obj["eos_token"]])
    return {"schema_version": 1, "source_git_blob_sha1": expected_blob,
            "source_sha256": sha256(data).hexdigest(), "labels": labels,
            "special_ids": sorted(set(special_ids))}


@dataclass(frozen=True)
class SpellPolicy:
    min_word_length: int
    long_word_length: int
    short_max_edits: int
    long_max_edits: int
    max_relative_edit: float
    minimum_probability: float
    minimum_log_probability_gain: float
    neighbours_per_side: int
    min_vertical_overlap: float
    max_horizontal_gap_heights: float
    max_caption_words: int

    def __post_init__(self):
        for value in (self.min_word_length, self.long_word_length, self.short_max_edits,
                      self.long_max_edits, self.neighbours_per_side, self.max_caption_words):
            if type(value) is not int or value < 1:
                raise ValueError("Spell policy integer bounds must be positive")
        if self.long_word_length < self.min_word_length:
            raise ValueError("Long word boundary must be >= minimum word length")
        for value in (self.max_relative_edit, self.min_vertical_overlap, self.minimum_probability):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Spell ratio/probability bounds must be in (0,1]")
        for value in (self.minimum_log_probability_gain, self.max_horizontal_gap_heights):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Spell gain/gap bounds must be finite and nonnegative")


def select_spelling(original: str, evidence: dict, policy: SpellPolicy) -> dict:
    """Two predeclared controls over the same neural prediction, retaining KEEP."""
    keep = {"text": original, "reason": evidence["status"]}
    if evidence["status"] != "scored":
        return {"native_head": keep, "conservative": dict(keep)}
    proposed = evidence["prediction"]
    probability = evidence["probability"]
    if (not isinstance(proposed, str) or not proposed
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("Invalid NeuSpell prediction or probability")
    # Special classes mean pass-through; they never become visible OCR text.
    if evidence["special_prediction"]:
        return {"native_head": {"text": original, "reason": "special_class_KEEP"},
                "conservative": {"text": original, "reason": "special_class_KEEP"}}
    native = {"text": proposed, "reason": "native_word_head"}
    selected = {"text": original, "reason": "KEEP"}
    if proposed.casefold() == original.casefold():
        selected["reason"] = "same_word_KEEP"
        return {"native_head": native, "conservative": selected}
    if (not re.fullmatch(r"[A-Za-z]+", proposed)
            or len(proposed) < policy.min_word_length):
        selected["reason"] = "candidate_shape_KEEP"
        return {"native_head": native, "conservative": selected}
    distance = Levenshtein.distance(original.casefold(), proposed.casefold())
    bound = policy.long_max_edits if len(original) >= policy.long_word_length else policy.short_max_edits
    if distance > bound or distance / max(len(original), len(proposed)) > policy.max_relative_edit:
        selected["reason"] = "edit_bound_KEEP"
    elif probability < policy.minimum_probability:
        selected["reason"] = "probability_bound_KEEP"
    elif evidence["reference_log_probability"] is not None:
        # Sum actual label probabilities of all casefold-equivalent KEEP labels.
        gain = evidence["log_probability"] - evidence["reference_log_probability"]
        if not math.isfinite(gain):
            raise ValueError("Non-finite NeuSpell KEEP-reference gain")
        if gain > policy.minimum_log_probability_gain:
            selected = {"text": proposed, "reason": "in_vocab_gain", "gain": gain}
        else:
            selected["reason"] = "reference_gain_KEEP"
    else:
        # The word head cannot score an arbitrary OOV original string. Do not
        # pretend the UNK score is the probability of that original string.
        selected = {"text": proposed, "reason": "oov_reference_probability_gate", "gain": None}
    if selected["text"] != original:
        if original.isupper():
            selected["text"] = proposed.upper()
        elif original.istitle():
            selected["text"] = proposed.title()
        else:
            selected["text"] = proposed.lower()
    return {"native_head": native, "conservative": selected}


def load_word_head(snapshot: Path, *, device: str, expected_labels: int,
                   allowed_unused_keys: Sequence[str]):
    """Genuine trained classifier; strict loading after explicitly audited dead keys."""
    import torch
    from transformers import BertConfig, BertForTokenClassification

    config = BertConfig.from_pretrained(snapshot, local_files_only=True)
    if config.model_type != "bert" or config.num_labels != expected_labels:
        raise ValueError("NeuSpell BERT config/classifier label count mismatch")
    model = BertForTokenClassification(config)
    state = torch.load(snapshot / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if (not isinstance(state, dict)
            or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items())):
        raise ValueError("Expected a tensor-only NeuSpell state dictionary")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    mismatch = [k for k in set(state) & set(expected) if tuple(state[k].shape) != tuple(expected[k].shape)]
    regenerated = {}
    position_key = "bert.embeddings.position_ids"
    if position_key in extra and position_key in allowed_unused_keys:
        saved, current = state[position_key], model.bert.embeddings.position_ids
        if saved.dtype != current.dtype or saved.shape != current.shape or not torch.equal(saved, current):
            raise RuntimeError("NeuSpell legacy position_ids differs from the regenerated BERT buffer")
        regenerated[position_key] = {"shape": list(saved.shape), "dtype": str(saved.dtype),
                                    "exact_equality_with_current_buffer": True}
    audit = {"missing_keys": missing, "unused_keys": extra, "mismatched_keys": sorted(mismatch),
             "verified_regenerated_buffers": regenerated,
             "source_tensor_dtypes": sorted({str(v.dtype) for v in state.values()}),
             "classifier_weight_shape": list(expected["classifier.weight"].shape),
             "config_architectures": config.architectures,
             "load_state_dict_strict": True, "weights_only": True}
    if missing or mismatch or set(extra) - set(allowed_unused_keys):
        raise RuntimeError(f"NeuSpell checkpoint compatibility failure: {audit}")
    model.load_state_dict({k: v for k, v in state.items() if k in expected}, strict=True)
    del state
    model.to(device).eval()
    return model, audit


class NeuSpellWordHead:
    def __init__(self, tokenizer, model, vocabulary: dict, *, device: str,
                 batch_size: int, max_input_tokens: int, saved_top_k: int):
        if not tokenizer.is_fast or model.config.model_type != "bert":
            raise ValueError("NeuSpell requires genuine BERT and a fast offset tokenizer")
        labels = vocabulary["labels"]
        if (len(labels) != model.config.num_labels or len(set(labels)) != len(labels)
                or not 1 <= saved_top_k <= len(labels) or batch_size < 1
                or not 3 <= max_input_tokens <= model.config.max_position_embeddings):
            raise ValueError("NeuSpell vocabulary/batch/token bounds mismatch")
        self.tokenizer, self.model, self.labels = tokenizer, model.eval(), labels
        self.special_ids = set(vocabulary["special_ids"])
        if any(type(i) is not int or not 0 <= i < len(labels) for i in self.special_ids):
            raise ValueError("NeuSpell special label IDs are invalid")
        self.aliases = defaultdict(list)
        for i, label in enumerate(labels):
            if i not in self.special_ids:
                self.aliases[label.casefold()].append(i)
        self.device, self.batch_size = device, batch_size
        self.max_input_tokens, self.saved_top_k = max_input_tokens, saved_top_k
        self.input_shapes = set()

    def encode_target(self, original: str, left: Sequence[str], right: Sequence[str],
                      caption: str | None) -> tuple[dict | None, dict]:
        prefix = " ".join(left)
        if caption:
            prefix = caption + ". " + prefix
        text = (prefix + " " if prefix else "") + original
        start, end = len(text) - len(original), len(text)
        if right:
            text += " " + " ".join(right)
        encoded = self.tokenizer(text, return_offsets_mapping=True, truncation=False)
        ids = encoded["input_ids"]
        observable = {"encoder_text": text, "input_tokens": len(ids),
                      "target_character_span": [start, end]}
        if len(ids) > self.max_input_tokens:
            return None, {**observable, "status": "input_token_limit"}
        positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
                     if a < end and b > start]
        if not positions or any(ids[i] in self.tokenizer.all_special_ids for i in positions):
            return None, {**observable, "status": "unknown_or_special_subword"}
        observable.update(target_subword_ids=[ids[i] for i in positions], target_positions=positions)
        return {"input_ids": ids, "attention_mask": encoded["attention_mask"],
                "positions": positions}, {**observable, "status": "scored"}

    def predict(self, requests: Sequence[dict]) -> list[dict]:
        import torch

        tasks, output = [], []
        self.input_shapes = set()
        for i, request in enumerate(requests):
            task, observable = self.encode_target(request["original"], request["left"],
                                                   request["right"], request["caption"])
            output.append(observable)
            if task is not None:
                tasks.append((i, task))
        for start in range(0, len(tasks), self.batch_size):
            group = tasks[start:start + self.batch_size]
            batch = self.tokenizer.pad([{k: t[k] for k in ("input_ids", "attention_mask")}
                                       for _, t in group], padding=True, return_tensors="pt")
            self.input_shapes.add(tuple(batch["input_ids"].shape))
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.inference_mode():
                hidden = self.model.bert(**batch, return_dict=True).last_hidden_state
                pooled = torch.stack([hidden[i, t["positions"]].mean(0)
                                      for i, (_, t) in enumerate(group)])
                # In eval, dropout is identity, as in the author's pooled head.
                logits = self.model.classifier(self.model.dropout(pooled)).float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("NeuSpell word head produced non-finite logits")
                logps = logits.log_softmax(-1)
                top_values, top_ids = torch.topk(logps, self.saved_top_k, dim=-1)
                for j, (index, _) in enumerate(group):
                    reference_ids = self.aliases.get(requests[index]["original"].casefold(), [])
                    reference = float(torch.logsumexp(logps[j, reference_ids], dim=0)) if reference_ids else None
                    ids = top_ids[j].cpu().tolist()
                    values = top_values[j].cpu().tolist()
                    # Match the author's argmax, including its first-index tie rule.
                    winner = int(logps[j].argmax())
                    winner_logp = float(logps[j, winner])
                    output[index].update(
                        prediction=self.labels[winner], label_id=winner, log_probability=winner_logp,
                        probability=math.exp(winner_logp), special_prediction=winner in self.special_ids,
                        reference_label_ids=list(reference_ids), reference_log_probability=reference,
                        reference_probability=math.exp(reference) if reference is not None else None,
                        top_labels=[{"label_id": k, "text": self.labels[k],
                                     "log_probability": v, "probability": math.exp(v)}
                                    for k, v in zip(ids, values)])
        return output


def correct_blocks(blocks: Sequence[dict], *, condition: str, caption: str | None,
                   decoder: NeuSpellWordHead, policy: SpellPolicy) -> list[dict]:
    if condition not in {"word_only", "local_ocr", "local_ocr_caption"}:
        raise ValueError("Unknown NeuSpell input condition")
    if condition == "local_ocr_caption" and not caption:
        raise ValueError("NeuSpell caption condition requires a bound caption")
    requests, indices, contexts = [], [], []
    for i, block in enumerate(blocks):
        left, right = ([], []) if condition == "word_only" else local_context(blocks, i, policy)
        scene = " ".join(caption.split()[:policy.max_caption_words]) if condition == "local_ocr_caption" else None
        contexts.append({"left": left, "right": right, "caption": scene})
        if eligible_word(block["text"], policy):
            indices.append(i)
            requests.append({"original": block["text"], "left": left, "right": right, "caption": scene})
    neural = decoder.predict(requests)
    if len(neural) != len(indices):
        raise RuntimeError("NeuSpell crop prediction cardinality mismatch")
    by_index = dict(zip(indices, neural))
    trace = []
    for i, block in enumerate(blocks):
        evidence = by_index.get(i, {"status": "protected_word_shape"})
        trace.append({**{k: block[k] for k in ("rank", "bbox", "crop_rgb_sha256")},
                      "original_text": block["text"], "context": contexts[i], "neural": evidence,
                      "selections": select_spelling(block["text"], evidence, policy)})
    return trace
