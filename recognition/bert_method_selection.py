"""Auditable, training-free BERT method controls; inference never consumes GT."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from rapidfuzz.distance import Levenshtein


@dataclass(frozen=True)
class MethodPolicy:
    minimum_source_length: int
    insertion_max_edits: int
    minimum_insertion_probability: float
    candidate_minimum_probability: float
    maximum_candidates: int
    pll_edit_penalty: float
    pll_minimum_gain: float
    rerank_oov_only: bool

    def __post_init__(self):
        for x in (self.minimum_source_length,self.insertion_max_edits,self.maximum_candidates):
            if type(x) is not int or x < 1:
                raise ValueError("Invalid BERT method integer bound")
        for x in (self.minimum_insertion_probability,self.candidate_minimum_probability):
            if not math.isfinite(x) or not 0 < x <= 1:
                raise ValueError("Invalid BERT method probability bound")
        for x in (self.pll_edit_penalty,self.pll_minimum_gain):
            if not math.isfinite(x) or x < 0:
                raise ValueError("Invalid BERT PLL penalty/gain")
        if type(self.rerank_oov_only) is not bool:
            raise ValueError("Invalid OOV reranking policy")


def restore_case(text,original):
    return text.upper() if original.isupper() else text.title() if original.istitle() else text.lower()


def only_insertions(original,candidate,max_edits):
    left,right=original.casefold(),candidate.casefold()
    if not 1 <= len(right)-len(left) <= max_edits:
        return False
    i=0
    for char in right:
        if i < len(left) and char == left[i]:
            i+=1
    return i==len(left)


def insertion_selection(original,neural,policy):
    keep={"text":original,"reason":"KEEP"}
    if neural["status"] != "scored":
        return {**keep,"reason":neural["status"]}
    if len(original) < policy.minimum_source_length:
        return {**keep,"reason":"short_source_KEEP"}
    if neural["reference_label_ids"]:
        return {**keep,"reason":"known_word_KEEP"}
    candidate=neural["prediction"]
    if (neural["special_prediction"] or not re.fullmatch(r"[A-Za-z]+",candidate)
            or not only_insertions(original,candidate,policy.insertion_max_edits)):
        return {**keep,"reason":"not_missing_letters_KEEP"}
    if not math.isfinite(neural["probability"]) or not 0 <= neural["probability"] <= 1:
        raise ValueError("Invalid head probability in insertion selection")
    if neural["probability"] < policy.minimum_insertion_probability:
        return {**keep,"reason":"insertion_probability_KEEP"}
    return {"text":restore_case(candidate,original),"reason":"oov_missing_letters",
            "inserted_characters":len(candidate)-len(original),"probability":neural["probability"]}


def bounded_candidates(original,neural,method_policy,spell_policy):
    output=[{"text":original,"edit_distance":0,"source":"immutable_OCR_KEEP"}]
    if (method_policy.maximum_candidates == 1 or neural["status"] != "scored" or len(original) < method_policy.minimum_source_length
            or method_policy.rerank_oov_only and neural["reference_label_ids"]):
        return output
    seen={original.casefold()}
    bound=spell_policy.long_max_edits if len(original)>=spell_policy.long_word_length else spell_policy.short_max_edits
    for label in neural["top_labels"]:
        text=label["text"]
        if label["probability"] < method_policy.candidate_minimum_probability:
            continue
        if not re.fullmatch(r"[A-Za-z]+",text) or len(text)<spell_policy.min_word_length or text.casefold() in seen:
            continue
        distance=Levenshtein.distance(original.casefold(),text.casefold())
        if distance > bound or distance/max(len(original),len(text)) > spell_policy.max_relative_edit:
            continue
        seen.add(text.casefold())
        output.append({"text":restore_case(text,original),"edit_distance":distance,
                       "source":"trained_head_top_labels","label_id":label["label_id"],
                       "head_probability":label["probability"]})
        if len(output) >= method_policy.maximum_candidates:
            break
    return output


def pll_selection(candidates,scores,policy):
    if not candidates or candidates[0]["edit_distance"] != 0:
        raise ValueError("PLL candidate zero must be immutable OCR KEEP")
    keep={"text":candidates[0]["text"],"reason":"KEEP","selected_index":0,"gain":0.0}
    if len(candidates)==1:
        if scores:
            raise ValueError("No-alternative PLL trace must not contain model scores")
        return {**keep,"reason":"no_eligible_alternative"}
    if len(candidates) != len(scores):
        raise ValueError("PLL candidate/score cardinality differs")
    reference=scores[0]["mean_log_probability"]
    if reference is None:
        return {**keep,"reason":scores[0]["status"]}
    if not math.isfinite(reference):
        raise ValueError("Nonfinite PLL reference score")
    best,gain=0,policy.pll_minimum_gain
    for i,(candidate,score) in enumerate(zip(candidates[1:],scores[1:]),1):
        value=score["mean_log_probability"]
        if value is None:
            continue
        if not math.isfinite(value):
            raise ValueError("Nonfinite PLL candidate score")
        change=value-reference-policy.pll_edit_penalty*candidate["edit_distance"]
        if change > gain:
            best,gain=i,change
    if not best:
        return keep
    return {"text":candidates[best]["text"],"reason":"head_candidate_PLL_gain",
            "selected_index":best,"gain":gain}
