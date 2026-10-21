"""APPROXIMATE filtering of frozen high-recall PARSeq outputs with CLIP.

No new boxes/readings, annotations, dictionaries, or per-image word budgets.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

from recognition.clip_innovative_verifier import font_identity, render_word
from recognition.clip_word_verifier import unit
from recognition.group_word_selection_runtime import square_canvas
from scripts.crop_clip_runtime import _embeddings, array_sha


FEATURES = ("clip_written", "clip_ensemble", "clip_glyph", "clip_textness",
            "clip_word_advantage", "clip_glyph_residual")
JOINT = {
    "native_ensemble": ("native_rank", "clip_ensemble"),
    "native_glyph": ("native_rank", "clip_glyph"),
    "native_textness": ("native_rank", "clip_textness"),
    "native_word_advantage": ("native_rank", "clip_word_advantage"),
    "native_glyph_residual": ("native_rank", "clip_glyph_residual"),
    "native_glyph_textness": ("native_rank", "clip_glyph", "clip_textness"),
    "native_residual_textness": ("native_rank", "clip_glyph_residual", "clip_textness"),
}


def validate_rows(rows: list[dict], image_wh: tuple[int, int]) -> None:
    ids, groups = set(), set()
    for row in rows:
        cid, gid = row["candidate_id"], row["group_id"]
        if cid in ids or gid in groups:
            raise ValueError("duplicate selected crop/group identity")
        ids.add(cid)
        groups.add(gid)
        confidence = row["confidence"]
        if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("invalid native PARSeq confidence")
        if not math.isfinite(row["physical"]) or not 0 <= row["physical"] <= 1:
            raise ValueError("invalid inherited physical evidence")
        x1, y1, x2, y2 = row["bbox"]
        if (any(type(v) is not int for v in row["bbox"])
            or not 0 <= x1 < x2 <= image_wh[0] or not 0 <= y1 < y2 <= image_wh[1]):
            raise ValueError("selected crop must lie in original-image coordinates")
        if not row["raw_text"] or not row["word"]:
            raise ValueError("empty selected reading")


def fit_calibration(records: list[dict]) -> dict[str, list[float]]:
    """Label-free frozen TRAIN empirical CDF; a rank, never a probability."""
    if not records:
        return {}
    values = {key: [] for key in ("native_rank", *FEATURES)}
    for row in records:
        if not math.isfinite(row["confidence"]) or not 0 <= row["confidence"] <= 1:
            raise ValueError("invalid native calibration confidence")
        values["native_rank"].append(row["confidence"])
        for feature in FEATURES:
            value = row["clip"][feature]
            if not math.isfinite(value):
                raise ValueError("nonfinite CLIP feature")
            values[feature].append(value)
    return {key: sorted(v) for key, v in values.items()}


def percentile(reference: list[float], value: float) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    if (reference.ndim != 1 or not len(reference) or not math.isfinite(value)
        or not np.isfinite(reference).all() or np.any(np.diff(reference) < 0)):
        raise ValueError("invalid frozen rank calibration")
    # Mid-CDF gives identical scores to ties and works on future inputs too.
    return float((np.searchsorted(reference, value, side="left")
                  + np.searchsorted(reference, value, side="right")) / (2 * len(reference)))


def admission_scores(row: dict, calibration: dict[str, list[float]], floor: float) -> dict:
    scores = {"confidence": row["confidence"], "physical": row["physical"]}
    active = row["confidence"] >= floor
    if row["clip_status"] != ("SCORED" if active else "NATIVE_PREFILTER_REJECTED"):
        raise ValueError("CLIP status differs from explicit prefilter")
    expected = {f"rank_{feature}" for feature in FEATURES} | set(JOINT)
    if not active:
        # None is an explicit exclusion; threshold zero cannot admit an unscored crop.
        return {**scores, **{key: None for key in expected}}
    ranks = {"native_rank": percentile(calibration["native_rank"], row["confidence"])}
    for feature in FEATURES:
        ranks[feature] = percentile(calibration[feature], row["clip"][feature])
        scores[f"rank_{feature}"] = ranks[feature]
    for name, features in JOINT.items():
        scores[name] = math.prod(ranks[key] for key in features) ** (1 / len(features))
    return scores


def filter_rows(rows: list[dict], scores: list[dict], family: str, threshold: float) -> list[dict]:
    if len(rows) != len(scores) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("invalid admission population/threshold")
    return [row for row, values in zip(rows, scores)
            if values[family] is not None and values[family] >= threshold]


def choose_operating_point(summary: dict, family: str, minimum_retention: float,
                           baseline_correct: int) -> str | None:
    """Development selection protects the same baseline GT identities."""
    eligible = [(name, values) for name, values in summary.items()
                if name.startswith(family + "@") and
                values["retained_baseline_correct"] >= minimum_retention * baseline_correct]
    if not eligible:
        return None
    return min(eligible, key=lambda item: (-item[1]["precision_proxy"],
               -item[1]["retained_baseline_correct"], item[1]["predictions"], item[0]))[0]


class CropAdmissionScorer:
    def __init__(self, model, processor, *, config: dict, fonts: dict[str, Path], device: str):
        self.model, self.processor, self.config, self.device = model, processor, config, device
        self.batch_size = config["execution"]["batch_size"]
        self.fonts = {name: ImageFont.truetype(str(path), size=config["glyph"]["font_size"])
                      for name, path in fonts.items()}
        font_identity(config["glyph"], fonts)
        self.glyphs, self.texts, self.metadata = {}, {}, {}
        self.probes = None

    def encode_images(self, images):
        def processor(*, images, return_tensors):
            return self.processor(images=images, return_tensors=return_tensors,
                                  input_data_format="channels_last")
        return _embeddings(self.model, processor, images=images, texts=None,
                           device=self.device, batch_size=self.batch_size)

    def prepare(self, words: list[str]):
        if self.probes is None:
            probes = self.config["textness_prompts"]
            prompts = probes["positive"] + probes["negative"]
            vectors = _embeddings(self.model, self.processor, images=None, texts=prompts,
                                 device=self.device, batch_size=self.batch_size)
            split = len(probes["positive"])
            blank = [Image.new("RGB", (self.model.config.vision_config.image_size,) * 2,
                               tuple(color)) for color in self.config["glyph"]["blank_colors"]]
            self.probes = {"positive": unit(vectors[:split].mean(axis=0)),
                           "negative": unit(vectors[split:].mean(axis=0)),
                           "blank": unit(self.encode_images(blank).mean(axis=0))}
        missing = sorted(set(words) - self.glyphs.keys())
        images, keys = [], []
        for word in missing:
            self.metadata[word] = []
            for name, font in self.fonts.items():
                for polarity in self.config["glyph"]["polarities"]:
                    rendered = render_word(word, font, padding=self.config["glyph"]["padding"],
                                           foreground=polarity["foreground"], background=polarity["background"])
                    canvas = square_canvas(np.asarray(rendered))
                    images.append(canvas)
                    keys.append(word)
                    self.metadata[word].append({"font": name, "polarity": polarity["name"],
                        "shape": list(np.asarray(canvas).shape), "rgb_sha256": array_sha(np.asarray(canvas))})
        if images:
            grouped = {word: [] for word in missing}
            for word, vector in zip(keys, self.encode_images(images)):
                grouped[word].append(vector)
            self.glyphs.update({word: unit(np.mean(v, axis=0)) for word, v in grouped.items()})
            templates = self.config["word_templates"]
            prompts = [template.format(text=word) for word in missing for template in templates]
            vectors = _embeddings(self.model, self.processor, images=None, texts=prompts,
                                 device=self.device, batch_size=self.batch_size)
            for i, word in enumerate(missing):
                block = vectors[i*len(templates):(i+1)*len(templates)]
                self.texts[word] = {"written": block[0], "ensemble": unit(block.mean(axis=0))}

    def score(self, rows: list[dict], crops: list[np.ndarray]) -> list[dict]:
        if len(rows) != len(crops):
            raise ValueError("crop/scoring population differs")
        self.prepare([row["raw_text"] for row in rows])
        vectors = self.encode_images([square_canvas(crop) for crop in crops]) if crops else []
        output = []
        for row, vector in zip(rows, vectors):
            word = row["raw_text"]
            values = {"clip_written": float(vector @ self.texts[word]["written"]),
                "clip_ensemble": float(vector @ self.texts[word]["ensemble"]),
                "clip_glyph": float(vector @ self.glyphs[word]),
                "clip_textness": float(vector @ (self.probes["positive"] - self.probes["negative"])),
                "clip_word_advantage": float(vector @ (self.texts[word]["ensemble"] - self.probes["negative"])),
                "clip_glyph_residual": float(vector @ (self.glyphs[word] - self.probes["blank"]))}
            if not all(math.isfinite(v) for v in values.values()):
                raise RuntimeError("nonfinite CLIP admission evidence")
            output.append(values)
        return output
