"""APPROXIMATE CLIP hypotheses: letter removal, glyph retrieval and stretching.

Candidate strings and prediction boxes remain frozen PARSeq outputs. Rendered
words are probes of the frozen image encoder, never generated OCR predictions.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from recognition.clip_word_verifier import best_word, selections, unit
from recognition.group_word_selection_runtime import square_canvas
from scripts.crop_clip_runtime import _embeddings, array_sha, crop_from_row, sha


def counterfactual_scores(scored: dict) -> dict:
    result = copy.deepcopy(scored)
    for anchor in result["scores"].values():
        for scores in anchor.values():
            for prompt in ("raw", "written", "ensemble", "context_residual"):
                value = scores[f"surround_{prompt}"] - scores[f"masked_surround_{prompt}"]
                if not math.isfinite(value):
                    raise ValueError("nonfinite counterfactual score")
                scores[f"letter_effect_{prompt}"] = value
    return result


def font_identity(config: dict, paths: dict) -> dict:
    identities = {}
    for name, expected in config["font_sha256"].items():
        path = paths[name]
        if sha(path) != expected:
            raise ValueError(f"font checksum differs: {name}: {path}")
        identities[name] = {"path": str(path), "sha256": expected}
    return identities


def render_word(text: str, font: ImageFont.FreeTypeFont, *, padding: int,
                foreground: list[int], background: list[int]) -> Image.Image:
    if not text or not text.isascii() or any(ord(c) < 32 or ord(c) > 126 for c in text):
        raise ValueError("glyph rendering requires complete nonempty printable ASCII PARSeq text")
    if padding < 1:
        raise ValueError("render padding must be positive")
    left, top, right, bottom = font.getbbox(text)
    if right <= left or bottom <= top:
        raise ValueError("candidate has no drawable glyph; refusing a blank substitute")
    image = Image.new("RGB", (right-left+2*padding, bottom-top+2*padding), tuple(background))
    ImageDraw.Draw(image).text((padding-left, padding-top), text, font=font, fill=tuple(foreground))
    return image


def warp_square(image: Image.Image, size: int) -> Image.Image:
    if size < 1:
        raise ValueError("invalid declared CLIP input size")
    return image.resize((size, size), resample=Image.Resampling.BICUBIC)


class GlyphScorer:
    """Reuse exact word/template features within one model's isolated process."""
    def __init__(self, model, processor, *, config: dict, base_config: dict,
                 font_paths: dict[str, Path], device: str):
        self.model, self.processor, self.config, self.base_config = model, processor, config, base_config
        self.device = device
        self.batch_size = config["execution"]["batch_size"]
        self.size = model.config.vision_config.image_size
        self.fonts = {name: ImageFont.truetype(str(font_paths[name]), size=config["font_size"])
                      for name in config["font_sha256"]}
        self.glyphs, self.render_metadata, self.texts = {}, {}, {}

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        def rgb_processor(*, images, return_tensors):
            return self.processor(images=images, return_tensors=return_tensors,
                                  input_data_format="channels_last")
        return _embeddings(self.model, rgb_processor, images=images, texts=None,
                           device=self.device, batch_size=self.batch_size)

    def prepare_words(self, words: list[str]) -> None:
        missing = [word for word in words if word not in self.glyphs]
        images, keys = [], []
        metadata = {word: [] for word in missing}
        for word in missing:
            for name, font in self.fonts.items():
                for polarity in self.config["polarities"]:
                    rendered = render_word(word, font, padding=self.config["render_padding"],
                        foreground=polarity["foreground"], background=polarity["background"])
                    for style, image in (("square", square_canvas(np.asarray(rendered))),
                                         ("stretch", warp_square(rendered, self.size))):
                        images.append(image)
                        keys.append((word, style))
                        metadata[word].append({"font": name, "polarity": polarity["name"],
                            "style": style, "shape": list(np.asarray(image).shape),
                            "rgb_array_sha256": array_sha(np.asarray(image))})
        if images:
            vectors = self.encode_images(images)
            grouped = {word: {"square": [], "stretch": []} for word in missing}
            for (word, style), vector in zip(keys, vectors):
                grouped[word][style].append(vector)
            for word, styles in grouped.items():
                self.glyphs[word] = {style: unit(np.mean(vectors, axis=0)) for style, vectors in styles.items()}
            self.render_metadata.update(metadata)
        missing_texts = [word for word in words if word not in self.texts]
        templates = self.base_config["templates"]
        prompts = sorted({template.format(text=word) for word in missing_texts for template in templates.values()})
        if prompts:
            vectors = _embeddings(self.model, self.processor, images=None, texts=prompts,
                                  device=self.device, batch_size=self.batch_size)
            by_prompt = dict(zip(prompts, vectors))
            for word in missing_texts:
                self.texts[word] = {"written": by_prompt[templates["written"].format(text=word)],
                    "ensemble": unit(np.mean([by_prompt[templates[name].format(text=word)]
                                                for name in self.base_config["ensemble_templates"]], axis=0))}

    def score(self, scored: dict, image: np.ndarray) -> dict:
        result = counterfactual_scores(scored)
        words = sorted({word for pool in result["pools"] for word in pool["words"] if word})
        controls = sorted({word[::-1] for word in words}) if self.config["reverse_word_control"] else []
        self.prepare_words(sorted(set(words) | set(controls)))
        images = []
        for pool in result["pools"]:
            crop = crop_from_row(image, pool["reference"])
            images.extend((square_canvas(crop), warp_square(Image.fromarray(crop), self.size)))
        vectors = self.encode_images(images) if images else []
        for index, pool in enumerate(result["pools"]):
            original, stretched = vectors[2*index:2*index+2]
            for word, scores in result["scores"][str(pool["anchor_rank"])].items():
                glyph = self.glyphs[word]
                scores.update({"glyph_square": float(original @ glyph["square"]),
                    "glyph_stretch": float(stretched @ glyph["stretch"]),
                    "glyph_multiview": float((original @ glyph["square"] + stretched @ glyph["stretch"]) / 2),
                    "stretch_written": float(stretched @ self.texts[word]["written"]),
                    "stretch_ensemble": float(stretched @ self.texts[word]["ensemble"])})
                if self.config["reverse_word_control"]:
                    reverse = self.glyphs[word[::-1]]
                    scores["glyph_order_effect"] = float((original @ (glyph["square"]-reverse["square"])
                        + stretched @ (glyph["stretch"]-reverse["stretch"])) / 2)
                if not all(math.isfinite(value) for value in scores.values()):
                    raise RuntimeError("nonfinite glyph/stretch score")
        result["render_metadata"] = {word: self.render_metadata[word] for word in sorted(set(words) | set(controls))}
        result["reverse_word_controls"] = {word: word[::-1] for word in words} if controls else {}
        return result


def innovative_selections(scored: dict, base_config: dict, config: dict) -> dict:
    chosen = selections(scored, base_config)
    has_glyphs = "render_metadata" in scored
    gates = {"letter_effect_gate": {}}
    if has_glyphs:
        gates["glyph_verified_gate"] = {}
    features = [f"letter_effect_{p}" for p in ("raw", "written", "ensemble", "context_residual")]
    if has_glyphs:
        features += ["glyph_square", "glyph_stretch", "glyph_multiview", "stretch_written", "stretch_ensemble"]
        if config["reverse_word_control"]:
            features.append("glyph_order_effect")
    for feature in features:
        chosen.setdefault(feature, {}).update({str(p["anchor_rank"]): chosen[feature].get(
            str(p["anchor_rank"]), p["baseline_text"]) for p in scored["pools"]})
    for pool in scored["pools"]:
        key, base = str(pool["anchor_rank"]), pool["baseline_text"]
        native = chosen["parseq_confidence"][key]
        vals = {word: v["letter_effect_written"] for word, v in scored["scores"][key].items()}
        grounded = best_word(pool, vals, margin=config["letter_effect_margin"])
        gates["letter_effect_gate"][key] = grounded if grounded == native else base
        glyph = {word: v["glyph_multiview"] for word, v in scored["scores"][key].items() if "glyph_multiview" in v}
        glyph_word = best_word(pool, glyph, margin=config["glyph_margin"])
        # Same raw word must win PARSeq, visible-letter effect, and synthetic glyph retrieval.
        if has_glyphs:
            gates["glyph_verified_gate"][key] = (glyph_word if glyph and glyph_word == grounded == native else base)
    chosen.update(gates)
    return chosen
