"""Shared hash-checked crop, PARSeq/CLIP runtime utilities.

No dataset, GT or model path is selected here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
from PIL import Image

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def contained(root: Path, relative: str | Path) -> Path:
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("artifact path must be relative")
    value = (root / part).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes declared root")
    return value

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def cached(path: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contract") != contract:
        raise RuntimeError(f"existing shard contract mismatch: {path}")
    return value

def crop_from_row(image: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    x1, y1, x2, y2 = row["bbox"]
    crop = np.ascontiguousarray(image[y1:y2, x1:x2])
    if crop.size == 0 or list(crop.shape) != row["crop_shape"] or array_sha(crop) != row["crop_array_sha256"]:
        raise RuntimeError(f"crop pixels changed: {row['candidate_id']}")
    return crop

def read_image(data_root: Path, sample: Any, expected_sha: str | None = None) -> tuple[np.ndarray, str]:
    path = contained(data_root, sample.image_path)
    image_sha = sha(path)
    if expected_sha is not None and image_sha != expected_sha:
        raise RuntimeError(f"image SHA256 changed: {sample.sample_id}")
    with Image.open(path) as opened:
        image = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
    return image, image_sha

def _cuda_peak(device: str) -> dict[str, int | None]:
    if not device.startswith("cuda"):
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    import torch
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}

def _embeddings(model: Any, processor: Any, *, images: list[Image.Image] | None,
                texts: list[str] | None, device: str, batch_size: int) -> np.ndarray:
    import torch
    values = images if images is not None else texts
    assert values is not None
    vectors = []
    for start in range(0, len(values), batch_size):
        batch = values[start:start+batch_size]
        if images is not None:
            inputs = processor(images=batch, return_tensors="pt")
            fn = model.get_image_features
            kwargs = {"pixel_values": inputs["pixel_values"].to(device)}
        else:
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=False)
            if inputs["input_ids"].shape[1] > model.config.text_config.max_position_embeddings:
                raise ValueError("CLIP prompt exceeds model context; refusing truncation")
            fn = model.get_text_features
            kwargs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            vector = fn(**kwargs)
            vector = torch.nn.functional.normalize(vector.float(), dim=-1)
        vectors.append(vector.cpu().numpy())
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, model.config.projection_dim), np.float32)

def clip_scores(model: Any, processor: Any, candidates: list[dict[str, Any]],
                image: np.ndarray, *, device: str, batch_size: int,
                templates: dict[str, str]) -> dict[str, dict[str, float]]:
    crops = [Image.fromarray(crop_from_row(image, row), mode="RGB") for row in candidates]
    image_vectors = _embeddings(model, processor, images=crops, texts=None,
                                device=device, batch_size=batch_size)
    strings = [row["ocr"]["raw_text"] for row in candidates]
    shifted = strings.copy()
    by_anchor: dict[int, list[int]] = {}
    for index, row in enumerate(candidates):
        by_anchor.setdefault(row["anchor_rank"], []).append(index)
    for indices in by_anchor.values():
        for offset, index in enumerate(indices):
            shifted[index] = strings[indices[(offset+1) % len(indices)]]
    prompts = []
    for template in templates.values():
        prompts.extend(template.format(text=text) for text in strings)
    prompts.extend(templates["written"].format(text=text) for text in shifted)
    unique = list(dict.fromkeys(prompts))
    text_vectors = _embeddings(model, processor, images=None, texts=unique,
                               device=device, batch_size=batch_size)
    by_text = dict(zip(unique, text_vectors))
    output = {}
    for index, row in enumerate(candidates):
        score = {name: float(image_vectors[index] @ by_text[template.format(text=strings[index])])
                 for name, template in templates.items()}
        score["shuffled_written"] = float(
            image_vectors[index] @ by_text[templates["written"].format(text=shifted[index])])
        if not all(np.isfinite(value) for value in score.values()):
            raise RuntimeError("nonfinite CLIP similarity")
        output[row["candidate_id"]] = score
    return output


def clip_snapshot_file_hashes(model_dir: Path) -> dict[str, str]:
    """Hash model files, excluding mutable Hugging Face local-dir metadata."""
    return {str(path.relative_to(model_dir)): sha(path)
            for path in sorted(model_dir.rglob("*"))
            if path.is_file() and ".cache" not in path.relative_to(model_dir).parts}

def run_clip(samples: list[Any], prepared: dict[str, dict[str, Any]],
             ocr: dict[str, dict[str, Any]], *, data_root: Path, output_dir: Path,
             model_dir: Path, revision: str, device: str, batch_size: int,
             templates: dict[str, str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("CLIP revision must be a full 40-character commit SHA")
    if model_dir.name != revision:
        raise ValueError("CLIP model directory must be a snapshot named by its full commit SHA")
    safetensors = model_dir / "model.safetensors"
    pytorch_bin = model_dir / "pytorch_model.bin"
    if safetensors.is_file():
        use_safetensors = True
    elif pytorch_bin.is_file():
        use_safetensors = False
    else:
        raise FileNotFoundError(f"local pinned CLIP weights missing: {model_dir}")
    files = clip_snapshot_file_hashes(model_dir)
    for required in ("config.json", "preprocessor_config.json", "tokenizer.json"):
        if required not in files:
            raise FileNotFoundError(f"CLIP snapshot missing {required}: {model_dir}")
    identity = {"revision": revision, "files_sha256": files,
                "weight_format": "safetensors" if use_safetensors else "pytorch_bin",
                "templates": templates}
    from transformers import CLIPModel, CLIPProcessor
    import torch
    model = None
    output = {}
    shard_peaks = []
    try:
        for number, sample in enumerate(samples, 1):
            shard_key = sample.sample_id.replace(":", "__")
            ocr_path = output_dir / "ocr" / f"{shard_key}.json"
            contract = {"ocr_sha256": sha(ocr_path), "clip": identity,
                        "device": device, "batch_size": batch_size, "precision": "float32"}
            path = output_dir / "clip" / f"{shard_key}.json"
            shard = cached(path, contract)
            rows = [{**row, "ocr": ocr[sample.sample_id][row["candidate_id"]]}
                    for row in prepared[sample.sample_id]["candidates"]]
            ids = [row["candidate_id"] for row in rows]
            if shard is None:
                if model is None:
                    if device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True,
                                                      use_safetensors=use_safetensors).eval().to(device)
                    vision = model.config.vision_config
                    if (vision.patch_size != 32 or vision.hidden_size != 768
                            or vision.num_hidden_layers != 12 or model.config.projection_dim != 512):
                        raise RuntimeError("CLIP snapshot is not the declared ViT-B/32 architecture")
                    processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)
                image, _ = read_image(data_root, sample, prepared[sample.sample_id]["contract"]["image_sha256"])
                scores = clip_scores(model, processor, rows, image, device=device,
                                     batch_size=batch_size, templates=templates)
                shard = {"contract": contract, "reproduction_mode": "approximate",
                         "claim_label": "approximate reproduction; TRAIN diagnostic intermediate",
                         "rows": [{"candidate_id": candidate_id, "scores": scores[candidate_id]}
                                  for candidate_id in ids], "cuda_peak": _cuda_peak(device)}
                write_json(path, shard)
            if [row["candidate_id"] for row in shard["rows"]] != ids:
                raise RuntimeError(f"CLIP shard candidate ordering mismatch: {sample.sample_id}")
            if device.startswith("cuda") and shard.get("cuda_peak", {}).get("peak_reserved_bytes") is None:
                raise RuntimeError("CLIP CUDA shard lacks memory peak")
            shard_peaks.append(shard.get("cuda_peak", {}))
            output[sample.sample_id] = {row["candidate_id"]: row["scores"] for row in shard["rows"]}
            print(f"[CLIP] {number}/{len(samples)} sample={sample.sample_id}", flush=True)
        peak = {key: max((v[key] for v in shard_peaks if v.get(key) is not None), default=None)
                for key in ("peak_allocated_bytes", "peak_reserved_bytes")}
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return output, {**identity, **peak}
