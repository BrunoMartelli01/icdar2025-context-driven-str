"""Content-addressed caption cache with all scientifically relevant key inputs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .blip2_captioner import BLIP2Captioner, CaptionResult, stable_hash


def image_sha256(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def caption_cache_key(
    image_hash: str, model_revision: str, prompt_hash: str, generation_config_hash: str
) -> str:
    return stable_hash({
        "image_hash": image_hash,
        "model_revision": model_revision,
        "prompt_hash": prompt_hash,
        "generation_config_hash": generation_config_hash,
    })


class CaptionCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CaptionResult | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        return CaptionResult(**json.loads(path.read_text(encoding="utf-8"))["result"])

    def put(self, key: str, result: CaptionResult, key_inputs: dict[str, str]) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"cache_key": key, "key_inputs": key_inputs, "result": asdict(result)}
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            temporary = Path(stream.name)
        os.replace(temporary, path)
        return path

    def get_or_generate(
        self,
        *,
        image_bytes: bytes,
        model_revision: str,
        prompt_hash: str,
        generation_config_hash: str,
        generate: Callable[[], CaptionResult],
    ) -> tuple[CaptionResult, bool]:
        inputs = {
            "image_hash": image_sha256(image_bytes),
            "model_revision": model_revision,
            "prompt_hash": prompt_hash,
            "generation_config_hash": generation_config_hash,
        }
        key = caption_cache_key(**inputs)
        cached = self.get(key)
        if cached is not None:
            return cached, True
        result = generate()
        self.put(key, result, inputs)
        return result, False


class CachedCaptioner:
    """Captioner-compatible wrapper using a canonical decoded-pixel image hash."""

    def __init__(self, captioner: BLIP2Captioner, cache: CaptionCache) -> None:
        self.captioner, self.cache = captioner, cache
        self.last_cache_hit = False

    def caption(self, image_rgb: Any) -> CaptionResult:
        config = self.captioner.config
        canonical = (
            str((image_rgb.shape, image_rgb.dtype.str)).encode("utf-8")
            + image_rgb.tobytes()
        )
        result, self.last_cache_hit = self.cache.get_or_generate(
            image_bytes=canonical,
            model_revision=config.revision,
            prompt_hash=stable_hash(config.prompt),
            generation_config_hash=stable_hash(config.generation),
            generate=lambda: self.captioner.caption(image_rgb),
        )
        return result

    def release(self) -> dict[str, Any]:
        return self.captioner.release()
