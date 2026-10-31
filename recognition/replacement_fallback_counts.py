"""Explicit hash-bound teacher counts for approximate router retraining."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=None)
def load_counts(path, digest):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError("Replacement teacher count archive changed")
    data=json.loads(raw)
    if data["mode"]!="approximate" or data["claim_label"]!="approximate reproduction":
        raise ValueError("Explicit approximate replacement teacher required")
    return data


def replacement_count(config,record):
    ds=record["sample_id"].split(":")[0]
    spec=config["replacement_fallback_counts"][ds]
    archive=load_counts(spec["path"],spec["sha256"])
    row=archive["samples"][record["sample_id"]]
    if row["image_sha256"]!=record["image_sha256"]:
        raise ValueError("Replacement teacher RGB identity differs")
    return dict(row["counts"])
