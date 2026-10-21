"""Download versioned detector weights, check upstream digests and freeze full SHA256."""
from __future__ import annotations

import argparse
from hashlib import md5, sha256
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

from scripts.analyze_totaltext_crop_word_groups import freeze

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/approximate/totaltext_text_detectors_v1.json"
DEFAULT_PATHS = REPO / "configs/approximate/totaltext_text_detectors_local_paths_v1.json"


def artifact_source(config: dict) -> dict:
    """Checkpoint identity excludes inference controls so weights can be reused."""
    fields = ("package","version","filename","url","archive_member","official_md5","official_sha256_prefix")
    return {key:config[key] for key in fields if key in config}


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = sha256() if algorithm == "sha256" else md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_weight(path: Path, config: dict) -> str:
    actual = file_digest(path)
    if "official_md5" in config and file_digest(path, "md5") != config["official_md5"]:
        raise RuntimeError(f"upstream detector MD5 mismatch: {path}")
    if "official_sha256_prefix" in config and not actual.startswith(config["official_sha256_prefix"]):
        raise RuntimeError(f"upstream detector SHA256 mismatch: {path}")
    if "official_md5" not in config and "official_sha256_prefix" not in config:
        raise ValueError("detector download has no upstream integrity evidence")
    return actual


def prepare(config_path: Path, paths_path: Path, names: list[str] | None = None) -> Path:
    if config_path.resolve().parent != REPO / "configs/approximate":
        raise ValueError("configuration must be in configs/approximate")
    config, paths = json.loads(config_path.read_text()), json.loads(paths_path.read_text())
    if config["reproduction_mode"] != "approximate" or paths["reproduction_mode"] != "approximate":
        raise ValueError("external detectors are approximate experiments")
    root = Path(paths["artifacts_root"])
    root.mkdir(parents=True, exist_ok=True)
    names = names if names is not None else list(config["models"])
    for name in names:
        model = config["models"][name]
        path = root / model["filename"]
        if not path.exists():
            print(f"[download] {name}: {model['url']}", flush=True)
            with tempfile.TemporaryDirectory(prefix=f"download-{name}-", dir=root) as temporary:
                downloaded = Path(temporary) / "download"
                with urllib.request.urlopen(model["url"], timeout=60) as response, downloaded.open("wb") as target:
                    shutil.copyfileobj(response, target, 1024 * 1024)
                candidate = downloaded
                if "archive_member" in model:
                    with zipfile.ZipFile(downloaded) as archive:
                        matching = [member for member in archive.namelist()
                                    if Path(member).name == model["archive_member"]]
                        if len(matching) != 1:
                            raise RuntimeError("expected exactly one qualified CRAFT archive member")
                        candidate = Path(temporary) / model["filename"]
                        with archive.open(matching[0]) as source, candidate.open("wb") as target:
                            shutil.copyfileobj(source, target)
                verify_weight(candidate, model)
                candidate.replace(path)
        digest = verify_weight(path, model)
        freeze(root / f"{name}.json", {"schema_version": 1, "reproduction_mode": "approximate",
            "name": name, "path": str(path.resolve()), "sha256": digest,
            "bytes": path.stat().st_size, "source": artifact_source(model),
            "integrity_evidence": "upstream MD5 or filename SHA256 prefix plus frozen full SHA256"})
        print(f"[verified] {name} sha256={digest}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--models", nargs="+", choices=("craft","db_resnet50","fast_tiny"))
    args = parser.parse_args()
    prepare(args.config, args.paths, args.models)


if __name__ == "__main__":
    main()
