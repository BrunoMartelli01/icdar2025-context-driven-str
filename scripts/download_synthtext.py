"""Acquire the author-linked original SynthText archive without package installs.

This is dataset acquisition, not a scientific evaluation. The existing training
runner remains responsible for its deterministic, SHA-bound 50k image subset.
"""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import time
import urllib.request
import zipfile
import zlib


INFO_HASH = "2dba9518166cbd141534cbf381aa3e99a087e83c"
TORRENT_URL = f"https://academictorrents.com/download/{INFO_HASH}.torrent"
ARIA_URL = "https://github.com/abcfy2/aria2-static-build/releases/download/1.37.0/aria2-x86_64-linux-musl_static.zip"
ARIA_SHA256 = "e0a09b12ef67f35f8a8e4fdddbec851d235b7c31da549d0578bff459032b499a"
GIB = 1024**3


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fetch(url, path, *, max_bytes):
    """Only small metadata/tool files; torrent payloads use aria2 resume."""
    path = Path(path)
    if path.exists():
        return
    temporary = Path(str(path) + ".part")
    print(f"[download] {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "context-str-SynthText-acquisition/1"})
    with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as stream:
        count = 0
        while chunk := response.read(1024**2):
            count += len(chunk)
            if count > max_bytes:
                raise ValueError("Metadata/tool download exceeds its declared size limit")
            stream.write(chunk)
    temporary.replace(path)


def safe_parts(parts):
    if not parts or any(not p or p in (".", "..") or "/" in p or "\\" in p or "\x00" in p for p in parts):
        raise ValueError("Unsafe archive/torrent path")
    return tuple(parts)


def torrent_archive(data, *, expected_info_hash=INFO_HASH):
    """Validate the pinned raw info dictionary and derive the selected index."""
    span = []

    def parse(pos, depth=0):
        if depth > 20 or pos >= len(data):
            raise ValueError("Malformed torrent metadata")
        token = data[pos:pos + 1]
        if token == b"i":
            end = data.index(b"e", pos + 1)
            return int(data[pos + 1:end]), end + 1
        if token in (b"l", b"d"):
            values = [] if token == b"l" else {}
            pos += 1
            while pos < len(data) and data[pos:pos + 1] != b"e":
                if token == b"l":
                    value, pos = parse(pos, depth + 1)
                    values.append(value)
                else:
                    key, pos = parse(pos, depth + 1)
                    if not isinstance(key, bytes) or key in values:
                        raise ValueError("Malformed torrent dictionary")
                    start = pos
                    value, pos = parse(pos, depth + 1)
                    values[key] = value
                    if depth == 0 and key == b"info":
                        span.append((start, pos))
            if pos >= len(data):
                raise ValueError("Truncated torrent metadata")
            return values, pos + 1
        colon = data.index(b":", pos)
        size = int(data[pos:colon])
        end = colon + 1 + size
        if size < 0 or end > len(data):
            raise ValueError("Malformed torrent string")
        return data[colon + 1:end], end

    metadata, end = parse(0)
    if end != len(data) or not isinstance(metadata, dict) or len(span) != 1:
        raise ValueError("Malformed torrent root")
    start, stop = span[0]
    if hashlib.sha1(data[start:stop]).hexdigest() != expected_info_hash:
        raise ValueError("Torrent info hash differs from the author-linked SynthText dataset")
    info = metadata[b"info"]
    if not isinstance(info, dict) or not isinstance(info.get(b"files"), list):
        raise ValueError("Expected the original multi-file SynthText torrent")
    root = safe_parts([info[b"name"].decode("utf-8")])
    matches = []
    for index, entry in enumerate(info[b"files"], 1):
        parts = safe_parts([p.decode("utf-8") for p in entry[b"path"]])
        length = entry[b"length"]
        if type(length) is not int or length <= 0:
            raise ValueError("Invalid torrent file size")
        if parts[-1] == "SynthText.zip":
            matches.append({"index": index, "path": str(PurePosixPath(*root, *parts)), "bytes": length})
    if len(matches) != 1:
        raise ValueError("Torrent must contain exactly one SynthText.zip")
    return matches[0]


def aria_binary(cache, explicit=None):
    if explicit:
        binary = Path(explicit).resolve()
    elif shutil.which("aria2c"):
        binary = Path(shutil.which("aria2c")).resolve()
    else:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("Automatic portable aria2 requires Linux x86_64; supply --aria2-binary")
        archive = cache / "aria2-1.37.0-x86_64.zip"
        fetch(ARIA_URL, archive, max_bytes=16 * 1024**2)
        if digest_file(archive) != ARIA_SHA256:
            raise ValueError("Portable aria2 SHA256 differs from the pinned GitHub release asset")
        binary = cache / "aria2c"
        with zipfile.ZipFile(archive) as opened:
            members = [m for m in opened.infolist() if PurePosixPath(m.filename).name == "aria2c" and not m.is_dir()]
            if len(members) != 1 or members[0].file_size > 32 * 1024**2:
                raise ValueError("Portable aria2 archive does not contain one bounded aria2c executable")
            payload = opened.read(members[0])  # ZIP CRC checked by zipfile.
        if binary.exists():
            if digest_file(binary) != hashlib.sha256(payload).hexdigest():
                raise ValueError("Cached portable aria2 executable changed")
        else:
            temporary = cache / "aria2c.part"
            temporary.write_bytes(payload)
            temporary.chmod(0o755)
            temporary.replace(binary)
    version = subprocess.run([str(binary), "--version"], check=True, capture_output=True, text=True).stdout
    if "BitTorrent" not in version:
        raise RuntimeError("Selected aria2 executable lacks BitTorrent support")
    return binary, {"path": str(binary), "sha256": digest_file(binary), "version": version}


def zip_entries(opened, parent):
    """Check every central-directory path before publishing any member."""
    parent = Path(parent).resolve()
    root = parent / "SynthText"
    if root.is_symlink():
        raise ValueError("SynthText extraction root must not be a symlink")
    entries = []
    seen = set()
    for entry in opened.infolist():
        parts = safe_parts(entry.filename.rstrip("/").split("/"))
        # The original author-linked ZIP also contains this root-level license.
        # Preserve its bytes beside gt.mat; arbitrary root files still fail.
        if parts == ("license.txt",) and not entry.is_dir():
            parts = ("SynthText", "license.txt")
        if parts[0] != "SynthText" or parts in seen or ((entry.external_attr >> 16) & 0o170000) == 0o120000:
            raise ValueError("Unexpected/duplicate/symlink SynthText ZIP member")
        seen.add(parts)
        destination = parent.joinpath(*parts)
        if (destination != root and root not in destination.resolve().parents) or destination.is_symlink():
            raise ValueError("ZIP destination escapes the dataset parent")
        if not entry.is_dir():
            entries.append((entry, destination))
    if not any(e.filename == "SynthText/gt.mat" for e, _ in entries):
        raise ValueError("Original SynthText archive is missing SynthText/gt.mat")
    return entries


def extract_member(opened, entry, destination):
    """Check existing bytes or atomically publish a CRC-verified ZIP member."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        crc = 0
        with destination.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                crc = zlib.crc32(chunk, crc)
        if destination.stat().st_size != entry.file_size or crc != entry.CRC:
            raise ValueError(f"Existing dataset file differs; preserved: {destination}")
    else:
        if shutil.disk_usage(destination.parent).free < entry.file_size + 12 * GIB:
            raise RuntimeError("Insufficient disk for the next retained file plus12GiB experiment reserve")
        temporary = Path(str(destination) + ".synthtext-part")
        with opened.open(entry) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024**2)
        temporary.replace(destination)


def extract_verified(archive, parent):
    """Atomic full extraction; mismatching existing files are never replaced."""
    parent = Path(parent).resolve()
    with zipfile.ZipFile(archive) as opened:
        entries = zip_entries(opened, parent)
        required = sum(e.file_size for e, p in entries if not p.exists())
        if shutil.disk_usage(parent).free < required + 12 * GIB:
            raise RuntimeError("Insufficient disk for extracted images plus the experiment reserve")
        for index, (entry, destination) in enumerate(entries, 1):
            extract_member(opened, entry, destination)
            if index % 10000 == 0:
                print(f"[extract] {index}/{len(entries)} files", flush=True)
    return len(entries)


def subset_settings(config, word_root):
    from scripts.run_unet_detection_methods import read_config
    from segmentation.data import load_segmentation_manifest
    c, _ = read_config(Path(config), "approximate")
    manifest_path = Path(word_root) / c["train_manifest"]
    if digest_file(manifest_path) != c["word_manifest_sha256"]:
        raise ValueError("Qualified Total-Text manifest differs before subset acquisition")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    return {"max_images": c["synthtext"]["max_images"], "seed": c["seed"],
            "coordinate_origin": c["synthtext"]["coordinate_origin"],
            "excluded_image_hashes": sorted({s.image_sha256 for s in manifest.samples}),
            "config_sha256": digest_file(config), "word_manifest_sha256": c["word_manifest_sha256"]}


def extract_subset_verified(archive, parent, settings):
    """Reuse the scientific selector while materializing only visited originals.

    Rejected overlap/duplicate candidates are retained as well: subsequent
    ordinary preparation must encounter the same bytes and make the same skips.
    """
    from segmentation.synthtext_words import prepare_synthtext
    parent = Path(parent).resolve()
    root = parent / "SynthText"
    with zipfile.ZipFile(archive) as opened:
        entries = zip_entries(opened, parent)
        by_path = {path: entry for entry, path in entries}
        annotation = root / "gt.mat"
        extract_member(opened, by_path[annotation], annotation)
        retained = {annotation}
        license_path = root / "license.txt"
        if license_path in by_path:
            extract_member(opened, by_path[license_path], license_path)
            retained.add(license_path)
        def materialize(path):
            if path not in by_path:
                raise FileNotFoundError(f"Original SynthText image absent from ZIP: {path}")
            extract_member(opened, by_path[path], path)
            retained.add(path)
        manifest = prepare_synthtext(root, max_images=settings["max_images"], seed=settings["seed"],
            coordinate_origin=settings["coordinate_origin"],
            excluded_image_hashes=set(settings["excluded_image_hashes"]), materialize_image=materialize)
    return {"manifest": manifest, "extracted_files": len(retained),
            "retained_bytes": sum(p.stat().st_size for p in retained)}


def discard_download_archive(cache, receipt):
    """Remove only this helper's verified cached ZIP after successful acquisition."""
    cache = Path(cache).resolve()
    payload_root = cache / "torrent-data"
    parts = safe_parts(receipt["selected_archive"]["path"].split("/"))
    archive = payload_root.joinpath(*parts)
    if (receipt["torrent_info_hash"] != INFO_HASH or parts[-1] != "SynthText.zip"
            or payload_root not in archive.resolve().parents or archive.is_symlink()):
        raise ValueError("Archive deletion destination differs from the verified acquisition cache")
    if not archive.exists() and receipt.get("archive_retained") is False:
        return
    if archive.exists():
        if (archive.stat().st_size != receipt["selected_archive"]["bytes"]
                or digest_file(archive) != receipt["archive_sha256"]):
            raise ValueError("Cached ZIP changed; preserved rather than deleted")
        archive.unlink()
    receipt.update(archive_retained=False, archive_deleted_unix_time=time.time())
    write_json(cache / "complete.json", receipt)
    print("[SynthText] verified cached ZIP removed; original GT and selected images retained", flush=True)


def validate_root(root, minimum_images):
    from scipy.io import whosmat
    fields = {name: shape for name, shape, _ in whosmat(root / "gt.mat")}
    populations = [fields.get(name) for name in ("imnames", "wordBB", "txt")]
    if any(s is None or len(s) != 2 or s[0] != 1 for s in populations) or len(set(populations)) != 1:
        raise ValueError("SynthText gt.mat must contain aligned1xN imnames, wordBB and txt cells")
    if populations[0][1] < minimum_images:
        raise ValueError(f"SynthText has fewer than the configured {minimum_images} images")
    return {"annotation_images": populations[0][1], "gt_mat_sha256": digest_file(root / "gt.mat")}


def acquire(parent, *, minimum_images, explicit_aria=None, subset=None, discard_archive=False):
    if subset is not None and subset["max_images"] != minimum_images:
        raise ValueError("Subset acquisition must use the configured image count")
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    cache = parent / "synthtext-downloads"
    cache.mkdir(exist_ok=True)
    with (cache / "download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A SynthText download/preparation is already running") from error
        root = parent / "SynthText"
        if root.is_symlink():
            raise ValueError("SynthText root must not be a symlink")
        marker = cache / "extraction_contract.json"
        complete = cache / "complete.json"
        if complete.exists():
            receipt = json.loads(complete.read_text())
            if receipt.get("subset") != subset:
                raise ValueError("Completed subset/full acquisition contract differs; choose another dataset parent")
            if subset is not None and digest_file(cache / "subset_manifest.json") != receipt.get("subset_manifest_sha256"):
                raise ValueError("Completed subset manifest changed")
            check = validate_root(root, minimum_images)
            if (receipt.get("torrent_info_hash") != INFO_HASH or receipt.get("root") != str(root)
                    or receipt.get("zip_crc_verified_on_acquisition") is not True
                    or any(receipt.get(k) != v for k, v in check.items())):
                raise ValueError("Completed SynthText acquisition contract/data changed")
            if discard_archive:
                discard_download_archive(cache, receipt)
            print(f"[SynthText] already acquired and annotation hash verified: {root}", flush=True)
            return root
        if root.exists() and not marker.exists() and any(root.iterdir()):
            raise ValueError("SynthText directory is unmanaged and nonempty; choose another dataset parent")
        torrent = cache / f"{INFO_HASH}.torrent"
        fetch(TORRENT_URL, torrent, max_bytes=8 * 1024**2)
        selected = torrent_archive(torrent.read_bytes())
        contract = {"torrent_info_hash": INFO_HASH, "selected_archive": selected, "root": str(root)}
        if subset is not None:
            contract["subset"] = subset
        if marker.exists() and json.loads(marker.read_text()) != contract:
            raise ValueError("Existing extraction contract differs; choose another dataset parent")
        write_json(marker, contract)
        reserve = (selected["bytes"] + 28 * GIB) if subset is not None else (3 * selected["bytes"] + 12 * GIB)
        if shutil.disk_usage(parent).free < reserve:
            raise RuntimeError(f"Insufficient disk; acquisition requires about{reserve / GIB:.1f}GiB free")
        binary, tool_receipt = aria_binary(cache, explicit_aria)
        payload = cache / "torrent-data"
        payload.mkdir(exist_ok=True)
        command = [str(binary), "--no-conf=true", "--enable-rpc=false", "--check-integrity=true", "--continue=true",
                   "--auto-file-renaming=false", "--file-allocation=none", "--seed-time=0", "--max-upload-limit=512K",
                   "--bt-stop-timeout=600", "--summary-interval=30", "--enable-dht6=false",
                   f"--dht-file-path={cache / 'dht.dat'}", f"--dht-file-path6={cache / 'dht6.dat'}",
                   f"--log={cache / 'download.log'}", "--log-level=notice",
                   f"--select-file={selected['index']}", f"--dir={payload}", str(torrent)]
        if Path("/etc/ssl/certs/ca-certificates.crt").is_file():
            command.insert(1, "--ca-certificate=/etc/ssl/certs/ca-certificates.crt")
        print(f"[SynthText] selected {selected['path']} ({selected['bytes'] / GIB:.2f}GiB); transfer is resumable", flush=True)
        subprocess.run(command, check=True, cwd=cache)
        archive = payload / selected["path"]
        if not archive.is_file() or archive.stat().st_size != selected["bytes"] or Path(str(archive) + ".aria2").exists():
            raise ValueError("SynthText archive download is incomplete; training was not started")
        print("[SynthText] hashing archive and checking ZIP CRC during extraction", flush=True)
        archive_sha = digest_file(archive)
        retained = {}
        if subset is None:
            files = extract_verified(archive, parent)
        else:
            retained = extract_subset_verified(archive, parent, subset)
            write_json(cache / "subset_manifest.json", retained.pop("manifest"))
            retained["subset_manifest_sha256"] = digest_file(cache / "subset_manifest.json")
            files = retained["extracted_files"]
            print(f"[SynthText] retained original files: {retained['retained_bytes'] / GIB:.2f}GiB", flush=True)
        annotation = validate_root(root, minimum_images)
        receipt = {**contract, **annotation, **retained, "archive_sha256": archive_sha, "extracted_files": files,
                             "torrent_sha256": digest_file(torrent), "aria2": tool_receipt,
                             "acquired_unix_time": time.time(), "zip_crc_verified_on_acquisition": True,
                             "zip_crc_scope": "all members" if subset is None else "retained members",
                             "zip_auxiliary_path_policy": {"license.txt": "SynthText/license.txt"},
                             "archive_retained": True, "scientific_evaluation": False}
        write_json(complete, receipt)
        if discard_archive:
            discard_download_archive(cache, receipt)
        print(f"[SynthText] ready: {root}", flush=True)
        return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-parent", type=Path, required=True)
    parser.add_argument("--minimum-images", type=int, required=True)
    parser.add_argument("--aria2-binary", type=Path)
    parser.add_argument("--subset-config", type=Path, help="Explicit approximate experiment configuration for reduced extraction")
    parser.add_argument("--word-root", type=Path, help="Qualified target dataset supplying the same RGB overlap exclusions")
    parser.add_argument("--discard-archive", action="store_true", help="Remove the verified cached ZIP after successful acquisition")
    args = parser.parse_args()
    if args.minimum_images < 1:
        parser.error("--minimum-images must be positive")
    if (args.subset_config is None) != (args.word_root is None):
        parser.error("--subset-config and --word-root must be supplied together")
    subset = subset_settings(args.subset_config, args.word_root) if args.subset_config else None
    acquire(args.dataset_parent, minimum_images=args.minimum_images, explicit_aria=args.aria2_binary,
            subset=subset, discard_archive=args.discard_archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
