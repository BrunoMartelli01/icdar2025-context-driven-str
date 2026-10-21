"""Offline engineering fixtures, never genuine SynthText qualification."""

import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest
from scipy.io import savemat
from scipy.io.matlab import MatReadError

from scripts import download_synthtext as download


def bencode(value):
    if isinstance(value, bytes):
        return str(len(value)).encode() + b":" + value
    if isinstance(value, int):
        return b"i" + str(value).encode() + b"e"
    if isinstance(value, list):
        return b"l" + b"".join(bencode(x) for x in value) + b"e"
    return b"d" + b"".join(bencode(k) + bencode(v) for k, v in sorted(value.items())) + b"e"


def metadata(files):
    info = {b"name": b"SynthText", b"files": files, b"piece length": 1024, b"pieces": b"x" * 20}
    return bencode({b"info": info}), hashlib.sha1(bencode(info)).hexdigest()


def make_archive():
    stream = io.BytesIO()
    annotations = io.BytesIO()
    cells = np.empty((1, 2), dtype=object)
    cells[0, 0] = cells[0, 1] = np.ones((2, 4))
    savemat(annotations, {"imnames": np.array([["1/a.png", "1/b.png"]], dtype=object),
                         "txt": np.array([["a", "b"]], dtype=object), "wordBB": cells})
    image = io.BytesIO()
    Image.new("RGB", (5, 5), "red").save(image, format="PNG")
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("SynthText/gt.mat", annotations.getvalue())
        for name in ("1/a.png", "1/b.png"):
            archive.writestr("SynthText/" + name, image.getvalue())
    return stream.getvalue()


def test_archive_index_is_derived_and_original_info_hash_is_required():
    data, info_hash = metadata([{b"path": [b"readme.txt"], b"length": 5},
                               {b"path": [b"SynthText.zip"], b"length": 42},
                               {b"path": [b"depth.h5"], b"length": 500}])
    assert download.torrent_archive(data, expected_info_hash=info_hash) == {
        "index": 2, "path": "SynthText/SynthText.zip", "bytes": 42}
    with pytest.raises(ValueError, match="info hash"):
        download.torrent_archive(data)
    with pytest.raises(ValueError):
        download.torrent_archive(data + b"trailing", expected_info_hash=info_hash)


def test_torrent_paths_and_ambiguous_archives_fail_before_download():
    for files in ([{b"path": [b"..", b"SynthText.zip"], b"length": 1}],
                  [{b"path": [b"SynthText.zip"], b"length": 1}] * 2):
        data, info_hash = metadata(files)
        with pytest.raises(ValueError):
            download.torrent_archive(data, expected_info_hash=info_hash)


@pytest.mark.parametrize("name,symlink", [("SynthText/../outside.txt", False),
                                          ("Other/gt.mat", False), ("SynthText/link", True)])
def test_zip_rejects_unsafe_members_before_extracting_any_file(tmp_path, name, symlink):
    source = tmp_path / "bad.zip"
    with zipfile.ZipFile(source, "w") as opened:
        opened.writestr("SynthText/gt.mat", b"untouched")
        member = zipfile.ZipInfo(name)
        if symlink:
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
        opened.writestr(member, b"payload")
    with pytest.raises(ValueError):
        download.extract_verified(source, tmp_path)
    assert not (tmp_path / "SynthText").exists()


def test_extraction_resumes_valid_files_and_preserves_conflicting_files(tmp_path):
    source = tmp_path / "good.zip"
    source.write_bytes(make_archive())
    assert download.extract_verified(source, tmp_path) == 3
    annotation = tmp_path / "SynthText/gt.mat"
    before = annotation.stat().st_mtime_ns
    assert download.extract_verified(source, tmp_path) == 3
    assert annotation.stat().st_mtime_ns == before
    image = tmp_path / "SynthText/1/a.png"
    image.write_bytes(b"user content")
    with pytest.raises(ValueError, match="preserved"):
        download.extract_verified(source, tmp_path)
    assert image.read_bytes() == b"user content"


def test_corrupt_zip_never_publishes_a_partially_extracted_file(tmp_path):
    source = tmp_path / "bad_crc.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as opened:
        opened.writestr("SynthText/gt.mat", b"unique-payload")
    source.write_bytes(source.read_bytes().replace(b"unique-payload", b"broken-payload", 1))
    with pytest.raises(zipfile.BadZipFile):
        download.extract_verified(source, tmp_path)
    assert not (tmp_path / "SynthText/gt.mat").exists()


def test_portable_tool_checksum_is_verified_before_executing(tmp_path, monkeypatch):
    (tmp_path / "aria2-1.37.0-x86_64.zip").write_bytes(b"wrong tool")
    monkeypatch.setattr(download.shutil, "which", lambda _: None)
    monkeypatch.setattr(download.platform, "system", lambda: "Linux")
    monkeypatch.setattr(download.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(download.subprocess, "run", lambda *a, **k: pytest.fail("Unverified binary executed"))
    with pytest.raises(ValueError, match="SHA256"):
        download.aria_binary(tmp_path)


def test_small_download_uses_atomic_file_and_never_accepts_oversize(tmp_path, monkeypatch):
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"abcdef"))
    target = tmp_path / "metadata"
    with pytest.raises(ValueError, match="size limit"):
        download.fetch("https://example.test/metadata", target, max_bytes=3)
    assert not target.exists()
    download.fetch("https://example.test/metadata", target, max_bytes=6)
    assert target.read_bytes() == b"abcdef"


def install_acquisition_fixture(tmp_path, monkeypatch):
    archive = make_archive()
    metadata_bytes, fixture_hash = metadata([{b"path": [b"SynthText.zip"], b"length": len(archive)}])
    real_parse = download.torrent_archive
    monkeypatch.setattr(download, "INFO_HASH", fixture_hash)
    monkeypatch.setattr(download, "torrent_archive", lambda data: real_parse(data, expected_info_hash=fixture_hash))
    def fetch(url, path, **kwargs):
        Path(path).write_bytes(metadata_bytes)
    monkeypatch.setattr(download, "fetch", fetch)
    monkeypatch.setattr(download, "aria_binary", lambda *a: (Path("fixture-aria2"), {"fixture": True}))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert "--check-integrity=true" in command and "--select-file=1" in command
        target = tmp_path / "synthtext-downloads/torrent-data/SynthText/SynthText.zip"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(download.subprocess, "run", run)
    return calls


def test_full_offline_fixture_acquisition_receipt_and_cached_resume(tmp_path, monkeypatch):
    calls = install_acquisition_fixture(tmp_path, monkeypatch)
    root = download.acquire(tmp_path, minimum_images=2)
    assert root == tmp_path / "SynthText" and len(calls) == 1
    receipt = json.loads((tmp_path / "synthtext-downloads/complete.json").read_text())
    assert receipt["annotation_images"] == 2 and receipt["zip_crc_verified_on_acquisition"]
    assert receipt["scientific_evaluation"] is False
    monkeypatch.setattr(download, "fetch", lambda *a, **k: pytest.fail("Completed acquisition downloaded again"))
    assert download.acquire(tmp_path, minimum_images=2) == root and len(calls) == 1
    (root / "gt.mat").write_bytes(b"corrupt")
    with pytest.raises(MatReadError, match="truncated"):
        download.acquire(tmp_path, minimum_images=2)


@pytest.mark.parametrize("failure", ["exit", "incomplete"])
def test_transfer_failure_never_claims_completion_or_starts_extraction(tmp_path, monkeypatch, failure):
    install_acquisition_fixture(tmp_path, monkeypatch)
    def run(*args, **kwargs):
        if failure == "exit":
            raise subprocess.CalledProcessError(1, args[0])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(download.subprocess, "run", run)
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        download.acquire(tmp_path, minimum_images=2)
    assert not (tmp_path / "synthtext-downloads/complete.json").exists()
    assert not (tmp_path / "SynthText").exists()


def test_unmanaged_dataset_and_insufficient_space_are_preserved(tmp_path, monkeypatch):
    root = tmp_path / "SynthText"
    root.mkdir()
    (root / "user.txt").write_text("existing data")
    with pytest.raises(ValueError, match="unmanaged"):
        download.acquire(tmp_path, minimum_images=2)
    assert (root / "user.txt").read_text() == "existing data"
    parent = tmp_path / "new"
    install_acquisition_fixture(parent, monkeypatch)
    monkeypatch.setattr(download.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    monkeypatch.setattr(download, "aria_binary", lambda *a: pytest.fail("Downloaded tool without sufficient disk"))
    with pytest.raises(RuntimeError, match="disk"):
        download.acquire(parent, minimum_images=2)


def ordered_subset_archive(seed=71):
    """Eight originals; seeded prefix has overlap, accepted, duplicate, accepted."""
    names, boxes, texts = (np.empty((1, 8), object) for _ in range(3))
    payloads = []
    def png(rgb):
        stream = io.BytesIO()
        Image.new("RGB", (5, 5), rgb).save(stream, format="PNG")
        return stream.getvalue()
    for index in range(8):
        names[0, index] = f"1/word{index}.png"
        boxes[0, index] = np.array([[0, 4, 4, 0], [0, 0, 4, 4]], float)
        texts[0, index] = f"word{index}"
        payloads.append(png((40 + index, 100, 20)))
    order = np.random.default_rng(seed).permutation(8)
    payloads[order[0]] = png((255, 0, 0))
    payloads[order[1]] = payloads[order[2]] = png((0, 255, 0))
    payloads[order[3]] = png((0, 0, 255))
    annotations = io.BytesIO()
    savemat(annotations, {"imnames": names, "wordBB": boxes, "txt": texts})
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as opened:
        opened.writestr("SynthText/gt.mat", annotations.getvalue())
        for index, payload in enumerate(payloads):
            opened.writestr("SynthText/" + names[0, index], payload)
    return stream.getvalue(), {"max_images": 2, "seed": seed, "coordinate_origin": 0,
                               "excluded_image_hashes": [hashlib.sha256(payloads[order[0]]).hexdigest()]}


def comparable_manifest(manifest):
    return {k: ([{name: value for name, value in row.items() if name != "image_path"}
                 for row in v] if k == "samples" else v)
            for k, v in manifest.items() if k != "root"}


def test_reduced_extraction_and_plain_replay_match_full_ids_labels_hashes_exclusions(tmp_path):
    from segmentation.synthtext_words import prepare_synthtext
    archive = tmp_path / "subset.zip"
    payload, settings = ordered_subset_archive()
    archive.write_bytes(payload)
    full, reduced = tmp_path / "full", tmp_path / "reduced"
    full.mkdir(); reduced.mkdir()
    download.extract_verified(archive, full)
    arguments = {k: settings[k] for k in ("max_images", "seed", "coordinate_origin")}
    arguments["excluded_image_hashes"] = set(settings["excluded_image_hashes"])
    original = prepare_synthtext(full / "SynthText", **arguments)
    result = download.extract_subset_verified(archive, reduced, settings)
    replayed = prepare_synthtext(reduced / "SynthText", **arguments)
    assert comparable_manifest(original) == comparable_manifest(result["manifest"]) == comparable_manifest(replayed)
    assert result["extracted_files"] == 5  # GT plus two selected and two rejected candidates.
    assert len(list((reduced / "SynthText").rglob("*.png"))) == 4
    assert len(list((full / "SynthText").rglob("*.png"))) == 8
    assert result["retained_bytes"] == sum(p.stat().st_size for p in (reduced / "SynthText").rglob("*") if p.is_file())
    assert len(result["manifest"]["excluded"]) == 2


def test_subset_acquisition_discards_only_verified_zip_and_resumes_without_downloading(tmp_path, monkeypatch):
    calls = install_acquisition_fixture(tmp_path, monkeypatch)
    settings = {"max_images": 1, "seed": 71, "coordinate_origin": 0, "excluded_image_hashes": []}
    root = download.acquire(tmp_path, minimum_images=1, subset=settings, discard_archive=True)
    cache = tmp_path / "synthtext-downloads"
    receipt = json.loads((cache / "complete.json").read_text())
    assert receipt["archive_retained"] is False and receipt["zip_crc_scope"] == "retained members"
    assert not (cache / "torrent-data/SynthText/SynthText.zip").exists()
    assert (root / "gt.mat").is_file() and len(list(root.rglob("*.png"))) == 1
    assert receipt["subset_manifest_sha256"] == download.digest_file(cache / "subset_manifest.json")
    unchanged = (cache / "complete.json").read_bytes()
    monkeypatch.setattr(download, "fetch", lambda *a, **k: pytest.fail("Subset downloaded again"))
    monkeypatch.setattr(download, "aria_binary", lambda *a, **k: pytest.fail("Tool loaded after ZIP deletion"))
    assert download.acquire(tmp_path, minimum_images=1, subset=settings, discard_archive=True) == root
    assert len(calls) == 1 and (cache / "complete.json").read_bytes() == unchanged
    with pytest.raises(ValueError, match="contract differs"):
        download.acquire(tmp_path, minimum_images=1)
    changed = {**settings, "seed": 72}
    with pytest.raises(ValueError, match="contract differs"):
        download.acquire(tmp_path, minimum_images=1, subset=changed)
    (cache / "subset_manifest.json").write_text("modified")
    with pytest.raises(ValueError, match="subset manifest changed"):
        download.acquire(tmp_path, minimum_images=1, subset=settings)


def test_failed_subset_selection_never_discards_its_zip(tmp_path, monkeypatch):
    install_acquisition_fixture(tmp_path, monkeypatch)
    settings = {"max_images": 2, "seed": 71, "coordinate_origin": 0, "excluded_image_hashes": []}
    # This fixture has identical image bytes; two distinct eligible images do not exist.
    with pytest.raises(ValueError, match="Not enough eligible"):
        download.acquire(tmp_path, minimum_images=2, subset=settings, discard_archive=True)
    assert (tmp_path / "synthtext-downloads/torrent-data/SynthText/SynthText.zip").is_file()
    assert not (tmp_path / "synthtext-downloads/complete.json").exists()


@pytest.mark.parametrize("failure", ["changed", "outside", "symlink"])
def test_archive_removal_refuses_changed_or_unbound_files(tmp_path, failure):
    path = tmp_path / "torrent-data/SynthText/SynthText.zip"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original")
    receipt = {"torrent_info_hash": download.INFO_HASH,
               "selected_archive": {"path": "SynthText/SynthText.zip", "bytes": 8},
               "archive_sha256": download.digest_file(path)}
    if failure == "changed":
        path.write_bytes(b"modified")
    elif failure == "outside":
        receipt["selected_archive"]["path"] = "../SynthText.zip"
    else:
        other = tmp_path / "other.zip"
        path.replace(other)
        path.symlink_to(other)
    with pytest.raises(ValueError):
        download.discard_download_archive(tmp_path, receipt)
    assert path.exists()
    assert not (tmp_path / "complete.json").exists()


def test_subset_settings_bind_config_and_checksum_checked_overlap_manifest(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("{}"); (tmp_path / "manifest.yaml").write_text("fixture-manifest")
    c = {"train_manifest": "manifest.yaml", "word_manifest_sha256": download.digest_file(tmp_path / "manifest.yaml"),
         "seed": 71, "synthtext": {"max_images": 2, "coordinate_origin": 0}}
    monkeypatch.setattr("scripts.run_unet_detection_methods.read_config", lambda *a: (c, {}))
    def manifest(path, require_checksums):
        assert require_checksums
        return SimpleNamespace(samples=[SimpleNamespace(image_sha256=x) for x in ["b"*64, "a"*64, "a"*64]])
    monkeypatch.setattr("segmentation.data.load_segmentation_manifest", manifest)
    settings = download.subset_settings(config, tmp_path)
    assert settings["max_images"] == 2 and settings["seed"] == 71 and settings["coordinate_origin"] == 0
    assert settings["excluded_image_hashes"] == ["a"*64, "b"*64]
    assert settings["word_manifest_sha256"] == c["word_manifest_sha256"]
    (tmp_path / "manifest.yaml").write_text("changed")
    with pytest.raises(ValueError, match="manifest differs"):
        download.subset_settings(config, tmp_path)


def test_original_root_level_license_is_preserved_in_full_and_subset_modes(tmp_path):
    archive = tmp_path / "licensed.zip"
    payload, settings = ordered_subset_archive()
    archive.write_bytes(payload)
    license_text = b"Original dataset license, preserved exactly.\n"
    with zipfile.ZipFile(archive, "a") as opened:
        opened.writestr("license.txt", license_text)
    full, reduced = tmp_path / "full", tmp_path / "reduced"
    full.mkdir(); reduced.mkdir()
    assert download.extract_verified(archive, full) == 10  # Eight images, GT, license.
    result = download.extract_subset_verified(archive, reduced, settings)
    assert result["extracted_files"] == 6
    assert (full / "SynthText/license.txt").read_bytes() == license_text
    assert (reduced / "SynthText/license.txt").read_bytes() == license_text
    assert not (tmp_path / "license.txt").exists()


@pytest.mark.parametrize("kind", ["arbitrary", "symlink", "collision"])
def test_root_license_exception_does_not_admit_other_files_symlinks_or_collisions(tmp_path, kind):
    archive = tmp_path / "unexpected.zip"
    with zipfile.ZipFile(archive, "w") as opened:
        opened.writestr("SynthText/gt.mat", b"annotation")
        member = zipfile.ZipInfo("license.txt" if kind != "arbitrary" else "unrelated.txt")
        if kind == "symlink":
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
        opened.writestr(member, b"licensing")
        if kind == "collision":
            opened.writestr("SynthText/license.txt", b"ambiguous second license")
    with pytest.raises(ValueError):
        download.extract_verified(archive, tmp_path)
    assert not (tmp_path / "SynthText").exists()
