import base64
import csv
import hashlib
import io
import struct
import zipfile

import pytest

from diagnostics.repair_decord_wheel import NAME, RECORD, WHEEL, repair, sha


def upstream(tmp_path, native_override=None):
    native = bytearray(64)
    native[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", native, 18, 62)
    path = tmp_path / "source.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(WHEEL, "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp36-cp36m-manylinux2010_x86_64\n")
        wheel.writestr(RECORD, "old-record\n")
        wheel.writestr("decord/_ffi/base.py", "library = ctypes.CDLL(path)\n")
        wheel.writestr("decord/libdecord.so", bytes(native) if native_override is None else native_override)
        wheel.writestr("decord/__init__.py", "__version__ = '0.6.0'\n")
    return path


def test_only_metadata_changes_and_record_hashes_are_valid(tmp_path):
    source = upstream(tmp_path)
    target = tmp_path / NAME
    result = repair(source, target, expected_sha=sha(source))
    assert result["changed_members"] == [WHEEL, RECORD]
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target) as new:
        assert "Tag: py3-none-manylinux2010_x86_64" in new.read(WHEEL).decode()
        for name in old.namelist():
            if name not in (WHEEL, RECORD):
                assert old.read(name) == new.read(name)
        for name, checksum, size in csv.reader(io.StringIO(new.read(RECORD).decode())):
            if name == RECORD:
                assert checksum == size == ""
            else:
                data = new.read(name)
                assert int(size) == len(data)
                assert checksum == "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")


def test_wrong_source_checksum_fails(tmp_path):
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        repair(upstream(tmp_path), tmp_path / NAME, expected_sha="invalid")


def test_rebuild_is_byte_reproducible(tmp_path):
    source = upstream(tmp_path)
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    repair(source, first, expected_sha=sha(source))
    repair(source, second, expected_sha=sha(source))
    assert sha(first) == sha(second)


def test_cpython_extension_cannot_be_retagged(tmp_path):
    source = upstream(tmp_path, native_override=b"PyInit_decord")
    with pytest.raises(RuntimeError, match="native library"):
        repair(source, tmp_path / NAME, expected_sha=sha(source))


def test_unexpected_tag_fails(tmp_path):
    source = upstream(tmp_path)
    data = tmp_path / "wrong.whl"
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(data, "w") as new:
        for name in old.namelist():
            payload = old.read(name)
            if name == WHEEL:
                payload = payload.replace(b"cp36-cp36m", b"cp310-cp310")
            new.writestr(name, payload)
    with pytest.raises(RuntimeError, match="upstream tag"):
        repair(data, tmp_path / NAME, expected_sha=sha(data))
