"""Repair the upstream Linux wheel's internal tag; preserve all runtime bytes."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import struct
import zipfile
from pathlib import Path

from diagnostics.lavis_captioner_candidates import download, sha, write_json

URL = "https://files.pythonhosted.org/packages/11/79/936af42edf90a7bd4e41a6cac89c913d4b47fa48a26b042d5129a9242ee3/decord-0.6.0-py3-none-manylinux2010_x86_64.whl"
ORIGINAL_SHA = "51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976"
NAME = "decord-0.6.0-1tagfix-py3-none-manylinux2010_x86_64.whl"
WHEEL = "decord-0.6.0.dist-info/WHEEL"
RECORD = "decord-0.6.0.dist-info/RECORD"
OLD_TAG = "cp36-cp36m-manylinux2010_x86_64"
NEW_TAG = "py3-none-manylinux2010_x86_64"


def repair(source, output, expected_sha=ORIGINAL_SHA):
    if sha(source) != expected_sha:
        raise RuntimeError("Original decord wheel SHA256 mismatch")
    payload_hashes = {}
    with zipfile.ZipFile(source) as original:
        names = original.namelist()
        if len(names) != len(set(names)) or any(n.endswith((".jws", ".p7s")) for n in names):
            raise RuntimeError("Unexpected duplicate members or signed wheel")
        wheel = original.read(WHEEL).decode()
        if re.findall(r"^Tag: (.+)$", wheel, re.M) != [OLD_TAG]:
            raise RuntimeError("Unexpected upstream tag; refuse speculative retagging")
        if b"ctypes.CDLL" not in original.read("decord/_ffi/base.py"):
            raise RuntimeError("Cannot establish a ctypes ABI independent of CPython")
        native = original.read("decord/libdecord.so")
        if (native[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", native, 18)[0] != 62
                or b"PyInit_" in native):
            raise RuntimeError("Unexpected native library architecture or CPython extension")
        if any(n.endswith(".so") and n != "decord/libdecord.so" for n in names):
            raise RuntimeError("Unexpected additional native extension")
        fixed = re.sub(r"^Tag: .+$", "Tag: " + NEW_TAG, wheel, flags=re.M)
        fixed = fixed.rstrip() + "\nBuild: 1tagfix\n"
        rows = []
        part = output.with_suffix(".whl.part")
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED) as patched:
            for info in original.infolist():
                if info.filename == RECORD or info.is_dir():
                    continue
                data = fixed.encode() if info.filename == WHEEL else original.read(info.filename)
                patched.writestr(info, data)
                h = hashlib.sha256(data).digest()
                rows.append((info.filename, "sha256=" + base64.urlsafe_b64encode(h).decode().rstrip("="), str(len(data))))
                if info.filename != WHEEL:
                    payload_hashes[info.filename] = h.hex()
            rows.append((RECORD, "", ""))
            text = io.StringIO(newline="")
            csv.writer(text, lineterminator="\n").writerows(rows)
            patched.writestr(original.getinfo(RECORD), text.getvalue().encode())
    with zipfile.ZipFile(part) as patched:
        for name, expected in payload_hashes.items():
            if hashlib.sha256(patched.read(name)).hexdigest() != expected:
                raise RuntimeError(f"Runtime payload changed: {name}")
    part.replace(output)
    return {"original_url": URL, "original_sha256": expected_sha,
            "repaired_sha256": sha(output), "changed_members": [WHEEL, RECORD],
            "runtime_payload_sha256": payload_hashes,
            "old_tag": OLD_TAG, "new_tag": NEW_TAG,
            "classification": "ASSUMPTION", "scope": "dependency packaging only; no runtime/model bytes changed"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / NAME
    receipt = args.output_dir / "decord-tag-repair.json"
    if output.is_file() and receipt.is_file():
        data = json.loads(receipt.read_text())
        if data["original_sha256"] != ORIGINAL_SHA or sha(output) != data["repaired_sha256"]:
            raise RuntimeError("Corrupt repaired wheel cache")
        print(f"[decord packaging] verified reusable wheel {output}", flush=True)
        return 0
    source = download(URL, args.output_dir / URL.rsplit("/", 1)[1], expected=ORIGINAL_SHA)
    write_json(receipt, repair(source, output))
    print(f"[decord packaging] repaired WHEEL/RECORD only; runtime bytes unchanged: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
