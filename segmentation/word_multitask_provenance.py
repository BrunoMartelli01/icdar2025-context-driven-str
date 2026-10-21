"""JSON-stable contracts and a narrow, auditable repair for smoke-only runs.

Training artifacts never enter the compatibility path. Scientific source,
configuration, dataset, encoder and split checks remain exact.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json


RUNNER_SOURCE = "scripts/run_word_multitask.py"
PROVENANCE_SOURCE = "segmentation/word_multitask_provenance.py"
# Exact pre-fix runner used by the user's three completed GPU smoke tests.
# This is source provenance, not a scientific parameter or a generic bypass.
LEGACY_RUNNER_SHA256 = "298b9023dfb7fe2fb6d9e8cf4234da515fffe953c7595341c63197f453bff38c"


def json_contract(value):
    """Use the representation actually persisted in JSON, rejecting NaN/Inf."""
    return json.loads(json.dumps(value, allow_nan=False))


def _canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _difference_paths(saved, current, path=""):
    if isinstance(saved, dict) and isinstance(current, dict):
        result = []
        for key in sorted(saved.keys() | current.keys()):
            location = path + "/" + key
            if key not in saved or key not in current:
                result.append(location)
            else:
                result.extend(_difference_paths(saved[key], current[key], location))
        return result
    return [] if saved == current else [path]


def _legacy_contract(current):
    legacy = deepcopy(current)
    legacy["source_sha256"][RUNNER_SOURCE] = LEGACY_RUNNER_SHA256
    legacy["source_sha256"].pop(PROVENANCE_SOURCE, None)
    return legacy


def _verified_smoke(directory, expected):
    path = directory / "smoke.json"
    report = json.loads(path.read_text())
    if (report.get("run_contract") != expected or report.get("result_class") != "smoke-test result"
        or report.get("train", {}).get("steps") != 1
        or report.get("train", {}).get("samples") != expected["config"]["batch_size"]
        or report.get("memory", {}).get("oom") is not False):
        raise ValueError("Existing smoke result is incomplete or its provenance differs: " + str(path))
    return report


def _transition_receipt(directory, current):
    return {
        "schema_version": 1, "kind": "known_runner_smoke_only_json_contract_repair",
        "classification": "ASSUMPTION",
        "rationale": "Canonical JSON Adam beta encoding; only launcher/provenance sources change. No training artifacts exist.",
        "legacy_runner_sha256": LEGACY_RUNNER_SHA256,
        "current_runner_sha256": current["source_sha256"][RUNNER_SOURCE],
        "current_contract_canonical_sha256": _canonical_hash(current),
        "archived_contract_file_sha256": sha256_file(directory / "smoke_run_contract.json"),
        "smoke_file_sha256": sha256_file(directory / "smoke.json"),
    }


def ensure_branch_contract(directory, current):
    """Write/verify a contract, or archive the known pre-fix smoke-only contract.

    The old smoke JSON remains byte-for-byte intact. Archival and transition
    records are idempotent across interruption before the new contract commit.
    """
    directory = Path(directory)
    current = json_contract(current)
    path = directory / "run_contract.json"
    if not path.exists():
        if any(directory.iterdir()):
            raise ValueError("Nonempty branch directory without provenance; refusing overwrite")
        _json(path, current)
        return
    saved = json.loads(path.read_text())
    if saved == current:
        return
    legacy = _legacy_contract(current)
    allowed = {"run_contract.json", "smoke.json", "smoke_run_contract.json", "contract_transition.json"}
    entries = {p.name for p in directory.iterdir()}
    if saved != legacy or "smoke.json" not in entries or entries - allowed:
        differences = ", ".join(_difference_paths(saved, current)[:8])
        raise ValueError("Existing branch contract differs; no safe smoke-only upgrade. "
                         "Differences: {}. Preserve this run and select a NEW output directory.".format(differences))
    _verified_smoke(directory, saved)
    original = path.read_bytes()
    archive = directory / "smoke_run_contract.json"
    if archive.exists():
        if archive.read_bytes() != original:
            raise ValueError("Archived smoke contract differs; refusing to replace provenance")
    else:
        with archive.open("xb") as stream:
            stream.write(original)
    transition = _transition_receipt(directory, current)
    receipt = directory / "contract_transition.json"
    if receipt.exists():
        if json.loads(receipt.read_text()) != transition:
            raise ValueError("Contract transition receipt differs")
    else:
        _json(receipt, transition)
    _json(path, current)
    print("[contract repair] archived smoke-only provenance: " + str(archive), flush=True)


def reusable_smoke(directory, current, device):
    """Reuse an unchanged same-contract smoke or the explicitly archived legacy smoke."""
    directory = Path(directory)
    path = directory / "smoke.json"
    if not path.exists():
        return None
    current = json_contract(current)
    report = json.loads(path.read_text())
    if report.get("run_contract") == current:
        report = _verified_smoke(directory, current)
    else:
        archive, receipt = directory / "smoke_run_contract.json", directory / "contract_transition.json"
        if not archive.exists() or not receipt.exists():
            raise ValueError("Existing smoke has a different contract and no verified transition")
        transition = json.loads(receipt.read_text())
        legacy = json.loads(archive.read_text())
        if legacy != _legacy_contract(current) or transition != _transition_receipt(directory, current):
            raise ValueError("Archived smoke provenance/checksum differs")
        report = _verified_smoke(directory, legacy)
    if report["memory"].get("device") != str(device):
        raise ValueError("Existing smoke used a different device; preserve it and select a NEW output directory")
    return report
