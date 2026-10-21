"""Preserve the one known CUDA-initialization failure, before any training.

This is NOT checkpoint compatibility or a generic source-hash bypass. Only the
exact original runner's failed first CUDA smoke may update launcher provenance.
All scientific settings, source artifacts and completed results stay immutable.
"""

from copy import deepcopy
import json
from pathlib import Path

from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _json
from segmentation.word_multitask_provenance import json_contract, _canonical_hash


RUNNER = "scripts/run_text_recall_resolution.py"
PROVENANCE = "segmentation/text_recall_provenance.py"
CUDA_HELPER = "segmentation/evaluate.py"
LEGACY_RUNNER_SHA256 = "75fb6e2d25552f8d8837d3abf385e261ff85571325e132575142ffa6693560b2"
CORRECTED_RUNNER_SHA256 = "79913cc005e955476bf58dd0559a8510cc910f883e642a09cee2f20f32c4447c"
QUALIFIED_CUDA_HELPER_SHA256 = "f0af56d95c5991c21245bafe194ebca6aab5c5cc443bb04e28cbf7a0071d51f4"
ARCHIVE = "cuda_initialization_archive"
RECEIPT = "cuda_initialization_transition.json"
STUB = "smoke/cuda_0/control512/run_contract.json"


def legacy_contract(current):
    previous = deepcopy(json_contract(current))
    previous["source_sha256"][RUNNER] = LEGACY_RUNNER_SHA256
    previous["source_sha256"].pop(PROVENANCE, None)
    previous["source_sha256"].pop(CUDA_HELPER, None)
    return previous


def _failure_proof(directory):
    failures = sorted(directory.glob("failure-*.json"))
    if len(failures) != 1:
        raise ValueError("CUDA repair requires exactly one recorded first-smoke failure")
    failure = json.loads(failures[0].read_text())
    if (failure.get("status") != "FAILED" or failure.get("result_class") != "smoke-test result"
        or failure.get("action") != "smoke" or failure.get("branch") != "control512"
        or failure.get("device") != "cuda:0" or failure.get("type") != "RuntimeError"
        or failure.get("message", "").strip() != "Invalid device argument"
        or failure.get("oom") is not False or failure.get("precision") != "float32"
        or failure.get("peak_allocated_bytes") != 0 or failure.get("peak_reserved_bytes") != 0):
        raise ValueError("Failure is not the known pre-allocation CUDA initialization error")
    return failures[0], failure


def _check_entries(directory, failure_path):
    allowed = {"experiment_contract.json", "target_qualification.json", "historical_reference.json",
               "runner.lock", STUB, "smoke/cuda_0/control512/warm_start.json", failure_path.name,
               RECEIPT}
    originals = {"experiment_contract.json", "target_qualification.json", STUB}
    allowed |= {ARCHIVE+"/"+name for name in originals}
    allowed_dirs = {"smoke", "smoke/cuda_0", "smoke/cuda_0/control512", ARCHIVE,
                   ARCHIVE+"/smoke", ARCHIVE+"/smoke/cuda_0", ARCHIVE+"/smoke/cuda_0/control512"}
    for path in directory.rglob("*"):
        name = path.relative_to(directory).as_posix()
        if path.is_symlink() or (path.is_dir() and name not in allowed_dirs) or (path.is_file() and name not in allowed):
            raise ValueError("CUDA repair refuses training, completed smoke or unexpected artifacts: "+name)


def ensure_experiment_contract(directory, current, historical, stub_current):
    """Archive and repair ONLY a fully verified, unsuccessful first CUDA smoke.

    The caller holds runner.lock. Source transitions are recorded BEFORE commits;
    experiment_contract is committed LAST so interrupted migrations can resume.
    Original preflight, failure and contract bytes remain auditable.
    """
    directory = Path(directory)
    current = json_contract(current)
    path = directory/"experiment_contract.json"
    if not path.exists():
        if any(p.name != "runner.lock" for p in directory.iterdir()):
            raise ValueError("Nonempty experiment directory without a run contract")
        _json(path, current)
        return
    saved = json.loads(path.read_text())
    if saved == current:
        receipt_path = directory/RECEIPT
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("current_contract_sha256") != _canonical_hash(current)
                or receipt.get("current_runner_sha256") != current["source_sha256"][RUNNER]):
                raise ValueError("CUDA transition receipt does not match the active contract")
            files = {ARCHIVE+"/"+name: digest for name, digest in receipt["archived_file_sha256"].items()}
            files.update({receipt["failure_file"]: receipt["failure_sha256"],
                "historical_reference.json": receipt["historical_reference_sha256"],
                "smoke/cuda_0/control512/warm_start.json": receipt["warm_start_sha256"]})
            for name, digest in files.items():
                if sha256_file(directory/name) != digest:
                    raise ValueError("CUDA transition evidence checksum differs: "+name)
        return
    previous = legacy_contract(current)
    if saved != previous:
        raise ValueError("Existing contract differs beyond the known CUDA launcher fix; use a NEW output directory")
    if (current["source_sha256"].get(RUNNER) != CORRECTED_RUNNER_SHA256
        or current["source_sha256"].get(CUDA_HELPER) != QUALIFIED_CUDA_HELPER_SHA256):
        raise ValueError("CUDA migration requires the exact qualified corrected runner and allocator helper")
    stub_current = json_contract(stub_current)
    stub_previous = {**previous, "branch": stub_current["branch"],
                     "training_config": stub_current["training_config"],
                     "execution_device": stub_current["execution_device"]}
    if stub_previous["branch"] != "control512" or stub_previous["execution_device"] != "cuda:0":
        raise ValueError("Only the first CUDA smoke stub is eligible")
    failure_path, failure = _failure_proof(directory)
    _check_entries(directory, failure_path)
    if (failure["input_shape"] != [current["config"]["training"]["batch_size"], 3, 512, 512]
        or failure["batch_size"] != current["config"]["training"]["batch_size"]):
        raise ValueError("Failure batch/shape does not match the unchanged experiment")
    if json.loads((directory/"historical_reference.json").read_text()) != json_contract(historical):
        raise ValueError("Historical reference differs; refusing preflight reuse")
    warm = json.loads((directory/"smoke/cuda_0/control512/warm_start.json").read_text())
    if (warm.get("source_checkpoint_sha256") != current["base_checkpoint_sha256"]
        or warm.get("source_epoch") != current["config"]["base_checkpoint_epoch"]
        or warm.get("architecture") != current["config"]["architecture_identity"]
        or warm.get("parameters") != 58155081):
        raise ValueError("Warm-start receipt differs from the fixed baseline")
    preflight_path = directory/"target_qualification.json"
    archive = directory/ARCHIVE
    receipt_path = directory/RECEIPT
    original_preflight = archive/"target_qualification.json" if receipt_path.exists() else preflight_path
    preflight = json.loads(original_preflight.read_text())
    if (preflight.get("status") != "PASS" or preflight.get("result_class") != "smoke-test result"
        or preflight.get("run_contract") != previous or preflight.get("augmentation") != "disabled"
        or set(preflight.get("resolutions", {})) != {"512", "768"}
        or any(r.get("images") != current["config"]["dataset"]["images"]
               or len(r.get("per_image", [])) != r["images"] for r in preflight["resolutions"].values())):
        raise ValueError("Original target qualification is incomplete or incompatible")
    replacements = {"experiment_contract.json": current, STUB: stub_current,
                    "target_qualification.json": {**preflight, "run_contract": current}}
    originals = {"experiment_contract.json": previous, STUB: stub_previous, "target_qualification.json": preflight}
    # Validate EVERY active and archived file before writing anything. Active
    # metadata may be old or new during interruption recovery, nothing else.
    for name in replacements:
        active = directory/name
        if json.loads(active.read_text()) not in (originals[name], replacements[name]):
            raise ValueError("Active CUDA-transition metadata was modified: "+name)
        backed_up = archive/name
        if backed_up.exists() and json.loads(backed_up.read_text()) != originals[name]:
            raise ValueError("Archived CUDA-transition metadata differs: "+name)
        if receipt_path.exists() and not backed_up.is_file():
            raise ValueError("Interrupted transition has an incomplete provenance archive")
    for name in replacements:
        backed_up = archive/name
        if not backed_up.exists():
            backed_up.parent.mkdir(parents=True, exist_ok=True)
            with backed_up.open("xb") as stream:
                stream.write((directory/name).read_bytes())
    receipt = {"schema_version": 1, "kind": "known_pretraining_cuda_allocator_initialization_fix",
        "classification": "ASSUMPTION",
        "rationale": "Initialize CUDA allocator before resetting integer-device peaks; no scientific settings or weights changed",
        "legacy_runner_sha256": LEGACY_RUNNER_SHA256,
        "current_runner_sha256": current["source_sha256"][RUNNER],
        "legacy_contract_sha256": _canonical_hash(previous), "current_contract_sha256": _canonical_hash(current),
        "archived_file_sha256": {name: sha256_file(archive/name) for name in replacements},
        "failure_file": failure_path.name, "failure_sha256": sha256_file(failure_path),
        "historical_reference_sha256": sha256_file(directory/"historical_reference.json"),
        "warm_start_sha256": sha256_file(directory/"smoke/cuda_0/control512/warm_start.json")}
    if receipt_path.exists():
        if json.loads(receipt_path.read_text()) != receipt:
            raise ValueError("CUDA initialization transition receipt differs")
    else:
        _json(receipt_path, receipt)
    for name in ("target_qualification.json", STUB, "experiment_contract.json"):
        _json(directory/name, replacements[name])
    print("[CUDA contract repair] archived original preflight/failure provenance; training has not started", flush=True)
