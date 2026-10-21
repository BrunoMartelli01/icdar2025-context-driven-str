from __future__ import annotations

import json
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import deepsolo.candidates as candidate_module
from deepsolo.adapter import (
    STAGED_ENTRYPOINT_NAME, STAGED_PACKAGE_NAME, DeepSoloDockerAdapter,
    DeepSoloProcessError, DeepSoloSubprocessAdapter, _stage_worker_bundle,
)
from deepsolo.candidates import DEEPSOLO_SOURCE_REVISION
from deepsolo.contracts import DeepSoloRequest, DeepSoloResponse
from deepsolo.worker import _serialize_instances, run as run_worker


CANDIDATE = "ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE"


def is_staged_worker_command(command):
    return any(str(value).endswith("/" + STAGED_ENTRYPOINT_NAME) for value in command)


def adapter_files(monkeypatch, tmp_path: Path, suffix: str = ".bin"):
    image = tmp_path / ("image" + suffix)
    manifest = tmp_path / "runtime.json"
    source = tmp_path / "DeepSolo"
    image.write_bytes(b"image")
    source.mkdir()
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "source_root": "/workspace/DeepSolo",
        "source_revision": DEEPSOLO_SOURCE_REVISION,
        "candidates": {},
        "runtime": {},
    }), encoding="utf-8")
    monkeypatch.setattr(
        candidate_module, "source_revision", lambda root: DEEPSOLO_SOURCE_REVISION
    )
    return image, manifest, source


def response_for(request, *, success=True):
    return {
        "schema_version": 1,
        "request_id": request["request_id"],
        "success": success,
        "detections": [] if not success else [{
            "source_index": 0, "text": "WORD", "score": 0.9,
            "score_type": "fixture instance score",
            "geometry": {"boundary_points": [[0, 0], [1, 1]]},
            "raw": {"fixture": True},
        }],
        "recognized_strings": [] if not success else ["WORD"],
        "selected_output": "" if not success else "WORD",
        "reduction_policy": request["reduction_policy"],
        "artifact": {},
        "latency": {},
        "placement": {},
        "memory": {},
        "error": None if success else {
            "type": "RuntimeError", "message": "fixture failure", "traceback": "trace"
        },
    }


def test_request_round_trip_and_strict_mode_rejection(tmp_path: Path):
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    request = DeepSoloRequest.create(
        image, CANDIDATE,
        {"device": "cuda:0", "score_threshold": None, "lexicon_type": None},
        "highest_score", request_id="request-1",
    )
    assert DeepSoloRequest.from_dict(request.to_dict()) == request
    strict = request.to_dict()
    strict["reproduction_mode"] = "STRICT"
    with pytest.raises(ValueError, match="APPROXIMATE"):
        DeepSoloRequest.from_dict(strict)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0"])
def test_contract_accepts_canonical_devices(tmp_path: Path, device: str):
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    request = DeepSoloRequest.create(
        image, CANDIDATE,
        {"device": device, "score_threshold": 0.4, "lexicon_type": 1},
        "source_order_space_join",
    )
    assert request.inference_options["device"] == device


def test_response_schema_rejects_missing_fields():
    with pytest.raises(ValueError, match="fields"):
        DeepSoloResponse.from_dict({"schema_version": 1})


def test_subprocess_adapter_uses_json_boundary_without_importing_deepsolo_runtime(
    monkeypatch, tmp_path: Path,
):
    image, manifest, source = adapter_files(monkeypatch, tmp_path)

    def run(command, **kwargs):
        request_path = Path(command[command.index("--request") + 1])
        response_path = Path(command[command.index("--response") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response_path.write_text(json.dumps(response_for(request)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    adapter = DeepSoloSubprocessAdapter(
        Path(sys.executable), manifest, host_source_root=source
    )
    response = adapter.infer(
        image_path=image,
        candidate_id=CANDIDATE,
        inference_options={"device": "cuda:0", "score_threshold": None, "lexicon_type": None},
        reduction_policy="highest_score",
    )
    assert response.success and response.selected_output == "WORD"


def test_subprocess_adapter_preserves_virtualenv_python_symlink(
    monkeypatch, tmp_path: Path,
):
    image, manifest, source = adapter_files(monkeypatch, tmp_path)
    base_python = tmp_path / "base-python"
    base_python.write_bytes(b"python fixture")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    captured = {}

    def run(command, **kwargs):
        captured["executable"] = command[0]
        request_path = Path(command[command.index("--request") + 1])
        response_path = Path(command[command.index("--response") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response_path.write_text(json.dumps(response_for(request)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    adapter = DeepSoloSubprocessAdapter(
        venv_python, manifest, host_source_root=source
    )
    response = adapter.infer(
        image_path=image,
        candidate_id=CANDIDATE,
        inference_options={"device": "cuda:0"},
        reduction_policy="highest_score",
    )

    assert response.success
    assert adapter.python_executable == venv_python.absolute()
    assert captured["executable"] == str(venv_python.absolute())
    assert Path(captured["executable"]).resolve() == base_python.resolve()


def test_subprocess_failure_is_propagated_with_structured_response(monkeypatch, tmp_path: Path):
    image, manifest, source = adapter_files(monkeypatch, tmp_path)

    def run(command, **kwargs):
        request_path = Path(command[command.index("--request") + 1])
        response_path = Path(command[command.index("--response") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response_path.write_text(json.dumps(response_for(request, success=False)), encoding="utf-8")
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="failure")

    monkeypatch.setattr(subprocess, "run", run)
    adapter = DeepSoloSubprocessAdapter(
        Path(sys.executable), manifest, host_source_root=source
    )
    with pytest.raises(DeepSoloProcessError) as captured:
        adapter.infer(
            image_path=image, candidate_id=CANDIDATE,
            inference_options={"device": "cuda:0"}, reduction_policy="highest_score",
        )
    assert captured.value.response is not None
    assert captured.value.response.payload["error"]["traceback"] == "trace"


def test_docker_adapter_preserves_contract_and_copies_container_visible_image(
    monkeypatch, tmp_path: Path,
):
    image, manifest, source = adapter_files(monkeypatch, tmp_path, ".png")
    captured = {}

    def run(command, **kwargs):
        if command[1:2] == ["cp"] and command[-1] == "qualified-container:/tmp":
            bundle = Path(command[-2])
            request = json.loads((bundle / "request.json").read_text(encoding="utf-8"))
            captured["request"] = request
            captured["manifest"] = json.loads(
                (bundle / "runtime_manifest.json").read_text(encoding="utf-8")
            )
            captured["response"] = response_for(request)
        elif command[1:2] == ["exec"] and is_staged_worker_command(command):
            captured["worker_command"] = command
        elif command[1:2] == ["cp"] and command[-2].endswith("/response.json"):
            Path(command[-1]).write_text(json.dumps(captured["response"]), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    adapter = DeepSoloDockerAdapter(
        "qualified-container",
        manifest,
        host_source_root=source,
        runtime_environment={
            "CONTEXT_STR_DEEPSOLO_IMAGE_TAG": "qualified-image",
            "CONTEXT_STR_DEEPSOLO_IMAGE_ID": "sha256:qualified",
        },
    )
    response = adapter.infer(
        image_path=image,
        candidate_id=CANDIDATE,
        inference_options={"device": "cuda:0"},
        reduction_policy="highest_score",
    )
    assert response.selected_output == "WORD"
    assert captured["request"]["image_path"].startswith("/tmp/context-str-deepsolo-")
    assert captured["manifest"]["runtime"]["source_identity_verification"]["verified"] is True
    command = captured["worker_command"]
    command_text = " ".join(command)
    worker_path = command[command.index("/opt/context-str-deepsolo/bin/python") + 1]
    assert worker_path.endswith("/" + STAGED_ENTRYPOINT_NAME)
    assert "/scripts/run_deepsolo_worker.py" not in command_text
    assert not any(value.startswith("PYTHONPATH=") for value in command)
    assert command[command.index(worker_path) + 1] == "--request"
    assert "CONTEXT_STR_DEEPSOLO_IMAGE_ID=sha256:qualified" in command_text


def test_docker_adapter_propagates_structured_failure_and_traceback(
    monkeypatch, tmp_path: Path,
):
    image, manifest, source = adapter_files(monkeypatch, tmp_path, ".png")
    captured = {}

    def run(command, **kwargs):
        if command[1:2] == ["cp"] and command[-1] == "qualified-container:/tmp":
            bundle = Path(command[-2])
            request = json.loads((bundle / "request.json").read_text(encoding="utf-8"))
            captured["response"] = response_for(request, success=False)
        elif command[1:2] == ["exec"] and is_staged_worker_command(command):
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="failure")
        elif command[1:2] == ["cp"] and command[-2].endswith("/response.json"):
            Path(command[-1]).write_text(
                json.dumps(captured["response"]), encoding="utf-8"
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    adapter = DeepSoloDockerAdapter(
        "qualified-container", manifest, host_source_root=source
    )
    with pytest.raises(DeepSoloProcessError) as captured_error:
        adapter.infer(
            image_path=image,
            candidate_id=CANDIDATE,
            inference_options={"device": "cuda:0"},
            reduction_policy="highest_score",
        )
    assert captured_error.value.response is not None
    assert captured_error.value.response.payload["error"]["traceback"] == "trace"


def test_staged_entrypoint_cannot_select_an_ambient_deepsolo_cli(tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    entrypoint = _stage_worker_bundle(Path(__file__).resolve().parents[2], bundle)
    ambient = tmp_path / "ambient"
    (ambient / "deepsolo").mkdir(parents=True)
    (ambient / "deepsolo" / "__init__.py").write_text("", encoding="utf-8")
    (ambient / "deepsolo" / "worker.py").write_text(
        "import argparse\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--config-file')\n"
        "    return parser.parse_args()\n",
        encoding="utf-8",
    )
    image = tmp_path / "input.bin"
    image.write_bytes(b"image")
    request = DeepSoloRequest.create(
        image, CANDIDATE, {"device": "cpu"}, "highest_score", request_id="dispatch"
    )
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    manifest_path = tmp_path / "manifest.json"
    request_path.write_text(json.dumps(request.to_dict()), encoding="utf-8")
    manifest_path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ambient)
    completed = subprocess.run(
        [
            sys.executable,
            str(entrypoint),
            "--request", str(request_path),
            "--response", str(response_path),
            "--runtime-manifest", str(manifest_path),
        ],
        cwd=ambient,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    response = DeepSoloResponse.from_dict(json.loads(response_path.read_text(encoding="utf-8")))
    assert "schema_version" in response.payload["error"]["message"]
    bootstrap = response.payload["placement"]["worker_bootstrap"]
    assert Path(bootstrap["entrypoint_path"]) == entrypoint.resolve()
    assert Path(bootstrap["package_path"]).is_relative_to(bundle.resolve())
    assert Path(bootstrap["worker_module_path"]).is_relative_to(bundle.resolve())
    assert (bundle / STAGED_PACKAGE_NAME / "worker.py").is_file()


def test_staged_entrypoint_fails_explicitly_if_deepsolo_resolves_outside_bundle(
    tmp_path: Path,
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    entrypoint = _stage_worker_bundle(Path(__file__).resolve().parents[2], bundle)
    (bundle / STAGED_PACKAGE_NAME).rename(bundle / "removed_staged_package")
    ambient = tmp_path / "ambient"
    (ambient / "deepsolo").mkdir(parents=True)
    (ambient / "deepsolo" / "__init__.py").write_text("", encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ambient)
    completed = subprocess.run(
        [sys.executable, str(entrypoint), "--help"],
        cwd=ambient,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "resolved deepsolo package outside staged bundle root" in completed.stderr


def test_docker_adapter_nonzero_without_response_is_explicit(monkeypatch, tmp_path: Path):
    image, manifest, source = adapter_files(monkeypatch, tmp_path, ".png")

    def run(command, **kwargs):
        if command[1:2] == ["exec"] and is_staged_worker_command(command):
            return subprocess.CompletedProcess(
                command, 2, stdout="", stderr="fixture parser failure"
            )
        if command[1:2] == ["cp"] and command[-2].endswith("/response.json"):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="missing")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    adapter = DeepSoloDockerAdapter(
        "qualified-container", manifest, host_source_root=source
    )
    with pytest.raises(
        DeepSoloProcessError,
        match="produced no response.*exit 2.*fixture parser failure",
    ):
        adapter.infer(
            image_path=image,
            candidate_id=CANDIDATE,
            inference_options={"device": "cuda:0"},
            reduction_policy="highest_score",
        )


def test_real_worker_error_path_writes_traceback_without_legacy_imports(tmp_path: Path):
    image = tmp_path / "image.bin"
    image.write_bytes(b"image")
    request = DeepSoloRequest.create(
        image, CANDIDATE, {"device": "cpu"}, "highest_score", request_id="worker-test"
    )
    request_path, response_path = tmp_path / "request.json", tmp_path / "response.json"
    manifest_path = tmp_path / "manifest.json"
    request_path.write_text(json.dumps(request.to_dict()), encoding="utf-8")
    manifest_path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    assert run_worker(request_path, response_path, manifest_path) == 2
    response = DeepSoloResponse.from_dict(json.loads(response_path.read_text(encoding="utf-8")))
    assert response.success is False
    assert "schema_version" in response.payload["error"]["message"]
    assert "deepsolo/worker.py" in response.payload["error"]["traceback"]


def test_worker_preserves_predicted_class_and_raw_recognition_fields():
    import numpy as np

    class Value:
        def __init__(self, value):
            self.value = np.asarray(value)
        def tolist(self):
            return self.value.tolist()
        def numpy(self):
            return self.value

    class Instances:
        scores = Value([0.9])
        pred_classes = Value([7])
        recs = Value([[0, 1, 36]])
        rec_scores = Value([[[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]]])
        bd = Value([[[0, 0, 1, 0], [0, 1, 1, 1]]])
        ctrl_points = Value([[[0, 0], [1, 1]]])
        def to(self, device):
            assert device == "cpu"
            return self
        def has(self, field):
            return hasattr(self, field)

    cfg = type("Cfg", (), {
        "MODEL": type("Model", (), {
            "TRANSFORMER": type("Transformer", (), {
                "VOC_SIZE": 37,
                "CUSTOM_DICT": "",
            })(),
        })(),
    })()
    detection = _serialize_instances(Instances(), cfg)[0]
    assert detection["raw"]["predicted_class"] == 7
    assert detection["raw"]["recognition_token_ids"] == [0, 1, 36]
    assert detection["raw"]["recognition_token_confidences"] == [0.8, 0.9, 0.6]


def test_worker_boundary_files_parse_as_python_38():
    root = Path(__file__).resolve().parents[2]
    for path in sorted((root / "deepsolo").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
    ast.parse(
        (root / "scripts" / STAGED_ENTRYPOINT_NAME).read_text(encoding="utf-8"),
        feature_version=(3, 8),
    )
