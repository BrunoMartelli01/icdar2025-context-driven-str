"""Main-environment adapter that invokes DeepSolo only through a subprocess."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

from .candidates import verify_host_source_identity
from .contracts import DeepSoloRequest, DeepSoloResponse


STAGED_PACKAGE_NAME = "deepsolo"
STAGED_ENTRYPOINT_NAME = "context_str_m6_deepsolo_json_worker.py"


def _stage_worker_bundle(repository_root: Path, bundle: Path) -> Path:
    """Stage the repository package beside its path-verifying M6 bootstrap."""
    package = bundle / STAGED_PACKAGE_NAME
    package.mkdir()
    for source in sorted((repository_root / "deepsolo").glob("*.py")):
        shutil.copy2(source, package / source.name)
    entrypoint = bundle / STAGED_ENTRYPOINT_NAME
    shutil.copy2(repository_root / "scripts" / STAGED_ENTRYPOINT_NAME, entrypoint)
    return entrypoint


class DeepSoloProcessError(RuntimeError):
    def __init__(self, message: str, response: Optional[DeepSoloResponse] = None) -> None:
        super().__init__(message)
        self.response = response


class DeepSoloSubprocessAdapter:
    """Execute one request in the isolated runtime using atomic JSON files."""

    def __init__(
        self,
        python_executable: Path,
        runtime_manifest: Path,
        *,
        host_source_root: Path,
        timeout_seconds: float = 600.0,
        repository_root: Optional[Path] = None,
    ) -> None:
        # Keep the venv entry-point path intact.  Resolving this symlink to the
        # base interpreter (for example /usr/bin/python3.8) bypasses pyvenv.cfg
        # discovery and silently drops the isolated DeepSolo site-packages.
        self.python_executable = python_executable.absolute()
        self.runtime_manifest = runtime_manifest.resolve()
        self.host_source_root = host_source_root.resolve()
        self.timeout_seconds = timeout_seconds
        self.repository_root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
        if not self.python_executable.is_file():
            raise ValueError("DeepSolo Python executable is missing")
        if not self.runtime_manifest.is_file():
            raise ValueError("DeepSolo runtime manifest is missing")
        if not self.host_source_root.is_dir():
            raise ValueError("DeepSolo host source root is missing")

    def infer(
        self,
        *,
        image_path: Path,
        candidate_id: str,
        inference_options: Mapping[str, Any],
        reduction_policy: str,
    ) -> DeepSoloResponse:
        verified_manifest = verify_host_source_identity(
            self.runtime_manifest, self.host_source_root
        )
        request = DeepSoloRequest.create(
            image_path, candidate_id, inference_options, reduction_policy
        )
        with tempfile.TemporaryDirectory(prefix="context-str-deepsolo-") as directory:
            root = Path(directory)
            request_path, response_path = root / "request.json", root / "response.json"
            manifest_path = root / "runtime_manifest.json"
            request_path.write_text(
                json.dumps(request.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            manifest_path.write_text(
                json.dumps(verified_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            entrypoint = _stage_worker_bundle(self.repository_root, root)
            command = [
                str(self.python_executable), str(entrypoint),
                "--request", str(request_path), "--response", str(response_path),
                "--runtime-manifest", str(manifest_path),
            ]
            completed = subprocess.run(
                command,
                cwd=self.repository_root,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if not response_path.is_file():
                raise DeepSoloProcessError(
                    "DeepSolo worker produced no response (exit {}): {}".format(
                        completed.returncode, completed.stderr.strip()
                    )
                )
            response = DeepSoloResponse.from_dict(
                json.loads(response_path.read_text(encoding="utf-8"))
            )
            if response.payload["request_id"] != request.request_id:
                raise DeepSoloProcessError("DeepSolo response request_id mismatch", response)
            if completed.returncode != 0 or not response.success:
                error = response.payload.get("error") or {}
                raise DeepSoloProcessError(
                    "DeepSolo worker failed: {}".format(error.get("message", completed.stderr.strip())),
                    response,
                )
            return response


class DeepSoloDockerAdapter:
    """Run the unchanged JSON worker inside the qualified persistent container.

    A request bundle is copied to a unique container temporary directory. This
    does not require Detectron2 imports, repository mounts, or host image paths
    to be visible inside the container.
    """

    def __init__(
        self,
        container_name: str,
        runtime_manifest: Path,
        *,
        host_source_root: Path,
        docker_executable: str = "docker",
        container_python: str = "/opt/context-str-deepsolo/bin/python",
        container_workdir: str = "/workspace/DeepSolo",
        timeout_seconds: float = 600.0,
        runtime_environment: Optional[Mapping[str, str]] = None,
        repository_root: Optional[Path] = None,
    ) -> None:
        if not container_name or any(character.isspace() for character in container_name):
            raise ValueError("A non-empty Docker container name without whitespace is required")
        self.container_name = container_name
        self.runtime_manifest = runtime_manifest.resolve()
        self.host_source_root = host_source_root.resolve()
        self.docker_executable = docker_executable
        self.container_python = container_python
        self.container_workdir = container_workdir
        self.timeout_seconds = timeout_seconds
        self.runtime_environment = dict(runtime_environment or {})
        self.repository_root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
        if not self.runtime_manifest.is_file():
            raise ValueError("DeepSolo runtime manifest is missing")
        if not self.host_source_root.is_dir():
            raise ValueError("DeepSolo host source root is missing")

    def _run(self, command: list[str], *, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            cwd=self.repository_root,
            capture_output=True,
            text=True,
            timeout=timeout or self.timeout_seconds,
        )

    def infer(
        self,
        *,
        image_path: Path,
        candidate_id: str,
        inference_options: Mapping[str, Any],
        reduction_policy: str,
    ) -> DeepSoloResponse:
        verified_manifest = verify_host_source_identity(
            self.runtime_manifest, self.host_source_root
        )
        source_image = image_path.resolve()
        request_id = str(uuid.uuid4())
        directory_name = "context-str-deepsolo-{}".format(request_id)
        container_root = "/tmp/{}".format(directory_name)
        with tempfile.TemporaryDirectory(prefix="context-str-deepsolo-docker-") as temporary:
            temporary_root = Path(temporary)
            bundle = temporary_root / directory_name
            bundle.mkdir()
            image_name = "input{}".format(source_image.suffix.lower() or ".image")
            bundled_image = bundle / image_name
            shutil.copy2(source_image, bundled_image)
            request = DeepSoloRequest.create(
                bundled_image,
                candidate_id,
                inference_options,
                reduction_policy,
                request_id=request_id,
            )
            request_value = request.to_dict()
            request_value["image_path"] = "{}/{}".format(container_root, image_name)
            request = DeepSoloRequest.from_dict(request_value)
            (bundle / "request.json").write_text(
                json.dumps(request.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (bundle / "runtime_manifest.json").write_text(
                json.dumps(verified_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _stage_worker_bundle(self.repository_root, bundle)

            copy_in = self._run([
                self.docker_executable, "cp", str(bundle), "{}:/tmp".format(self.container_name),
            ])
            if copy_in.returncode != 0:
                raise DeepSoloProcessError(
                    "Could not copy DeepSolo request into container: {}".format(
                        copy_in.stderr.strip()
                    )
                )
            try:
                command = [self.docker_executable, "exec", "-w", self.container_workdir]
                for key, value in sorted(self.runtime_environment.items()):
                    command.extend(["-e", "{}={}".format(key, value)])
                command.extend([
                    self.container_name,
                    self.container_python,
                    "{}/{}".format(container_root, STAGED_ENTRYPOINT_NAME),
                    "--request", "{}/request.json".format(container_root),
                    "--response", "{}/response.json".format(container_root),
                    "--runtime-manifest", "{}/runtime_manifest.json".format(container_root),
                ])
                completed = self._run(command)
                host_response = temporary_root / "response.json"
                copy_out = self._run([
                    self.docker_executable, "cp",
                    "{}:{}/response.json".format(self.container_name, container_root),
                    str(host_response),
                ])
                if copy_out.returncode != 0 or not host_response.is_file():
                    raise DeepSoloProcessError(
                        "DeepSolo Docker worker produced no response (exit {}): {}".format(
                            completed.returncode, completed.stderr.strip()
                        )
                    )
                response = DeepSoloResponse.from_dict(
                    json.loads(host_response.read_text(encoding="utf-8"))
                )
                if response.payload["request_id"] != request.request_id:
                    raise DeepSoloProcessError("DeepSolo response request_id mismatch", response)
                if completed.returncode != 0 or not response.success:
                    error = response.payload.get("error") or {}
                    raise DeepSoloProcessError(
                        "DeepSolo Docker worker failed: {}".format(
                            error.get("message", completed.stderr.strip())
                        ),
                        response,
                    )
                return response
            finally:
                # Delete only the UUID-scoped directory created by this call.
                self._run([
                    self.docker_executable, "exec", self.container_name,
                    self.container_python, "-c",
                    "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)",
                    container_root,
                ], timeout=30.0)
