import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.collaboration import (
    REQUIRED, SourceError, check_source, export_source, forbidden_reason,
)


REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    for name in REQUIRED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "# Example\n", encoding="utf-8")
    (root / ".gitignore").write_text("artifacts/\n.venvs/\n.aws/\n*.pt\n.env\n", encoding="utf-8")
    # Use the actual bounded catalog implementation, without any ML dependency.
    (root / "research").mkdir(exist_ok=True)
    for name in ("__init__.py", "__main__.py", "comparison.py", "documents.py", "library.py"):
        (root / "research" / name).write_bytes((REPOSITORY / "research" / name).read_bytes())
    return root


def test_ignored_models_and_credentials_never_enter_export(source, tmp_path):
    for name in ("artifacts/large.json", ".venvs/pkg.py", ".aws/credentials", "weights.pt", ".env"):
        path = source / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(b"private local artifact")
    files, _ = check_source(source)
    assert not any(name.startswith(("artifacts/", ".aws/", ".venvs/")) for name in files)
    target = tmp_path / "export"
    export_source(source, target)
    for name in ("artifacts", ".venvs", ".aws", "weights.pt", ".env", ".git"):
        assert not (target / name).exists()


def test_force_tracked_checkpoint_is_rejected_even_when_ignored(source):
    path = source / "weights.pt"
    path.write_bytes(b"checkpoint")
    subprocess.run(["git", "-C", str(source), "add", "-f", "weights.pt"], check=True)
    with pytest.raises(SourceError, match="external artifact"):
        check_source(source)


def test_secret_is_rejected_without_printing_its_value(source):
    token = "ghp_" + "a" * 36
    (source / "docs/leak.txt").write_text(token)
    with pytest.raises(SourceError, match="possible credential") as error:
        check_source(source)
    assert token not in str(error.value)


def test_symlink_cannot_leak_an_external_file(source, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("external")
    (source / "docs/leak.txt").symlink_to(outside)
    with pytest.raises(SourceError, match="symlinks"):
        check_source(source)


def test_oversized_source_is_explicit_error(source):
    (source / "docs/large.txt").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    with pytest.raises(SourceError, match="exceeds 2 MiB"):
        check_source(source)


@pytest.mark.parametrize("name", ["artifacts/results.json", "docs/../escape.py", "/tmp/a.py", "docker/deepsolo/.env"])
def test_forbidden_paths_are_rejected(name):
    assert forbidden_reason(name)


@pytest.mark.parametrize("name", ["tests/data/test_dataset.py", "configs/approximate/datasets/sources.yaml"])
def test_dataset_source_and_test_directories_are_allowed(name):
    assert forbidden_reason(name) is None


def test_export_preserves_original_and_records_hashes_modes_and_new_catalog(source, tmp_path):
    before = (source / "docs/AUDIT_INDEX.json").read_bytes()
    script = source / "research/__main__.py"
    script.chmod(0o755)
    target = tmp_path / "export"
    export_source(source, target)
    assert (source / "docs/AUDIT_INDEX.json").read_bytes() == before
    manifest = json.loads((target / "COLLABORATION_EXPORT.json").read_text())
    for entry in manifest["files"]:
        path = target / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert bool(path.stat().st_mode & 0o111) == entry["executable"]
    row = next(entry for entry in manifest["files"] if entry["path"] == "docs/AUDIT_INDEX.json")
    assert row["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert row["source_sha256"] != row["sha256"]
    subprocess.run(["python3", "-m", "research", "catalog", "--source-only", "--check"], cwd=target, check=True, capture_output=True)


def test_export_refuses_existing_destination_or_source_child(source, tmp_path):
    with pytest.raises(SourceError, match="already exists"):
        export_source(source, tmp_path)
    with pytest.raises(SourceError, match="outside"):
        export_source(source, source / "export")


def test_unlisted_source_is_not_silently_dropped(source):
    (source / "unknown.txt").write_text("new source")
    with pytest.raises(SourceError, match="outside the documented source allowlist"):
        check_source(source)


def test_small_result_tables_are_allowed_only_in_summary_directory(source):
    result = source / "experiments/results/example.csv"
    result.parent.mkdir(parents=True)
    result.write_text("variant,tp\nexample,1\n")
    assert "experiments/results/example.csv" in check_source(source)[0]
    (source / "docs/dataset.csv").write_text("raw,data\n")
    with pytest.raises(SourceError, match="unsupported source format"):
        check_source(source)


def test_result_csv_bytes_survive_git_storage(source):
    (source / ".gitattributes").write_bytes((REPOSITORY / ".gitattributes").read_bytes())
    result = source / "experiments/results/example.csv"
    result.parent.mkdir(parents=True)
    raw = b"variant,tp\r\nexample,1\r\n"
    result.write_bytes(raw)
    subprocess.run(["git", "-C", str(source), "add", ".gitattributes", "experiments/results/example.csv"], check=True)
    stored = subprocess.check_output(["git", "-C", str(source), "show", ":experiments/results/example.csv"])
    assert stored == raw
    subprocess.run(["git", "-C", str(source), "diff", "--cached", "--check"], check=True, capture_output=True)


def test_result_csv_still_rejects_trailing_spaces(source):
    (source / ".gitattributes").write_bytes((REPOSITORY / ".gitattributes").read_bytes())
    result = source / "experiments/results/example.csv"
    result.parent.mkdir(parents=True)
    result.write_bytes(b"variant,tp\r\nexample,1 \r\n")
    subprocess.run(["git", "-C", str(source), "add", ".gitattributes", "experiments/results/example.csv"], check=True)
    checked = subprocess.run(["git", "-C", str(source), "diff", "--cached", "--check"], capture_output=True, text=True)
    assert checked.returncode == 2
    assert "trailing whitespace" in checked.stdout
