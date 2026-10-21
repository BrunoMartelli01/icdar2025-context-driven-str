import json
from pathlib import Path

import pytest

from research.__main__ import main
from research.comparison import EvidenceError
from research.documents import headings, inventory, search, section


@pytest.fixture
def documents(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "provenance").mkdir()
    (docs / "history.md").write_text("# History\n\n## X-001 — open question\n\nOriginal evidence.\n\n### Detail\n\nMore evidence.\n\n## X-002\n\nOther evidence.\n")
    return dict(repository=tmp_path)


def test_catalog_hashes_sources_and_is_deterministic(documents):
    first = inventory(documents)
    assert inventory(documents) == first
    assert first["entries"][0]["headings"][1]["line"] == 3
    (documents["repository"] / "docs/history.md").write_text("Changed")
    assert inventory(documents)["entries"][0]["sha256"] != first["entries"][0]["sha256"]


def test_section_retains_original_text_and_nested_evidence(documents):
    result = section(documents, "X-001")
    assert result.startswith("repository:docs/history.md:3")
    assert "### Detail" in result and "More evidence." in result
    assert "X-002" not in result


def test_ambiguous_section_requires_disambiguation(documents):
    (documents["repository"] / "docs/other.md").write_text("# Other\n\n## X-001\n\nDifferent evidence.")
    with pytest.raises(EvidenceError, match="found 2"):
        section(documents, "X-001")
    assert "Original evidence" in section(documents, "X-001", "history.md")


def test_code_fences_do_not_create_sections():
    result = headings("# Real\n```python\n## Fake\n```\n~~~\n# Also fake\n~~~\n## Actual\n")
    assert [h["title"] for h in result] == ["Real", "Actual"]


def test_search_is_literal_and_bounded(documents):
    path = documents["repository"] / "docs/regex.md"
    path.write_text("# Regex\n\na.*b\nAB\na.*b again\n")
    matches = search(documents, "a.*b", limit=1)
    assert len(matches) == 1 and "regex.md:3" in matches[0]


def test_search_long_json_line_shows_the_actual_match(documents):
    path = documents["repository"] / "provenance/large.json"
    path.write_text(json.dumps({"padding": "x" * 1000, "status": "important finding"}))
    result = search(documents, "important finding")
    assert len(result) == 1 and "important finding" in result[0]
    assert len(result[0]) < 400


def test_cache_pruning_does_not_even_enter_sample_directories(documents, monkeypatch):
    root = documents["repository"] / "artifacts" / "audit-v1"
    root.mkdir(parents=True)
    (root / "audit.json").write_text('{"status":"FAILED"}')
    generated = root / "generations"
    generated.mkdir()
    (generated / "audit.json").write_text("Invalid JSON that must never be read")
    original = Path.iterdir
    def checked(path):
        if path.name == "generations":
            raise AssertionError("Attempt to enter generation cache")
        return original(path)
    monkeypatch.setattr(Path, "iterdir", checked)
    index = inventory(documents)
    receipt = next(row for row in index["entries"] if row["path"].endswith("audit.json"))
    assert receipt["reported_metadata"]["status"] == "FAILED"


def test_generated_indices_are_not_self_indexed(documents):
    for name in ("AUDIT_INDEX.json", "AUDIT_INDEX.md"):
        (documents["repository"] / "docs" / name).write_text("not a source")
    assert len(inventory(documents)["entries"]) == 1


def test_external_audits_require_explicit_root_and_do_not_follow_symlinks(documents, tmp_path):
    external = tmp_path / "separate"
    external.mkdir()
    (external / "audit.json").write_text('{"status":"PASS"}')
    (documents["repository"] / "docs/link.md").symlink_to(external / "audit.json")
    assert all(row["root"] == "repository" for row in inventory(documents)["entries"])
    result = inventory(dict(documents, external_artifacts=external))
    assert any(row["root"] == "external_artifacts" for row in result["entries"])
    assert not any("link.md" in row["path"] for row in result["entries"])


def test_cli_catalog_check_detects_changed_sources_without_writes(documents, capsys):
    root = documents["repository"]
    args = ["--repo-root", str(root), "catalog"]
    assert main(args) == 0
    assert main(args + ["--check"]) == 0
    index = root / "docs/AUDIT_INDEX.json"
    original = index.read_bytes()
    (root / "docs/history.md").write_text("New evidence")
    assert main(args + ["--check"]) == 1
    assert index.read_bytes() == original
    assert "Stale/missing" in capsys.readouterr().err
