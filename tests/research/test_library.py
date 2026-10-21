import json
from pathlib import Path

import pytest

from research.comparison import EvidenceError, digest
from research.library import inside, load_library, select_experiments, load_saved_result, check_saved_views
from research.__main__ import main
from scripts.build_experiment_library import generate_navigation, generate_results


@pytest.fixture
def library(tmp_path):
    (tmp_path / "experiments/results").mkdir(parents=True)
    (tmp_path / "research").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/example.md").write_text("# Example experiment\n\nRecorded limits.\n")
    row = dict(id="example", title="Example experiment", family="crops", guide="docs/example.md",
               code=[], configs=[], tests=[], result_groups=["sample"])
    catalog = dict(schema_version=1, families={"crops": "Crops"}, experiments=[row])
    (tmp_path / "experiments/catalog.json").write_text(json.dumps(catalog))
    registry = b'{"schema_version":1,"groups":[]}'
    (tmp_path / "research/experiments.json").write_bytes(registry)
    group = dict(id="sample", mode="approximate", result_class="approximate reproduction",
                 source=dict(root="runs", path="external.json", sha256="a" * 64))
    report = dict(group=group, source_sha256="a" * 64, rows=[dict(tp=1)])
    raw = json.dumps(report).encode()
    (tmp_path / "experiments/results/sample.json").write_bytes(raw)
    manifest = dict(schema_version=1, registry_sha256=digest(registry), groups=[dict(
        id="sample", path="experiments/results/sample.json", sha256=digest(raw), rows=1)])
    (tmp_path / "experiments/results/manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_navigation_and_saved_results_need_no_external_report(library):
    assert select_experiments(load_library(library), "crops", "example")[0]["id"] == "example"
    assert load_saved_result(library, "sample")["rows"][0]["tp"] == 1
    assert not (library / "external.json").exists()


def test_unknown_family_and_duplicate_ids_fail(library):
    catalog = load_library(library)
    with pytest.raises(EvidenceError, match="Unknown family"):
        select_experiments(catalog, "missing")
    catalog["experiments"].append(catalog["experiments"][0])
    (library / "experiments/catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(EvidenceError, match="Duplicate"):
        load_library(library)


def test_missing_code_reference_is_actionable(library):
    catalog = load_library(library)
    catalog["experiments"][0]["code"] = ["missing.py"]
    (library / "experiments/catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(EvidenceError, match="Missing experiment reference"):
        load_library(library)


@pytest.mark.parametrize("path", ["../outside.json", "/tmp/outside.json"])
def test_library_rejects_paths_outside_repository(library, path):
    with pytest.raises(EvidenceError, match="inside repository"):
        inside(library, path)


def test_library_rejects_symlink_escape(library, tmp_path):
    (library / "escape").symlink_to(tmp_path.parent)
    with pytest.raises(EvidenceError, match="inside repository"):
        inside(library, "escape/outside.json")


def test_changed_summary_is_never_silently_displayed(library):
    path = library / "experiments/results/sample.json"
    path.write_text(path.read_text().replace('"tp": 1', '"tp": 9'))
    with pytest.raises(EvidenceError, match="SHA256 mismatch"):
        load_saved_result(library, "sample")


def test_stale_registry_rejects_saved_summaries(library):
    (library / "research/experiments.json").write_text('{}')
    with pytest.raises(EvidenceError, match="registry changed"):
        load_saved_result(library, "sample")


def test_summary_cannot_be_promoted_to_strict_even_with_updated_checksum(library):
    path = library / "experiments/results/sample.json"
    report = json.loads(path.read_text())
    report["group"]["mode"] = "strict"
    raw = json.dumps(report).encode()
    path.write_bytes(raw)
    manifest_path = library / "experiments/results/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["groups"][0]["sha256"] = digest(raw)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(EvidenceError, match="scientific class"):
        load_saved_result(library, "sample")


def test_cli_lists_and_describes_experiments_without_models(library, capsys):
    assert main(["--repo-root", str(library), "experiments", "--family", "crops"]) == 0
    assert "example" in capsys.readouterr().out
    assert main(["--repo-root", str(library), "experiment", "example"]) == 0
    assert "not a completion claim" in capsys.readouterr().out


def test_navigation_build_respects_documented_links_and_spaces(library):
    (library / "experiments/families.json").write_text(json.dumps({"families": {
        "analysis": {"title": "Analysis", "prefixes": []}}}))
    (library / "docs/Guide with spaces.md").write_text("# A guide\n")
    (library / "scripts").mkdir()
    (library / "scripts/example.py").write_text('"""An entry point."""\n')
    (library / "docs/example.md").write_text("# Example\n\npython3 -m scripts.example\n")
    outputs = generate_navigation(library)
    assert "(<../../docs/Guide with spaces.md>)" in outputs["experiments/analysis/README.md"]
    catalog = json.loads(outputs["experiments/catalog.json"])
    assert next(row for row in catalog["experiments"] if row["id"] == "example")["code"] == ["scripts/example.py"]


def test_result_refresh_does_not_fabricate_missing_sources(library):
    path = library / "research/experiments.json"
    path.write_text(json.dumps({"schema_version": 1, "groups": [dict(
        id="missing", mode="approximate", result_class="approximate reproduction",
        source=dict(root="runs", path="missing.json", sha256="a" * 64))]}))
    before = (library / "experiments/results/sample.json").read_bytes()
    with pytest.raises(EvidenceError, match="root.*unavailable"):
        generate_results(library, None, None)
    assert (library / "experiments/results/sample.json").read_bytes() == before


def test_changed_csv_cannot_disagree_with_verified_summary(library):
    original = Path(__file__).resolve().parents[2]
    for name in ("research/experiments.json", "experiments/results/manifest.json",
                 "experiments/results/multitask-geometry.json", "experiments/results/multitask-geometry.md",
                 "experiments/results/multitask-geometry.csv"):
        (library / name).write_bytes((original / name).read_bytes())
    assert check_saved_views(library, "multitask-geometry")["rows"]
    path = library / "experiments/results/multitask-geometry.csv"
    path.write_text(path.read_text() + "changed data\n")
    with pytest.raises(EvidenceError, match="Stale saved-result csv"):
        check_saved_views(library, "multitask-geometry")
