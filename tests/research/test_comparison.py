"""Protect scientific reporting boundaries without inference or external data."""

import copy
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from research.__main__ import main
from research.comparison import EvidenceError, compare, digest, overview, pointer, render


@pytest.fixture
def evidence(tmp_path):
    document = {"status": "SUCCESS", "mode": "approximate", "rows": [
        {"name": "base", "tp": 5, "predictions": 10, "gt": 20, "images": 3, "f1": 1 / 3},
        {"name": "variant", "tp": 6, "predictions": 12, "gt": 20, "images": 3, "f1": 0.375},
    ]}
    source = tmp_path / "result.json"
    source.write_text(json.dumps(document))
    group = dict(id="fixture", title="Fixture", mode="approximate", result_class="approximate reproduction",
                 classification="ASSUMPTION", dataset="fixture", split="TRAIN holdout", metric="bbox_exact",
                 protocol="IoU>=0.5 and exact casefold; all care GT retained", limitations=["Fixture only"],
                 source=dict(root="repository", path="result.json", sha256=digest(source.read_bytes())),
                 checks={"doc:/status": "SUCCESS", "doc:/mode": "approximate"}, rows="/rows", name=["row:/name"],
                 counts={k: "row:/" + k for k in ("tp", "predictions", "gt", "images")},
                 population=dict(gt=20, images=3), expected_rows=2, verify_rates=dict(f1="row:/f1"))
    return group, dict(repository=tmp_path), source, document


def rewrite(evidence):
    group, _, source, document = evidence
    source.write_text(json.dumps(document))
    group["source"]["sha256"] = digest(source.read_bytes())


def test_complete_denominators_and_micro_metrics(evidence):
    group, roots, _, _ = evidence
    row = compare(group, roots)["rows"][0]
    assert (row["tp"], row["fp"], row["fn"], row["gt"]) == (5, 5, 15, 20)
    assert row["precision"] == 0.5
    assert row["recall"] == 0.25
    assert row["f1"] == pytest.approx(1 / 3)
    assert row["tn0"] == 0.2
    assert row["source_pointer"] == "/rows/0"


def test_hash_change_never_produces_partial_table(evidence):
    group, roots, source, _ = evidence
    source.write_text(source.read_text() + "\n")
    with pytest.raises(EvidenceError, match="SHA256 mismatch"):
        compare(group, roots)


def test_missing_source_is_actionable(evidence):
    group, roots, source, _ = evidence
    source.unlink()
    with pytest.raises(EvidenceError, match="Restore the pinned source"):
        compare(group, roots)


def test_roots_are_never_inferred(evidence):
    group, _, _, _ = evidence
    with pytest.raises(EvidenceError, match="root.*unavailable"):
        compare(group, {})


@pytest.mark.parametrize("path", ["../outside.json", "/outside.json"])
def test_source_cannot_escape_declared_root(evidence, path):
    group, roots, _, _ = evidence
    group["source"]["path"] = path
    with pytest.raises(EvidenceError, match="inside its declared root"):
        compare(group, roots)


def test_symlink_source_cannot_escape_root(evidence, tmp_path):
    group, roots, _, _ = evidence
    (tmp_path / "external.json").symlink_to(tmp_path.parent / "unrelated.json")
    group["source"]["path"] = "external.json"
    with pytest.raises(EvidenceError, match="inside its declared root"):
        compare(group, roots)


@pytest.mark.parametrize("field,value,match", [
    ("status", "FAILED", "qualification mismatch"),
    ("mode", "strict", "qualification mismatch"),
])
def test_saved_status_cannot_be_promoted(evidence, field, value, match):
    group, roots, _, document = evidence
    document[field] = value
    rewrite(evidence)
    with pytest.raises(EvidenceError, match=match):
        compare(group, roots)


def test_registry_cannot_promote_approximate_rows_to_strict(evidence):
    group, roots, _, _ = evidence
    group["mode"] = "strict"
    with pytest.raises(EvidenceError, match="approximate groups only"):
        compare(group, roots)


@pytest.mark.parametrize("field,value", [("gt", 21), ("images", 2)])
def test_changed_population_fails(evidence, field, value):
    group, roots, _, document = evidence
    document["rows"][1][field] = value
    rewrite(evidence)
    with pytest.raises(EvidenceError, match="Population mismatch"):
        compare(group, roots)


@pytest.mark.parametrize("value", [-1, True, 5.0, None, "5"])
def test_noninteger_and_negative_counts_fail(evidence, value):
    group, roots, _, document = evidence
    document["rows"][0]["tp"] = value
    rewrite(evidence)
    with pytest.raises(EvidenceError, match="nonnegative integer"):
        compare(group, roots)


def test_impossible_matching_counts_fail(evidence):
    group, roots, _, document = evidence
    document["rows"][0]["tp"] = 11
    rewrite(evidence)
    with pytest.raises(EvidenceError, match="TP exceeds"):
        compare(group, roots)


@pytest.mark.parametrize("rate", [0.5, float("nan"), float("inf"), True])
def test_inconsistent_or_nonfinite_reported_rates_fail(evidence, rate):
    group, roots, _, document = evidence
    document["rows"][0]["f1"] = rate
    rewrite(evidence)
    with pytest.raises(EvidenceError, match="disagrees with counts"):
        compare(group, roots)


def test_missing_row_field_is_not_defaulted(evidence):
    group, roots, _, document = evidence
    del document["rows"][1]["tp"]
    rewrite(evidence)
    with pytest.raises(EvidenceError, match="Missing evidence.*tp"):
        compare(group, roots)


def test_duplicate_variants_are_not_silently_collapsed(evidence):
    group, roots, _, document = evidence
    document["rows"][1]["name"] = "base"
    rewrite(evidence)
    with pytest.raises(EvidenceError, match="Duplicate variant"):
        compare(group, roots)


def test_missing_registered_rows_fail(evidence):
    group, roots, _, document = evidence
    document["rows"].pop()
    rewrite(evidence)
    with pytest.raises(EvidenceError, match="Expected 2 rows"):
        compare(group, roots)


def test_explicit_filter_keeps_diagnostic_out_of_main_group(evidence):
    group, roots, _, document = evidence
    document["rows"].append(dict(document["rows"][0], name="oracle"))
    group["where"] = {"row:/name": "variant"}
    group["expected_rows"] = 1
    rewrite(evidence)
    assert [r["variant"] for r in compare(group, roots)["rows"]] == ["variant"]


def test_pointer_and_dynamic_key_preserve_slashes_and_tildes(evidence):
    group, roots, _, document = evidence
    row = document["rows"][0]
    document["rows"] = {"kernel/pad~2": row}
    document["epochs"] = {"kernel/pad~2": 17}
    group.update(name=["key:"], keys=["kernel/pad~2"], expected_rows=1,
                 extra={"epoch": "doc:/epochs/{key}"})
    rewrite(evidence)
    report = compare(group, roots)
    assert report["rows"][0]["source_pointer"] == "/rows/kernel~1pad~02"
    assert report["rows"][0]["extra"]["epoch"] == 17
    assert pointer(document, report["rows"][0]["source_pointer"]) == row
    group["keys"].append("missing")
    with pytest.raises(EvidenceError, match="Missing registered rows"):
        compare(group, roots)


@pytest.mark.parametrize("path", ["/rows/-1", "/rows/01", "/rows/x", "/bad~2key"])
def test_invalid_json_pointer_cannot_select_unintended_rows(evidence, path):
    with pytest.raises(EvidenceError, match="Missing evidence"):
        pointer(evidence[3], path)


def test_paired_gain_and_loss_must_match_tp_difference(evidence):
    group, roots, _, document = evidence
    for row, gained, lost in ((document["rows"][0], 0, 0), (document["rows"][1], 3, 2)):
        row.update(gained=gained, lost=lost)
    group.update(paired_baseline="base", extra=dict(gained="row:/gained", lost="row:/lost"))
    rewrite(evidence)
    assert len(compare(group, roots)["rows"]) == 2
    document["rows"][1]["lost"] = 0
    rewrite(evidence)
    with pytest.raises(EvidenceError, match="Gained/lost counts disagree"):
        compare(group, roots)


def test_paired_baseline_cannot_come_from_different_group(evidence):
    group, roots, _, _ = evidence
    group["paired_baseline"] = "external baseline"
    with pytest.raises(EvidenceError, match="same group"):
        compare(group, roots)


@pytest.mark.parametrize("gt,predictions,expected_f1", [(20, 0, 0.0), (0, 0, None)])
def test_empty_predictions_retain_gt_and_undefined_rates(evidence, gt, predictions, expected_f1):
    group, roots, _, document = evidence
    document["rows"] = [dict(name="empty", tp=0, predictions=predictions, gt=gt, images=3)]
    group.update(population=dict(gt=gt, images=3), expected_rows=1, verify_rates={})
    rewrite(evidence)
    row = compare(group, roots)["rows"][0]
    assert row["fn"] == gt
    assert row["precision"] is None
    assert row["f1"] == expected_f1


def test_csv_and_json_keep_protocol_counts_and_provenance(evidence):
    group, roots, _, _ = evidence
    report = compare(group, roots)
    csvrow = list(csv.DictReader(io.StringIO(render(report, "csv"))))[0]
    assert csvrow["protocol"] == group["protocol"]
    assert csvrow["source_sha256"] == group["source"]["sha256"]
    assert csvrow["result_class"] == "approximate reproduction"
    assert json.loads(csvrow["source_qualification"]) == group["checks"]
    assert csvrow["fn"] == "15"
    assert json.loads(render(report, "json"))["rows"][0]["f1"] == 1 / 3


def test_markdown_escapes_variant_and_keeps_counts(evidence):
    group, roots, _, document = evidence
    document["rows"][0]["name"] = "a|b\nc"
    rewrite(evidence)
    result = render(compare(group, roots), "markdown")
    assert "a\\|b c" in result
    assert "approximate reproduction" in result
    assert "TRAIN holdout" in result


def write_registry(evidence):
    group, roots, _, _ = evidence
    path = roots["repository"] / "registry.json"
    path.write_text(json.dumps(dict(schema_version=1, groups=[group], overview_groups=[group["id"]])))
    return path


def test_cli_output_never_overwrites_existing_source(evidence, capsys):
    _, roots, source, _ = evidence
    registry = write_registry(evidence)
    before = source.read_bytes()
    status = main(["--repo-root", str(roots["repository"]), "--registry", str(registry),
                   "compare", "fixture", "--format", "csv", "--output", str(source)])
    assert status == 1
    assert "File exists" in capsys.readouterr().err
    assert source.read_bytes() == before


def test_cli_check_reports_every_failed_group(evidence, capsys):
    group, roots, _, _ = evidence
    registry = write_registry(evidence)
    missing = copy.deepcopy(group)
    missing["id"] = "missing"
    missing["source"]["path"] = "missing.json"
    registry.write_text(json.dumps(dict(schema_version=1, groups=[missing, dict(missing, id="also-missing"), group])))
    status = main(["--repo-root", str(roots["repository"]), "--registry", str(registry), "check"])
    output = capsys.readouterr()
    assert status == 1
    assert "FAIL missing" in output.err and "FAIL also-missing" in output.err
    assert "1 passed; 2 failed" in output.out


def test_overview_retains_separate_groups(evidence):
    group, roots, _, _ = evidence
    other = dict(group, id="different-protocol", title="Different matching", protocol="lexical matching")
    result = overview(dict(groups=[group, other], overview_groups=[group["id"], other["id"]]), roots)
    assert result.count("| variant | tp | predictions |") == 2
    assert "lexical matching" in result and "IoU>=0.5" in result


def test_groups_work_without_site_packages_models_or_external_roots():
    repository = Path(__file__).resolve().parents[2]
    code = "import runpy,sys; sys.argv=['research','groups']; runpy.run_module('research',run_name='__main__'); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-S", "-c", code], cwd=repository, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "native128-pad0" in result.stdout
