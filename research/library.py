"""Portable experiment navigation and verified saved-result summaries.

No model imports, dataset reads, inference, or changes to scientific protocols.
"""

import json
import re
from pathlib import Path

from research.comparison import EvidenceError, digest, render


def inside(repository, relative):
    root = Path(repository).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root not in path.parents:
        raise EvidenceError("Library path must remain inside repository: " + relative)
    return path


def load_library(repository):
    library = json.loads(inside(repository, "experiments/catalog.json").read_text(encoding="utf-8"))
    if library.get("schema_version") != 1:
        raise EvidenceError("Unsupported experiment library schema")
    ids = [row["id"] for row in library["experiments"]]
    if len(ids) != len(set(ids)):
        raise EvidenceError("Duplicate experiment library IDs")
    for row in library["experiments"]:
        if row["family"] not in library["families"]:
            raise EvidenceError("Unknown experiment family: " + row["family"])
        for name in [row["guide"]] + row["code"] + row["configs"] + row["tests"]:
            if not inside(repository, name).is_file():
                raise EvidenceError("Missing experiment reference: " + name)
    return library


def select_experiments(library, family=None, query=None):
    if family and family not in library["families"]:
        raise EvidenceError("Unknown family; choose from: " + ", ".join(library["families"]))
    rows = [row for row in library["experiments"] if not family or row["family"] == family]
    if query:
        rows = [row for row in rows if query.casefold() in json.dumps(row, ensure_ascii=False).casefold()]
    return rows


def experiment_markdown(row):
    lines = ["# " + row["title"], "", "ID: `" + row["id"] + "`; family: `" + row["family"] + "`.",
             "", "Guide: [" + row["guide"] + "](<" + row["guide"] + ">).", "",
             "Status and limitations are recorded in the dated guide/audits; membership is not a completion claim.", ""]
    for field, title in (("code", "Code"), ("configs", "Configurations"), ("tests", "Tests")):
        lines += ["## " + title, ""]
        lines += ["- [" + name + "](<" + name + ">)" for name in row[field]] or ["No explicit reference indexed; inspect the guide."]
        lines.append("")
    lines += ["## Saved numerical comparisons", ""]
    lines += ["- `python3 -m research results " + group + "`" for group in row["result_groups"]] or [
        "No comparison registered for this guide. Read its evidence; missing numerical summaries are not zero results."]
    return "\n".join(lines) + "\n"


def load_saved_result(repository, group_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", group_id):
        raise EvidenceError("Invalid comparison ID")
    manifest = json.loads(inside(repository, "experiments/results/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise EvidenceError("Unsupported saved-result manifest")
    registry = inside(repository, "research/experiments.json").read_bytes()
    if digest(registry) != manifest["registry_sha256"]:
        raise EvidenceError("Comparison registry changed; regenerate and review the saved-result summaries")
    entries = [row for row in manifest["groups"] if row["id"] == group_id]
    if len(entries) != 1:
        raise EvidenceError("Unknown saved comparison: " + group_id)
    entry = entries[0]
    raw = inside(repository, entry["path"]).read_bytes()
    if digest(raw) != entry["sha256"]:
        raise EvidenceError("Saved-result SHA256 mismatch: " + group_id)
    report = json.loads(raw)
    group = report["group"]
    if group["id"] != group_id or group["mode"] != "approximate" or group["result_class"] != "approximate reproduction":
        raise EvidenceError("Saved-result identity or scientific class mismatch")
    if len(report["rows"]) != entry["rows"] or report["source_sha256"] != group["source"]["sha256"]:
        raise EvidenceError("Saved-result rows or source identity mismatch")
    return report


def saved_result_markdown(report):
    content = render(report, "markdown")
    group = report["group"]
    location = group["source"]["root"] + ":" + group["source"]["path"]
    line = "Source: [saved report](<{}>); SHA256 `{}`.".format(report["source_path"], report["source_sha256"])
    content = content.replace(line, "Original evidence: `{}`; SHA256 `{}`.".format(location, report["source_sha256"]))
    return content + ("\nThis is a saved numerical summary, not a new evaluation. Its checksum is verified locally; "
                      "recheck original evidence with `python3 -m research compare " + group["id"] + "`.\n")


def check_saved_views(repository, group_id):
    report = load_saved_result(repository, group_id)
    for extension, expected in (("csv", render(report, "csv")), ("md", saved_result_markdown(report))):
        path = inside(repository, "experiments/results/" + group_id + "." + extension)
        if path.read_bytes() != expected.encode("utf-8"):
            raise EvidenceError("Stale saved-result " + extension + " view: " + group_id)
    return report
