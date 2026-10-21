"""Bounded audit discovery and section reads, without walking sample caches."""

import json
import re
from pathlib import Path

from research.comparison import EvidenceError, digest


SKIP_DIRS = {
    ".git", ".aws", ".venvs", "__pycache__", "generations", "cache", "ocr",
    "crops", "images", "samples", "shards", "per_image", "exports", "masks",
    "probabilities", "checkpoints", "datasets", "dataset", "data", "visuals",
    "proposals", "features", "predictions", "responses", "weights", "node_modules",
}
GENERATED = {"AUDIT_INDEX.md", "AUDIT_INDEX.json"}


def walk(root, max_depth):
    def visit(directory, depth):
        for path in sorted(directory.iterdir()):
            if path.is_symlink():
                continue
            if path.is_dir() and depth < max_depth and path.name not in SKIP_DIRS:
                yield from visit(path, depth + 1)
            elif path.is_file():
                yield path
    if root.is_dir():
        yield from visit(root, 0)


def document_paths(roots):
    repository = Path(roots["repository"])
    for folder in ("docs", "provenance"):
        for path in walk(repository / folder, 3):
            if path.name not in GENERATED and path.suffix.lower() in (".md", ".json", ".yaml", ".txt"):
                yield "repository", path.relative_to(repository).as_posix(), path
    for name, base in (("repository", repository / "artifacts"),
                       ("runs", roots.get("runs")), ("external_artifacts", roots.get("external_artifacts"))):
        if base is None:
            continue
        base = Path(base)
        for path in walk(base, 5):
            if path.suffix.lower() not in (".md", ".json", ".txt"):
                continue
            filename = path.name.lower()
            named_audit = "audit" in filename or "review" in filename
            audit_directory = any("audit" in part.lower() or "review" in part.lower()
                                  for part in path.relative_to(base).parts[:-1])
            audit_summary = audit_directory and filename in {
                "results.json", "verification.json", "provenance.json", "summary.json",
                "readme.md", "report.md", "policy_ablation.json", "metric_sensitivity.json",
            }
            if named_audit or filename == "report.md" or audit_summary:
                root = repository if name == "repository" else base
                yield name, path.relative_to(root).as_posix(), path


def headings(text):
    result = []
    fence = None
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            char = marker.group(1)[0]
            if fence is None:
                fence = char
            elif fence == char:
                fence = None
            continue
        if fence is None:
            match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if match:
                result.append(dict(line=line_number, level=len(match.group(1)), title=match.group(2)))
    return result


def inventory(roots):
    entries = []
    for root, relative, path in document_paths(roots):
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        entry = dict(root=root, path=relative, bytes=len(raw), sha256=digest(raw))
        if path.suffix == ".md":
            entry["headings"] = headings(text)
            entry["title"] = entry["headings"][0]["title"] if entry["headings"] else path.stem
        else:
            entry["title"] = path.stem
        if path.suffix == ".json":
            doc = json.loads(text)
            if isinstance(doc, dict):
                entry["reported_metadata"] = {
                    k: doc[k] for k in ("status", "result_class", "reproduction_mode", "claim_label")
                    if k in doc and isinstance(doc[k], (str, int, bool))
                }
        entries.append(entry)
    return dict(schema_version=1, scope={
        "roots": sorted(k for k, v in roots.items() if v is not None),
        "discovery": "docs/provenance depth3; named audit/review and report summaries depth5; no symlinks or sample caches",
        "excluded_directories": sorted(SKIP_DIRS),
        "verification": "inventory and content hashes only; reported statuses are historical claims, not rerun qualifications",
    }, entries=sorted(entries, key=lambda e: (e["root"], e["path"])))


def index_markdown(index, roots):
    lines = ["# Audit and evidence index", "", "Generated with `python3 -m research catalog`; preserve original sources.",
             "", "{} documents. {}.".format(len(index["entries"]), index["scope"]["discovery"]),
             "", "Hashes and all Markdown section positions: [AUDIT_INDEX.json](AUDIT_INDEX.json).",
             "Reported statuses below are copied metadata, not new scientific qualifications.", ""]
    for root in index["scope"]["roots"]:
        lines += ["## " + root, "", "| Source | Reported status | Bytes |", "| --- | --- | ---: |"]
        for entry in index["entries"]:
            if entry["root"] != root:
                continue
            path = entry["path"]
            link = "../" + path if root == "repository" else str(Path(roots[root]) / path)
            status = entry.get("reported_metadata", {}).get("status", "—")
            lines.append("| [{}](<{}>) | {} | {} |".format(path, link, status, entry["bytes"]))
        lines.append("")
    lines += ["## Targeted reads", "", "```bash", "python3 -m research find 'kernel'",
              "python3 -m research section TABLE5-SOURCE-METRIC-AUDIT-001", "```", "",
              "Section output retains its exact source path and line numbers. External roots are opt-in.", ""]
    return "\n".join(lines)


def search(roots, query, limit=20):
    matches = []
    needle = query.casefold()
    for root, relative, path in document_paths(roots):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if needle in line.casefold():
                position = line.casefold().index(needle)
                start = max(0, position - 80)
                snippet = ("…" if start else "") + line[start:start + 300].strip()
                matches.append("{}:{}:{}: {}".format(root, relative, number, snippet))
                if len(matches) >= limit:
                    return matches
    return matches


def section(roots, query, file_filter=None):
    matches = []
    for root, relative, path in document_paths(roots):
        if path.suffix != ".md" or (file_filter and file_filter not in relative):
            continue
        text = path.read_text(encoding="utf-8")
        outline = headings(text)
        for index, heading in enumerate(outline):
            if query.casefold() in heading["title"].casefold():
                end = next((h["line"] - 1 for h in outline[index + 1:] if h["level"] <= heading["level"]), len(text.splitlines()))
                matches.append((root, relative, heading["line"], end, text.splitlines()))
    if len(matches) != 1:
        candidates = ["{}:{}:{}".format(*m[:3]) for m in matches]
        raise EvidenceError("Expected one section; found {}. Refine the heading or use --file. {}".format(len(matches), ", ".join(candidates)))
    root, path, start, end, lines = matches[0]
    return "{}:{}:{}\n\n{}\n".format(root, path, start, "\n".join(lines[start - 1:end]))
