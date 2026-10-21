"""Explicit projections of pinned reports; never load models or infer protocols."""

import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path


class EvidenceError(ValueError):
    """Missing, changed or inconsistent evidence; no partial comparison is valid."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pointer(value, path):
    """Resolve an RFC 6901 JSON pointer, including keys containing slashes."""
    if path == "":
        return value
    if not path.startswith("/"):
        raise EvidenceError("JSON pointer must start with '/': " + path)
    try:
        for token in path[1:].split("/"):
            if re.search(r"~(?:[^01]|$)", token):
                raise ValueError("Invalid JSON pointer escape")
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not re.fullmatch(r"0|[1-9][0-9]*", token):
                    raise ValueError("Invalid JSON pointer array index")
                value = value[int(token)]
            else:
                value = value[token]
        return value
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise EvidenceError("Missing evidence at JSON pointer " + path) from exc


def escape_key(key):
    return str(key).replace("~", "~0").replace("/", "~1")


def select(expression, document, row, key):
    if expression == "key:":
        return key
    scope, sep, path = expression.partition(":")
    if not sep or scope not in ("doc", "row"):
        raise EvidenceError("Use doc:/pointer, row:/pointer or key:; got " + expression)
    return pointer(document if scope == "doc" else row,
                   path.replace("{key}", escape_key(key)))


def load_registry(path):
    registry = read_json(path)
    if registry.get("schema_version") != 1:
        raise EvidenceError("Unsupported comparison registry schema")
    ids = [g["id"] for g in registry["groups"]]
    if len(set(ids)) != len(ids):
        raise EvidenceError("Duplicate comparison group IDs")
    return registry


def source_path(source, roots):
    root = roots.get(source["root"])
    if root is None:
        raise EvidenceError("Source root '{}' is unavailable; supply its CLI root option".format(source["root"]))
    root = Path(root).resolve()
    path = (root / source["path"]).resolve()
    if root not in path.parents:
        raise EvidenceError("Source must remain inside its declared root: " + source["path"])
    return path


def count(value, name):
    if type(value) is not int or value < 0:
        raise EvidenceError("{} must be a nonnegative integer, got {!r}".format(name, value))
    return value


def compare(group, roots):
    # This registry is an approximate reporting interface, never a strict execution config.
    if group["mode"] != "approximate" or group["result_class"] != "approximate reproduction":
        raise EvidenceError("This reporting schema supports explicitly approximate groups only")
    path = source_path(group["source"], roots)
    if not path.is_file():
        raise EvidenceError("Missing saved report: {}. Restore the pinned source; no inference is started.".format(path))
    data = path.read_bytes()
    actual_hash = digest(data)
    if actual_hash != group["source"]["sha256"]:
        raise EvidenceError("SHA256 mismatch for {}. Review the changed report before updating the registry.".format(path))
    document = json.loads(data)
    for expression, expected in group["checks"].items():
        if select(expression, document, None, "") != expected:
            raise EvidenceError("Source qualification mismatch: " + expression)
    collection = pointer(document, group["rows"])
    if not isinstance(collection, (list, dict)):
        raise EvidenceError("The registered row pointer must select a list or object")
    if "keys" in group:
        if not isinstance(collection, dict):
            raise EvidenceError("Explicit row keys require an object")
        if len(set(group["keys"])) != len(group["keys"]):
            raise EvidenceError("Duplicate registered row keys")
        missing = set(group["keys"]) - set(collection)
        if missing:
            raise EvidenceError("Missing registered rows: " + str(sorted(missing)))
        items = [(key, collection[key]) for key in group["keys"]]
    else:
        items = list(collection.items()) if isinstance(collection, dict) else list(enumerate(collection))
    rows = []
    names = set()
    for key, row in items:
        get = lambda expression: select(expression, document, row, key)
        if any(get(expr) != expected for expr, expected in group.get("where", {}).items()):
            continue
        name = " / ".join(str(get(expr)) for expr in group["name"])
        if name in names:
            raise EvidenceError("Duplicate variant identity: " + name)
        names.add(name)
        counts = {k: count(get(v), k) for k, v in group["counts"].items()}
        for key_name, expected in group["population"].items():
            if counts[key_name] != expected:
                raise EvidenceError("Population mismatch for {}: {} != {}".format(name, counts[key_name], expected))
        tp, predictions, gt = counts["tp"], counts["predictions"], counts["gt"]
        if tp > min(predictions, gt):
            raise EvidenceError("TP exceeds predictions or GT for " + name)
        fp, fn = predictions - tp, gt - tp
        # Undefined empty populations are not silently assigned a perfect or zero score.
        rates = {
            "precision": tp / predictions if predictions else None,
            "recall": tp / gt if gt else None,
            "f1": 2 * tp / (predictions + gt) if predictions + gt else None,
            "tn0": tp / (tp + fp + fn) if tp + fp + fn else None,
        }
        for metric, expression in group.get("verify_rates", {}).items():
            recorded = get(expression)
            calculated = rates[metric]
            if (type(recorded) not in (int, float) or not math.isfinite(recorded)
                    or calculated is None or not math.isclose(recorded, calculated, rel_tol=1e-10, abs_tol=1e-12)):
                raise EvidenceError("Recorded {} disagrees with counts for {}".format(metric, name))
        extra = {k: get(v) for k, v in group.get("extra", {}).items()}
        if "gained" in extra and "lost" in extra:
            count(extra["gained"], "gained")
            count(extra["lost"], "lost")
        rows.append(dict(variant=name, source_pointer=group["rows"] + "/" + escape_key(key),
                         **counts, fp=fp, fn=fn, **rates, extra=extra))
    if len(rows) != group["expected_rows"]:
        raise EvidenceError("Expected {} rows, found {} in {}".format(group["expected_rows"], len(rows), group["id"]))
    if not rows:
        raise EvidenceError("An empty comparison cannot qualify")
    if "paired_baseline" in group:
        baseline = [row for row in rows if row["variant"] == group["paired_baseline"]]
        if len(baseline) != 1:
            raise EvidenceError("Paired baseline must be present in the same group")
        for row in rows:
            extra = row["extra"]
            if extra["gained"] - extra["lost"] != row["tp"] - baseline[0]["tp"]:
                raise EvidenceError("Gained/lost counts disagree with the paired baseline for " + row["variant"])
    metadata = {k: v for k, v in group.items() if k not in
                ("counts", "rows", "keys", "where", "name", "verify_rates", "extra", "checks")}
    return dict(schema_version=1, group=metadata, source_path=str(path),
                source_sha256=actual_hash, verification="source hash, declared population and registered rates; no model requalification",
                source_qualification=group["checks"], rows=rows)


def flat_rows(report):
    group = report["group"]
    for row in report["rows"]:
        result = {k: v for k, v in row.items() if k != "extra"}
        result.update(row["extra"])
        result.update(group_id=group["id"], result_class=group["result_class"],
                      dataset=group["dataset"], split=group["split"], metric=group["metric"],
                      protocol=group["protocol"], source_path=report["source_path"],
                      source_sha256=report["source_sha256"],
                      source_qualification=json.dumps(report["source_qualification"], sort_keys=True))
        yield result


def render(report, fmt):
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if fmt == "csv":
        rows = list(flat_rows(report))
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()
    group = report["group"]
    columns = ["variant", "tp", "predictions", "gt", "precision", "recall", "f1", "tn0"]
    columns += list(report["rows"][0]["extra"])
    def cell(value, metric):
        if value is None:
            return "UNRESOLVED"
        if metric in ("precision", "recall", "f1", "tn0"):
            return "{:.2%}".format(value)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# " + group["title"], "", group["result_class"], "",
             "{}; {}. {}".format(group["dataset"], group["split"], group["protocol"]), ""]
    if "paired_baseline" in group:
        lines += ["Gained/lost baseline: `" + group["paired_baseline"] + "`.", ""]
    if "performance_scope" in group:
        lines += ["Performance scope: " + group["performance_scope"], ""]
    lines += [
             "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in report["rows"]:
        values = dict(row, **row["extra"])
        lines.append("| " + " | ".join(cell(values[k], k) for k in columns) + " |")
    lines += ["", "P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).",
              "", "Source: [saved report](<{}>); SHA256 `{}`.".format(report["source_path"], report["source_sha256"]), ""]
    lines += ["- " + note for note in group["limitations"]]
    return "\n".join(lines) + "\n"


def overview(registry, roots):
    """Render separate reviewed groups, with no merged ranking or metric conversion."""
    by_id = {group["id"]: group for group in registry["groups"]}
    lines = ["# Saved model comparisons", "", "Approximate reproduction — read-only reporting snapshot.", "",
             "Each table retains its own dataset, matching, ignore policy and cost boundary.",
             "No global winner or paper reproduction is inferred. These are existing saved results; no new inference.", "",
             "All groups: `python3 -m research groups`. Usage and limits: [research guide](README.md).", ""]
    for group_id in registry["overview_groups"]:
        if group_id not in by_id:
            raise EvidenceError("Unknown overview group: " + group_id)
        report = compare(by_id[group_id], roots)
        part = render(report, "markdown")
        lines.append("#" + part)
    return "\n".join(lines)
