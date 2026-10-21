"""Build the English experiment library; optionally refresh verified result summaries.

Navigation uses only source files. --refresh-results requires the original pinned
reports and explicit external roots, and performs no model inference.
"""

import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from research.comparison import EvidenceError, compare, digest, load_registry, render
from research.library import check_saved_views, saved_result_markdown


ROOT = Path(__file__).resolve().parents[1]
META = {"README", "AUDIT_INDEX", "RESULTS_OVERVIEW", "REFACTOR_AUDIT", "GETTING_STARTED",
        "REPOSITORY_MAP", "EXTERNAL_ASSETS", "HOST_OPERATIONS", "PAPER_AMBIGUITIES",
        "ENGLISH_REFACTOR", "LANGUAGE_POLICY"}
CODE_ROOTS = ("caption", "cli", "deepsolo", "diagnostics", "localize", "models",
              "pipeline", "recognition", "research", "scoring", "scripts", "segmentation", "tests")


def encode(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def generate_navigation(root):
    organization = json.loads((root / "experiments/families.json").read_text())
    families = organization["families"]
    registry = load_registry(root / "research/experiments.json")
    records = []
    for guide in sorted((root / "docs").glob("*.md")):
        if guide.stem in META:
            continue
        text = guide.read_text(encoding="utf-8")
        title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), guide.stem)
        family = next((key for key, spec in families.items() if any(
            guide.stem.casefold().startswith(prefix.casefold()) for prefix in spec["prefixes"])), "analysis")
        referenced = set(re.findall(r"(?:caption|cli|configs|deepsolo|diagnostics|localize|models|pipeline|recognition|research|scoring|scripts|segmentation|tests)/[A-Za-z0-9_./ -]+?\.(?:py|sh|json|ya?ml)\b", text))
        referenced.update(module.replace(".", "/") + ".py" for module in re.findall(
            r"(?:-m\s+)((?:scripts|segmentation|diagnostics|research)\.[A-Za-z0-9_.]+)", text))
        for name in organization.get("guide_links", {}).get(guide.stem, []):
            if not (root / name).is_file():
                raise EvidenceError("Missing explicit guide link: " + name)
            referenced.add(name)
        paths = sorted(name for name in referenced if (root / name).is_file() and ".." not in Path(name).parts)
        groups = [group["id"] for group in registry["groups"] if guide.relative_to(root).as_posix() in group.get("evidence", [])]
        records.append(dict(id=re.sub(r"[^a-z0-9]+", "-", guide.stem.lower()).strip("-"),
                            title=title, family=family, guide=guide.relative_to(root).as_posix(),
                            guide_sha256=digest(guide.read_bytes()),
                            code=[name for name in paths if not name.startswith(("configs/", "tests/"))],
                            configs=[name for name in paths if name.startswith("configs/")],
                            tests=[name for name in paths if name.startswith("tests/")], result_groups=groups))
    if len({row["id"] for row in records}) != len(records):
        raise EvidenceError("Experiment guide IDs are not unique")
    modules = []
    for package in CODE_ROOTS:
        for path in sorted((root / package).rglob("*.py")):
            if "__pycache__" in path.parts or path.is_symlink():
                continue
            summary = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
            relative = path.relative_to(root).as_posix()
            modules.append(dict(path=relative, package=package, summary=summary.splitlines()[0] if summary else "No module docstring; see linked source and tests",
                                experiments=[row["id"] for row in records if relative in row["code"] + row["tests"]]))
    catalog = dict(schema_version=1, scope="Navigation only; dated guides/audits establish experimental status",
                   families={key: value["title"] for key, value in families.items()}, experiments=records, modules=modules)
    outputs = {"experiments/catalog.json": encode(catalog)}
    overview = ["# Experiment library", "", "Browse a family, then follow its guide to code, configurations, tests, and saved results.",
                "The catalog does not infer completion or merge incompatible protocols.", "",
                "| Family | Guides | Scope |", "| --- | ---: | --- |"]
    for key, spec in families.items():
        rows = [row for row in records if row["family"] == key]
        overview.append("| [{}]({}/README.md) | {} | {} |".format(key, key, len(rows), spec["title"]))
        lines = ["# " + spec["title"], "", "[All families](../README.md) · [Code index](../CODE_INDEX.md) · [Saved results](../results/README.md)", "",
                 "Category membership is navigation, not a shared evaluation protocol or a completion claim.", ""]
        for row in rows:
            lines += ["## " + row["title"], "", "ID: `" + row["id"] + "`. [Guide](<../../" + row["guide"] + ">).", ""]
            for field, label in (("code", "Code"), ("configs", "Configuration"), ("tests", "Tests")):
                if row[field]:
                    lines += [label + ": " + ", ".join("[" + name + "](<../../" + name + ">)" for name in row[field]) + ".", ""]
            if row["result_groups"]:
                lines += ["Numerical results: " + ", ".join("[" + group + "](../results/" + group + ".md)" for group in row["result_groups"]) + ".", ""]
            else:
                lines += ["Results/status: read the dated guide and its source audits; no numerical comparison registered here.", ""]
        outputs["experiments/" + key + "/README.md"] = "\n".join(lines)
    overview += ["", "```bash", "python3 -m research experiments --family word-instances", "python3 -m research experiment mask-crop-three-tests",
                 "python3 -m research results native128-pad0", "```", "",
                 "[Code by package](CODE_INDEX.md) · [Saved numerical results](results/README.md) · [Language and evidence policy](../docs/LANGUAGE_POLICY.md)", "",
                 "Regenerate navigation: `python3 -m scripts.build_experiment_library`. Add `--check` to detect stale views.", "",
                 "Refresh numerical summaries only with the original reports: `--refresh-results --runs-root PATH --artifacts-root PATH`.",
                 "A missing source remains an error. No training or inference runs.", ""]
    outputs["experiments/README.md"] = "\n".join(overview)
    lines = ["# Code index", "", "Modules grouped by package. Summaries come from source docstrings; no module is imported.", "",
             "[Experiment families](README.md) · [Repository responsibilities](../docs/REPOSITORY_MAP.md)", ""]
    for package in CODE_ROOTS:
        lines += ["## " + package, "", "| Source | Purpose | Experiment guides |", "| --- | --- | --- |"]
        for row in modules:
            if row["package"] == package:
                lines.append("| [{}](../{}) | {} | {} |".format(row["path"], row["path"], row["summary"].replace("|", "\\|"), ", ".join(row["experiments"]) or "See source/tests"))
        lines.append("")
    outputs["experiments/CODE_INDEX.md"] = "\n".join(lines)
    return outputs


def generate_results(root, runs_root, artifacts_root):
    registry_path = root / "research/experiments.json"
    registry = load_registry(registry_path)
    roots = dict(repository=root, runs=runs_root, external_artifacts=artifacts_root)
    # Validate every source before writing anything; never publish a partial set.
    reports = [compare(group, roots) for group in registry["groups"]]
    outputs, entries = {}, []
    for report in reports:
        group = report["group"]
        report["source_path"] = group["source"]["root"] + ":" + group["source"]["path"]
        report["summary_scope"] = "Saved numerical projection; no new evaluation; original report hash checked at generation"
        text = encode(report)
        path = "experiments/results/" + group["id"]
        outputs[path + ".json"] = text
        outputs[path + ".csv"] = render(report, "csv")
        outputs[path + ".md"] = saved_result_markdown(report)
        entries.append(dict(id=group["id"], path=path + ".json", sha256=digest(text.encode()),
                            rows=len(report["rows"]), source=group["source"]))
    manifest = dict(schema_version=1, captured_utc=datetime.now(timezone.utc).isoformat(),
                    registry_sha256=digest(registry_path.read_bytes()), groups=entries)
    outputs["experiments/results/manifest.json"] = encode(manifest)
    lines = ["# Saved numerical results", "", "Approximate reproduction. These are small projections of hash-verified saved reports, not new evaluations.",
             "Each table keeps its own protocol, denominator, split, ignore policy, and cost limitations.",
             "Original per-sample outputs/checkpoints remain external; no global ranking is valid.", "",
             "| Comparison | Rows | Dataset / split | Metric |", "| --- | ---: | --- | --- |"]
    for report in reports:
        group = report["group"]
        lines.append("| [{}]({}.md) · [JSON]({}.json) · [CSV]({}.csv) | {} | {} / {} | {} |".format(
            group["id"], group["id"], group["id"], group["id"], len(report["rows"]), group["dataset"], group["split"], group["metric"]))
    lines += ["", "Read offline: `python3 -m research results GROUP`.",
              "Recheck original sources: `python3 -m research compare GROUP` with the required external roots.",
              "[Restore original evidence](../../docs/EXTERNAL_ASSETS.md#restore-saved-comparisons).", ""]
    outputs["experiments/results/README.md"] = "\n".join(lines)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--refresh-results", action="store_true")
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--artifacts-root", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.repo_root.resolve()
        if args.check and args.refresh_results:
            raise EvidenceError("Use --check or --refresh-results, not both")
        outputs = generate_navigation(root)
        if args.refresh_results:
            outputs.update(generate_results(root, args.runs_root, args.artifacts_root))
        if args.check:
            stale = [name for name, text in outputs.items() if not (root / name).is_file() or (root / name).read_text(encoding="utf-8") != text]
            if stale:
                raise EvidenceError("Stale experiment library: " + ", ".join(stale))
            registry = load_registry(root / "research/experiments.json")
            for group in registry["groups"]:
                check_saved_views(root, group["id"])
        else:
            for name, text in outputs.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
        print("Experiment library " + ("verified" if args.check else "written") + ": " + str(len(outputs)) + " generated views")
        return 0
    except (EvidenceError, OSError, ValueError, KeyError, TypeError, SyntaxError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
